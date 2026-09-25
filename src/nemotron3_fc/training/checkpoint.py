"""LoRA checkpoints with exact schedule and RNG restoration."""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

from nemotron3_fc.data.io import sha256_file, write_json
from nemotron3_fc.training.config import TrainingConfig
from nemotron3_fc.training.data import PreparedData, prefix_progress_hash, schedule_sha256
from nemotron3_fc.training.model import learning_rate_factor, load_exact_adapter, new_optimizer_scheduler


def capture_rng() -> dict[str, Any]:
    import numpy as np
    import torch

    # JSON-compatible RNG state is stored alongside optimizer and scheduler
    # state so resumption reproduces dropout and batch behavior exactly.
    numpy_state, python_state = np.random.get_state(), random.getstate()
    return {
        "python": [python_state[0], list(python_state[1]), python_state[2]],
        "numpy": [numpy_state[0], numpy_state[1].tolist(), numpy_state[2], numpy_state[3], numpy_state[4]],
        "torch_cpu": torch.get_rng_state().tolist(),
        "torch_cuda": [state.tolist() for state in torch.cuda.get_rng_state_all()],
    }


def restore_rng(state: dict[str, Any]) -> None:
    import numpy as np
    import torch

    random.setstate((state["python"][0], tuple(state["python"][1]), state["python"][2]))
    np.random.set_state(
        (
            state["numpy"][0],
            np.array(state["numpy"][1], dtype=np.uint32),
            state["numpy"][2],
            state["numpy"][3],
            state["numpy"][4],
        )
    )
    torch.set_rng_state(torch.tensor(state["torch_cpu"], dtype=torch.uint8))
    torch.cuda.set_rng_state_all([torch.tensor(item, dtype=torch.uint8) for item in state["torch_cuda"]])


def save_checkpoint(
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    config: TrainingConfig,
    data: PreparedData,
    position: int,
    tokens_processed: int,
    progress_hash: str,
    best_score: float,
) -> Path:
    """Save all continuation state before retiring an older complete checkpoint."""
    import peft
    import torch
    import transformers

    root = config.output_dir / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".checkpoint-step-{position:06d}.partial"
    complete = root / f"checkpoint-step-{position:06d}"
    if staging.exists() or complete.exists():
        raise FileExistsError(f"Checkpoint target already exists: {staging} or {complete}")
    free_gib = shutil.disk_usage(config.output_dir).free / 1024**3
    if free_gib < config.checkpoint_minimum_free_disk_gib:
        raise OSError(f"Only {free_gib:.2f} GiB free; retaining the previous checkpoint")
    staging.mkdir()
    try:
        model.save_pretrained(staging / "adapter", selected_adapters=["default"], safe_serialization=True)
        torch.save(optimizer.state_dict(), staging / "optimizer.pt")
        torch.save(scheduler.state_dict(), staging / "scheduler.pt")
        write_json(
            staging / "state.json",
            {
                "schema_version": 3,
                "next_position": position,
                "total_updates": len(data.schedule),
                "tokens_processed": tokens_processed,
                "progress_hash": progress_hash,
                "best_score_observed": best_score if math.isfinite(best_score) else None,
                "rng": capture_rng(),
                "source_run_number": config.run_number,
            },
        )
        # Hash the complete payload before publishing; the manifest is excluded
        # because it contains these hashes.
        files = {str(path.relative_to(staging)): sha256_file(path) for path in staging.rglob("*") if path.is_file()}
        write_json(
            staging / "manifest.json",
            {
                "schema_version": 3,
                "model_path": str(config.model_path),
                "training_identity_sha256": config.identity_sha256(),
                "schedule_sha256": data.schedule_sha256,
                "dataset_identities": data.dataset_identities,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peft": peft.__version__,
                "files": files,
            },
        )
        # Publishing is atomic on the same filesystem; incomplete checkpoints
        # retain the `.partial` name and are never considered resumable.
        staging.rename(complete)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # Retire an older checkpoint only after the replacement is complete and
    # only when its manifest proves it is a managed checkpoint directory.
    for old in root.iterdir():
        if old.is_dir() and old.name.startswith("checkpoint-step-") and old != complete:
            if old.resolve().parent != root.resolve() or not (old / "manifest.json").is_file():
                raise RuntimeError(f"Refusing unsafe checkpoint retirement: {old}")
            print(f"Retiring older working checkpoint: {old.name}")
            shutil.rmtree(old)
    remaining = shutil.disk_usage(config.output_dir).free / 1024**3
    print(f"CHECKPOINT step={position} path={complete} free_disk_gib={remaining:.2f}")
    return complete


def load_checkpoint(model: Any, config: TrainingConfig, data: PreparedData) -> tuple[dict[str, Any], Any, Any]:
    """Restore an exact schedule or extend only from a completed-epoch prefix."""
    import peft
    import torch
    import transformers

    checkpoint = config.previous_checkpoint
    if checkpoint is None:
        raise ValueError("No previous checkpoint configured")
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    manifest_schema = manifest.get("schema_version")
    if manifest_schema not in {2, 3} or manifest.get("model_path") != str(config.model_path):
        raise RuntimeError("Checkpoint model or schema does not match")
    if manifest_schema == 3 and manifest.get("training_identity_sha256") != config.identity_sha256():
        raise RuntimeError("Checkpoint training configuration differs from this run")
    if manifest_schema == 3 and manifest.get("dataset_identities") != data.dataset_identities:
        raise RuntimeError("Checkpoint dataset hashes or counts differ from this run")
    if manifest_schema == 2:
        # Legacy schema v2 omits repository identity fields. Compatibility is
        # therefore established through schedule, runtime, tensor, and state validation.
        print(
            "LEGACY CHECKPOINT schema=2. Validating model, packages, schedule, adapter tensors, "
            "optimizer, scheduler, and RNG state."
        )
    versions = (manifest.get("torch"), manifest.get("transformers"), manifest.get("peft"))
    current_versions = (torch.__version__, transformers.__version__, peft.__version__)
    if versions != current_versions:
        raise RuntimeError(f"Checkpoint stack {versions} differs from current stack {current_versions}")
    # Check both membership and content. Hashing only known files would miss an
    # unexpected payload inserted into the checkpoint directory.
    actual_files = {
        str(path.relative_to(checkpoint))
        for path in checkpoint.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual_files != set(manifest.get("files", {})):
        raise RuntimeError("Checkpoint file set changed")
    for relative, expected_hash in manifest["files"].items():
        if sha256_file(checkpoint / relative) != expected_hash:
            raise RuntimeError(f"Checkpoint file corrupted: {relative}")

    state = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    position, prior_total, total = state.get("next_position"), state.get("total_updates"), len(data.schedule)
    if (
        state.get("schema_version") != manifest_schema
        or not isinstance(position, int)
        or not isinstance(prior_total, int)
        or not (0 <= position <= prior_total <= total)
    ):
        raise RuntimeError("Invalid checkpoint position or schedule length")
    if state.get("source_run_number") != config.run_number:
        raise RuntimeError("Checkpoint belongs to another run number")
    if state.get("progress_hash") != prefix_progress_hash(data, position):
        raise RuntimeError("Checkpoint progress does not match the rebuilt batch order")
    # Resumption accepts either the identical schedule or a prior run ending at
    # a complete epoch boundary of the now-extended schedule.
    same_schedule = prior_total == total and manifest.get("schedule_sha256") == data.schedule_sha256
    completed_prefix = (
        0 < prior_total < total
        and position == prior_total
        and data.schedule[prior_total - 1][0] != data.schedule[prior_total][0]
        and manifest.get("schedule_sha256") == schedule_sha256(data.schedule, data.batch_items, prior_total)
    )
    if not same_schedule and not completed_prefix:
        raise RuntimeError("Checkpoint is neither the exact current schedule nor a completed-epoch prefix")

    load_exact_adapter(model, checkpoint / "adapter" / "adapter_model.safetensors")
    optimizer, scheduler = new_optimizer_scheduler(model, config)
    optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True))
    scheduler.load_state_dict(torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True))
    if scheduler.last_epoch != position:
        raise RuntimeError(f"Scheduler step {scheduler.last_epoch} does not match data step {position}")
    optimizer_lr, scheduler_lr = optimizer.param_groups[0]["lr"], scheduler.get_last_lr()[0]
    expected_lr = config.learning_rate * learning_rate_factor(position, config.warmup_steps)
    if not math.isclose(optimizer_lr, scheduler_lr, rel_tol=1e-6, abs_tol=1e-10):
        raise RuntimeError(f"Optimizer LR {optimizer_lr} differs from restored scheduler LR {scheduler_lr}")
    if not math.isclose(optimizer_lr, expected_lr, rel_tol=1e-5, abs_tol=1e-9):
        raise RuntimeError(f"Restored LR {optimizer_lr} differs from continuous schedule LR {expected_lr}")
    restore_rng(state["rng"])
    print(
        f"RUN RESTORED next_step={position} target_step={total} optimizer_lr={optimizer_lr} "
        f"optimizer_state_entries={len(optimizer.state)}"
    )
    print(f"SCHEDULE {'verified completed-epoch prefix' if completed_prefix else 'exact current schedule'}")
    return state, optimizer, scheduler


def initial_progress_hash() -> str:
    return hashlib.sha256(b"").hexdigest()
