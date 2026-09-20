"""Nemotron BF16 loading, LoRA attachment, batching, and optimizer setup."""

from __future__ import annotations

import math
import random
from typing import Any, Iterable, Mapping, Sequence

from nemotron3_fc.training.config import TrainingConfig
from nemotron3_fc.training.encoding import EncodedWindow


def load_tokenizer(config: TrainingConfig) -> Any:
    """Load the local fast tokenizer required for assistant-span offsets."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True, local_files_only=True)
    if not tokenizer.is_fast:
        raise RuntimeError("The tokenizer lacks offsets required for assistant-only masking")
    print(f"Native-chat tokenizer: {type(tokenizer).__name__}")
    return tokenizer


def build_training_model(config: TrainingConfig) -> Any:
    """Load the BF16 base model and attach the configured LoRA parameter set."""
    import numpy as np
    import torch
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    base = AutoModelForCausalLM.from_pretrained(config.model_path, dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True, device_map="cuda")
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    lora = config.lora
    adapted = get_peft_model(base, PeftLoraConfig(r=lora.rank, lora_alpha=lora.alpha, lora_dropout=lora.dropout, bias="none", target_modules=list(lora.target_modules), task_type="CAUSAL_LM"))
    counts = {
        target: sum(name.endswith("." + target) and hasattr(module, "lora_A") and "default" in module.lora_A for name, module in adapted.get_base_model().named_modules())
        for target in lora.target_modules
    }
    trainable = sum(parameter.numel() for parameter in adapted.parameters() if parameter.requires_grad)
    print(f"LoRA module counts: {counts} | trainable parameters: {trainable}")
    if lora.expected_module_counts is not None and counts != dict(lora.expected_module_counts):
        raise RuntimeError(f"LoRA module coverage changed: expected {dict(lora.expected_module_counts)}, found {counts}")
    if lora.expected_trainable_parameters is not None and trainable != lora.expected_trainable_parameters:
        raise RuntimeError(f"LoRA trainable parameter count changed: expected {lora.expected_trainable_parameters}, found {trainable}")
    return adapted


def collate_windows(windows: Sequence[EncodedWindow], indices: Sequence[int], pad_id: int) -> dict[str, Any]:
    """Right-pad independent conversations without training on padding."""
    import torch

    selected = [windows[index] for index in indices]
    width = max(item.tokens for item in selected)
    input_ids = torch.full((len(selected), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(selected), width), dtype=torch.long)
    labels = torch.full((len(selected), width), -100, dtype=torch.long)
    for row, item in enumerate(selected):
        length = item.tokens
        input_ids[row, :length] = torch.tensor(item.input_ids, dtype=torch.long)
        attention_mask[row, :length] = 1
        labels[row, :length] = torch.tensor(item.labels, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def gpu_batch(window: EncodedWindow) -> dict[str, Any]:
    import torch

    return {
        "input_ids": torch.tensor([window.input_ids], dtype=torch.long, device="cuda"),
        "attention_mask": torch.ones((1, window.tokens), dtype=torch.long, device="cuda"),
        "labels": torch.tensor([window.labels], dtype=torch.long, device="cuda"),
    }


def verify_batch_loss_parity(model: Any, windows: Sequence[EncodedWindow], pad_id: int) -> None:
    """Compare batched assistant-only loss with supervised-token-weighted singles."""
    import torch

    if len(windows) < 2:
        raise RuntimeError("At least two training windows are required for batch loss verification")
    ordered = sorted(range(len(windows)), key=lambda index: windows[index].tokens)
    indices = (ordered[len(ordered) // 4], ordered[3 * len(ordered) // 4])
    prior_mode = model.training
    model.eval()
    try:
        with torch.inference_mode():
            singles = [float(model(**gpu_batch(windows[index])).loss.item()) for index in indices]
            combined = {key: value.to("cuda") for key, value in collate_windows(windows, indices, pad_id).items()}
            batched = float(model(**combined).loss.item())
        weights = [windows[index].supervised for index in indices]
        reference = sum(loss * weight for loss, weight in zip(singles, weights)) / sum(weights)
        difference = abs(batched - reference)
        print(f"BATCH LOSS PARITY weighted_singles={reference} batch={batched} difference={difference}")
        if difference > 0.01 + 0.02 * abs(reference):
            raise RuntimeError("Batched loss differs materially from individual losses")
    finally:
        model.train(prior_mode)


def adapter_parameters(model: Any) -> dict[str, Any]:
    return {name: parameter for name, parameter in model.named_parameters() if ".lora_A.default." in name or ".lora_B.default." in name}


def snapshot_adapter(model: Any) -> dict[str, Any]:
    return {name: parameter.detach().cpu().clone() for name, parameter in adapter_parameters(model).items()}


def restore_adapter(model: Any, snapshot: Mapping[str, Any]) -> None:
    import torch

    with torch.no_grad():
        live = adapter_parameters(model)
        if set(live) != set(snapshot):
            raise RuntimeError("Adapter snapshot does not match live LoRA parameters")
        for name, parameter in live.items():
            parameter.copy_(snapshot[name])


def load_exact_adapter(model: Any, weights_file: Any) -> int:
    """Map every safetensors LoRA tensor to exactly one live parameter."""
    from safetensors import safe_open
    import torch

    parameters, mapping = adapter_parameters(model), {}
    expected = set(parameters)
    with safe_open(str(weights_file), framework="pt", device="cpu") as saved:
        for saved_key in saved.keys():
            marker = ".lora_A." if ".lora_A." in saved_key else (".lora_B." if ".lora_B." in saved_key else None)
            if marker is None:
                raise RuntimeError(f"Unexpected adapter tensor: {saved_key}")
            renamed = saved_key.replace(marker, marker + "default.", 1)
            matches = [name for name in (renamed, "base_model.model." + renamed, "base_model." + renamed) if name in expected]
            if len(matches) != 1:
                raise RuntimeError(f"Missing or ambiguous adapter target: {saved_key}")
            target = matches[0]
            if target in mapping.values() or tuple(saved.get_slice(saved_key).get_shape()) != tuple(parameters[target].shape):
                raise RuntimeError(f"Duplicate or shape-mismatched tensor: {saved_key}")
            mapping[saved_key] = target
        if set(mapping.values()) != expected:
            raise RuntimeError(f"Adapter incomplete: {len(mapping)}/{len(expected)} tensors")
        with torch.no_grad():
            for saved_key, target in mapping.items():
                parameters[target].copy_(saved.get_tensor(saved_key))
    print(f"Strict adapter tensors loaded: {len(mapping)}")
    return len(mapping)


def learning_rate_factor(step: int, warmup_steps: int) -> float:
    """Warm up linearly to one, then remain constant without decay."""
    # LambdaLR receives completed optimizer steps; +1 gives the first update
    # its first non-zero warmup fraction.
    return min(1.0, (step + 1) / warmup_steps)


def new_optimizer_scheduler(model: Any, config: TrainingConfig) -> tuple[Any, Any]:
    """Create fused AdamW and the checkpointable warmup-then-constant schedule."""
    import torch

    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=config.learning_rate, weight_decay=config.weight_decay, fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: learning_rate_factor(step, config.warmup_steps))
    return optimizer, scheduler


def accumulate_microbatch(model: Any, windows: Sequence[EncodedWindow], indices: Sequence[int], pad_id: int, total_supervised: int) -> tuple[float, int]:
    batch = collate_windows(windows, indices, pad_id)
    padded_tokens = batch["input_ids"].numel()
    batch = {key: value.to("cuda") for key, value in batch.items()}
    loss = model(**batch).loss
    loss_value = float(loss.detach().item())
    if not math.isfinite(loss_value):
        raise FloatingPointError("Non-finite batched training loss")
    supervised = sum(windows[index].supervised for index in indices)
    # The model loss is a mean over supervised tokens. Scaling by each
    # microbatch's token share makes accumulation equivalent to one full batch
    # loss even when padding differs between microbatches.
    fraction = supervised / total_supervised
    (loss * fraction).backward()
    return loss_value * fraction, padded_tokens


def train_one_step(model: Any, windows: Sequence[EncodedWindow], indices: Sequence[int], pad_id: int, total_supervised: int, optimizer: Any, scheduler: Any, gradient_clip: float) -> dict[str, Any]:
    """Train one scheduled batch, reducing only its microbatch size after CUDA OOM."""
    import gc
    import time
    import torch

    model.set_adapter("default")
    model.train()
    micro_limit = len(indices)
    started = time.monotonic()
    while True:
        optimizer.zero_grad(set_to_none=True)
        optimizer_started = False
        try:
            weighted_loss, padded_tokens = 0.0, 0
            for offset in range(0, len(indices), micro_limit):
                chunk = indices[offset : offset + micro_limit]
                part_loss, part_padded = accumulate_microbatch(model, windows, chunk, pad_id, total_supervised)
                weighted_loss += part_loss
                padded_tokens += part_padded
            grad_norm = torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters() if parameter.requires_grad), gradient_clip, error_if_nonfinite=True)
            optimizer_started = True
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            return {
                "loss": weighted_loss,
                "padded_tokens": padded_tokens,
                "microbatches": math.ceil(len(indices) / micro_limit),
                "grad_norm": float(grad_norm.item()),
                "lr": scheduler.get_last_lr()[0],
                "seconds": time.monotonic() - started,
            }
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            if optimizer_started or micro_limit == 1:
                raise
            micro_limit = max(1, micro_limit // 2)
            gc.collect()
            torch.cuda.empty_cache()
            print(f"BATCH OOM microbatch_size={micro_limit}")
