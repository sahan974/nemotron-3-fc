#!/usr/bin/env python3
"""Bootstrap the correct isolated environment, then run a project command."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from nemotron3_fc.artifacts import load_artifact_manifest, resolve_artifacts, write_artifact_map
from nemotron3_fc.environment import RuntimeContract, detect_platform, load_runtime_contract
from nemotron3_fc.paths import ARTIFACT_MAP_ENV

ROLE_BY_COMMAND = {
    "train": "training",
    "evaluate": "serving",
    "serve": "serving",
    "verify-serving": "serving",
    "benchmark-serving": "serving",
}
PACKAGES = {
    "training": [
        "torch==2.10.0",
        "transformers==5.5.0",
        "accelerate==1.15.0",
        "peft==0.21.0",
        "mamba-ssm==2.3.1",
        "causal-conv1d==1.6.1",
    ],
    "serving": [
        "vllm==0.18.0",
        "torch==2.10.0+cu128",
        "torchaudio==2.10.0+cu128",
        "torchvision==0.25.0+cu128",
        "transformers==4.57.6",
    ],
}
MARKER_NAME = ".nemotron3-fc-environment.json"
RUNTIME_PROBES = {
    "training": """
import torch
import transformers
import accelerate
import peft
import mamba_ssm
import causal_conv1d

assert torch.cuda.is_available()
print(torch.__version__, transformers.__version__, peft.__version__)
""".strip(),
    "serving": """
import torch
import transformers
import vllm

assert torch.cuda.is_available()
print(torch.__version__, transformers.__version__, vllm.__version__)
""".strip(),
}


# Command interface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare-data", "check-serving", *sorted(ROLE_BY_COMMAND)])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--platform", choices=["auto", "kaggle", "gpu-machine"], default="auto")
    parser.add_argument("--platform-profile", type=Path)
    parser.add_argument("--artifact-manifest", type=Path, default=REPO_ROOT / "configs" / "artifacts.json")
    parser.add_argument("--artifact-cache", type=Path)
    parser.add_argument("--artifact-source", choices=["auto", "mounted", "kaggle"], default="auto")
    parser.add_argument("--package-source", choices=["auto", "mounted", "kaggle", "index"], default="auto")
    parser.add_argument("--force-artifacts", action="store_true")
    parser.add_argument("--rebuild-environment", action="store_true")
    parser.add_argument("--bootstrap-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, help="Override the serving report and log directory")
    return parser.parse_args()


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


# Runtime lifecycle


def contract_fingerprint(contract: RuntimeContract) -> str:
    requirements = REPO_ROOT / "environments" / contract.role / "requirements.txt"
    payload = {
        "role": contract.role,
        "requirements": requirements.read_text(encoding="utf-8"),
        "wheel_dirs": [str(path) for path in contract.wheel_dirs],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_wheel_dirs(contract: RuntimeContract) -> None:
    missing = [str(path) for path in contract.wheel_dirs if not path.is_dir() or not any(path.rglob("*.whl"))]
    if missing:
        raise FileNotFoundError("Missing wheel-containing directories: " + ", ".join(missing))


def wheel_search_dirs(roots: tuple[Path, ...]) -> list[Path]:
    """Return every concrete directory containing wheels below artifact roots."""
    return sorted({wheel.parent for root in roots for wheel in root.rglob("*.whl")})


def reset_managed_runtime(contract: RuntimeContract) -> Path:
    """Reset a runtime directory after verifying launcher ownership."""
    runtime = contract.runtime_dir.resolve()
    marker = runtime / MARKER_NAME
    protected = {Path(runtime.anchor), Path.home().resolve(), REPO_ROOT.resolve()}
    if runtime in protected:
        raise RuntimeError(f"Refusing unsafe runtime directory: {runtime}")
    if runtime.exists() and any(runtime.iterdir()):
        managed = False
        if marker.is_file():
            try:
                managed = json.loads(marker.read_text(encoding="utf-8")).get("managed_by") == "nemotron3-fc"
            except (OSError, ValueError):
                managed = False
        if contract.platform == "kaggle" and Path("/kaggle/working") in runtime.parents:
            managed = True
        if not managed:
            raise RuntimeError(f"Refusing to replace non-project directory {runtime}; select a new runtime_dir")
        shutil.rmtree(runtime)
    runtime.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"managed_by": "nemotron3-fc", "state": "installing", "role": contract.role}, indent=2),
        encoding="utf-8",
    )
    return marker


def kaggle_bootstrap(contract: RuntimeContract, rebuild: bool) -> Path:
    """Install a role-specific Kaggle target directory strictly from attached wheels."""
    validate_wheel_dirs(contract)
    marker = contract.runtime_dir / MARKER_NAME
    fingerprint = contract_fingerprint(contract)

    # Runtime reuse requires an exact match of pinned requirements and wheel locations.
    if marker.is_file() and not rebuild:
        state = json.loads(marker.read_text(encoding="utf-8"))
        if state.get("fingerprint") == fingerprint:
            print(f"Reusing validated {contract.role} runtime: {contract.runtime_dir}", flush=True)
            return Path(sys.executable)
    marker = reset_managed_runtime(contract)

    # Kaggle preloads packages that conflict with the isolated pinned runtime.
    if contract.role == "training":
        run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "torchao"])

    # Wheel filenames are normalized in a staging directory before offline installation.
    with tempfile.TemporaryDirectory(prefix="nemotron3-fc-wheels-", dir="/kaggle/working") as temporary:
        staging = Path(temporary)
        serving_torch: Path | None = None
        source_dirs = contract.wheel_dirs if contract.role == "training" else contract.wheel_dirs[:1]

        for wheel_dir in source_dirs:
            for wheel in wheel_dir.rglob("*.whl"):
                filename = wheel.name

                if contract.role == "serving":
                    if filename.startswith("torch-"):
                        continue
                    filename = filename.replace("torchaudio-2.10.0cu128-", "torchaudio-2.10.0+cu128-")
                    filename = filename.replace("torchvision-0.25.0cu128-", "torchvision-0.25.0+cu128-")
                destination = staging / filename

                if not destination.exists():
                    try:
                        destination.symlink_to(wheel)
                    except OSError:
                        shutil.copy2(wheel, destination)

        if contract.role == "serving":
            candidates = [
                wheel for wheel_dir in contract.wheel_dirs[1:] for wheel in wheel_dir.rglob("torch-2.10.0+cu128-*.whl")
            ]

            if len(candidates) == 1:
                serving_torch = candidates[0]

            if serving_torch is None:
                raise RuntimeError(
                    "The serving runtime requires the torch-2.10.0+cu128 wheel from the training-wheel dataset"
                )

            destination = staging / serving_torch.name

            try:
                destination.symlink_to(serving_torch)
            except OSError:
                shutil.copy2(serving_torch, destination)
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-index",
                "--find-links",
                str(staging),
                "--target",
                str(contract.runtime_dir),
                "--ignore-installed",
                "--upgrade",
                *PACKAGES[contract.role],
            ]
        )

    # Runtime publication requires successful role-specific imports and CUDA validation.
    run([sys.executable, "-c", RUNTIME_PROBES[contract.role]], env=runtime_environment(contract))

    marker.write_text(
        json.dumps(
            {
                "managed_by": "nemotron3-fc",
                "state": "ready",
                "fingerprint": fingerprint,
                "platform": contract.platform,
                "role": contract.role,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return Path(sys.executable)


def venv_python(runtime_dir: Path) -> Path:
    return runtime_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def gpu_machine_bootstrap(contract: RuntimeContract, rebuild: bool) -> Path:
    """Create or reuse a dedicated virtual environment on Vast.ai or another GPU host."""
    python = venv_python(contract.runtime_dir)
    marker = contract.runtime_dir / MARKER_NAME
    fingerprint = contract_fingerprint(contract)

    reusable = (
        python.is_file()
        and marker.is_file()
        and json.loads(marker.read_text(encoding="utf-8")).get("fingerprint") == fingerprint
    )

    if reusable and not rebuild:
        print(f"Reusing validated {contract.role} environment: {contract.runtime_dir}", flush=True)
        return python

    marker = reset_managed_runtime(contract)
    run([sys.executable, "-m", "venv", str(contract.runtime_dir)])
    python = venv_python(contract.runtime_dir)

    requirement_file = REPO_ROOT / "environments" / contract.role / "requirements.txt"
    install = [str(python), "-m", "pip", "install", "-r", str(requirement_file)]

    if contract.wheel_dirs:
        validate_wheel_dirs(contract)
        install[4:4] = [
            "--no-index",
            *sum((["--find-links", str(path)] for path in wheel_search_dirs(contract.wheel_dirs)), []),
        ]
    else:
        install.extend(["--extra-index-url", "https://download.pytorch.org/whl/cu128"])
    run(install)

    run([str(python), "-m", "pip", "install", "--no-deps", "-e", str(REPO_ROOT)])
    run([str(python), "-c", RUNTIME_PROBES[contract.role]])

    marker.write_text(
        json.dumps(
            {
                "managed_by": "nemotron3-fc",
                "state": "ready",
                "fingerprint": fingerprint,
                "platform": contract.platform,
                "role": contract.role,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return python


def runtime_environment(contract: RuntimeContract, artifact_map: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    prefixes = [str(SRC_ROOT)]

    if contract.platform == "kaggle":
        prefixes.insert(0, str(contract.runtime_dir))
        env.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONNOUSERSITE": "1",
                "USE_TF": "0",
                "USE_FLAX": "0",
            }
        )
        if contract.role == "serving":
            env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    existing = env.get("PYTHONPATH")

    env["PYTHONPATH"] = os.pathsep.join(prefixes + ([existing] if existing else []))

    if artifact_map is not None:
        env[ARTIFACT_MAP_ENV] = str(artifact_map)
    return env


# Artifact and package resolution

def configure_package_source(contract: RuntimeContract, requested: str, specs: dict, force: bool) -> RuntimeContract:
    """Choose attached wheels, Kaggle-downloaded wheels, or normal package indexes."""
    source = ("mounted" if contract.platform == "kaggle" else "index") if requested == "auto" else requested

    if source == "index":
        if contract.platform == "kaggle":
            raise RuntimeError(
                "Package-index installation is unavailable in the offline GPU environment. "
                "Use mounted or downloaded wheel bundles."
            )
        return replace(contract, wheel_dirs=())

    if source == "mounted":
        if not contract.wheel_dirs:
            raise RuntimeError(
                f"The {contract.platform} profile has no mounted wheel_dirs for the {contract.role} role"
            )
        validate_wheel_dirs(contract)
        return contract

    names = ["training-wheels"] if contract.role == "training" else ["serving-wheels", "training-wheels"]
    wheels = resolve_artifacts(specs, names, "kaggle", force)

    return replace(contract, wheel_dirs=tuple(wheels[name] for name in names))


def config_artifact_names(path: Path) -> list[str]:
    """Collect only artifact references actually used by a run configuration."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    names: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, str) and value.startswith("artifact://"):
            names.add(value[len("artifact://") :].split("/", 1)[0])
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    visit(raw)
    return sorted(names)


# Command dispatch

def main() -> int:
    args = parse_args()
    if args.output_dir and args.command not in {"serve", "verify-serving", "check-serving", "benchmark-serving"}:
        raise ValueError("--output-dir applies only to serving commands")
    cache = args.artifact_cache.resolve() if args.artifact_cache else None
    specs = load_artifact_manifest(args.artifact_manifest.resolve(), REPO_ROOT, cache)

    if args.command in {"prepare-data", "check-serving"}:
        if args.bootstrap_only:
            raise ValueError("--bootstrap-only applies only to GPU runtime commands")

        platform = detect_platform(args.platform)
        resolved = resolve_artifacts(
            specs, config_artifact_names(args.config.resolve()), args.artifact_source, args.force_artifacts
        )

        map_root = Path("/kaggle/working") if platform == "kaggle" else (cache or REPO_ROOT / ".artifacts")
        artifact_map = write_artifact_map(map_root / "resolved.json", resolved)

        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join([str(SRC_ROOT), *([existing] if existing else [])])
        env[ARTIFACT_MAP_ENV] = str(artifact_map)

        command = [sys.executable, "-m", "nemotron3_fc.cli", args.command, "--config", str(args.config.resolve())]
        if args.output_dir:
            command.extend(["--output-dir", str(args.output_dir.resolve())])
        run(command, env=env)

        return 0

    role = ROLE_BY_COMMAND[args.command]
    profile = args.platform_profile.resolve() if args.platform_profile else None

    contract = load_runtime_contract(REPO_ROOT, role, args.platform, profile)
    contract = configure_package_source(contract, args.package_source, specs, args.force_artifacts)

    print(
        f"Platform: {contract.platform} | role: {contract.role} | profile: {contract.profile_path}",
        flush=True,
    )

    print(f"Artifact source: {args.artifact_source} | package source: {args.package_source}", flush=True)
    print(f"Runtime: {contract.runtime_dir}", flush=True)

    python = (
        kaggle_bootstrap(contract, args.rebuild_environment)
        if contract.platform == "kaggle"
        else gpu_machine_bootstrap(contract, args.rebuild_environment)
    )

    if args.bootstrap_only:
        return 0

    resolved = resolve_artifacts(
        specs, config_artifact_names(args.config.resolve()), args.artifact_source, args.force_artifacts
    )

    map_root = Path("/kaggle/working") if contract.platform == "kaggle" else (cache or REPO_ROOT / ".artifacts")

    artifact_map = write_artifact_map(map_root / "resolved.json", resolved)

    command = [str(python), "-m", "nemotron3_fc.cli", args.command, "--config", str(args.config.resolve())]
    if args.output_dir:
        command.extend(["--output-dir", str(args.output_dir.resolve())])
    run(command, env=runtime_environment(contract, artifact_map))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
