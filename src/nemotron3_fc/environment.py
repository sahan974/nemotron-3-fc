"""Platform and isolated runtime contracts for training and serving."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Platform = Literal["kaggle", "gpu-machine"]
Role = Literal["training", "serving"]


@dataclass(frozen=True)
class RuntimeContract:
    """Resolved interpreter/runtime location for one platform and one role."""

    platform: Platform
    role: Role
    runtime_dir: Path
    wheel_dirs: tuple[Path, ...]
    profile_path: Path


def detect_platform(requested: str = "auto") -> Platform:
    """Resolve an explicit platform or detect Kaggle without importing GPU libraries."""
    if requested not in {"auto", "kaggle", "gpu-machine"}:
        raise ValueError(f"Unsupported platform: {requested}")
    if requested != "auto":
        return requested  # type: ignore[return-value]

    # Either marker identifies Kaggle across interactive and scheduled execution.
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/input").is_dir():
        return "kaggle"
    return "gpu-machine"


def load_runtime_contract(
    repo_root: Path, role: Role, requested_platform: str = "auto", profile_path: Path | None = None
) -> RuntimeContract:
    """Load one role from a platform profile and resolve its local paths."""
    platform = detect_platform(requested_platform)
    profile = profile_path or repo_root / "configs" / "platforms" / (
        "kaggle.json" if platform == "kaggle" else "gpu-machine.example.json"
    )
    profile = profile.resolve()
    data = json.loads(profile.read_text(encoding="utf-8"))
    if data.get("platform") != platform:
        raise ValueError(f"Profile {profile} declares platform={data.get('platform')!r}, expected {platform!r}")
    if role not in data:
        raise ValueError(f"Profile {profile} has no {role!r} runtime")
    entry = data[role]

    # Relative profile entries resolve against the repository. Absolute entries
    # support externally mounted artifacts and machine-local runtimes.
    def resolve(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (repo_root / path).resolve()

    return RuntimeContract(
        platform=platform,
        role=role,
        runtime_dir=resolve(entry["runtime_dir"]),
        wheel_dirs=tuple(resolve(value) for value in entry.get("wheel_dirs", [])),
        profile_path=profile,
    )
