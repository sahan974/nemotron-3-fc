"""Portable configuration paths, including resolved project artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ARTIFACT_MAP_ENV = "NEMOTRON3_FC_ARTIFACT_MAP"


def resolve_path(base: Path, value: Any, label: str) -> Path:
    """Resolve a normal path or an ``artifact://name/subpath`` reference."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a path string")

    # Launcher-generated artifact mappings decouple configuration from host paths.
    if value.startswith("artifact://"):
        reference = value[len("artifact://") :]
        name, separator, suffix = reference.partition("/")
        mapping_file = os.environ.get(ARTIFACT_MAP_ENV)
        if not mapping_file:
            raise RuntimeError(
                f"{label} uses {value!r}, but {ARTIFACT_MAP_ENV} is not set. "
                "Use scripts/run.py or scripts/fetch_artifacts.py."
            )
        mapping = json.loads(Path(mapping_file).read_text(encoding="utf-8"))
        if name not in mapping:
            raise KeyError(f"Artifact {name!r} is absent from {mapping_file}")
        root = Path(mapping[name]).resolve()
        result = (root / suffix).resolve() if separator else root

        # Artifact subpaths are constrained to the declared root.
        if result != root and root not in result.parents:
            raise ValueError(f"{label} escapes artifact root: {value!r}")
        return result
    candidate = Path(value).expanduser()
    return (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
