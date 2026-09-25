"""Resolve project inputs from mounted paths, local caches, or Kaggle."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ArtifactSource = Literal["auto", "mounted", "kaggle"]
ARTIFACT_MARKER = ".nemotron3-fc-artifact.json"


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    kind: str
    handle: str
    mounted_path: Path
    cache_path: Path
    required: tuple[str, ...]
    expand_zip: bool


def load_artifact_manifest(path: Path, repo_root: Path, cache_root: Path | None = None) -> dict[str, ArtifactSpec]:
    """Load portable artifact declarations without touching the network."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    root = (cache_root or repo_root / raw.get("cache_root", ".artifacts")).resolve()
    result: dict[str, ArtifactSpec] = {}

    # Mounted paths remain platform-specific inputs. Relative cache paths are
    # resolved against the configured repository cache root.
    for name, value in raw["artifacts"].items():
        mounted = Path(value["mounted_path"]).expanduser()
        cache = Path(value.get("cache_path", name)).expanduser()

        result[name] = ArtifactSpec(
            name=name,
            kind=value["kind"],
            handle=value["handle"],
            mounted_path=mounted,
            cache_path=cache if cache.is_absolute() else root / cache,
            required=tuple(value.get("required", [])),
            expand_zip=bool(value.get("expand_zip", False)),
        )

    return result


def artifact_is_valid(spec: ArtifactSpec, root: Path) -> bool:
    return root.is_dir() and all(any(root.glob(pattern)) for pattern in spec.required)


def _kaggle_command(spec: ArtifactSpec, destination: Path) -> list[str]:
    if spec.kind == "dataset":
        return [
            "kaggle",
            "datasets",
            "download",
            spec.handle,
            "--path",
            str(destination),
            "--unzip",
            "--force",
            "--quiet",
        ]
    if spec.kind == "model":
        return [
            "kaggle",
            "models",
            "variations",
            "versions",
            "download",
            spec.handle,
            "--path",
            str(destination),
            "--untar",
            "--force",
            "--quiet",
        ]
    if spec.kind == "notebook-output":
        return ["kaggle", "kernels", "output", spec.handle, "--path", str(destination)]
    raise ValueError(f"Unsupported artifact kind for {spec.name}: {spec.kind}")


def _clear_managed_destination(destination: Path) -> None:
    """Remove an old cache only when its marker proves this project owns it."""
    if not destination.exists():
        return

    if not any(destination.iterdir()):
        destination.rmdir()
        return

    # Recursive replacement requires the project ownership marker.
    marker = destination / ARTIFACT_MARKER
    if not marker.is_file():
        raise RuntimeError(f"Refusing to replace non-project artifact directory: {destination}")

    shutil.rmtree(destination)


def _extract_zip_archives(root: Path) -> None:
    """Expand downloaded wheel bundles after rejecting paths outside the extraction directory."""
    for archive in sorted(root.rglob("*.zip")):
        extraction = archive.parent / archive.stem
        extraction.mkdir(exist_ok=True)
        target = extraction.resolve()

        with zipfile.ZipFile(archive) as package:
            # Pre-extraction path validation prevents archive traversal.
            for member in package.infolist():
                member_path = (target / member.filename).resolve()
                if member_path != target and target not in member_path.parents:
                    raise RuntimeError(f"Unsafe archive member in {archive}: {member.filename}")

            package.extractall(extraction)


def _write_artifact_marker(staging: Path, spec: ArtifactSpec) -> None:
    marker = {
        "name": spec.name,
        "kind": spec.kind,
        "handle": spec.handle,
    }
    (staging / ARTIFACT_MARKER).write_text(json.dumps(marker, indent=2), encoding="utf-8")


def _download_from_kaggle(spec: ArtifactSpec, force: bool) -> Path:
    """Download one artifact into a validated, atomically published project cache."""
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle download requested but the 'kaggle' CLI is unavailable. "
            "Install the bootstrap client with: python -m pip install kaggle"
        )

    destination = spec.cache_path.resolve()
    if artifact_is_valid(spec, destination) and not force:
        return destination

    _clear_managed_destination(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Download into a sibling staging directory so publication can use an
    # atomic rename on the same filesystem.
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))

    try:
        command = _kaggle_command(spec, staging)
        print("FETCH:", " ".join(command), flush=True)
        subprocess.run(command, check=True)

        if spec.expand_zip and not artifact_is_valid(spec, staging):
            _extract_zip_archives(staging)

        if not artifact_is_valid(spec, staging):
            raise RuntimeError(f"Downloaded artifact {spec.name!r} is missing required content: {spec.required}")

        _write_artifact_marker(staging, spec)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return destination


def resolve_artifacts(
    specs: dict[str, ArtifactSpec], names: Iterable[str], source: ArtifactSource, force: bool = False
) -> dict[str, Path]:
    """Resolve named artifacts deterministically and validate their expected files."""
    resolved: dict[str, Path] = {}
    for name in names:
        if name not in specs:
            raise KeyError(f"Unknown artifact: {name}")
        spec = specs[name]
        # Automatic resolution order is mounted input, validated cache, then download.
        candidates = []
        if source in {"auto", "mounted"}:
            candidates.append(spec.mounted_path)
        if source == "auto":
            candidates.append(spec.cache_path)
        selected = next((path.resolve() for path in candidates if artifact_is_valid(spec, path)), None)
        if selected is None and source in {"auto", "kaggle"}:
            selected = _download_from_kaggle(spec, force)
        if selected is None:
            raise FileNotFoundError(f"Artifact {name!r} was not found at mounted path {spec.mounted_path}")
        resolved[name] = selected
        print(f"ARTIFACT {name} | {selected}", flush=True)
    return resolved


def write_artifact_map(path: Path, resolved: dict[str, Path]) -> Path:
    """Merge resolved artifact paths into the map consumed by run configurations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    existing.update({name: str(value) for name, value in resolved.items()})
    path.write_text(json.dumps(dict(sorted(existing.items())), indent=2), encoding="utf-8")
    return path
