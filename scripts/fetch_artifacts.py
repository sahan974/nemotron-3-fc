#!/usr/bin/env python3
"""Resolve or download named project artifacts and write a reusable path map."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nemotron3_fc.artifacts import load_artifact_manifest, resolve_artifacts, write_artifact_map


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", help="Artifact names from configs/artifacts.json")
    parser.add_argument("--source", choices=["auto", "mounted", "kaggle"], default="auto")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "configs" / "artifacts.json")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / ".artifacts" / "resolved.json")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    specs = load_artifact_manifest(
        args.manifest.resolve(), REPO_ROOT, args.cache_root.resolve() if args.cache_root else None
    )

    resolved = resolve_artifacts(specs, args.names, args.source, args.force)
    output = write_artifact_map(args.output.resolve(), resolved)

    print("ARTIFACT MAP:", output, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
