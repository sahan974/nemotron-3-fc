import json
from pathlib import Path

from nemotron3_fc.artifacts import load_artifact_manifest, resolve_artifacts, write_artifact_map
from nemotron3_fc.paths import ARTIFACT_MAP_ENV, resolve_path


def test_mounted_artifact_and_portable_path(tmp_path: Path, monkeypatch) -> None:
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    (mounted / "test.jsonl").write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "artifacts.json"
    manifest.write_text(
        json.dumps(
            {
                "cache_root": "cache",
                "artifacts": {
                    "sample": {
                        "kind": "dataset",
                        "handle": "owner/sample",
                        "mounted_path": str(mounted),
                        "required": ["test.jsonl"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    specs = load_artifact_manifest(manifest, tmp_path)
    resolved = resolve_artifacts(specs, ["sample"], "mounted")
    mapping = write_artifact_map(tmp_path / "resolved.json", resolved)
    monkeypatch.setenv(ARTIFACT_MAP_ENV, str(mapping))
    assert resolve_path(tmp_path, "artifact://sample/test.jsonl", "test") == (mounted / "test.jsonl").resolve()
