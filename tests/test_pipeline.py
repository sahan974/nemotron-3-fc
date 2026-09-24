import json
from pathlib import Path

from nemotron3_fc.cli import default_registry
from nemotron3_fc.data.io import read_jsonl
from nemotron3_fc.data.pipeline import load_preparation_config, prepare_data


def _xlam_record(index: int) -> dict:
    word = chr(ord("a") + index)
    tool = {
        "name": f"lookup_{word}",
        "description": f"Lookup {word}",
        "parameters": {"query": {"type": "str"}},
    }
    return {
        "id": f"source-{word}",
        "query": f"ask {word}",
        "tools": json.dumps([tool]),
        "answers": json.dumps([{"name": tool["name"], "arguments": {"query": word}}]),
    }


def test_prepare_data_writes_verified_nonempty_splits(tmp_path: Path):
    source = tmp_path / "xlam.json"
    source.write_text(json.dumps([_xlam_record(index) for index in range(20)]), encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "output_dir": "prepared",
                "seed": 2026,
                "splits": {"train": 0.6, "validation": 0.2, "test": 0.2},
                "sources": [{"name": "xlam", "adapter": "xlam", "path": "xlam.json", "expected_records": 20}],
            }
        ),
        encoding="utf-8",
    )

    config = load_preparation_config(config_path)
    report = prepare_data(config, default_registry())

    assert config.output_dir == tmp_path / "prepared"
    assert report["leakage_checks"] == {
        "cross_split_exact_queries": 0,
        "cross_split_offered_tool_definitions": 0,
        "cross_split_query_templates": 0,
    }
    counts = []
    for split in ("train", "validation", "test"):
        rows = list(read_jsonl(config.output_dir / "xlam" / f"{split}.jsonl"))
        assert rows
        counts.extend(row["id"] for row in rows)
    assert len(counts) == len(set(counts)) == 20
    manifest = json.loads((config.output_dir / "xlam" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["raw_records"] == 20
    assert manifest["files"]["rejected.jsonl"]["records"] == 0
