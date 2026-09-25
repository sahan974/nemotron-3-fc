import json
from pathlib import Path
from types import SimpleNamespace

from nemotron3_fc.data.io import write_json, write_jsonl
from nemotron3_fc.training.config import load_training_config
from nemotron3_fc.training.data import BatchItem, make_epoch_batches, prepare_data, schedule_sha256
from nemotron3_fc.training.encoding import EncodedWindow


class CharacterTokenizer:
    is_fast = True

    def apply_chat_template(self, messages, tools, tokenize, add_generation_prompt, **kwargs):
        del tools, tokenize, kwargs
        rendered = "".join(
            f"<{message['role']}>{message.get('content', '')}</{message['role']}>" for message in messages
        )
        return rendered + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens, return_offsets_mapping):
        assert not add_special_tokens and return_offsets_mapping
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _window(index: int, source: str, tokens: int) -> EncodedWindow:
    return EncodedWindow(
        source=source,
        record_id=f"{source}:{index}",
        id=f"{source}:{index}#w0",
        input_ids=tuple(range(tokens)),
        labels=tuple(range(tokens)),
        tokens=tokens,
        supervised=tokens,
        no_call=False,
        multi_turn=False,
        has_tool_result=False,
    )


def test_epoch_batches_cover_every_window_once_and_do_not_mix_sources():
    windows = [
        _window(0, "toolace", 10),
        _window(1, "toolace", 12),
        _window(2, "xlam", 8),
        _window(3, "xlam", 9),
    ]
    groups = [[0], [1], [2], [3]]
    config = SimpleNamespace(seed=2026, length_bucket_records=128, max_batch_examples=4, max_batch_padded_tokens=100)
    batches = make_epoch_batches(windows, groups, 0, config)
    assert sorted(index for batch in batches for index in batch) == [0, 1, 2, 3]
    assert all(len({windows[index].source for index in batch}) == 1 for batch in batches)


def test_schedule_hash_changes_when_batch_identity_changes():
    first = BatchItem(source="toolace", id="a", indices=(0,), tokens=10, supervised=5, examples=1)
    second = BatchItem(source="toolace", id="b", indices=(1,), tokens=11, supervised=6, examples=1)
    schedule = ((0, 0), (0, 1))
    original = schedule_sha256(schedule, (first, second))
    changed = schedule_sha256(
        schedule,
        (first, BatchItem(source="toolace", id="changed", indices=(1,), tokens=11, supervised=6, examples=1)),
    )
    assert original != changed


def test_prepare_training_data_verifies_manifests_and_builds_both_validation_scopes(tmp_path: Path):
    data_root = tmp_path / "fixture-data"
    data_root.mkdir()

    def row(index: int) -> dict:
        return {
            "schema_version": "1.0",
            "id": f"fixture:{index}",
            "source": "fixture",
            "tools": [],
            "messages": [
                {"role": "user", "content": f"question {index}"},
                {"role": "assistant", "content": f"answer {index}"},
            ],
        }

    train_info = write_jsonl(data_root / "train.jsonl", [row(index) for index in range(12)])
    validation_info = write_jsonl(data_root / "validation.jsonl", [row(index + 100) for index in range(8)])
    write_json(
        data_root / "manifest.json",
        {"source": "fixture", "files": {"train.jsonl": train_info, "validation.jsonl": validation_info}},
    )
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "model_path": "model",
                "output_dir": "output",
                "epochs": 2,
                "best_score_weights": {"fixture": 1.0},
                "datasets": [
                    {
                        "name": "fixture",
                        "root": "fixture-data",
                        "train": True,
                        "evaluate": True,
                        "expected_train_records": 12,
                        "expected_validation_records": 8,
                        "monitor_validation_records": 4,
                    }
                ],
                "lora": {"target_modules": ["q_proj"]},
            }
        ),
        encoding="utf-8",
    )
    config = load_training_config(config_path)
    prepared = prepare_data(config, CharacterTokenizer())
    assert prepared.full_record_counts == {"fixture": 8}
    assert prepared.monitor_record_counts == {"fixture": 4}
    assert len({window.record_id for window in prepared.validation_full}) == 8
    assert len({window.record_id for window in prepared.validation_monitor}) == 4
    assert prepared.counts["updates_per_epoch"][0] in {
        prepared.counts["updates_per_epoch"][1],
        prepared.counts["updates_per_epoch"][1] + 1,
    }
    assert len(prepared.schedule) == sum(prepared.counts["updates_per_epoch"])
