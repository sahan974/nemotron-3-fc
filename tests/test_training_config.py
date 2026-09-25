import json
from pathlib import Path

import pytest

from nemotron3_fc.training.config import load_training_config


def _config(tmp_path: Path) -> dict:
    return {
        "model_path": "model",
        "output_dir": "output",
        "mode": "full",
        "epochs": 2,
        "best_score_weights": {"toolace": 0.8, "toolace:no_call": 0.2},
        "datasets": [
            {
                "name": "toolace",
                "root": "data",
                "train": True,
                "evaluate": True,
                "expected_train_records": 10,
                "expected_validation_records": 4,
                "monitor_validation_records": 2,
            }
        ],
        "lora": {"rank": 16, "alpha": 32, "target_modules": ["q_proj"]},
    }


def test_training_config_resolves_paths_and_preserves_notebook_schedule(tmp_path: Path):
    path = tmp_path / "train.json"
    path.write_text(json.dumps(_config(tmp_path)), encoding="utf-8")
    config = load_training_config(path)
    assert config.model_path == (tmp_path / "model").resolve()
    assert config.output_dir == (tmp_path / "output").resolve()
    assert config.learning_rate == 1e-4
    assert config.warmup_steps == 100
    assert config.monitor_seed == 3031
    assert config.epochs == 2
    assert config.train_datasets[0].name == "toolace"


def test_training_config_rejects_resume_without_both_artifacts(tmp_path: Path):
    raw = _config(tmp_path)
    raw.update({"start_from": "checkpoint", "previous_checkpoint": "checkpoint"})
    path = tmp_path / "train.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="require previous_checkpoint and previous_best"):
        load_training_config(path)


def test_training_identity_allows_completed_epoch_extension(tmp_path: Path):
    raw = _config(tmp_path)
    first = tmp_path / "first.json"
    first.write_text(json.dumps(raw), encoding="utf-8")
    first_config = load_training_config(first)
    raw["epochs"] = 5
    second = tmp_path / "second.json"
    second.write_text(json.dumps(raw), encoding="utf-8")
    second_config = load_training_config(second)
    assert first_config.identity_sha256() == second_config.identity_sha256()
