import json
from pathlib import Path

import pytest

from nemotron3_fc.environment import detect_platform, load_runtime_contract


def test_explicit_platform_selection() -> None:
    assert detect_platform("kaggle") == "kaggle"
    assert detect_platform("gpu-machine") == "gpu-machine"
    with pytest.raises(ValueError):
        detect_platform("unknown")


def test_runtime_contract_keeps_roles_separate(tmp_path: Path) -> None:
    profile = tmp_path / "platform.json"
    profile.write_text(
        json.dumps(
            {
                "platform": "gpu-machine",
                "training": {"runtime_dir": ".train", "wheel_dirs": []},
                "serving": {"runtime_dir": ".serve", "wheel_dirs": []},
            }
        ),
        encoding="utf-8",
    )
    training = load_runtime_contract(tmp_path, "training", "gpu-machine", profile)
    serving = load_runtime_contract(tmp_path, "serving", "gpu-machine", profile)
    assert training.runtime_dir == (tmp_path / ".train").resolve()
    assert serving.runtime_dir == (tmp_path / ".serve").resolve()
    assert training.runtime_dir != serving.runtime_dir
