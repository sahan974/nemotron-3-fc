import json

from nemotron3_fc.training.encoding import encode_record, record_windows
from nemotron3_fc.training.model import learning_rate_factor


class CharacterTokenizer:
    is_fast = True

    def apply_chat_template(self, messages, tools, tokenize, add_generation_prompt, **kwargs):
        del tools, tokenize, kwargs
        rendered = ""
        for message in messages:
            if message["role"] == "assistant":
                value = message.get("content") or json.dumps(message.get("tool_calls"), sort_keys=True)
                rendered += f"<a>{value}</a>"
            else:
                rendered += f"<{message['role']}>{message.get('content', '')}</{message['role']}>"
        if add_generation_prompt:
            rendered += "<a>"
        return rendered

    def __call__(self, text, add_special_tokens, return_offsets_mapping):
        assert not add_special_tokens and return_offsets_mapping
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _row():
    return {
        "id": "fixture:1",
        "source": "fixture",
        "tools": [],
        "messages": [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "second answer"},
        ],
    }


def test_assistant_only_mask_excludes_user_text():
    ids, labels, _ = encode_record(_row(), CharacterTokenizer())
    supervised_text = "".join(chr(token) for token, label in zip(ids, labels) if label != -100)
    assert "first answer" in supervised_text
    assert "second answer" in supervised_text
    assert "question" not in supervised_text


def test_overlapping_windows_supervise_each_target_once():
    ids, labels, _ = encode_record(_row(), CharacterTokenizer())
    windows = record_windows(_row(), "fixture", CharacterTokenizer(), max_tokens=45, overlap_tokens=10)
    assert len(windows) > 1
    assert sum(window.supervised for window in windows) == sum(value != -100 for value in labels[1:])
    assert all(window.multi_turn for window in windows)


def test_learning_rate_warms_up_then_stays_constant():
    assert learning_rate_factor(0, 100) == 0.01
    assert learning_rate_factor(99, 100) == 1.0
    assert learning_rate_factor(1000, 100) == 1.0
