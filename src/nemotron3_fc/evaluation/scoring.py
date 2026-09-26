"""BFCL-v3-style AST scoring adapted to canonical tool conversations.

This follows the BFCL evaluator reference implementation. Prompts and references
come from the selected canonical dataset, and native Nemotron calls require decoding.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

BFCL_SOURCE_COMMIT = "7c0efb120e9ce0bd016033a0a0c7f2154c8b8cc8"
TYPE_MAPPING = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "any": str,
}
NESTED_TYPES = {"array", "tuple"}
CALL_PATTERN = r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>"
PARAMETER_PATTERN = r"<parameter=([^>\n]+)>(.*?)</parameter>"


def clean_assistant_text(value: str) -> str:
    value = re.sub(r"<think>\s*</think>", "", value)
    for marker in ("<|im_end|>", "<|endoftext|>"):
        value = value.replace(marker, "")
    return value.strip()


def _normalize_parameter(value: str) -> str:
    value = value.strip()
    try:
        return json.dumps(json.loads(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError):
        return value


def parse_native_calls(text: str) -> dict[str, Any]:
    """Parse native calls for strict exact-call diagnostics."""
    cleaned = clean_assistant_text(text)
    # This parser intentionally retains normalized strings for strict native
    # exact-match diagnostics; BFCL scoring uses the typed parser below.
    matches = list(re.finditer(CALL_PATTERN, cleaned, flags=re.DOTALL))
    has_markup = "<tool_call" in cleaned or "<function=" in cleaned
    if not matches:
        return {
            "calls": [],
            "valid": not has_markup,
            "has_call": has_markup,
            "error": "malformed_tool_call" if has_markup else None,
        }
    calls = []
    for match in matches:
        body = match.group(2)
        parameter_matches = list(re.finditer(PARAMETER_PATTERN, body, flags=re.DOTALL))
        if re.sub(PARAMETER_PATTERN, "", body, flags=re.DOTALL).strip():
            return {"calls": calls, "valid": False, "has_call": True, "error": "unparsed_function_body"}
        parameters = {}
        for parameter in parameter_matches:
            name = parameter.group(1).strip()
            if name in parameters:
                return {"calls": calls, "valid": False, "has_call": True, "error": "duplicate_parameter"}
            parameters[name] = _normalize_parameter(parameter.group(2))
        calls.append({"name": match.group(1).strip(), "parameters": parameters})
    valid = not re.sub(CALL_PATTERN, "", cleaned, flags=re.DOTALL).strip()
    return {
        "calls": calls,
        "valid": valid,
        "has_call": True,
        "error": None if valid else "text_outside_tool_calls",
    }


def parse_native_calls_typed(text: str) -> list[dict[str, dict[str, Any]]]:
    """Decode native calls into the structured AST used by BFCL-style checks."""
    cleaned = clean_assistant_text(text)
    matches = list(re.finditer(CALL_PATTERN, cleaned, flags=re.DOTALL))
    if not matches:
        if "<tool_call" in cleaned or "<function=" in cleaned:
            raise ValueError("malformed_tool_call")
        return []
    decoded = []
    for match in matches:
        body = match.group(2)
        parameter_matches = list(re.finditer(PARAMETER_PATTERN, body, flags=re.DOTALL))
        if re.sub(PARAMETER_PATTERN, "", body, flags=re.DOTALL).strip():
            raise ValueError("unparsed_function_body")
        parameters = {}
        for parameter in parameter_matches:
            name = parameter.group(1).strip()
            if name in parameters:
                raise ValueError(f"duplicate_parameter:{name}")
            raw = parameter.group(2).strip()
            # Valid JSON preserves numeric, boolean, list, and object types.
            # Plain strings remain strings when the model omits JSON quoting.
            try:
                parameters[name] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                parameters[name] = raw
        decoded.append({match.group(1).strip(): parameters})
    if re.sub(CALL_PATTERN, "", cleaned, flags=re.DOTALL).strip():
        raise ValueError("text_outside_tool_calls")
    return decoded


def _schema_type(schema: dict[str, Any]) -> str:
    declared = schema.get("type", "any")
    if isinstance(declared, list):
        non_null = [item for item in declared if item != "null"]
        declared = non_null[0] if len(non_null) == 1 else "any"
    return {"object": "dict", "number": "float"}.get(declared, declared)


def _adapt_tool(tool: dict[str, Any]) -> dict[str, Any]:
    definition = tool.get("function", tool)
    if not isinstance(definition, dict) or "name" not in definition:
        raise ValueError("invalid_tool_definition")
    parameters = definition.get("parameters") or {"type": "object", "properties": {}}
    properties = {}
    # BFCL's checker consumes a smaller type vocabulary than JSON Schema.
    for name, schema in (parameters.get("properties") or {}).items():
        kind = _schema_type(schema)
        adapted = {"type": kind}
        if kind in NESTED_TYPES:
            adapted["items"] = {"type": _schema_type(schema.get("items") or {"type": "any"})}
        properties[name] = adapted
    return {
        "name": definition["name"],
        "description": definition.get("description", ""),
        "parameters": {
            "type": "dict",
            "properties": properties,
            "required": list(parameters.get("required") or []),
        },
    }


def _ground_truth_structure(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: [_ground_truth_structure(item)] for key, item in value.items()}
    if isinstance(value, list):
        return [_ground_truth_structure(item) for item in value]
    return value


def _ground_truth(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {call["name"]: {key: [_ground_truth_structure(value)] for key, value in call["parameters"].items()}}
        for call in calls
    ]


def _standardize(value: str) -> str:
    return re.sub(r"[ \,\.\/\-\_\*\^]", "", value).lower().replace("'", '"')


def _result(valid: bool, error_type: str | None = None, error: Any = None) -> dict[str, Any]:
    return {
        "valid": valid,
        "error": [] if error is None else error if isinstance(error, list) else [error],
        "error_type": error_type,
    }


def _type_check(
    parameter: str, value: Any, answers: list[Any], kind: str, expected: type, nested: type | None
) -> dict[str, Any]:
    possible_type = next((type(answer) for answer in answers if answer != ""), None)
    variable = possible_type is not None and possible_type != expected
    if type(value) is expected:
        if nested is None:
            return {**_result(True), "is_variable": variable}
        for answer in answers:
            valid = True
            if isinstance(answer, list):
                for item in value:
                    if not _type_check(parameter, item, answer, str(nested), nested, None)["valid"]:
                        valid = False
                        break
            if valid:
                return {**_result(True), "is_variable": variable}
        return {
            **_result(False, "type_error:nested", f"Nested type checking failed for parameter {parameter!r}."),
            "is_variable": variable,
        }
    if possible_type is not None and type(value) is possible_type:
        return {**_result(True), "is_variable": True}
    return {
        **_result(
            False,
            "type_error:simple",
            f"Incorrect type for parameter {parameter!r}: expected {kind}, got {type(value).__name__}.",
        ),
        "is_variable": variable,
    }


def _dict_check(value: dict[str, Any], answers: list[Any]) -> dict[str, Any]:
    last = _result(False, "dict_checker:no_match", "No dictionary answer matched.")
    for answer in answers:
        if answer == "":
            continue
        candidate = _result(True)
        for key, item in value.items():
            if key not in answer:
                candidate = _result(False, "value_error:dict_key", f"Unexpected dictionary key: {key!r}.")
                break
            actual = _standardize(item) if isinstance(item, str) else item
            expected = [_standardize(a) if isinstance(a, str) else a for a in answer[key]]
            if actual not in expected:
                candidate = _result(
                    False,
                    "value_error:dict_value",
                    f"Invalid dictionary value for {key!r}: {item!r}; expected one of {expected!r}.",
                )
                break
        if candidate["valid"]:
            for key, options in answer.items():
                if key not in value and "" not in options:
                    candidate = _result(False, "value_error:dict_key", f"Missing dictionary key: {key!r}.")
                    break
        if candidate["valid"]:
            return candidate
        last = candidate
    return last


def _simple_check(description: dict[str, Any], output: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    name = description["name"]
    if name not in output:
        return _result(
            False,
            "simple_function_checker:wrong_func_name",
            f"Function name {name!r} not found in model output.",
        )
    expected_parameters = next(iter(answer.values()))
    actual_parameters = output[name]
    schemas = description["parameters"]["properties"]
    for parameter in description["parameters"]["required"]:
        if parameter not in actual_parameters:
            return _result(
                False,
                "simple_function_checker:missing_required",
                f"Missing required parameter: {parameter!r}.",
            )
    # Validate schema and type before comparing values. BFCL treats values with
    # reference types unlike the declared schema as variables.
    for parameter, value in actual_parameters.items():
        if parameter not in schemas or parameter not in expected_parameters:
            return _result(False, "simple_function_checker:unexpected_param", f"Unexpected parameter: {parameter!r}.")
        schema = schemas[parameter]
        kind = schema["type"]
        if kind not in TYPE_MAPPING:
            return _result(
                False,
                "toolace_adapter:unsupported_schema_type",
                f"Unsupported BFCL parameter type: {kind!r}.",
            )
        expected_type = TYPE_MAPPING[kind]
        nested = None
        if kind in NESTED_TYPES:
            nested_kind = schema.get("items", {}).get("type", "any")
            if nested_kind not in TYPE_MAPPING:
                return _result(
                    False,
                    "toolace_adapter:unsupported_nested_type",
                    f"Unsupported BFCL nested type: {nested_kind!r}.",
                )
            nested = TYPE_MAPPING[nested_kind]
        if kind == "float" and type(value) is int:
            value = float(value)
        choices = expected_parameters[parameter]
        checked = _type_check(parameter, value, choices, kind, expected_type, nested)
        if not checked["valid"]:
            return checked
        if not checked["is_variable"]:
            if expected_type is dict:
                result = _dict_check(value, choices)
                if not result["valid"]:
                    return result
                continue
            if expected_type is list and nested is dict:
                result = _result(False, "list_dict_checker:no_match", "No list-of-dictionaries answer matched.")
                for choice in choices:
                    if len(value) != len(choice):
                        result = _result(
                            False, "value_error:list_dict_count", "Wrong number of dictionaries in the list."
                        )
                        continue
                    result = _result(True)
                    for actual, expected in zip(value, choice):
                        result = _dict_check(actual, [expected])
                        if not result["valid"]:
                            break
                    if result["valid"]:
                        break
                if not result["valid"]:
                    return result
                continue
            if expected_type is str:
                options = [_standardize(item) for item in choices if isinstance(item, str)]
                if _standardize(value) not in options:
                    return _result(
                        False,
                        "value_error:string",
                        f"Invalid value for parameter {parameter!r}: {value!r}; expected one of {choices!r}.",
                    )
                continue
            if expected_type is list:
                actual = [_standardize(item) if isinstance(item, str) else item for item in value]
                options = [
                    [_standardize(item) if isinstance(item, str) else item for item in choice] for choice in choices
                ]
                if actual not in options:
                    return _result(
                        False,
                        "value_error:list/tuple",
                        f"Invalid list for parameter {parameter!r}: {value!r}; expected one of {choices!r}.",
                    )
                continue
        if value not in choices:
            return _result(
                False,
                "value_error:others",
                f"Invalid value for parameter {parameter!r}: {value!r}; expected one of {choices!r}.",
            )
    for parameter, options in expected_parameters.items():
        if parameter not in actual_parameters and "" not in options:
            return _result(
                False,
                "simple_function_checker:missing_optional",
                f"Parameter {parameter!r} was not provided and was not marked optional.",
            )
    return _result(True)


def _ast_check(
    descriptions: list[dict[str, Any]],
    output: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    category: str,
) -> dict[str, Any]:
    # Parallel calls are matched without order, but every prediction can satisfy
    # only one expected call.
    if "parallel" in category:
        if len(output) != len(answers):
            return _result(False, "parallel_function_checker_no_order:wrong_count", "Wrong number of functions.")
        matched = set()
        for index, expected in enumerate(answers):
            name = next(iter(expected))
            description = next((item for item in descriptions if item["name"] == name), None)
            if description is None:
                return _result(
                    False,
                    "toolace_adapter:missing_tool_definition",
                    f"No tool definition found for {name!r}.",
                )
            errors = []
            for predicted_index, predicted in enumerate(output):
                if predicted_index not in matched:
                    result = _simple_check(description, predicted, expected)
                    if result["valid"]:
                        matched.add(predicted_index)
                        break
                    errors.append(result)
            else:
                return _result(
                    False,
                    "parallel_function_checker_no_order:cannot_find_match",
                    [f"No predicted call matched expected call {index}.", *errors],
                )
        return _result(True)
    if len(output) != 1 or len(answers) != 1:
        return _result(False, f"{category}_function_checker:wrong_count", "Wrong number of functions.")
    name = next(iter(answers[0]))
    description = next((item for item in descriptions if item["name"] == name), None)
    if description is None:
        return _result(False, "toolace_adapter:missing_tool_definition", f"No tool definition found for {name!r}.")
    return _simple_check(description, output[0], answers[0])


def score_bfcl_metrics(task: dict[str, Any], generation: dict[str, Any]) -> dict[str, Any]:
    expected_call = task["expected_call"]
    try:
        predicted = parse_native_calls_typed(generation["generated_text"])
        decode_valid, decode_error = True, None
    except (ValueError, TypeError, KeyError) as error:
        predicted, decode_valid, decode_error = [], False, str(error)
    predicted_nonempty = decode_valid and bool(predicted)
    result = {
        "bfcl_source_commit": BFCL_SOURCE_COMMIT,
        "bfcl_decode_valid": decode_valid,
        "bfcl_predicted_nonempty_call": predicted_nonempty,
        "bfcl_relevance_correct": predicted_nonempty if expected_call else None,
        "bfcl_irrelevance_correct": not predicted_nonempty if not expected_call else None,
        "bfcl_ast_correct": None,
        "bfcl_ast_category": None,
        "bfcl_ast_error_type": None,
        "bfcl_ast_errors": [],
        "bfcl_decode_error": decode_error,
    }
    # No-call records use the irrelevance decision and have no AST reference.
    if not expected_call:
        return result
    reference = parse_native_calls_typed(task["reference_text"])
    reference_calls = [{"name": next(iter(call)), "parameters": next(iter(call.values()))} for call in reference]
    try:
        descriptions = [_adapt_tool(tool) for tool in task["tools"]]
        category = (
            "parallel_multiple" if len(reference_calls) > 1 else "multiple" if len(task["tools"]) > 1 else "simple"
        )
        checked = _ast_check(descriptions, predicted, _ground_truth(reference_calls), category)
        result.update(
            {
                "bfcl_ast_correct": bool(checked["valid"]),
                "bfcl_ast_category": category,
                "bfcl_ast_error_type": checked.get("error_type"),
                "bfcl_ast_errors": checked.get("error", []),
            }
        )
    except (ValueError, TypeError, KeyError) as error:
        result.update(
            {
                "bfcl_ast_correct": False,
                "bfcl_ast_category": "adapter_error",
                "bfcl_ast_error_type": "toolace_adapter:exception",
                "bfcl_ast_errors": [str(error)],
            }
        )
    return result


def score_generation(task: dict[str, Any], generation: dict[str, Any]) -> dict[str, Any]:
    """Return BFCL-style primary metrics and strict native-call diagnostics."""
    reference = parse_native_calls(task["reference_text"])
    prediction = parse_native_calls(generation["generated_text"])
    expected = task["expected_call"]
    if expected and (not reference["valid"] or not reference["calls"]):
        raise RuntimeError(f"Cannot parse reference calls for {task['task_id']}")
    names = (
        expected
        and prediction["valid"]
        and Counter(call["name"] for call in reference["calls"])
        == Counter(call["name"] for call in prediction["calls"])
    )
    exact = (
        expected
        and prediction["valid"]
        and Counter(json.dumps(call, sort_keys=True, ensure_ascii=False) for call in reference["calls"])
        == Counter(json.dumps(call, sort_keys=True, ensure_ascii=False) for call in prediction["calls"])
    )
    # Native exactness remains a diagnostic beside BFCL-style primary metrics;
    # it is deliberately stricter than functional AST correctness.
    diagnostics = {
        "expected_call": expected,
        "predicted_call": prediction["has_call"],
        "decision_correct": prediction["has_call"] == expected,
        "syntax_valid": prediction["valid"],
        "function_names_exact": bool(names) if expected else None,
        "native_calls_exact": bool(exact) if expected else None,
        "text_exact": clean_assistant_text(task["reference_text"])
        == clean_assistant_text(generation["generated_text"]),
        "prediction_parse_error": prediction["error"],
        "reference_calls": reference["calls"],
        "predicted_calls": prediction["calls"],
    }
    return {**score_bfcl_metrics(task, generation), **diagnostics}
