"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from nemotron3_fc.data.adapters.toolace import ToolACEAdapter
from nemotron3_fc.data.adapters.xlam import XLAMAdapter
from nemotron3_fc.data.io import load_raw_records
from nemotron3_fc.data.pipeline import load_preparation_config, prepare_data
from nemotron3_fc.data.registry import DatasetRegistry


def default_registry() -> DatasetRegistry:
    """Build the registry of adapters shipped with this package."""
    registry = DatasetRegistry()
    registry.register("toolace", ToolACEAdapter)
    registry.register("xlam", XLAMAdapter)
    return registry


def inspect_dataset(adapter_name: str, path: Path) -> int:
    """Convert a raw source in memory and print a compact inventory."""
    adapter = default_registry().create(adapter_name)
    raw_records = load_raw_records(path)
    records, rejected, assistant_turns, tool_call_turns = 0, 0, 0, 0
    for index, raw in enumerate(raw_records):
        try:
            record = adapter.convert_record(raw, index)
        except (ValueError, TypeError, KeyError, IndexError, RecursionError):
            rejected += 1
            continue
        records += 1
        for message in record.messages:
            if message.role == "assistant":
                assistant_turns += 1
                tool_call_turns += bool(message.tool_calls)
    print(
        f"adapter={adapter_name} raw={len(raw_records)} records={records} rejected={rejected} "
        f"assistant_turns={assistant_turns} tool_call_turns={tool_call_turns}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nemotron3-fc")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect-dataset", help="Convert and summarize a raw dataset source")
    inspect.add_argument("--adapter", required=True)
    inspect.add_argument("--path", required=True, type=Path)
    prepare = commands.add_parser("prepare-data", help="Convert, split, audit, and write configured datasets")
    prepare.add_argument("--config", required=True, type=Path)
    train = commands.add_parser("train", help="Run configured BF16 LoRA training")
    train.add_argument("--config", required=True, type=Path)
    evaluate = commands.add_parser("evaluate", help="Run vLLM inference and BFCL-style turn-level scoring")
    evaluate.add_argument("--config", required=True, type=Path)
    serve = commands.add_parser("serve", help="Launch the vLLM OpenAI-compatible server")
    serve.add_argument("--config", required=True, type=Path)
    verify = commands.add_parser("verify-serving", help="Start, validate, and stop the vLLM API")
    verify.add_argument("--config", required=True, type=Path)
    check = commands.add_parser("check-serving", help="Validate a running local serving API")
    check.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Keep heavy GPU stacks out of lightweight data commands. Imports happen
    # only after argparse has selected training or evaluation.
    if args.command == "inspect-dataset":
        return inspect_dataset(args.adapter, args.path)
    if args.command == "prepare-data":
        prepare_data(load_preparation_config(args.config), default_registry())
        return 0
    if args.command == "train":
        from nemotron3_fc.training.config import load_training_config
        from nemotron3_fc.training.trainer import run_training

        summary = run_training(load_training_config(args.config))
        return 0 if summary["status"] in {"completed", "session_time_limit"} else 1
    if args.command == "evaluate":
        from nemotron3_fc.evaluation.config import load_evaluation_config
        from nemotron3_fc.evaluation.runner import run_evaluation

        run_evaluation(load_evaluation_config(args.config))
        return 0
    if args.command in {"serve", "verify-serving"}:
        from nemotron3_fc.serving.config import load_serving_config
        from nemotron3_fc.serving.server import run_server

        return run_server(load_serving_config(args.config), verify=args.command == "verify-serving")
    if args.command == "check-serving":
        from nemotron3_fc.serving.config import load_serving_config
        from nemotron3_fc.serving.check import run_serving_check

        run_serving_check(load_serving_config(args.config, validate_inputs=False))
        return 0
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
