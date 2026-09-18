"""Command line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from nemotron3_fc.data.adapters.toolace import ToolACEAdapter
from nemotron3_fc.data.registry import DatasetRegistry


def default_registry() -> DatasetRegistry:
    """Build the registry of adapters shipped with this package."""
    registry = DatasetRegistry()
    registry.register("toolace", ToolACEAdapter)
    return registry


def inspect_dataset(adapter_name: str, path: Path) -> int:
    """Validate one canonicalized split and print a compact inventory."""
    adapter = default_registry().create(adapter_name)
    records = 0
    assistant_turns = 0
    tool_call_turns = 0
    for record in adapter.load_split(path):
        records += 1
        for message in record.messages:
            if message.role == "assistant":
                assistant_turns += 1
                tool_call_turns += bool(message.tool_calls)
    print(f"adapter={adapter_name} records={records} assistant_turns={assistant_turns} tool_call_turns={tool_call_turns}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nemotron3-fc")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect-dataset", help="Validate and summarize a source dataset split")
    inspect.add_argument("--adapter", required=True)
    inspect.add_argument("--path", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect-dataset":
        return inspect_dataset(args.adapter, args.path)
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

