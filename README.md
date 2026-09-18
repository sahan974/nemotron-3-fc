# Nemotron 3 FC

Nemotron 3 FC provides tool-calling dataset preparation, BF16 LoRA training, checkpointed resumption, deterministic evaluation, and vLLM serving for Nemotron-3-Nano-30B-A3B-BF16.

## Dataset preparation

The data pipeline converts source-specific records into one canonical conversation schema before splitting. Built-in adapters currently support raw ToolACE and xLAM 60K sources.

```bash
nemotron3-fc inspect-dataset --adapter toolace --path /data/toolace/data.json
nemotron3-fc prepare-data --config configs/prepare-data.example.json
```

`prepare-data` performs source conversion, rejection accounting, exact deduplication, whole-conversation grouping, deterministic split assignment, numeric-template quarantine, cross-split leakage auditing, JSONL round-trip validation, and SHA-256 manifest generation.

Each configured source receives its own directory containing:

- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`
- `rejected.jsonl`
- `manifest.json`

The output root also contains `preparation.json` and `split-report.json`. Dataset paths, adapters, expected source identities, split ratios, random seed, and multi-turn holdout requirements are configuration values.
