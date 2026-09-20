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

## LoRA training

Training is configured with JSON and consumes canonical split directories produced by the data pipeline.

```bash
nemotron3-fc train --config configs/train-toolace.example.json
```

The training command provides assistant-only supervision, over-length record windowing, deterministic token-budget batching, BF16 LoRA, fused AdamW, linear warmup followed by a constant learning rate, stratified monitoring, full epoch-end validation, atomic checkpoints, exact resumption, best-adapter selection, JSONL metrics, and separate-scale training plots.

For a short integration run, copy the example configuration and change `mode` to `quick-test`. Full runs use every configured training and validation record. Resume runs use `start_from: "checkpoint"` and explicitly provide both `previous_checkpoint` and `previous_best` from the same run number.
