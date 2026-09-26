# Nemotron 3 FC

Nemotron 3 FC is a Python project for preparing function calling data, training
LoRA adapters for Nemotron 3 Nano 30B A3B BF16, and evaluating trained adapters
with vLLM and BFCL metrics.

Training and evaluation use separate pinned environments so each workflow runs
with its validated dependency versions.

## Features

- ToolACE and xLAM conversion with leakage checks
- BF16 LoRA training with assistant response masking
- Atomic checkpoints and exact training resumption
- Training metrics, periodic validation, and complete validation after each epoch
- vLLM evaluation with BFCL metrics and exact call diagnostics
- Portable resolution of models, datasets, adapters, and wheel bundles

## Repository layout

```text
configs/                     Run and artifact configurations
environments/training/       Training dependency contract
environments/serving/        Evaluation dependency contract
scripts/run.py               Main launcher
scripts/fetch_artifacts.py   Artifact download utility
src/nemotron3_fc/data/       Dataset preparation
src/nemotron3_fc/training/   LoRA training and checkpoint management
src/nemotron3_fc/evaluation/ vLLM inference and evaluation
tests/                       CPU tests
```

## Environments

Training uses Transformers 5.5.0 with PEFT 0.21.0. Evaluation uses vLLM 0.18.0
with Transformers 4.57.6. The launcher selects the correct isolated environment
before importing either stack.

Run configurations use portable references such as `artifact://model` and
`artifact://toolace`. Artifact locations are declared in
`configs/artifacts.json`.

## Prepare data

The preparation pipeline converts the ToolACE and xLAM source datasets and keeps
their processed outputs separate.

```bash
python scripts/run.py prepare-data --config configs/prepare-data.example.json
```

The pipeline verifies record counts, file hashes, record identity, split
isolation, and JSONL round trips before writing the output.

## Train

Use the same command on any supported GPU host. The launcher detects the current
environment, resolves the configured artifacts, and prepares the pinned training
environment.

```bash
python scripts/run.py train --config configs/train-toolace.portable.json
```

Set `mode` to `quick-test` for a short integration check or `full` for a complete
run. A resumed run must reference the previous checkpoint and its matching best
adapter.

## Evaluate

Evaluation performs greedy vLLM generation for every assistant turn in the
selected test split. BFCL metrics are primary. Exact native call comparisons are
retained as diagnostics.

```bash
python scripts/run.py evaluate --config configs/evaluate-two-epochs.portable.json
```

Each adapter receives its own prediction file, identity record, and summary.
Evaluation of multiple adapters also produces paired results and a comparison
summary. Compatible completed predictions are reused automatically.

## Artifact sources

Models, processed datasets, adapters, and wheel bundles can be supplied through
local paths, mounted directories, or the artifact cache. Kaggle datasets can be
used as an optional artifact registry without changing the training or evaluation
code.

Download selected artifacts into the local cache:

```bash
python scripts/fetch_artifacts.py model toolace training-wheels --source kaggle
```

Select Kaggle explicitly as the artifact source for a normal run when required:

```bash
python scripts/run.py train --artifact-source kaggle --package-source kaggle --config configs/train-toolace.portable.json
```

Use `--artifact-cache` to place large downloads on a persistent volume. Hosts
without network access can use already mounted artifacts and wheel bundles.

## Tests

Install the test dependency and run the CPU test suite:

```bash
python -m pip install -e ".[test]"
python -m pytest
```
