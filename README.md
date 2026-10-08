# Nemotron 3 FC

Nemotron 3 FC fine tunes LoRA adapters for NVIDIA Nemotron 3 Nano 30B A3B BF16
on ToolACE function calling conversations. Nemotron 3 Nano is a 30B parameter
large language model with 3B active parameters per forward pass. Its hybrid
architecture combines Mamba 2 state space layers, Transformer attention layers
and sparse mixture of experts layers instead of using a Transformer only design.

The project evaluates the base model and trained adapters on the held out
ToolACE test split. BFCL AST accuracy is the primary evaluation metric. vLLM
serves the selected adapter through an OpenAI compatible API.

## What this project does

- Train resumable LoRA fine tuning runs on the ToolACE training split.
- Evaluate the base model and trained adapters on the held out ToolACE test split.
- Measure BFCL AST accuracy and show generated calls next to their references.
- Launch an OpenAI compatible vLLM endpoint with a selected adapter.
- Measure concurrent function calling requests and retain the resulting report.

## Repository layout

```text
configs/                     Training, evaluation, serving and artifact configuration
environments/                Separate dependency specifications for training and serving
reports/serving/             Recorded serving benchmark results and plots
scripts/run.py               Command launcher
scripts/fetch_artifacts.py   Artifact download utility
src/nemotron3_fc/training/   LoRA training, validation, checkpoints and training reports
src/nemotron3_fc/evaluation/ vLLM generation, BFCL AST scoring and evaluation reports
src/nemotron3_fc/serving/    vLLM server checks, API verification and benchmarks
src/nemotron3_fc/data/       ToolACE input schemas and supporting data utilities
tests/                       CPU test suite
```

## Setup

Training and serving have separate pinned dependency specifications because the
training stack and the validated vLLM runtime require different package versions.
The launcher selects the relevant environment before importing either stack.

Models, the ToolACE split, adapters and wheel bundles are declared in
[`configs/artifacts.json`](configs/artifacts.json). A configuration refers to
them through `artifact://...` paths, so changing where artifacts are stored does
not require editing source code.

## Train

The training configuration defines the LoRA configuration, run mode, validation
schedule, checkpoint schedule and source of a resumed run.

```bash
python scripts/run.py train --config configs/train-toolace.portable.json
```

Use `mode: "quick-test"` for an integration check. Use `mode: "full"` for a
training run. To continue a previous run, set `start_from`, `previous_checkpoint`
and `previous_best` in the run configuration to the corresponding saved outputs.

## Evaluate

Evaluation generates responses for the selected ToolACE test split using vLLM.
BFCL AST accuracy is the primary quality result. The report also preserves the
reference call and generated call for every record so mismatches can be inspected.

```bash
python scripts/run.py evaluate --config configs/evaluate-two-epochs.portable.json
```

Each adapter receives its own predictions and summary. When multiple adapters
are evaluated together, the run also writes a direct comparison summary.

## Evaluation results

The base model and the selected fine tuned adapter were evaluated with greedy
vLLM generation on the same ToolACE test split: 1,077 records, 1,283 assistant
turns, 970 function calling turns and 313 no call turns. 

AST accuracy is the primary metric. The BFCL implementation's AST checker
and relevance checker were applied to ToolACE outputs. 

| Metric | Base model | Fine tuned adapter | Change |
| --- | ---: | ---: | ---: |
| **BFCL AST accuracy on function calling turns** | 59.28% | **65.77%** | **+6.49 pp** |
| BFCL relevance accuracy on function calling turns | 98.97% | 99.38% | +0.41 pp |
| BFCL irrelevance accuracy on no call turns | 75.08% | **90.10%** | **+15.02 pp** |
| BFCL decoder valid rate on all turns | 99.84% | 99.92% | +0.08 pp |
| Exact native function call on function calling turns | 55.77% | 66.49% | +10.72 pp |
| Call or no call decision accuracy on all turns | 93.30% | 97.04% | +3.74 pp |
| No call decision accuracy | 75.08% | 89.78% | +14.70 pp |
| Exact function name on function calling turns | 89.48% | 93.30% | +3.82 pp |

The fine tuned adapter added 104 exact native function calls out of the 970
function calling turns. The largest practical gain was no call behavior: it
avoids unnecessary calls when the prompt lacks required information or an earlier
tool result already answers the request.

### LLM-as-a-Judge

The BFCL AST checker compares each generated call with the recorded reference
call using fixed parser rules. It can recognize syntax and structural matches,
but it cannot decide whether a different argument value, optional parameter or
valid call plan still satisfies the user's request. The LLM as a Judge used
GPT-5.6-Luna (Low) to review the 332 calls rejected by the AST checker with the
user request, available tool definition, reference call and generated call. It
assigned one of three labels: functionally correct, incorrect or underdetermined.

| LLM as a Judge outcome for the 332 AST rejected calls | Cases |
| --- | ---: |
| Functionally correct | 149 |
| Incorrect | 70 |
| Underdetermined | 113 |

Combining the 638 calls accepted by the BFCL AST checker with the 149 rejected
calls judged functionally correct gives **787 acceptable calls out of 970**, or
**81.13% judge adjusted functional correctness**. This is a supplementary
judgment result for the fine tuned adapter. The reproducible primary metric
remains BFCL AST accuracy of **65.77%**.

## Serve

Serve the base model and a selected LoRA adapter through an OpenAI compatible
vLLM API. Set the adapter, port and output directory in
[`configs/serve.example.json`](configs/serve.example.json).

```bash
python scripts/run.py serve --config configs/serve.example.json
```

Validate a running server from another terminal:

```bash
python scripts/run.py check-serving --config configs/serve.example.json
```

Run a complete server lifecycle check, including a tool call, tool result and
final answer:

```bash
python scripts/run.py verify-serving --config configs/serve.example.json
```

## Serving benchmark

The benchmark measures concurrent complete interactions: user request, generated
tool call, deterministic tool result and streamed final response. It records
completion status, request rate, token rate, latency percentiles, first-content
latency and GPU memory use.

```bash
python scripts/run.py benchmark-serving --config configs/benchmark-serving.example.json
```

Each run writes `summary.json` and `requests.csv` to a timestamped directory
inside the configured output location. The recorded RTX PRO 6000 benchmark is
available in [`reports/serving`](reports/serving/README.md).


## Tests

```bash
python -m pip install -e ".[test]"
python -m pytest
```

## References

- [NVIDIA Nemotron 3 Nano technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf)
- [Berkeley Function Calling Leaderboard implementation](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [Improving Large Language Models Function Calling and Interpretability via Guided Structured Templates](https://aclanthology.org/2025.emnlp-main.1242.pdf)

