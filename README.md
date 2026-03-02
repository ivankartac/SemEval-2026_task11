# UFAL-CUNI at SemEval-2026 Task 11: An Efficient Modular Neuro-symbolic Method for Syllogistic Reasoning

## Project structure

```
├── configs/              # YAML configs for each subtask
├── data/                 # Input data
├── prompt_templates/     
├── scripts/              # Inference and evaluation scripts
│   ├── run_inference.py
│   └── run_evaluation.py
├── src/
│   ├── evaluation.py     # Evaluation metrics
│   ├── orchestrator.py   # GPU worker queue
│   ├── utils.py
│   └── pipeline/         # System components
└── requirements.txt
```

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Inference requires a running [Ollama](https://ollama.ai) instance:

```bash
ollama pull qwen3:4b-thinking-2507-fp16
ollama serve
```

## Configuration

Uses YAML files to configure inference parameters. See `configs` for the full list of options.

## Usage

### Inference

```bash
python scripts/run_inference.py <input.json> -c <config.yaml>
```

CLI arguments can be used to override config values.

### Evaluation

```bash
python scripts/run_evaluation.py <predictions.jsonl> <ground_truth.json> -o results.json
```

## Subtasks

Each subtask has a corresponding config file in `configs/`

| Subtask | Description | Config |
|---|---|---|
| 1 | Validity prediction (English) | `configs/subtask_1.yaml` |
| 2 | Validity + premise retrieval (English) | `configs/subtask_2.yaml` |
| 3 | Validity prediction (multilingual, with translation) | `configs/subtask_3.yaml` |
| 4 | Validity + premise retrieval (multilingual) | `configs/subtask_4.yaml` |

### Example (subtask 1)

Run inference:

```bash
python scripts/run_inference.py data/val_data_subtask_1.json -c configs/subtask_1.yaml
```

Evaluate results with bootstrap resampling:

```bash
python scripts/run_evaluation.py results/output.jsonl data/val_data_subtask_1.json -o results/eval.json
```
