# Self-Evolving Time-Series Agent (SE-TSA)

Official implementation of **"Self-Evolving Time-Series Agent: Automatic Architecture Discovery Based on Large Language Models"**.

## Overview

SE-TSA is a large language model (LLM)-driven programmatic evolution framework that automatically discovers and refines time-series forecasting architectures without human intervention in architecture design. Unlike traditional hand-crafted models or foundation models that rely on predefined assumptions, SE-TSA iteratively generates and optimizes model architectures tailored to the target dataset's intrinsic characteristics.

### Key Contributions

- **Dual-Strategy Evolution (EXPLORE / EXPLOIT)**: Dynamically balances architectural diversity exploration with performance-oriented refinement.
- **MAP-Elites Quality-Diversity Archive**: Maintains a behavioral grid indexed by architecture family and preprocessing type to mitigate premature convergence and preserve diverse high-performing candidates.
- **Reflection & Principle Accumulation**: Extracts and deduplicates interpretable design principles across iterations via semantic embedding, enabling cross-iteration meta-learning.
- **Bagging Ensemble**: Combines globally best models with behaviorally diverse elites for robust final predictions.

## Repository Structure

```
.
├── src/
│   ├── main.py                    # Main entry point: dataset loading, evolution, testing
│   ├── config.yaml                # Default hyperparameters
│   ├── agent/
│   │   ├── run.py                 # Core evolution loop (MAP-Elites + reflection)
│   │   ├── generate.py            # EXPLORE / EXPLOIT / FIX model generation
│   │   ├── reflect.py             # Design principle extraction & deduplication
│   │   ├── client.py              # Anthropic LLM API client with retry logic
│   │   └── prompts/               # LLM prompt templates (markdown)
│   ├── models/
│   │   ├── base.py                # BaseForecastModel abstract class
│   │   ├── RevINLinear.py         # Predefined baseline model
│   │   └── layers/
│   │       └── revin.py           # RevIN (Reversible Instance Normalization)
│   └── utils/
│       ├── load_dataset.py        # Data loading, BERT encoding, sliding-window
│       ├── evaluate_model.py      # Subprocess-isolated training & evaluation
│       ├── test_model.py          # Final ensemble testing & visualization
│       ├── parse_args.py          # YAML + CLI argument parser
│       └── generate_summary.py    # Results summarizer
├── experiments/                   # Per-dataset experiment shell scripts
│   ├── TEST.sh
│   ├── Agriculture.sh
│   ├── Climate.sh
│   ├── Economy.sh
│   ├── Energy.sh
│   ├── Environment.sh
│   ├── PublicHealth.sh
│   ├── Security.sh
│   ├── SocialGood.sh
│   └── Traffic.sh
├── datasets/
│   └── raw/                       # Place raw CSV datasets here
├── outputs/                       # Evolution artifacts, models, stats, plots
├── pyproject.toml                 # Python dependencies (uv-based)
└── .env_sample                    # Template for LLM API credentials
```

## Environment Setup

### Prerequisites

- **Python** >= 3.12
- **CUDA** (for PyTorch GPU support; CPU mode is supported but slower)
- **uv** package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))
- An **Anthropic API key** (or compatible LLM endpoint) for architecture generation

### Installation

```bash
# Create virtual environment and install dependencies
uv venv --python 3.12
source .venv/bin/activate
uv sync

# Configure LLM API credentials
cp .env_sample .env
# Edit .env with your API key and endpoint
```

### Environment Variables (`.env`)

```bash
ANTHROPIC_API_KEY=your-api-key
ANTHROPIC_BASE_URL=your-base-url        # Optional: for custom/proxy endpoints
ANTHROPIC_MODEL=claude-opus-4-7         # Or your preferred model
LLM_MAX_TOKENS=196608
LLM_RETRY_ATTEMPTS=3
LLM_RETRY_BASE_DELAY=2.0
```

## Data Preparation

### Dataset Format

Place raw CSV files under `datasets/raw/`. Each CSV must contain:

- A `date` or `Date` column (parsed as datetime index).
- A target column named `OT` (moved to the first position automatically).
- Optional exogenous numeric and text columns.

Example:
```csv
date,OT,exog_temp,exog_humidity,text_info
2023-01-01,1.2,22.5,60.0,"sunny day"
...
```

### Supported Datasets

The paper evaluates SE-TSA on nine cross-domain benchmarks:

| Dataset       | Features | Input/Output Combinations                  |
|---------------|----------|--------------------------------------------|
| Agriculture   | 28       | (8,6), (8,8), (8,10), (8,12)               |
| Climate       | 29       | (8,6), (8,8), (8,10), (8,12)               |
| Economy       | 29       | (8,6), (8,8), (8,10), (8,12)               |
| Energy        | 24       | (36,12), (36,24), (36,36), (36,48)         |
| Environment   | 41       | (96,48), (96,96), (96,192), (96,336)       |
| PublicHealth  | 44       | (36,12), (36,24), (36,36), (36,48)         |
| Security      | 26       | (8,6), (8,8), (8,10), (8,12)               |
| SocialGood    | 26       | (8,6), (8,8), (8,10), (8,12)               |
| Traffic       | 17       | (8,6), (8,8), (8,10), (8,12)               |

> **Note**: Download links for the benchmark datasets will provided soon.

### Preprocessing Pipeline

1. **Temporal indexing**: Parse `date`/`Date` column and sort chronologically.
2. **Auxiliary timestamp conversion**: Convert `start_date`, `end_date`, or `Month` to numeric values.
3. **Target-column reordering**: Move `OT` to the first column.
4. **Missing-value imputation**: Forward fill, then zero fill.
5. **Chronological split**: Train (70%) / Validation (10%) / Test (20%).
6. **Text-feature encoding**: Encode text columns via `sentence-transformers/all-MiniLM-L6-v2`, then reduce to 3 dimensions via truncated SVD.
7. **Sliding-window construction**: Generate `(X, Y)` pairs for each `(input_len, output_len)` condition.

Processed datasets are cached in `datasets/processed/` for reuse.

## Training & Evaluation

### Quick Test (5-minute run)

```bash
bash experiments/TEST.sh
```

This runs SE-TSA on a small `TEST` dataset with 2 input/output conditions and a 5-minute time budget.

### Full Experiment (Single Dataset)

```bash
bash experiments/Agriculture.sh
```

Or run directly:

```bash
uv run python -m src.main \
    --dataset Agriculture \
    --io-pairs "8,6;8,8;8,10;8,12"
```

### Custom Run

```bash
uv run python -m src.main \
    --dataset [DATASET_NAME] \
    --io-pairs "8,6;8,8;8,10;8,12" \
    --split-ratio "0.7,0.1,0.2" \
    --max-time 360
```

**Key Arguments:**

| Argument       | Description                                      | Default       |
|----------------|--------------------------------------------------|---------------|
| `--dataset`    | Dataset name (matches CSV filename)              | `TEST`        |
| `--io-pairs`   | Semicolon-separated `input_len,output_len` pairs | `"8,6;8,8"`   |
| `--split-ratio`| Train/val/test split ratios                      | `"0.7,0.1,0.2"`|
| `--max-time`   | Total evolution time budget in minutes           | `360`         |

### Test-Only Mode (Re-evaluate Saved Models)

If you have already run evolution and want to re-run final testing with the saved ensemble:

```bash
uv run python -m src.main \
    --dataset Agriculture \
    --io-pairs "8,6;8,8;8,10;8,12" \
    --test-only
```

### Hyperparameters

Default hyperparameters are defined in `src/config.yaml`:

| Hyperparameter               | Value   | Description                              |
|------------------------------|---------|------------------------------------------|
| `max_time`                   | 360 min | Total evolution time budget              |
| `max_iterations`             | 10      | Maximum evolution iterations             |
| `n_candidates`               | 5       | Candidates per EXPLORE iteration         |
| `exploit_top_k`              | 3       | Top-k families for EXPLOIT               |
| `n_fix_retries`              | 2       | Auto-repair attempts for failed models   |
| `batch_size`                 | 32      | Training batch size                      |
| `lr`                         | 0.001   | Learning rate (AdamW)                    |
| `train_max_epochs`           | 10      | Max training epochs per candidate        |
| `train_patience`             | 3       | Early stopping patience                  |
| `quick_search_max_epochs`    | 3       | Quick hyperparameter search epochs       |
| `quick_search_max_combinations` | 8    | Max configs in quick search              |
| `ensemble_top_n_overall`     | 3       | Top-N global models in final ensemble    |
| `ensemble_top_n_map_elites`  | 3       | Top-N MAP-Elites elites in ensemble      |
| `exploration_rate_base`      | 0.9     | Initial exploration rate                 |
| `exploration_rate_decay`     | 0.9     | Exploration decay factor                 |
| `exploration_rate_min`       | 0.1     | Minimum exploration rate                 |

### Outputs

After a run completes, the following artifacts are saved under `outputs/{dataset}_multi/`:

- `models/` — Generated PyTorch model `.py` files from evolution.
- `stats.json` — Full evolution history, MAP-Elites grid, and per-candidate metrics.
- `summary.json` — Condensed results with aggregated test metrics.
- `test_prediction_*.png` — Visualization plots of ensemble predictions vs. ground truth.

## Acknowledgments

- Time-MMD: https://github.com/AdityaLab/Time-MMD

- TimeCopilot: https://github.com/TimeCopilot/timecopilot
