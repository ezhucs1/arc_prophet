# ARC — Agent Research & Curation

A ReAct agent evaluation system for binary forecasting on Polymarket questions.
The agent answers yes/no prediction market questions by searching a Reddit knowledge base,
and is evaluated in a time-series manner — once per historical price timepoint per question —
to compare agent probability estimates against human (market) probabilities over time.

## Overview

```
Polymarket JSONL dataset
        ↓
agent/evals.py          ← benchmark harness (single-pass or time-series)
        ↓ per question / per timepoint
agent/react_agent.py    ← ReAct while-loop (LLM + tools, max 20 iterations)
        ↓
shared/llm.py           ← OpenAI-compatible client (vLLM / Qwen3-14B-AWQ)
        ↓ tool calls
agent/tools.py          ← 5 tools: search, post, comment, thread, author history
        ↓
d_agent_client/client.py ← IPC TCP client → Text2SQL server → Reddit DB
        ↓
agent/results/*.jsonl   ← one record per (question, timepoint)
        ↓ optional
eval/eval_forecasting.py ← calibration, classification, operational metrics
```

## Project Structure

```
arc/
├── agent/
│   ├── react_agent.py      # ReAct loop: LLM → tool calls → repeat → final answer
│   ├── tools.py            # 5 OpenAI-schema tools for Reddit DB access
│   ├── prompts.py          # System prompt: reasoning discipline + reflection
│   ├── evals.py            # Benchmark harness: single-pass and time-series modes
│   └── results/            # Output JSONL files (gitignored)
├── database/
│   ├── loader.py           # Polymarket dataset loaders
│   ├── polymarket_binary_yesno.jsonl         # All binary markets (~361 MB, gitignored)
│   ├── polymarket_binary_weekly_plus.jsonl   # Markets open ≥1 week (~212 MB, gitignored)
│   ├── polymarket_binary_monthly_plus.jsonl  # Markets open ≥1 month (~47 MB, gitignored)
│   └── polymarket_binary_yearly_plus.jsonl   # Markets open ≥1 year (~382 KB, gitignored)
├── d_agent_client/
│   └── client.py           # IPC TCP client to Text2SQL server (127.0.0.1:61001)
├── eval/
│   └── eval_forecasting.py # Full metrics: Brier, ECE, MCC, F1, per-topic breakdown
├── shared/
│   └── llm.py              # LLM client init from .env
├── run.py                  # Single interactive query CLI
├── pyproject.toml
└── requirements.txt
```

## Setup

### Requirements

- Python 3.12
- vLLM server running locally (default: `http://127.0.0.1:8000/v1`)
- Text2SQL IPC server running locally (default: `127.0.0.1:61001`)

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment

Create a `.env` file in the project root:

```env
VLLM_API_BASE=http://127.0.0.1:8000/v1
VLLM_MODEL_NAME=Qwen/Qwen3-14B-AWQ

# Optional: Langfuse tracing
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=https://us.cloud.langfuse.com
```

## Usage

### Single interactive query

```bash
python run.py --query "Will Bitcoin hit 100k by Feb 2026?" --cutoff_time 2026-02-01
python run.py --query "Will inflation drop below 3%?" --cutoff_time 2026-01-01 --options Yes No
```

### Single-pass benchmark (one result per question)

```bash
# Run 10 questions
python agent/evals.py --max_questions 10

# Run full dataset
python agent/evals.py

# Resume from existing results file
python agent/evals.py --results_file agent/results/results_XXXXXX.jsonl

# Summarize existing results
python agent/evals.py --summarize agent/results/results_XXXXXX.jsonl
```

### Time-series benchmark (one result per question per timepoint)

```bash
# Full time series — recommended for first run to collect all data
python agent/evals.py --timeseries --max_questions 10

# Cap timepoints per question (evenly spaced)
python agent/evals.py --timeseries --max_questions 100 --max_timepoints 10

# Use a different dataset (weekly/monthly/yearly)
python agent/evals.py --timeseries --dataset database/polymarket_binary_monthly_plus.jsonl

# Resume a time-series run
python agent/evals.py --timeseries --results_file agent/results/results_ts_XXXXXX.jsonl

# Custom timeout per timepoint (default: 300s)
python agent/evals.py --timeseries --max_questions 50 --timeout 180
```

Time-series result files are named `results_ts_YYYYMMDD_HHMMSS.jsonl` to distinguish them
from single-pass results (`results_YYYYMMDD_HHMMSS.jsonl`).

### Comprehensive metrics analysis

```bash
python eval/eval_forecasting.py agent/results/results_ts_XXXXXX.jsonl
python eval/eval_forecasting.py agent/results/results_ts_XXXXXX.jsonl --output report.json
python eval/eval_forecasting.py agent/results/results_ts_XXXXXX.jsonl --format all --output-dir ./eval_results/
```

## Output Schema

### Single-pass result record

| Field | Description |
|---|---|
| `question_id` | Sequential question index |
| `question` | Question text |
| `options` | Always `["Yes", "No"]` |
| `ground_truth` | Resolved market outcome |
| `cutoff_date` | Market close date (agent's cutoff) |
| `topic` | Market category |
| `predicted` | Agent's selected option |
| `correct` | Whether prediction matches ground truth |
| `confidence` | Agent's stated confidence (0.0–1.0) |
| `agent_prob_yes` / `agent_prob_no` | Agent probabilities derived from confidence |
| `market_prob_yes` / `market_prob_no` | Human market probability at close date |
| `reflection` | Last reflection block from agent reasoning |
| `self_critique` | Agent's pre-commit counter-argument |
| `tool_call_count` | Number of tool calls made |
| `latency_sec` | Wall-clock seconds |
| `error` | Error message or `null` |
| `timestamp` | UTC timestamp of record creation |
| `final_answer` | Raw LLM output (reasoning + answer) |

### Time-series result record

Same as above, with these differences/additions:

| Field | Description |
|---|---|
| `timepoint_index` | Position in the time series (1-based) |
| `timepoint_date` | Date used as agent's cutoff (NOT the close date) |
| `total_timepoints` | Total timepoints for this question |
| `close_date` | Market resolution date (stored for reference, never sent to agent) |
| `human_prob_yes` / `human_prob_no` | Market probability at this specific timepoint |

## Agent Design

### ReAct Loop (`agent/react_agent.py`)

Plain while-loop, no frameworks. Each iteration:
1. Call LLM with accumulated messages + tool schemas
2. If response has tool calls → execute → append results → continue
3. If no tool calls → return final answer

Max 20 iterations. 300-second timeout per question/timepoint.

### Tools (`agent/tools.py`)

| Tool | Purpose |
|---|---|
| `search_database` | Hybrid semantic + keyword search across Reddit posts/comments |
| `get_post_core_info` | Full content and metadata for a specific post |
| `get_comment_core_info` | Full content and metadata for a specific comment |
| `get_post_comments_list` | Thread context around a comment (ancestors + descendants) |
| `get_author_history_list` | Chronological post/comment history for an author |

All tools respect `cutoff_time` — results are filtered to dates before the cutoff.

### Reflection (`agent/prompts.py`)

The agent is instructed to produce a structured `Reflection:` block every 3 tool calls:

```
Reflection:
  Evidence for Yes: ...
  Evidence for No: ...
  Gaps / uncertainties: ...
  Current belief: Yes=0.X | No=0.X
  Next action: GATHER MORE — reason | CONCLUDE — reason
```

Before the final answer, the agent also writes a `Self-critique:` — the strongest
counter-argument to its conclusion, with an explicit check for absence-of-evidence
vs evidence-of-absence confusion.

Both fields are extracted and stored in the result record for research analysis.

### Datasets

All datasets are Polymarket binary yes/no markets with daily `history_prices` snapshots.
The `_plus` variants filter for markets that were open for at least that duration,
giving questions with more historical timepoints.

| Dataset | Size | Description |
|---|---|---|
| `polymarket_binary_yesno.jsonl` | ~361 MB | All binary markets |
| `polymarket_binary_weekly_plus.jsonl` | ~212 MB | Markets open ≥1 week |
| `polymarket_binary_monthly_plus.jsonl` | ~47 MB | Markets open ≥1 month |
| `polymarket_binary_yearly_plus.jsonl` | ~382 KB | Markets open ≥1 year |

### Data Leakage Prevention

In time-series mode, the agent receives only the **timepoint date** as its cutoff —
the market's close/resolution date is never passed to the agent. This prevents the
agent from searching for news about the outcome on or after resolution day.
