# ARC — Agent Research & Curation

A retrieval-augmented LLM agent evaluation system for binary forecasting on Polymarket questions, targeting EMNLP 2026. The agent (Qwen3-14B-AWQ) answers yes/no prediction market questions by searching a Reddit knowledge base. Evaluated across **7 experimental setups** that ablate tools, reflection, and carry-forward belief revision.

## Overview

```
Polymarket JSONL dataset (939 binary yes/no questions)
        ↓
agent/evals.py          ← benchmark harness (question-by-question or time-series)
        ↓ per question (parallel via --parallel N)
agent/react_agent.py    ← ReAct while-loop (LLM + tools, max 20 iterations)
        ↓
shared/llm.py           ← OpenAI-compatible client (vLLM / Qwen3-14B-AWQ)
        ↓ tool calls
agent/tools.py          ← 5 tools: hybrid/vector search, post, comment, thread, author
        ↓
d_agent_client/client.py ← IPC TCP client → Text2SQL server → Reddit DB (19 subreddits)
        ↓
agent/results/*.jsonl   ← one record per (question[, timepoint])
        ↓
eval/eval_forecasting.py ← metrics + statistical comparison (bootstrap CI, McNemar, Wilcoxon)
```

## 7-Setup Experiment Framework

| Setup | Name | Tools | Reflection | Carry-Forward |
|-------|------|-------|------------|---------------|
| 1 | Zero-Shot | No | No | No |
| 2 | ReAct | Yes | No | No |
| 3 | ReAct + Reflection | Yes | Yes | No |
| 4 | ReAct + Conclusion Carry | Yes | No | Conclusion |
| 5 | ReAct + Self-Critique Carry | Yes | No | Critique |
| 6 | ReAct + Both Carry | Yes | No | Both |
| 7 | ReAct + Full | Yes | Yes | Both |

**Research questions answered by each comparison:**
- Setup 1 vs 2: Does retrieval improve calibration? (RQ1)
- Setup 2 vs 3: Does reflection improve prediction quality? (RQ2)
- Setup 2 vs 4/5/6: Does cross-timepoint memory help? (RQ3)
- Setup 4 vs 5: Which carry-forward type is more effective? (RQ4)
- Setup 6 vs 7: Does combining all components help? (RQ5)

## Project Structure

```
arc/
├── agent/
│   ├── react_agent.py      # ReAct loop + carry-forward generation
│   ├── tools.py            # 5 tools (hybrid/vector search, post, comment, thread, author)
│   ├── prompts.py          # 7 setup prompts + carry-forward templates + XML output
│   ├── evals.py            # Benchmark harness: question-by-question + time-series, parallel
│   └── results/            # Output JSONL files (gitignored)
├── database/
│   ├── loader.py           # Polymarket dataset loaders (single + time-series)
│   └── *.jsonl             # Polymarket datasets (gitignored, ~361 MB primary)
├── d_agent_client/
│   └── client.py           # IPC TCP client to Text2SQL server (127.0.0.1:61001)
├── eval/
│   ├── eval_forecasting.py # Metrics suite + statistical comparison mode
│   └── results_to_html.py  # JSONL → HTML viewer with collapsible tool/thinking sections
├── shared/
│   └── llm.py              # LLM client init from .env
├── paper/                  # Research documentation (see below)
├── run.py                  # Single interactive query CLI
├── pyproject.toml
└── requirements.txt
```

## Setup

### Requirements

- Python 3.12+
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
```

### Start Infrastructure

```bash
# vLLM server (separate terminal, keep running)
vllm serve Qwen/Qwen3-14B-AWQ \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 32768 --quantization awq \
  --enforce-eager --enable-auto-tool-choice \
  --tool-call-parser hermes --generation-config vllm

# Text2SQL IPC server (separate terminal, keep running)
# See d_agent_client/d_agent_usage.md for details
```

## Usage

### Quick Start — Question-by-Question (Recommended)

```bash
# Zero-shot baseline (thinking ON is fine — single fast LLM call)
python agent/evals.py --setup 1 --max_questions 939 --parallel 8

# ReAct agent (--no_thinking recommended for parallel runs — see note below)
python agent/evals.py --setup 2 --max_questions 939 --parallel 8 --no_thinking

# Run as background job with logging
nohup python -u agent/evals.py --setup 2 --max_questions 939 --parallel 8 --no_thinking \
  > logs/react_s2_nothink.log 2>&1 &

# Monitor progress
tail -f logs/react_s2_nothink.log
wc -l agent/results/results_s2_*.jsonl

# All 7 setups
for s in 1 2 3 4 5 6 7; do
  python agent/evals.py --setup $s --max_questions 939 --parallel 8 --no_thinking
done
```

> **Why `--no_thinking` for ReAct?** With thinking ON + `--parallel 8`, GPU contention causes each LLM call to take ~150s. The agent only manages 1-2 tool calls before the 300s timeout — barely researching. With thinking OFF, calls take ~5-15s, enabling 5-8 tool calls within budget. More evidence gathered = better predictions. See `paper/FINDINGS.md` Finding 9.

### Time-Series Mode (Temporal Analysis)

```bash
# Time-series with carry-forward (needed for setups 4-7)
python agent/evals.py --timeseries --setup 7 --max_questions 100 --parallel 8

# Cap timepoints per question
python agent/evals.py --timeseries --setup 2 --max_questions 100 --max_timepoints 10 --parallel 8
```

### Evaluate and Compare

```bash
# Single setup metrics
python eval/eval_forecasting.py agent/results/results_s2_XXXXXX.jsonl

# Statistical comparison between two setups
python eval/eval_forecasting.py results_s1.jsonl --compare results_s2.jsonl

# Export all formats
python eval/eval_forecasting.py results.jsonl --format all --output-dir ./eval_results/
```

### Other Operations

```bash
# Resume interrupted run
python agent/evals.py --setup 2 --parallel 8 --results_file agent/results/results_s2_XXXXXX.jsonl

# Summarize existing results
python agent/evals.py --summarize agent/results/results_XXXXXX.jsonl

# Single interactive query
python run.py --query "Will Bitcoin hit 100k by Feb 2026?" --cutoff_time 2026-02-01

# Use a different dataset variant
python agent/evals.py --setup 2 --dataset database/polymarket_binary_monthly_plus.jsonl
```

### CLI Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--setup N` | Setup number 1-7 | 2 (ReAct) |
| `--parallel N` | Concurrent questions (vLLM batches automatically) | 1 (sequential) |
| `--max_questions N` | Limit number of questions | All |
| `--timeseries` | Time-series mode (multiple timepoints per question) | Off |
| `--max_timepoints N` | Cap timepoints per question (evenly spaced) | All |
| `--timeout N` | Per-question timeout in seconds | 300 |
| `--results_file PATH` | Resume from existing JSONL | Auto-generate |
| `--no_thinking` | Disable `<think>` blocks for faster inference | Thinking on |
| `--dataset PATH` | Path to dataset JSONL | `polymarket_binary_yesno.jsonl` |

## Output Format

All setups produce structured XML predictions (Prompt v4):
```xml
<prediction>
  <yes>0.72</yes>
  <no>0.28</no>
  <reasoning>2-4 sentence explanation</reasoning>
</prediction>
```

### Result Record Fields

| Field | Q-by-Q | Time-Series | Description |
|---|---|---|---|
| `question_id` | ✓ | ✓ | Sequential question index |
| `setup` | ✓ | ✓ | Setup number (1-7) |
| `question` | ✓ | ✓ | Question text |
| `options` | ✓ | ✓ | Always `["Yes", "No"]` |
| `ground_truth` | ✓ | ✓ | Resolved market outcome |
| `cutoff_date` | ✓ | — | Market close date (agent's cutoff) |
| `timepoint_index` | — | ✓ | Position in time series (1-based) |
| `timepoint_date` | — | ✓ | Date used as cutoff (NOT close date) |
| `total_timepoints` | — | ✓ | Total timepoints for this question |
| `close_date` | — | ✓ | Market resolution date (never sent to agent) |
| `predicted` | ✓ | ✓ | Agent's selected option ("Yes"/"No") |
| `agent_prob_yes` / `agent_prob_no` | ✓ | ✓ | Agent probabilities from XML |
| `market_prob_yes` / `market_prob_no` | ✓ | ✓ | Market probability |
| `reflection` | ✓ | ✓ | Reflection block (setups 3, 7) |
| `self_critique` | ✓ | ✓ | Self-critique text |
| `carry_conclusion` / `carry_critique` | — | ✓ | Carry-forward XML (setups 4-7) |
| `tool_call_count` | ✓ | ✓ | Number of tool calls made |
| `latency_sec` | ✓ | ✓ | Wall-clock seconds |
| `error` | ✓ | ✓ | `null`, `GRACEFUL_TIMEOUT`, `TIMEOUT`, `TOKEN_LIMIT`, or error message |

## Agent Design

### ReAct Loop (`agent/react_agent.py`)

Plain while-loop, no frameworks. Each iteration:
1. Call LLM with accumulated messages + tool schemas
2. If response has tool calls → execute → append results → continue
3. If no tool calls → return final answer

Max 20 iterations. 300-second timeout per question/timepoint. Full reasoning trace captured across all iterations.

**Timeout handling:** Uses `threading.Timer` + `threading.Event` for reliable timeout across all threads (SIGALRM is unreliable during C-extension calls). When the budget expires, a **graceful timeout** makes one final LLM call (no tools) to extract a prediction from evidence gathered so far. Results tagged `GRACEFUL_TIMEOUT` (got prediction) or `TIMEOUT` (extraction failed).

### Tools (`agent/tools.py`)

| Tool | Purpose |
|---|---|
| `search_database` | Vector semantic search across Reddit posts/comments (default). Hybrid mode available but slower. Supports `month`, `authors`, `subreddit` filters. |
| `get_post_core_info` | Full content and metadata for a specific post |
| `get_comment_core_info` | Full content and metadata for a specific comment |
| `get_post_comments_list` | Thread context around a comment (ancestors + descendants) |
| `get_author_history_list` | Chronological post/comment history for an author |

All tools respect `cutoff_time` — results are filtered to dates before the cutoff.

### Data Leakage Prevention

The agent receives only the **cutoff date** (timepoint date in time-series mode, close date in question-by-question mode). The market's resolution date is never passed to the agent, preventing the agent from searching for outcome-revealing news.

## Research Documentation

| File | Contents |
|------|----------|
| `paper/PROPOSAL.md` | Full research proposal for EMNLP 2026 |
| `paper/CHECKPOINT.md` | Project state, architecture, error history, workflow |
| `paper/EXPERIMENTS.md` | Result file → run condition mapping, planned runs |
| `paper/DECISIONS.md` | 10 design decisions (all resolved) |
| `paper/FINDINGS.md` | Empirical findings from prior runs |
| `paper/PROMPTS_VERSIONED.md` | Exact prompt text for each version (v2-v4) |
| `paper/DATASET.md` | Dataset documentation and statistics |
| `paper/REPRODUCIBILITY.md` | Hardware, software, setup guide for reviewers |
