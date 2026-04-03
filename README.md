# P-Agent: ReAct Forecasting Agent

A minimal ReAct agent that answers binary (Yes/No) prediction questions by
searching a Reddit corpus via semantic + full-text retrieval, then generating a
probability estimate. Built for evaluation against the Polymarket dataset as part
of a research paper on LLM-based forecasting.

No LangChain. No LangGraph. No LangFuse. Just Python + OpenAI API + a while-loop.

---

## Project Structure

```
p_agent_workspace/
│
├── agent/
│   ├── agent.py       ReAct while-loop — call LLM → tools → repeat
│   ├── tools.py       5 d_agent IPC tools + OpenAI function schemas
│   ├── prompts.py     System prompt (subreddit routing, output format)
│   └── evals.py       Benchmark harness + metrics summary
│
├── shared/
│   └── llm.py         OpenAI client pointed at local vLLM server
│
├── database/
│   ├── loader.py      Loads Polymarket JSONL → question dicts
│   └── polymarket_binary_yesno.jsonl   (31 K questions, ~345 MB)
│
├── d_agent_client/    IPC client for PostgreSQL + pgvector Reddit database
│
├── eval/
│   └── eval_forecasting.py   Post-run metrics (Brier, F1, calibration, topics)
│
├── run.py             Interactive single-query entry point
├── .env               API base URL, model name, secrets
└── requirements.txt   openai, python-dotenv
```

---

## How It Works

```
Question + cutoff_date
        │
        ▼
  agent/agent.py  ──── run_react() while-loop ────────────────────────────┐
        │                                                                   │
        │  1. Build messages = [system_prompt, user_question]              │
        │  2. Call vLLM via OpenAI API  ──► LLM returns tool_calls?        │
        │         Yes ──► execute tool via d_agent_client                  │
        │                 append result to messages                         │
        │                 loop back                                         │
        │         No  ──► extract final_answer (contains Confidence score) │
        │                                                                   │
        ▼                                                                   │
  Output dict ◄──────────────────────────────────────────────────────────┘
    final_answer    raw LLM text
    tool_call_count number of d_agent calls made
    latency_sec     wall-clock time
    error           None or error string

        │
        ▼
  agent/evals.py  ──── extract_selected_option()  →  "Yes" / "No"
                  ──── extract_confidence()        →  0.0 – 1.0
                  ──── derive agent_prob_yes/no    →  complement pair
                  ──── compare to ground_truth     →  correct: bool
                  ──── write JSONL record
```

The d_agent server is a separate service that wraps a PostgreSQL + pgvector
database of Reddit posts and comments. The agent queries it up to the
`cutoff_date` so it cannot see future information.

---

## JSONL Output Schema

Each evaluated question produces one record:

```json
{
  "question_id":     "42",
  "question":        "Will SEC approve the first spot Bitcoin ETF by Jan 8 2024?",
  "options":         ["Yes", "No"],
  "ground_truth":    "Yes",
  "cutoff_date":     "2024-01-08",
  "topic":           "",
  "final_answer":    "Reasoning: ...\nSelected Option: Yes\nConfidence: 0.82",
  "predicted":       "Yes",
  "correct":         true,
  "confidence":      0.82,
  "agent_prob_yes":  0.82,
  "agent_prob_no":   0.18,
  "market_prob_yes": 0.75,
  "market_prob_no":  0.25,
  "tool_call_count": 4,
  "latency_sec":     23.5,
  "error":           null,
  "timestamp":       "2026-04-02T10:00:00"
}
```

`agent_prob_yes/no` — the agent's own probability estimate from Reddit evidence  
`market_prob_yes/no` — the Polymarket crowd's implied probability at `cutoff_date`

---

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd p_agent_workspace
pip install -r requirements.txt
```

### 2. Configure `.env`

```bash
cp .env.example .env   # or edit .env directly
```

```ini
# .env
VLLM_API_BASE=http://127.0.0.1:8000/v1
VLLM_MODEL_NAME=Qwen/Qwen3-14B-AWQ
```

### 3. Start the vLLM server (GPU required)

The agent uses **Qwen3-14B-AWQ** served via vLLM. Start it once before running
any evaluation.

If you have multiple GPUs, set `CUDA_VISIBLE_DEVICES` to pick a specific one
before launching the server. Check available GPUs first:

```bash
nvidia-smi
```

Then start the server on the desired GPU (replace `0` with the GPU index you
want to use):

```bash
CUDA_VISIBLE_DEVICES=6 nohup python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 8000 \
  --model Qwen/Qwen3-14B-AWQ \
  --quantization awq \
  --max-model-len 16384 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enforce-eager \
  --generation-config vllm \
  > vllm.log 2>&1 &
```

To use a different GPU, change the index — e.g. `CUDA_VISIBLE_DEVICES=1` for
GPU 1, `CUDA_VISIBLE_DEVICES=2` for GPU 2, and so on.

Wait for the server to be ready (check `vllm.log` for `"Application startup complete"`):

```bash
tail -f vllm.log
# Or health-check:
curl http://127.0.0.1:8000/health
```

To stop the server:
```bash
pkill -f "vllm.entrypoints.openai.api_server"
```

### 4. Start the D-Agent server

The D-Agent serves the Reddit PostgreSQL + pgvector database over IPC:

```bash
# Start d_agent server (separate process, keep running during eval)
python d_agent_client/client.py --tcp 127.0.0.1:61001 --authkey secret123 --ping
```

---

## Running the Agent

### Interactive single query

```bash
python run.py \
  --query "Will Bitcoin reach $100k by February 2026?" \
  --cutoff_time 2026-02-01

# Custom options
python run.py \
  --query "Will the Fed cut rates in March 2025?" \
  --cutoff_time 2025-03-01 \
  --options "Yes" "No"
```

### Benchmark evaluation

```bash
# Quick smoke test — 10 questions
python agent/evals.py --max_questions 10

# Larger run — 100 questions
python agent/evals.py --max_questions 100

# Full dataset
python agent/evals.py

# Resume an interrupted run
python agent/evals.py --results_file agent/results/results_20260402_120000.jsonl

# Longer timeout per question (default 300s)
python agent/evals.py --max_questions 100 --timeout 600
```

Results are saved incrementally to `agent/results/results_YYYYMMDD_HHMMSS.jsonl`.
If the run is interrupted, re-run the same command with `--results_file` to resume
from where it left off (already-completed questions are skipped).

### View summary

```bash
python agent/evals.py --summarize agent/results/results_XXXXXX.jsonl
```

Output:
```
=======================================================
Results: results_20260402_120000.jsonl
=======================================================
  Total questions : 100
  Answered        : 97
  Correct         : 68
  Accuracy        : 70.1%
  Errors          : 2
  Timeouts        : 1
  Avg tool calls  : 4.3
  Avg latency     : 24.7s
  Avg confidence  : 0.714
=======================================================
```

---

## Evaluation Process

### Step 1 — Run benchmark

```bash
python agent/evals.py --max_questions 200
```

### Step 2 — Full metrics (Brier score, calibration, topic breakdown)

```bash
python eval/eval_forecasting.py agent/results/results_XXXXXX.jsonl
# Export to JSON
python eval/eval_forecasting.py agent/results/results_XXXXXX.jsonl --output metrics.json
# Export to CSV
python eval/eval_forecasting.py agent/results/results_XXXXXX.jsonl --format csv --output metrics.csv
```

Metrics computed:
- **Accuracy** — fraction correct (Yes/No)
- **Precision / Recall / F1** — binary classification quality
- **MCC / Cohen's Kappa** — balanced accuracy for imbalanced datasets
- **Brier Score** — probability calibration (`agent_prob_yes` vs ground truth)
- **Log Loss** — probabilistic accuracy
- **ECE** — expected calibration error
- **Topic-level breakdown** — per-category accuracy
- **Operational** — avg tool calls, avg latency, error/timeout rates

### Probability trajectory evaluation (timeline mode)

Each Polymarket question has daily price snapshots (`history_prices`). Running
the agent at multiple cutoff dates per question shows how the agent's probability
estimate evolves as the event approaches — comparable to the market's price curve.

```
Date points from history_prices:
  Feb 04  market=0.26  →  agent runs with cutoff=Feb 04  →  agent_prob_yes=?
  Feb 10  market=0.31  →  agent runs with cutoff=Feb 10  →  agent_prob_yes=?
  Feb 20  market=0.16  →  agent runs with cutoff=Feb 20  →  agent_prob_yes=?
  Feb 26  market=0.98  →  agent runs with cutoff=Feb 26  →  agent_prob_yes=?
```

This produces a per-question probability time-series for agent vs. market
comparison, which is the core evaluation contribution for the research paper.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| No LangGraph / LangFuse | A ReAct loop is a `while` loop. Frameworks add ~4 dependencies and ~500 lines for no functional gain. |
| Fresh IPC connection per tool call | Prevents stale-connection errors on long-running GPU queries. |
| `Confidence:` in output format | Converts binary Yes/No to a calibrated probability, enabling Brier score and log-loss evaluation. |
| `agent_prob_yes` = complement of `agent_prob_no` | Yes + No probabilities always sum to 1.0. Market prices confirm this (Yes price + No price ≈ 1.0). |
| `cutoff_time` on every d_agent call | Strict data leakage prevention — agent can only use Reddit posts published before the cutoff date. |
| Subreddit routing in prompt | Narrows search to the most relevant community (e.g., CryptoCurrency for Bitcoin questions), improving evidence quality. |
| max_tokens=1024, temp=0.1 | Deterministic, concise responses within the 16 384-token context window. |

---

## Dependencies

```
openai          OpenAI-compatible client → points at local vLLM
python-dotenv   Load .env config
vllm            (server only, not imported by agent code)
```

The `d_agent_client/` package is a local IPC client — no pip install needed.
