# Reproducibility Guide
**Last updated:** 2026-04-04

Everything a reviewer or collaborator needs to reproduce any result from scratch.
Items marked ⚠️ are gaps that must be filled before paper submission.

---

## 1. Hardware

| Component | Value |
|-----------|-------|
| GPU | ⚠️ *Not recorded — document which GPU(s), VRAM, and whether single or multi-card* |
| CPU / RAM | ⚠️ *Not recorded* |
| OS | Linux (Arch Linux, kernel 6.19.10-arch1-1) |
| Server | ASU HPC cluster (`ad.asu.edu` domain) |

**Action required:** Run `nvidia-smi` and record GPU model, VRAM, driver version. Add to this section.

---

## 2. Software Environment

### Python
```
Python >= 3.12
Virtual environment: .venv (uv or pip)
```

### Key packages

| Package | Version (from config) | Role |
|---------|----------------------|------|
| vllm | ≥ 0.16.0 (confirmed: 0.16.0) | LLM serving |
| python-dotenv | ≥ 1.2.2 | Config loading |
| langchain | ≥ 1.2.13 | *(present in requirements.txt, not used by core agent)* |
| langchain-core | ≥ 1.2.19 | *(present, not used by core agent)* |
| langchain-openai | ≥ 1.1.11 | *(present, not used by core agent)* |
| langgraph | ≥ 1.1.2 | *(present, not used by core agent)* |
| langfuse | ≥ 4.0.1 | Tracing (optional) |

⚠️ **Action required:** Run `pip freeze > requirements_frozen.txt` in the active venv and commit it. The current `requirements.txt` lists minimum versions only — exact installed versions are unknown. Reviewers need exact versions to reproduce.

### Model
| Field | Value |
|-------|-------|
| Model | Qwen/Qwen3-14B-AWQ |
| Quantization | AWQ (4-bit) |
| Source | Hugging Face Hub |
| Model revision | ⚠️ *Not pinned — record the exact HF commit hash* |

---

## 3. vLLM Server Launch Command

This is the exact command that must be run before any benchmark:

```bash
vllm serve Qwen/Qwen3-14B-AWQ \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 32768 \
  --quantization awq \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --generation-config vllm
```

**Why each flag is required:**
| Flag | Reason |
|------|--------|
| `--max-model-len 32768` | Context window. Previous session used 16384 — too small after raising max_tokens to 4096+. Must be ≥ 32768. |
| `--quantization awq` | Required for AWQ model weights |
| `--enforce-eager` | Disables CUDA graph capture; required for stability with this model |
| `--enable-auto-tool-choice` | Enables OpenAI-compatible tool/function calling |
| `--tool-call-parser hermes` | Required with `--enable-auto-tool-choice`; matches Qwen3's tool call format |
| `--generation-config vllm` | Use vLLM's generation config rather than model's default |

**Environment variables (`.env` file):**
```env
VLLM_API_BASE=http://127.0.0.1:8002/v1
VLLM_MODEL_NAME=Qwen/Qwen3-14B-AWQ
```

---

## 4. Text2SQL IPC Server

⚠️ **This component is not documented in this repository.** Reviewers will ask about it.

**What is known:**
- Listens on TCP `127.0.0.1:61001`
- Auth key: `"secret123"` (hardcoded in `agent/tools.py`)
- Accepts two engine types: `"hybrid"` (semantic + keyword) and `"sql"` (direct SQL)
- Contains Reddit posts and comments from 19 subreddits (see below)
- Filters results by `cutoff_time` to prevent data leakage

**Action required:**
- Document the server binary/repository name and version
- Document startup command
- Document whether it uses a separate vector index and what embedding model
- Document the SQL schema

---

## 5. Reddit Database

⚠️ **Data collection methodology is not documented in this repository.** This is a critical gap for reproducibility.

**What is known from the code:**
- 19 subreddits: CryptoCurrency, Economics, personalfinance, Entrepreneur, worldnews, politics, science, technology, space, Health, ChatGPT, hardware, music, movies, sports, gaming, weather, todayilearned, AskReddit
- Data includes: posts (submissions) and comments
- Each record has: author, date, content, post_id / comment_id
- The Text2SQL server enforces `cutoff_time` filtering

**Action required — document the following:**
- Reddit API used (PRAW? Pushshift? Arctic Shift? academic API?)
- Date range of collected data (earliest and latest post/comment date)
- Total record counts (posts, comments, per-subreddit)
- Sampling strategy (all posts? top N? score threshold?)
- Any preprocessing or filtering applied
- Whether historical deleted/removed content is included

---

## 6. Polymarket Dataset

**File:** `database/polymarket_binary_yesno.jsonl` (~361 MB, gitignored)

**Structure per record:**
```json
{
  "prediction": {
    "title": "Question text",
    "close_iso_utc": "2026-02-10T...",
    "result": "Yes" or "No",
    "source_payload": {
      "outcomes": "[\"Yes\", \"No\"]",
      "outcomePrices": "[\"1\", \"0\"]",
      "category": "topic string"
    }
  },
  "history_prices": [
    {"outcome_label": "Yes", "date_utc": "2026-02-04T...", "price": 0.9795},
    ...
  ]
}
```

**Filtering applied by `loader.py`:**
- Only records with `ground_truth ∈ {Yes, No}` are loaded
- Records with no `history_prices` are skipped
- Records with no `close_iso_utc` are skipped
- Close date is excluded from agent timepoints (data leakage prevention)

**Available dataset variants:**
| File | Size | Filter |
|------|------|--------|
| `polymarket_binary_yesno.jsonl` | ~361 MB | All binary markets |
| `polymarket_binary_weekly_plus.jsonl` | ~212 MB | Open ≥ 1 week |
| `polymarket_binary_monthly_plus.jsonl` | ~47 MB | Open ≥ 1 month |
| `polymarket_binary_yearly_plus.jsonl` | ~382 KB | Open ≥ 1 year |

⚠️ **Action required:**
- Document total question count and date range of the full dataset
- Document the Polymarket API / data source used to build this file
- Document whether markets are from a specific category or all categories
- Record the dataset download/snapshot date

---

## 7. Random Seeds and Determinism

| Parameter | Value | Notes |
|-----------|-------|-------|
| LLM temperature | 0.1 | Set in `react_agent.py` and `evals.py` |
| Python `random` seed | ⚠️ Not set | Affects nothing in current code |
| NumPy seed | ⚠️ Not set | Affects nothing in current code |
| vLLM seed | Default (0) | vLLM uses `seed=0` by default per engine config log |
| CUDA determinism | Not enforced | `--enforce-eager` disables cudagraph but not CUDA non-determinism |

**Practical reproducibility:** At temperature=0.1, outputs are nearly deterministic but not guaranteed identical across runs due to floating-point non-determinism in GPU ops. For full bit-exact reproducibility, temperature=0 would be needed, but that may hurt calibration quality.

---

## 8. Install and Run from Scratch

```bash
# 1. Clone repo
git clone <repo-url>
cd arc

# 2. Create environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Create .env
echo "VLLM_API_BASE=http://127.0.0.1:8000/v1" >> .env
echo "VLLM_MODEL_NAME=Qwen/Qwen3-14B-AWQ" >> .env

# 4. Start vLLM (separate terminal, keep running)
vllm serve Qwen/Qwen3-14B-AWQ \
  --host 127.0.0.1 --port 8002 \
  --max-model-len 32768 --quantization awq \
  --enforce-eager --enable-auto-tool-choice \
  --tool-call-parser hermes --generation-config vllm

# 5. Start Text2SQL IPC server (separate terminal, keep running)
# ⚠️ Command not yet documented — see REPRODUCIBILITY.md §4

# 6. Run zero-shot baseline
python agent/evals.py --zero_shot --timeseries --max_questions 100

# 7. Run ReAct agent
python agent/evals.py --timeseries --max_questions 100

# 8. Evaluate
python eval/eval_forecasting.py agent/results/<file>.jsonl
```
