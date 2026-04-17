# ARC — LLM Agent for Polymarket Forecasting (V2.1)

A ReAct LLM agent that forecasts binary Polymarket questions by retrieving evidence from a Reddit corpus. Built for an EMNLP 2026 analysis paper.

## Architecture

```
Polymarket question (binary yes/no, with cutoff date)
        ↓
agent/evals.py              ← eval harness (parallel, timeout, JSONL output)
        ↓
agent/react_agent.py        ← V2.1 ReAct loop (3 structured rounds)
        ↓                       Round: <state>/<plan>/<queries> → tools → <rethink>
shared/llm.py               ← OpenAI-compatible client (vLLM)
        ↓ tool calls
agent/tools.py              ← 5 tools: search, post, comment, thread, author
        ↓
d_agent_client/client.py    ← TCP client → Text2SQL vector engine → Reddit DB
        ↓
results/*.jsonl              ← one row per question
        ↓
eval/eval_forecasting.py    ← metrics + paired statistical tests
```

## V2.1 Pipeline — What Happens Per Question

1. **Eval harness** pulls a `(question, description, cutoff_time, ground_truth)` row
2. **Round 1**: LLM emits `<state>` (evidence/gaps/p_yes) → `<plan>` → `<queries>` (3-5 hypothesis-driven searches)
3. **Parallel tool execution**: `search_database()` calls hit the D-agent vector engine (~0.8s each)
4. **Auto-drilldown**: top-3 hits per search auto-fetched via `get_post_core_info` / `get_comment_core_info`
5. **Rethink**: LLM synthesizes findings in visible text (persists in message history)
6. **Rounds 2-3**: same pattern, targeting gaps identified in prior rounds
7. **Final prediction**: guided-regex forces valid `<prediction><yes>0.XX</yes><no>0.YY</no><reasoning>...</reasoning></prediction>` XML
8. **Extraction**: `extract_agent_prediction()` parses XML → JSONL row

## Key Files

| File | Purpose |
|---|---|
| `agent/react_agent.py` | ReAct loop — `run_react()` (legacy) and `run_react_v2()` (structured rounds) |
| `agent/evals.py` | Eval harness — parallel execution, timeout, JSONL output |
| `agent/prompts.py` | System prompts for 7 setups + round nudge templates |
| `agent/tools.py` | 5 Reddit tools (all vector-engine) + OpenAI tool schemas |
| `database/loader.py` | Dataset loader — `load_polymarket_timeseries()` |
| `eval/eval_forecasting.py` | Metrics (accuracy, F1, Brier) + paired tests (bootstrap, McNemar, Wilcoxon) |
| `d_agent_client/client.py` | TCP client for Text2SQL D-agent |
| `shared/llm.py` | OpenAI-compatible LLM client wrapper |

## Running

**Prerequisites:** vLLM serving a model on port 8000, D-agent on port 61001. Configure in `.env`.

```bash
# Single question test
python3 agent/evals.py --setup 2 --v2 --no_thinking --max_questions 1 --timeout 120

# 1000-question eval (V2.1, no thinking, 8 parallel)
python3 agent/evals.py --setup 2 --v2 --no_thinking \
    --max_questions 1000 --parallel 8 --timeout 120 \
    --results_file results/eval_v21_<stamp>.jsonl

# Compare two result files (paired stats)
python3 eval/eval_forecasting.py results/A.jsonl --compare results/B.jsonl --output analysis/compare.txt
```

## Code-Level Constraints

| Constraint | Location | Purpose |
|---|---|---|
| Guided-regex on `<prediction>` XML | `react_agent.py` `_guided_final_prediction` | Zero parse failures |
| Search result trimming | `react_agent.py` `_trim_search_result` | ~40% context savings |
| Forced 3-round minimum | `react_agent.py` round counter | Prevent premature prediction |
| Structured round nudge (FOUND/GAPS/P(Yes)) | `react_agent.py` round nudge injection | Force evidence synthesis between rounds |
| Fuzzy search dedup | `react_agent.py` `_is_near_duplicate` | Reject >60% word overlap queries |
| Auto-drilldown dedup | `react_agent.py` `seen_drilldowns` set | Don't re-fetch same document |
| Compound-id normalization | `tools.py` `_strip_compound_id` | Fix doc_id format mismatch |
| Vector engine forced | `tools.py` `engine="vector"` | 325× faster than hybrid engine |
| Text tool-call fallback | `react_agent.py` `_parse_text_tool_calls` | Catch plain-text tool calls |

## 7-Setup Experiment Framework

| Setup | Name | Tools | Reflection | Carry-Forward |
|---|---|---|---|---|
| 1 | Zero-Shot | No | No | No |
| 2 | ReAct (V2.1) | Yes | No | No |
| 3 | ReAct + Reflection | Yes | Yes | No |
| 4 | ZS + Carry-Forward | No | No | Yes |
| 5 | ReAct + CF | Yes | No | Yes |
| 6 | ReAct + Reflection + CF | Yes | Yes | Yes |
| 7 | ZS + CF (no tools) | No | No | Yes |

Current focus: **Setup 2 (V2.1)** — the reference agent pipeline.
