# ARC Project — Claude Code Instructions

## What This Project Is
ARC is a research eval system for an EMNLP paper. A ReAct LLM agent (Qwen3-14B-AWQ) uses Reddit retrieval to forecast binary Polymarket questions. Evaluated in time-series mode across 7 experimental setups with 3 ablation variables: tools, reflection, and carry-forward belief revision.

## How to Pick Up Where We Left Off
1. Read `paper/PROPOSAL.md` — the full research proposal with all design decisions, methodology, and submission strategy.
2. Read `paper/CHECKPOINT.md` §10-11 — known issues and decision status (all resolved).
3. The `paper/EXPERIMENTS.md` file maps every result file to its exact run conditions + planned runs.
4. The `paper/FINDINGS.md` file documents empirical findings from prior runs (v2 prompt, pre-framework).
5. All research docs live in `paper/` — do NOT create duplicates at the root.

## What Was Done Last (2026-04-09, session 7)
- **Fixed broken `_parse_text_tool_calls()` regex** — Qwen3 sometimes writes tool calls as plain-text `<tool_call>` blocks instead of using the function-calling interface. The fallback parser's regex `[^}]+` couldn't handle nested braces in the arguments JSON (`"arguments": {"query": "...", "cutoff_time": "..."}`), so it silently failed on every call. Replaced with brace-counting parser. This was the root cause of Q35 getting 0 tool calls across 20 iterations (177s wasted).
- **Trimmed search results for LLM context** — `_trim_search_result()` strips per-hit noise fields (doc_id, month, author, created_utc, created_at, title, distance) and top-level echo (question, cutoff_time, table, top_k, engine, filters). Raw results still used for auto-drilldown parsing. Measured across 44 real eval questions: **54% reduction** in search data (1,155K → 530K chars), saving ~3,556 tokens per question (11% of 32K context). Before trim: all 3 test questions hit CONTEXT_LIMIT. After trim: all 3 complete naturally with Error: None.
- **Split GRACEFUL_TIMEOUT into two error types:**
  - `GRACEFUL_TIMEOUT` — time budget (120s default) exceeded, wraps up with prediction from evidence
  - `CONTEXT_LIMIT` — context window full, wraps up with prediction from evidence
  - Updated `eval_forecasting.py` to track both types separately
- **Reduced default timeout** — `DEFAULT_TIMEOUT_SEC` 300→120 in `evals.py` and `test_react_v2.py`
- **Test results after fixes (thinking ON, Setup 2, 3 questions):**
  - Q43 Eichorn: CORRECT, No P(Yes)=0.1, 41 tools, 47.1s, Error: None
  - Q35 Trump deport: CORRECT, No P(Yes)=0.2, 43 tools, 52.7s, Error: None
  - BTC $100k: CORRECT, Yes P(Yes)=0.65, 34 tools, 61.6s, Error: None
  - All 3 complete naturally with 34-43 tool calls, 47-62s latency
- **44-question partial eval exists** — `agent/results/results_s2_20260409_213216.jsonl` (pre-trim, all Error: None)
- **HTML reports generated:**
  - `eval/react_v2_test_results.html` — pre-trim 3 questions (all GRACEFUL_TIMEOUT)
  - `eval/react_v2_trimmed_results.html` — post-trim 3 questions (all Error: None)

### How the ReAct loop works now
1. **Round 1**: LLM generates 5-6 hypothesis-driven searches in parallel → auto-drilldown fetches top 3 per search → code injects "Round 1 of 3 complete, identify NEW leads" message
2. **Round 2**: LLM sees all Round 1 evidence + think blocks → does 3-4 follow-up searches based on new leads → auto-drilldowns → "Round 2 of 3 complete" message
3. **Round 3**: LLM sees all accumulated evidence → does 2-3 final gap-filling searches → auto-drilldowns → LLM synthesizes and predicts
4. **All messages accumulate** — think blocks, search results (trimmed), drilldowns all persist across rounds. Round 3 sees everything from Rounds 1-2.
5. **Safety nets**: If context fills up → CONTEXT_LIMIT (predict from evidence). If time exceeds 120s → GRACEFUL_TIMEOUT (predict from evidence). If model writes tool calls as text → fallback parser catches them.

### Prior session (2026-04-09, session 6)
- **Deep JSONL analysis of S1 vs S2 results** — identified root causes for S2 underperformance:
  - Discordant cases: 83 S1-only correct vs 56 S2-only correct (retrieval hurts net 27 questions)
  - When retrieval flipped No→Yes: wrong 68% of the time (51/75). Reddit noise > signal.
  - S2 makes 3x more extreme predictions (178 vs 66 at p=0.0/1.0), higher overconfidence
  - Agent treats Reddit opinions as facts (e.g., "charged" → "guilty", "plans" → "will happen")
  - Absence-of-evidence fallacy worse with tools (85 vs 79): search returns nothing → predict No
- **Found doc_id format mismatch bug** — 10% of tool budget wasted:
  - search_database returns compound `doc_id`: `"2025-02:comment:mc806oy"`
  - get_comment_core_info expects plain ID: `"mc806oy"`
  - Agent passes compound format 25.3% of the time (345/1366 drilldown calls) — all fail
  - 345 of 422 not-found errors (81.8%) are from this format mismatch
  - Remaining 77 are IDs in vector index but missing from SQL lookup table
- **Tool usage analysis:** agent uses all 5 tools but skewed — 66% search, 24% comment lookup,
  9% post lookup, <1% thread/author tools. 61% of questions never drill down past search.
- **Full analysis documented** in `eval/results_summary.txt`

### Prior session (2026-04-09, session 5)
- **Overnight runs completed (S1 + S2, 1000q each):**
  - S1 (ZS): 75.0% acc, 56.3% balanced acc, MCC 0.191, Brier 0.198, BSS -2.45
  - S2 (ReAct, `--no_thinking --parallel 8`): 72.0% acc, 53.3% balanced acc, MCC 0.095, Brier 0.207, BSS -2.70
  - Statistical comparison: no significant difference (Brier p=0.37, McNemar p=0.065, Wilcoxon p=0.87)
  - ZS slightly outperforms ReAct on all metrics, but this is a `--no_thinking` proof-of-concept run, not final paper results
- **Fixed eval_forecasting.py error classification:**
  - `GRACEFUL_TIMEOUT` was falling into `other` bucket — misreported as 75.9% error rate
  - Reality: 2.0% hard errors (20 NoneType, no prediction), 73.9% graceful timeouts (739, all produced valid predictions)
  - Graceful timeout accuracy (72.4%) actually slightly higher than clean accuracy (71.0%)
  - Added: `hard_error_rate`, `graceful_timeout_rate`, accuracy split by completion type
  - Skips empty topic section when no real topics exist
  - `--compare --format json` now outputs structured JSON (was dumping raw text)
- **Added iteration tracking:**
  - `react_agent.py` now tracks and returns `iteration_count` (ReAct loop iterations, distinct from tool calls — 1 iteration can have multiple tool calls)
  - `evals.py` writes `iteration_count` to JSONL output
  - `eval_forecasting.py` reports avg/median/max ReAct iterations when the field is present
- **Eval results generated:** `eval/eval_s1/`, `eval/eval_s2/`, `eval/comparison_s1_vs_s2.json`

### Prior session (2026-04-08, session 4)
- **Standardized field naming:** Renamed `human_prob_yes`/`human_prob_no` → `market_prob_yes`/`market_prob_no` across all code and docs. Both q-by-q and time-series modes now use the same field name. Previous inconsistency: time-series loader used `human_prob_*`, q-by-q loader used `market_prob_*`, eval script only knew `human_prob_*` — market comparison silently skipped for q-by-q results.
- **Fixed market prob extraction for q-by-q mode:** `_extract_market_probs()` in `database/loader.py` now falls back to last available price when cutoff date predates price history (300/1000 questions were returning null).
- **Fixed eval field mismatch:** `eval/eval_forecasting.py` now reads `market_prob_yes` consistently (was `human_prob_yes`).
- **Backfilled S1 results:** Existing `results_s1_20260408_160848.jsonl` backfilled with correct market probs (300 nulls → 0 nulls).
- **Created `run_overnight.sh`:** Dual-GPU script — GPU 0 for ReAct (port 8000), GPU 1 for ZS (port 8001, 0.8 memory). Kills only user's processes (`pkill -u ezhu4`), waits for servers, launches both experiments.
- **S1 ZS baseline eval (1000q):** 74.0% acc, 55.2% balanced acc, MCC 0.151, Brier 0.200 vs market 0.055, BSS -2.63.

### Prior session (2026-04-08, session 3)
- **Found root cause of ALL timeout issues (Error 10):** `search_database` defaulted to hybrid engine (260s/query). Vector engine = 0.8s (325× faster). Confirmed with DB team: "Do not use hybrid. Using vector only search."
- **Switched default engine to vector** in `agent/tools.py` — function signature, fallback, and tool schema
- **Validated end-to-end:** Sequential no_thinking: 79s, 11 tool calls, completes normally. Parallel 8 no_thinking: 266s/batch, 5-6 tool calls, some graceful timeouts from GPU contention.
- **Thinking ON test results:** Sequential 89s/3 tools (completes), parallel 8: 518s/3-4 tools (mixed errors)

### Prior session (2026-04-08, session 2)
- **Fixed nested ThreadPoolExecutor deadlock (Error 5)** — removed nested executors, worker threads call `run_react()` directly
- **Fixed SIGALRM unreliability (Error 8)** — replaced with `threading.Timer` + `cancel_event`
- **Added graceful timeout (Error 9)** — final LLM call extracts prediction from partial evidence
- **Reduced per-call HTTP timeout** — 120s → `min(90s, timeout_sec)`
- **Updated all documentation**

### Prior session (2026-04-08, session 1)
- **Added parallel execution to question-by-question mode** — `--parallel N` now works in both `benchmark()` and `benchmark_timeseries()`, enabling concurrent question processing via ThreadPoolExecutor + vLLM continuous batching
- **Made `run_zero_shot()` thread-safe** — split into `_run_zero_shot_impl()` + timeout wrapper
- **Added `--setup` support to `benchmark()`** — all 7 setups now work in question-by-question mode
- **Diagnosed nested executor deadlock** — systematic isolation via test_vllm.py, test_react.py, test_run_single.py

### Prior session (2026-04-07)
- Designed 7-setup experiment framework — factorial ablation of tools × reflection × carry-forward
- Implemented all 7 setup prompts in `agent/prompts.py` with XML `<prediction>` output format
- Added carry-forward system — conclusion + self-critique generation, cross-timepoint context passing
- Added tool capabilities — vector search, month filter, authors filter in `agent/tools.py`
- Added parallel execution for timeseries mode — `--parallel N` in `benchmark_timeseries()`
- Added statistical tests — bootstrap CI, McNemar, Wilcoxon in `eval/eval_forecasting.py --compare`
- Created `paper/PROPOSAL.md` — full research proposal

## What's Next (Priority Order)

### Completed ✓
- ~~Overnight runs (S1 ZS + S2 ReAct, 1000q each)~~ — done, eval results in `eval/eval_s1/`, `eval/eval_s2/`
- ~~Eval comparison S1 vs S2~~ — done, `eval/comparison_s1_vs_s2.json`
- ~~Fix doc_id format mismatch~~ — `_strip_compound_id()` in `tools.py` strips compound prefix
- ~~Fix text tool call parser~~ — brace-counting parser replaces broken regex
- ~~Trim search results~~ — 54% smaller, eliminates CONTEXT_LIMIT on 3 test questions
- ~~Split GRACEFUL_TIMEOUT / CONTEXT_LIMIT~~ — separate error types for time vs context limits

### Next steps
1. **Run 100-question eval** — `python3 agent/evals.py --max_questions 100 --setup 2 --parallel 4 --timeout 120` to validate trim + parser fixes at scale
2. **Prompt debiasing** — agent predicts Yes only 9-10% vs 25.4% GT. Add NO-bias mitigation. See `eval/results_summary.txt` "FIXES TO APPLY" section.
3. **Consider testing Setup 3** (reflection addon) — current Setup 2 think blocks do planning across rounds but no self-critique. Setup 3 forces structured `<reflection>` blocks with error/bias checking.
4. **Reduce auto-drilldowns from 3→2 per search** — optional further context savings if needed
5. **Validation run:** All 7 setups on 20 questions to verify XML parsing + carry-forward
6. **Full run with thinking ON:** All 7 setups on 939 questions — `--parallel 4`. This is the final paper data.
7. **Clean up test files:** `test_vllm.py`, `test_react.py`, `test_run_single.py`, `test_bottleneck.py`, `test_db_speed.py`, `test_full_speed.py`, `test_thinking_speed.py`, `test_vllm_health.py`, `test_nothink_speed.py`

## Critical Config Notes
- **Vector engine MUST be default** — hybrid takes 260s/query, vector takes 0.8s. Changed in `agent/tools.py`.
- **Search results are trimmed** — `_trim_search_result()` in `react_agent.py` strips noise fields before sending to LLM. Raw results still used for auto-drilldown. Saves ~3,556 tokens/question.
- **Default timeout is 120s** — changed from 300s. Override with `--timeout N`.
- **Thinking ON for final paper results** — ~50s/q with trim (was 65s before trim). Use `--parallel 4` for thinking ON runs.
- **ZS does not need `--no_thinking`** — single LLM call, fast regardless (~17 min for 939q at parallel 8).
- **vLLM on port 8001**, D agent on port 61001. `.env` configured for both.

## Important Rules
- **Explain before executing.** Always present proposed changes for user review before modifying code. User makes research design decisions.
- **Do not run experiments** without confirming infrastructure is up (vLLM on port 8000, Text2SQL on port 61001).
- **Tag results with prompt version.** v2 = `Confidence:` format. v3 = `P(Yes):` format. v4 = `<prediction>` XML format (current HEAD). They are not directly comparable.
- **The .venv has a broken python symlink.** Use system `python3` or fix the venv if needed.
- **Thinking always on.** Do not use `--no_thinking`. Evaluate at LLM's best performance.

## Key Files
- `agent/react_agent.py` — ReAct loop + carry-forward generation
- `agent/evals.py` — benchmark harness (7 setups, parallel, carry orchestration)
- `agent/prompts.py` — 7 setup prompts + carry-forward templates + XML output format
- `agent/tools.py` — 5 Reddit tools (hybrid/vector search) + OpenAI schemas
- `eval/eval_forecasting.py` — metrics suite + statistical comparison mode
- `eval/results_to_html.py` — JSONL→HTML viewer
- `paper/PROPOSAL.md` — full research proposal
- `paper/CHECKPOINT.md` — project state, architecture, error history
- `paper/EXPERIMENTS.md` — result file → run condition mapping
- `paper/DECISIONS.md` — design decisions (all resolved)
- `paper/FINDINGS.md` — empirical findings from prior runs
- `paper/DATASET.md` — dataset documentation
- `paper/REPRODUCIBILITY.md` — reproducibility guide
