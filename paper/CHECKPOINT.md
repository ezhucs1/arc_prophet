# ARC Project Checkpoint
**Last updated:** 2026-04-09 (session 6)  
**Status:** Proof-of-concept complete — S1 (ZS) and S2 (ReAct) overnight runs done (1000q each, `--no_thinking`). Deep JSONL analysis revealed retrieval introduces confident noise (Finding 16), doc_id format bug wastes 10% of tool budget (Finding 17), and shallow tool usage pattern (Finding 18). Next: fix doc_id bug + prompt debiasing, then thinking ON runs for final paper data.

---

## 1. Purpose & Research Goal

**ARC** (Agent Research & Curation) evaluates whether a retrieval-augmented LLM agent can outperform prediction markets on binary forecasting questions.

**Core hypothesis:**  
A ReAct agent searching a Reddit knowledge base produces better-calibrated probability estimates than a zero-shot LLM, and can provide useful signal where the market has low liquidity or lagging price discovery.

**Target venue:** EMNLP (Empirical Methods in Natural Language Processing)

**Research questions:**
1. Does tool-augmented retrieval improve calibration over zero-shot? (RQ1)
2. Does within-timepoint structured reflection improve prediction quality? (RQ2)
3. Does cross-timepoint memory (belief revision) improve calibration over time? (RQ3)
4. What type of carry-forward is most effective: factual conclusions, reasoning self-critique, or both? (RQ4)
5. Does combining all components produce synergistic improvement? (RQ5)
6. On which question categories does social media RAG add value over prediction markets? (RQ6)

**Design principle — LLM autonomy:**  
The agent decides freely: which tools to call, how many times, in what order, when to stop, and what to conclude. No external messages are injected to guide the decision. No tools are removed mid-run. Any infrastructure limits (context window math) are handled transparently without touching the agent's decision space.

---

## 2. System Architecture

```
User Question (Polymarket binary yes/no)
    │
    ▼
agent/evals.py  (benchmark harness)
    ├── benchmark()            — single-pass, one prediction per question
    └── benchmark_timeseries() — N predictions per question (one per historical timepoint)
            │
            ▼
    agent/react_agent.py  (ReAct loop, no frameworks)
    ├── System prompt (agent/prompts.py) — REACT_SYSTEM_PROMPT or _REFLECTION variant
    ├── LLM: Qwen/Qwen3-14B-AWQ via vLLM (OpenAI-compatible API, port 8000)
    └── Tools (agent/tools.py) — 5 tools → Text2SQL IPC server (TCP 127.0.0.1:61001)
            │
            ▼
    Reddit database (hybrid semantic + keyword search)
            │
            ▼
    Output: JSONL records → agent/results/
            │
            ▼
    eval/eval_forecasting.py  — metrics suite
```

### Infrastructure
| Component | Detail |
|-----------|--------|
| LLM | Qwen/Qwen3-14B-AWQ |
| Serving | vLLM v0.16.0 |
| Context window | 32,768 tokens (`--max-model-len 32768`) |
| vLLM launch flags | `--tool-call-parser hermes --enable-auto-tool-choice` |
| DB backend | Text2SQL IPC server, TCP 127.0.0.1:61001 |
| Dataset | Polymarket binary yes/no questions (JSONL, gitignored) |

---

## 3. Evaluation Modes — 7-Setup Framework (Finalized 2026-04-07)

| Setup | Name | CLI | Tools | Reflection | Carry-Forward |
|-------|------|-----|-------|------------|---------------|
| 1 | Zero-Shot | `--setup 1` | No | No | No |
| 2 | ReAct | `--setup 2` | Yes | No | No |
| 3 | ReAct + Reflection | `--setup 3` | Yes | Yes | No |
| 4 | ReAct + Conclusion Carry | `--setup 4` | Yes | No | Conclusion |
| 5 | ReAct + Self-Critique Carry | `--setup 5` | Yes | No | Critique |
| 6 | ReAct + Both Carry | `--setup 6` | Yes | No | Both |
| 7 | ReAct + Full | `--setup 7` | Yes | Yes | Both |

### Evaluation Modes

**Question-by-question mode (default):** One prediction per question using the market close date as cutoff. No timepoints. Fastest way to evaluate all 7 setups.

**Time-series mode (`--timeseries`):** Agent evaluated at N historical price timepoints per question. Cutoff date = that timepoint's date (NOT the resolution date). Simulates real-time forecasting with no future leakage. Required for carry-forward setups (4-7) to function meaningfully.

**Carry-forward (time-series only):** For setups 4-7, after each timepoint a separate LLM call generates structured XML (conclusion and/or self-critique). This is passed as context to the next timepoint within the same question.

### Parallel Execution

**`--parallel N`** processes N questions concurrently in **both** modes:
- **Question-by-question:** N independent questions run simultaneously. All questions are independent.
- **Time-series:** N questions run simultaneously. Timepoints within each question remain sequential (required for carry-forward).

vLLM's continuous batching handles concurrent requests efficiently — multiple parallel requests are batched on the GPU automatically, giving much better GPU utilization than sequential processing.

**Default:** `--parallel 1` = sequential (original behavior). No code path changes when parallel is not used.

**Legacy flags:** `--zero_shot`, `--reflection` still work (map to setup 1, 3 respectively). `--setup` overrides them.

---

## 4. Output Schema (time-series JSONL)

```json
{
  "question_id":      "1",
  "setup":            2,
  "timepoint_index":  1,
  "timepoint_date":   "2026-02-04",
  "total_timepoints": 6,
  "question":         "Will DoorDash report less than 800M orders in Q4 2025?",
  "options":          ["Yes", "No"],
  "ground_truth":     "No",
  "close_date":       "2026-02-10",
  "topic":            "",
  "market_prob_yes":   0.0205,
  "market_prob_no":    0.9795,
  "predicted":        "No",
  "correct":          true,
  "confidence":       0.97,
  "agent_prob_yes":   0.03,
  "agent_prob_no":    0.97,
  "reflection":       null,
  "self_critique":    null,
  "carry_conclusion": null,
  "carry_critique":   null,
  "tool_call_count":  3,
  "latency_sec":      232.11,
  "error":            null,
  "timestamp":        "2026-04-03T07:06:59.834508+00:00",
  "final_answer":     "<think>...</think>\n\n<prediction><yes>0.03</yes><no>0.97</no>..."
}
```

### Output field semantics (IMPORTANT — evolved across sessions)

**`<prediction>` XML — current format (Prompt v4, after 2026-04-07):**  
All 7 setups output `<prediction><yes>0.72</yes><no>0.28</no><reasoning>...</reasoning></prediction>`.  
- Extraction priority: `extract_prediction_xml()` → `extract_prob_yes()` (P(Yes): text) → `extract_confidence_legacy()` (Confidence: text)  
- `predicted = "Yes" if agent_prob_yes > 0.5 else "No"` — always consistent

**New fields (v4):** `setup` (int 1-7), `carry_conclusion` (XML or null), `carry_critique` (XML or null)

**Legacy formats (backward-compatible):** `P(Yes): 0.75` (v3) and `Confidence: 0.75` + `Selected Option:` (v2) are still parsed via the fallback chain for old result files.

---

## 5. Complete Error History

### Error 0: vLLM missing `--tool-call-parser`
- **Session:** First launch attempt (nohup.out)
- **Symptom:** `TypeError: --enable-auto-tool-choice requires --tool-call-parser`
- **Fix:** Added `--tool-call-parser hermes` to vLLM launch command

### Error 1: Context window too small (`max_model_len=16384`)
- **Session:** April 2 (~20:18 UTC, vllm.log)
- **Symptom:** Tool-heavy conversations exceeded the effective input limit
- **Chain of events:**  
  1. Original code used `max_tokens=1024` → effective input limit = 16,384 − 1,024 = 15,360  
  2. "fix eval" commit (April 3) raised `max_tokens` to 4,096 → input limit dropped to 12,288, making overflow more frequent  
  3. Previous session fix: restarted vLLM with `--max-model-len 32768`  
- **Evidence:** vllm.log shows `max_model_len: 16384`; vllm_gpu2/3/4.log show `max_model_len: 32768`
- **Status:** Fixed (server config)

### Error 2: TOKEN_LIMIT (finish_reason=length)
- **Session:** April 3 runs (2 ReAct+Reflection records, 3 ZS records), resurfaced April 8 in ZS run
- **Symptom:** `error: "TOKEN_LIMIT"` — generation cut off mid-stream; `hermes_tool_parser` silently drops truncated tool call JSON
- **Root cause:** `max_tokens=4096` insufficient for Qwen3-14B's `<think>` blocks (~2,000 tokens) + visible output (~500 tokens)
- **Fix (April 4):** Default `max_tokens` raised from 4,096 → 8,192 in `react_agent.py`
- **Fix (April 8):** Same fix applied to `_run_zero_shot_impl()` in `evals.py` (was still at 4096)
- **Status:** Fixed in both code paths

### Error 3: 400 Context Overflow (Q5, timepoint 4)
- **Session:** April 3 ReAct+Reflection run
- **Symptom:** `BadRequestError: 400 — You passed 28,673 input tokens and requested 4,096 output tokens. Model context length is only 32,768.`
- **Root cause:** Agent made 12 tool calls, accumulating 28,673 input tokens. With `max_tokens=4,096`: 28,673 + 4,096 = 32,769 > 32,768 → API rejected entire request.
- **Note:** Raising `max_tokens` to 8,192 makes the threshold stricter (input limit: 24,576 instead of 28,672), so this error would now trigger after ~10 tool calls instead of ~12.

**Fix evolution (important for understanding the current approach):**

| Attempt | Approach | Problem |
|---------|----------|---------|
| First fix | Context guard: inject "stop now" user message + set `tools=None` | **Violates LLM autonomy** — directly overrides agent decisions |
| Reverted | Remove guard entirely | 400 error returns for any conversation >24,576 input tokens |
| **Final fix** | Dynamic `call_max_tokens = min(max_tokens, context_window − last_prompt_toks − 100)` | No messages injected, no tools removed, LLM decides freely |

**Why the final fix preserves autonomy:**  
The LLM does not "see" or "decide based on" `max_tokens`. It generates tokens until it decides to stop (end of response) or the budget runs out (truncation → TOKEN_LIMIT). Adjusting `max_tokens` per call is infrastructure arithmetic — equivalent to saying "you get however many tokens physically remain in the window." The LLM's choices about tools, reasoning, and conclusions are entirely unaffected.

**Why 8192 is safe for both ZS and ReAct:**  
- **ZS:** Single LLM call, input is ~500 tokens (system prompt + question). `500 + 8192 = 8692`, well within 32768. No overflow risk.
- **ReAct:** Multi-turn conversation that grows with each tool call. The dynamic budget (`react_agent.py:99-106`) auto-shrinks `call_max_tokens` each iteration: `min(8192, 32768 − current_prompt_tokens − 100)`. Starts at 8192, shrinks as conversation grows. No manual switching needed between modes.

**What happens when the conversation is very long:**  
If an agent makes many tool calls and `call_max_tokens` drops very low, the model may produce a truncated response → `finish_reason=length` → recorded as `TOKEN_LIMIT` error. This is valid research data: the agent used so many tool calls it exhausted its context. It is not masked or overridden.

- **Status:** Fixed (dynamic budget in ReAct, 8192 in ZS)

### Error 4: `TOKEN_LIMIT` misclassified as `recursion_limit` in eval
- **Session:** April 4 eval analysis
- **Symptom:** `eval_forecasting.py` classified `TOKEN_LIMIT` under `recursion_limit` because `'limit' in 'TOKEN_LIMIT'.lower()` matched the wrong branch
- **Fix (April 4):** Ordered error classification in `compute_operational_metrics`: check exact string `TOKEN_LIMIT` before generic `limit` substring
- **Status:** Fixed

### Error 5: Nested ThreadPoolExecutor deadlock in `--parallel` mode (CRITICAL)
- **Session:** April 8, first ReAct parallel run (`results_s2_20260408_164137.jsonl`)
- **Symptom:** With `--parallel 4` or `--parallel 8`, all ReAct questions timeout at 300s with 0 tool calls and empty `final_answer`. vLLM log shows it IS processing requests successfully (200 OK), but results never arrive back to the eval harness.
- **Root cause:** **Nested ThreadPoolExecutor deadlock.** When `--parallel N > 1`, `benchmark()` runs `_process_single_question()` in worker threads via `ThreadPoolExecutor(max_workers=N)`. Inside each worker, `run_single()` detected "not main thread" and created ANOTHER `ThreadPoolExecutor(max_workers=1)` to wrap `run_react()` with timeout. This nested executor deadlocked — Python's GIL + thread pool exhaustion caused the inner `future.result()` to never return, even though `run_react()` completed its work inside the inner thread.
- **Why ZS worked but ReAct didn't:** ZS finishes in ~8s (single LLM call), so the nested executor completed before GIL contention could develop. ReAct runs 5-10 LLM calls over 3-5 minutes — sustained GIL contention between inner and outer thread pools caused the deadlock.
- **Diagnosis steps:**
  1. `test_vllm.py` confirmed vLLM responds in 6.5s for simple requests
  2. `test_react.py` confirmed vLLM handles ReAct tool calls in 6.2s per iteration
  3. `test_run_single.py` isolated the bug: main thread (SIGALRM) completes in 57.8s, worker thread (nested executor) hangs forever
- **Evidence:** `results_s2_20260408_164137.jsonl` — 8 questions, 6 TIMEOUT at 300s with 0 tools, 1 success at 62s (Q7, happened to complete before timeout), 1 at 25s with 0 tools (Q15, model answered without calling tools)
- **Fix (April 8, two iterations):**
  - **Iteration 1:** Removed nested `ThreadPoolExecutor`. Worker threads call `run_react()` directly with `timeout_sec` param. Added per-call HTTP `timeout=120.0`. Validation showed worker thread completed in 29.7s (fix worked), but SIGALRM was unreliable (main thread ran 311s on 120s budget with no error — see Error 8).
  - **Iteration 2:** Replaced SIGALRM entirely with `threading.Timer` + `cancel_event` for ALL threads. Added `_timed_out()` helper checked at 4 points in the loop (top, after LLM call, after HTTP error, before each tool call). Reduced per-call HTTP timeout to `min(90s, timeout_sec)`. Removed `signal` import and `_Timeout`/`_alarm_handler` dead code.
- **Files changed:** `agent/react_agent.py`, `agent/evals.py`
- **Status:** **FIXED and validated** — `test_run_single.py` confirms both main thread (233s, TIMEOUT at budget) and worker thread (29.7s, completes normally) work correctly
- **Lesson:** Never nest `ThreadPoolExecutor` in Python — the inner executor's threads compete with the outer executor's threads for GIL time, causing deadlocks under sustained load. Use direct function calls with time budget checks instead.

### Error 8: SIGALRM unreliable during C-extension calls
- **Session:** April 8, during Error 5 fix validation
- **Symptom:** Main thread `run_single()` with `timeout_sec=120` completed in 311.7s with `error: None` — SIGALRM never fired (or was swallowed).
- **Root cause:** Python's `signal.alarm()` / SIGALRM is unreliable when the thread is blocked inside C extensions (httpx SSL/network I/O). The signal fires but the exception raised by the handler gets caught or suppressed by the C extension code. This is a known Python limitation.
- **Fix (April 8):** Replaced SIGALRM with `threading.Timer` + `cancel_event` for both main thread and worker threads. The timer sets a `threading.Event` after `timeout_sec`, and `run_react()` checks this event at multiple points in its loop. Per-call HTTP timeout (90s) ensures no single vLLM call blocks indefinitely, giving the event check a chance to fire.
- **Files changed:** `agent/evals.py` (removed `signal` import, `_Timeout`, `_alarm_handler`; `run_single()` and `run_zero_shot()` now use timer-based timeout), `agent/react_agent.py` (added `_timed_out()` helper, checked at 4 loop points)
- **Status:** **FIXED and validated**

### Error 9: TIMEOUT produces no prediction (data loss)
- **Session:** April 8, after Error 8 fix
- **Symptom:** When a question exceeds its time budget, the old code returned `error: TIMEOUT` with whatever partial trace existed — often no `<prediction>` XML. This meant the question produced no usable prediction, losing data.
- **Root cause:** The timeout handler simply returned immediately without giving the LLM a chance to produce a final answer from the evidence it had already gathered.
- **Fix (April 8):** Added **graceful timeout** mechanism in `run_react()`. When the time budget is exceeded, instead of returning immediately, one final LLM call is made with `tools=None` (no more tool calls) and a prompt: "TIME BUDGET EXCEEDED. Based on ALL evidence you have gathered so far, produce your final prediction NOW." This call has a 30s HTTP timeout and 1024 max_tokens — just enough for the `<prediction>` XML. If the final call succeeds, returns `error: GRACEFUL_TIMEOUT` (prediction available). If it fails, returns `error: TIMEOUT` (no prediction).
- **Impact on research:** Questions that finish naturally are completely unaffected. Graceful timeout predictions are tagged `GRACEFUL_TIMEOUT` so they can be reported separately or excluded in sensitivity analysis. The agent already did N tool calls — the forced conclusion uses real evidence, not a blind guess.
- **Does NOT violate LLM autonomy:** Unlike the rejected "context guard" (Error 3), this only fires AFTER the budget is exhausted. It's a recovery mechanism for data that would otherwise be lost, not mid-loop steering.
- **Files changed:** `agent/react_agent.py` (`_graceful_timeout()` function, replaces `_timeout_result()`)
- **Status:** Implemented, awaiting validation

### Error 10: Hybrid search engine 260s per query (CRITICAL — root cause of all timeout issues)
- **Session:** April 8, session 3
- **Symptom:** Every ReAct question timed out regardless of `--no_thinking`, `--parallel` settings, or timeout duration. Even sequential single questions with no thinking took 285s for 1 tool call.
- **Diagnosis:** `test_bottleneck.py` isolated each step:
  - LLM call 1: **0.8s** (fast)
  - Tool execution (`search_database`, hybrid engine): **259.5s** (bottleneck!)
  - LLM call 2: **0.9s** (fast)
- **Root cause:** The `search_database` tool defaulted to `engine="hybrid"` (semantic + keyword search). The hybrid engine on the Text2SQL IPC server was extremely slow (~260s per query). The `"vector"` engine (pure semantic search) returns in **0.8s** — 325× faster.
- **Confirmed with DB team:** "Do not use hybrid. Using vector only search."
- **Fix:** Changed default engine from `"hybrid"` to `"vector"` in `agent/tools.py` (function signature, fallback, and OpenAI tool schema).
- **Before/after:**
  | Engine | Query time | ReAct per question (sequential) | Tool calls in 300s budget |
  |--------|-----------|--------------------------------|--------------------------|
  | hybrid | 260s | 285s (1 tool call, always timeout) | 1 |
  | **vector** | **0.8s** | **79s (11 tool calls, completes normally)** | **11** |
- **Files changed:** `agent/tools.py` (3 lines: function default, fallback default, schema description)
- **Status:** **FIXED and validated**
- **Lesson:** Always isolate bottlenecks step-by-step (LLM call → tool execution → LLM call). The obvious suspects (thinking, parallelism, timeout logic) were all fine — the problem was in an infrastructure dependency.

### Error 6: vLLM crash on startup (GPU memory)
- **Session:** April 8, first vLLM launch
- **Symptom:** vLLM started on port 8000, served a few requests, then `EngineCore_DP0 died unexpectedly` after ~6 minutes. All subsequent requests got "Connection error."
- **Root cause:** Likely GPU memory pressure or another process on the same GPU
- **Fix:** Restart vLLM. Ensure `CUDA_VISIBLE_DEVICES` points to a free GPU.
- **Status:** Resolved by restart

### Error 7: `.env` port mismatch
- **Session:** April 8
- **Symptom:** All ZS questions returned "Connection error." — `results_s1_20260408_160011.jsonl` (30 questions, all errors)
- **Root cause:** `.env` had `VLLM_API_BASE=http://127.0.0.1:8002/v1` but vLLM was launched on port 8000
- **Fix:** Updated `.env` to port 8000
- **Status:** Fixed

### Error 11: Market probability field name mismatch + null values in q-by-q mode
- **Session:** April 8 (session 4)
- **Symptom:** `market_prob_yes`/`market_prob_no` were `null` in q-by-q results. Eval's MARKET COMPARISON section silently skipped (no Brier Skill Score computed).
- **Root cause (two issues):**
  1. **Field name mismatch:** Time-series loader wrote `human_prob_yes`, q-by-q loader wrote `market_prob_yes`, but eval expected `human_prob_yes` only.
  2. **Cutoff date filter too strict:** `_extract_market_probs()` only used prices at/before cutoff date. For 300/1000 questions, the cutoff (e.g. `2025-10-29`) predated all price history (starts `2026-02-04`), returning null.
- **Fix:**
  1. Standardized all code and docs to `market_prob_yes`/`market_prob_no` (renamed in `database/loader.py`, `agent/evals.py`, `eval/eval_forecasting.py`, `eval/results_to_html.py`, `README.md`, `paper/CHECKPOINT.md`, `paper/DATASET.md`).
  2. Added fallback in `_extract_market_probs()`: if no prices exist at/before cutoff, use last available price.
  3. Backfilled existing S1 results (300 nulls → 0).
- **Status:** Fixed

---

## 6. Key Discoveries (Research Findings)

### Discovery 1: Confidence/Probability Inconsistency (21–34% of records)
- **What:** In 21% of ReAct+Reflection records and 34% of ZS records, `predicted ≠ argmax(agent_prob_yes, agent_prob_no)`
- **Why:** Ambiguous prompt — `Confidence: 0.3` with `Selected Option: No` means "30% confident in No", making `agent_prob_yes = 0.7`. The agent chose the less-probable option.
- **Pattern:** All conflicts have `confidence ≤ 0.5`. The agent defaults to "No" under uncertainty even when its stated probability leans "Yes".
- **Empirical result:** In conflict cases, the literal prediction is MORE accurate (63.8%) than probability argmax (36.2%) — the "default to No" heuristic happens to work for this question distribution.
- **Fix:** Prompt now asks for `P(Yes)` explicitly; `predicted` is derived from it → consistent by construction.
- **Paper note:** The legacy inconsistency is a research finding in itself. Document both old and new formats. Old results evaluated correctly via backward-compat extraction.

### Discovery 2: Strong NO Prediction Bias
- **What:** ReAct −27.4% bias, ZS −13.4% bias (both predict No far more than ground truth warrants)
- **Why:** Absence-of-evidence fallacy — when Reddit has no relevant data, agent defaults to "No"
- **Paper note:** Flag as a systematic failure mode. `eval_forecasting.py` detects this in `analyze_error_patterns` ("absence_of_evidence_fallacy" pattern).

### Discovery 3: Market vs Agent Calibration Gap
- **What:** Market Brier Score = 0.044; Agent Brier Score = 0.281; Brier Skill Score = −5.39
- **Why:** Prediction markets aggregate vast information; Reddit search covers only specific communities and topics
- **Paper note:** Frame as expected baseline — the interesting question is whether the agent has *any* skill on specific question subsets (niche topics where market is thin).

### Discovery 4: Reflection Correlates with Accuracy
- **What:** Accuracy +16.8% when a Reflection block is present (69.5% vs 52.7%)
- **Caveat:** Selection bias — the agent is more likely to write a Reflection block when it found relevant evidence, which also correlates with correct prediction
- **Paper note:** Needs controlled analysis. Collect reflection-present rate per question type.

### Discovery 5: Majority Vote Beats Per-Timepoint Accuracy
- **What:** Per-timepoint accuracy 60.3%; majority vote per question 65.0%; first-timepoint accuracy 70.0%
- **Interpretation:** Early predictions (more uncertainty, less market information) happen to be more accurate — possibly because the agent uses general priors which are correct, while later timepoints show the agent is not tracking the market's evidence accumulation

### Discovery 6: Speed vs Quality Tradeoff
- **What:** With `enable_thinking=True` (default), Qwen3-14B generates ~2,000 token `<think>` blocks per call at 84 tok/s → ~24s per LLM call → ~329s for 3 tool call runs
- **With `enable_thinking=False` (`--no_thinking`):** Estimated ~65s avg
- **Paper note:** Run both conditions and report accuracy gap. Thinking likely helps calibration; we measure the tradeoff.

---

## 7. Files Changed Summary

### `agent/react_agent.py`
| Change | Reason |
|--------|--------|
| `max_tokens=4096` → `8192` | Fix TOKEN_LIMIT errors (Qwen3 needs more output budget) |
| Added `thinking: bool = True` param | Enable/disable Qwen3 `<think>` blocks for speed vs quality |
| Added `extra_body={"chat_template_kwargs": {"enable_thinking": thinking}}` | vLLM API for Qwen3 thinking control |
| Added `_CONTEXT_WINDOW = 32768`, `_CONTEXT_BUFFER = 100` | Constants for dynamic budget math |
| Dynamic `call_max_tokens = min(max_tokens, _CONTEXT_WINDOW − last_prompt_toks − _CONTEXT_BUFFER)` | Prevent 400 errors without touching LLM decisions |
| Track `last_prompt_toks` from `response.usage.prompt_tokens` | Input to budget math above |
| Added `cancel_event` + `timeout_sec` params to `run_react()` | Apr 8 | Timer-based timeout for all threads (replaces unreliable SIGALRM) |
| Added `_timed_out()` helper, checked at 4 loop points | Apr 8 | Reliable budget enforcement: loop top, after LLM call, after HTTP error, before each tool call |
| Added `_graceful_timeout()` | Apr 8 | When budget exceeded, one final LLM call (no tools, 1024 tokens, 30s HTTP timeout) extracts prediction from partial evidence. Returns `GRACEFUL_TIMEOUT` on success, `TIMEOUT` on failure |
| Per-call HTTP timeout 120s → `min(90s, timeout_sec)` | Apr 8 | Ensures no single vLLM call consumes the entire budget. Shorter timeout = more frequent budget checks |

**Not added (rejected):** Context guard with message injection and `tools=None` during active loop. Reason: violates LLM autonomy — directly overrides agent decisions mid-reasoning. (Note: graceful timeout is different — it fires AFTER budget exhaustion, not during active reasoning.)

### `agent/evals.py`
| Change | Date | Reason |
|--------|------|--------|
| Added `thinking: bool = True` param to `run_single`, `benchmark`, `benchmark_timeseries` | Apr 7 | Thread through to `run_react` |
| Added `--no_thinking` CLI flag | Apr 7 | Enable fast inference mode from command line |
| `extract_confidence()` → `extract_prob_yes()` + `extract_confidence_legacy()` | Apr 4 | New prompt uses `P(Yes):` label; old results use `Confidence:` |
| `predicted` derived from `P(Yes) > 0.5` | Apr 4 | Eliminates choice/probability inconsistency |
| `agent_prob_yes = P(Yes)` directly | Apr 4 | No more ambiguous direction inference |
| `run_zero_shot()` made thread-safe | Apr 8 | Split into `_run_zero_shot_impl()` + direct call in worker threads (no nested executor) |
| ZS `max_tokens` 4096 → 8192 | Apr 8 | Was causing TOKEN_LIMIT errors in ZS runs; safe because ZS input is ~500 tokens (no overflow risk). ReAct already uses 8192 with dynamic budget. |
| `_process_single_question()` extracted | Apr 8 | Per-question worker function for parallel question-by-question mode |
| `benchmark()` gains `setup` + `parallel` params | Apr 8 | Enables `--setup 1-7` and `--parallel N` in question-by-question mode (previously only available in timeseries mode) |
| CLI passes `setup` + `parallel` to `benchmark()` | Apr 8 | Wire up existing CLI flags to newly supported params |
| Removed nested ThreadPoolExecutor from `run_single()` | Apr 8 | Was causing deadlock in `--parallel` mode (Error 5). Worker threads now call `run_react()` directly with `timeout_sec` param. |
| Removed nested ThreadPoolExecutor from `run_zero_shot()` | Apr 8 | Same deadlock fix. Worker threads call `_run_zero_shot_impl()` directly. |
| Replaced SIGALRM with `threading.Timer` + `cancel_event` | Apr 8 | SIGALRM unreliable during C-extension calls (Error 8). Both `run_single()` and `run_zero_shot()` now use timer-based timeout. Removed `signal` import, `_Timeout` class, `_alarm_handler`. |
| Added per-call HTTP timeout to ZS | Apr 8 | `_run_zero_shot_impl()` now passes `timeout=float(timeout_sec)` to the HTTP client |

### `agent/prompts.py`
| Change | Reason |
|--------|--------|
| Output format: `Confidence: ...` → `P(Yes): ...` | Eliminates ambiguity between "certainty in answer" vs "probability of Yes" |
| Added consistency constraint: `P(Yes) > 0.5 → Yes, P(Yes) < 0.5 → No` | Forces `predicted` and `agent_prob_yes` to always be logically consistent |
| Reflection block: `Current belief: Yes=0.X | No=0.X` → `Current P(Yes): 0.X` | Consistency with new output format |

### `eval/eval_forecasting.py`
| Change | Reason |
|--------|--------|
| Fixed TOKEN_LIMIT error classification | Was matching `recursion_limit` branch via substring `'limit'` |
| Added `token_limit_rate` and `context_overflow_rate` fields | Separate the two distinct error types |
| `extract_confidence()` → `extract_prob_yes_from_record()` | Priority: `agent_prob_yes` field → `P(Yes):` text → legacy `Confidence + predicted` |
| Added `TimeSeriesMetrics` dataclass + `compute_timeseries_metrics()` | Accuracy by timepoint position, consistency rate, majority vote |
| Added `MarketComparisonMetrics` dataclass + `compute_market_comparison_metrics()` | Brier Skill Score, correlation, directional agreement vs market |
| Added `ReflectionQualityMetrics` dataclass + `compute_reflection_quality_metrics()` | Reflection/self-critique rates, accuracy split |
| Updated text and CSV formatters | Display new metric sections |

---

## 8. Workflow: How to Run

### Start Infrastructure
```bash
# vLLM server — exact flags required
vllm serve Qwen/Qwen3-14B-AWQ \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 32768 \
  --quantization awq \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --generation-config vllm

# Text2SQL IPC server (separate terminal, port 61001)
```

### Run Benchmarks (7-Setup Framework)

#### Question-by-Question Mode (recommended for initial evaluation)
```bash
# Single setup, 8 questions in parallel, thinking OFF (recommended for speed)
python agent/evals.py --setup 2 --max_questions 1000 --parallel 8 --no_thinking

# Same but with thinking ON (slower, deeper reasoning per call)
python agent/evals.py --setup 2 --max_questions 1000 --parallel 8

# Run as background job with logging
nohup python -u agent/evals.py --setup 2 --max_questions 1000 --parallel 8 --no_thinking \
  > logs/react_s2_nothink.log 2>&1 &

# Monitor progress
tail -f logs/react_s2_nothink.log
wc -l agent/results/results_s2_*.jsonl    # count completed questions

# Run all 7 setups with parallelism
for s in 1 2 3 4 5 6 7; do
  python agent/evals.py --setup $s --max_questions 939 --parallel 8 --no_thinking
done

# Resume interrupted run
python agent/evals.py --setup 2 --parallel 8 --no_thinking \
  --results_file agent/results/results_s2_XXXXXX.jsonl
```

#### Thinking ON vs OFF — When to Use Which (Updated with vector engine data)

| Flag | Sequential | Parallel 8 | Tool calls | Use case |
|------|-----------|-----------|-----------|----------|
| thinking ON | 89s/q, completes | 65s throughput, 50% timeout | 3 (sequential) / 3-4 (parallel) | **Final paper results** |
| `--no_thinking` | 79s/q, completes | 33s throughput, 87% timeout | 11 (sequential) / 5-6 (parallel) | **Proof of concept, overnight runs** |

**Key insight (after vector engine fix):** Thinking ON is now viable. At 89s sequential with 3 tool calls and clean completion, the agent does meaningful research. The prior conclusion that thinking was "actively hurting results" was wrong — the real bottleneck was the hybrid engine (260s/query), not thinking tokens.

**For final paper results:** Use thinking ON. At `--parallel 4` (estimated ~12 hrs for 1000q), should get 3+ tool calls with mostly clean completions.

**For proof of concept / overnight:** `--no_thinking --parallel 8` (9.2 hrs for 1000q). Agent makes 5-6 tool calls, most get predictions via graceful timeout.

**Switching between modes:** `--no_thinking` is a CLI flag only — nothing in the code, prompts, or saved results changes.

#### Time-Series Mode (for temporal analysis and carry-forward)
```bash
# Single setup with timepoints
python agent/evals.py --timeseries --setup 2 --max_questions 100 --parallel 8

# Cap timepoints per question
python agent/evals.py --timeseries --setup 7 --max_questions 100 --max_timepoints 10 --parallel 8

# Run all 7 setups in time-series
for s in 1 2 3 4 5 6 7; do
  python agent/evals.py --timeseries --setup $s --max_questions 939 --parallel 8
done
```

#### Legacy Flags (still supported)
```bash
python agent/evals.py --zero_shot --max_questions 100                  # = --setup 1
python agent/evals.py --reflection --max_questions 100                 # = --setup 3
python agent/evals.py --zero_shot --timeseries --max_questions 100     # = --setup 1 + timeseries
```

### Evaluate Results
```bash
# Single setup report
python eval/eval_forecasting.py agent/results/results_s2_ts_XXXXXX.jsonl

# Statistical comparison between two setups
python eval/eval_forecasting.py results_s2.jsonl --compare results_s7.jsonl

# All formats to directory
python eval/eval_forecasting.py results.jsonl --format all --output-dir ./eval_results/

# Quick comparison across all result files
for f in agent/results/*.jsonl; do
  echo "=== $(basename $f) ===" 
  python eval/eval_forecasting.py "$f" | grep -E "Accuracy|Brier Skill|MCC|Bias|Error Rate"
done
```

---

## 9. Current Results (as of 2026-04-04)

### ReAct + Reflection (20 questions, 233 timepoints)
| Metric | Value |
|--------|-------|
| Accuracy (excl. nulls) | 60.3% |
| Balanced Accuracy | 60.1% |
| MCC | 0.244 |
| Prediction bias | −27.4% (strong NO bias) |
| Agent Brier Score | 0.281 |
| Market Brier Score | 0.044 |
| Brier Skill Score | −5.39 (market wins) |
| Agent–market correlation r | 0.063 |
| Majority-vote accuracy | 65.0% |
| Consistency rate | 50.0% |
| Avg latency | 329s |
| Error rate | 1.3% (3 records) |

### Zero-Shot Baseline (939 questions, 8,957 timepoints)
| Metric | Value |
|--------|-------|
| Accuracy | 77.3% |
| Balanced Accuracy | 54.1% |
| MCC | 0.125 |
| Prediction bias | −13.4% |
| Agent Brier Score | 0.263 |
| Avg latency | 8.7s |
| Error rate | 0.03% (3 records) |

---

## 10. Known Issues & Open Questions

| Issue | Status | Notes |
|-------|--------|-------|
| NO prediction bias (−27% ReAct, −13% ZS) | Known | Absence-of-evidence fallacy; document in paper |
| topic field always empty | Known | Loader issue; no per-topic breakdown possible yet |
| Reflection blocks sometimes inside `<think>` | Workaround | Fallback extraction searches inside `<think>` (both XML and text formats) |
| Agent–market correlation r=0.06 | Expected | Reddit ≠ market data; investigate niche question subsets |
| Full 7-setup run (939 questions) not yet done | **Next step** | All code ready; need infra up to run |
| Resume + carry-forward limitation | Known | Carry state isn't persisted to disk; resume clears carry context |
| XML parsing untested on live Qwen3 output | **Validate first** | Test on 20 questions before full run |
| **Trace data loss (49% of reflection data)** | **FIXED** (Apr 4) | `react_agent.py` returns full trace |
| **Extraction first-match bug** | **FIXED** (Apr 4) | Uses last match for all extraction |
| **Missing tool capabilities** | **FIXED** (Apr 7) | vector search, Month, Authors added to tools.py |
| **Cross-timepoint memory** | **IMPLEMENTED** (Apr 7) | Setups 4-7 with carry-forward |
| **Inconsistent reflection output** | **RESOLVED** (Apr 7) | Reflection is a binary ablation variable (setups 3, 7) |
| **Output format** | **UPDATED** (Apr 7) | XML `<prediction>` format with fallback chain |
| **Statistical testing** | **IMPLEMENTED** (Apr 7) | Bootstrap CI, McNemar, Wilcoxon in eval_forecasting.py |
| **Sequential bottleneck in benchmark()** | **FIXED** (Apr 8) | `--parallel N` now works in both question-by-question and time-series modes |
| **run_zero_shot() not thread-safe** | **FIXED** (Apr 8) | Calls `_run_zero_shot_impl()` directly in worker threads (no nested executor) |
| **benchmark() missing --setup support** | **FIXED** (Apr 8) | All 7 setups now work in question-by-question mode |
| **Nested ThreadPoolExecutor deadlock** | **FIXED** (Apr 8) | See Error 5. Removed nested executors; direct calls + timer-based timeout |
| **SIGALRM unreliable in C extensions** | **FIXED** (Apr 8) | See Error 8. Replaced with `threading.Timer` + `cancel_event` for all threads |
| **TIMEOUT loses prediction data** | **FIXED** (Apr 8) | See Error 9. Graceful timeout: final LLM call extracts prediction from partial evidence |
| **Hybrid engine 260s/query** | **FIXED** (Apr 8) | See Error 10. Switched default to vector engine (0.8s/query). Root cause of all prior timeout issues. |
| **ZS max_tokens too low (4096)** | **FIXED** (Apr 8) | Raised to 8192 in `_run_zero_shot_impl()`. Safe for ZS (single call, ~500 input tokens) |
| **Doc_id format mismatch** | **BUG** (Apr 9) | search_database returns compound `doc_id` (`"2025-02:comment:mc806oy"`), drilldown tools expect plain ID (`"mc806oy"`). Agent passes compound format 25% of drilldowns → all fail. 345/422 not-found errors (82%). Wastes 10% of tool budget. Fix: strip prefix in tools.py or ask D agent team to accept compound format. |
| **Dangling IDs in vector index** | **Known** (Apr 9) | 77 IDs exist in vector search index but not in SQL lookup table → legitimate not-found errors on drilldown. Ask D agent team to clean up. |
| **NoneType bug in S2** | **Known** (Apr 9) | 20 NoneType errors (`'NoneType' object is not subscriptable`). Likely XML parsing edge case. Needs investigation. |
| **Retrieval introduces confident noise** | **Known** (Apr 9) | S2 retrieval flips predictions wrong more often than right (83 hurt vs 56 helped). Reddit opinions treated as facts. S2 makes 3× more extreme predictions. See FINDINGS.md Finding 16. |
| **NO bias not mitigated by retrieval** | **Known** (Apr 9) | Both S1 and S2 predict Yes only 9-10% vs 25.4% GT. Absence-of-evidence fallacy worse with tools (85 vs 79). Needs prompt debiasing. |
| **Shallow tool usage pattern** | **Known** (Apr 9) | 61% of questions use only search_database, never drill down. Thread context and author history tools barely used (22 and 21 calls total). See FINDINGS.md Finding 18. |

---

## 11. Design Decisions — All Finalized (2026-04-07)

All 10 decisions from DECISIONS.md have been finalized and implemented. See `paper/PROPOSAL.md` for the full research proposal.

| # | Decision | Resolution |
|---|----------|-----------|
| 1 | Cross-timepoint memory | 4 carry-forward variants (setups 4-7) |
| 2 | Consistent reflection | Reflection as binary ablation (setups 3, 7 only) |
| 3 | Expose all search tools | Done — vector, month, authors added |
| 4 | Information gain metric | Use ZS (setup 1) as proxy for prior |
| 5 | Compute strategy | Multi-GPU parallel, thinking always on |
| 6 | Conclusion content | prediction + market_probability + key_evidence + evidence_gaps + forward_prior |
| 7 | Self-critique content | search_quality + evidence_balance + reasoning_biases + confidence_justification + improvement_plan (no ground truth) |
| 8 | Number of setups | 7 (no additional cells) |
| 9 | Statistical testing | Bootstrap CI + McNemar + Wilcoxon |
| 10 | Market baseline | Evaluation anchor, not the novelty. Novelty = temporal framework + ablation + carry-forward |
