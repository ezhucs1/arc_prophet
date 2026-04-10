# Research Findings
**Last updated:** 2026-04-09 (session 6)

This file documents empirical findings from experiments — what was discovered, the evidence, what it means for the paper, and caveats for reviewers. Engineering notes live in CHECKPOINT.md; experiment conditions live in EXPERIMENTS.md.

---

## Finding 1 — Probability/Choice Inconsistency in LLM Output (21–34% of records)

### What was found
When Qwen3-14B-AWQ was prompted with `Confidence: <how certain you are>`, it interpreted this as "certainty in the selected option" rather than "P(Yes)". A response like:

```
Selected Option: No
Confidence: 0.3
```

meant the model was 30% confident in No — so P(Yes) = 0.7 — yet it still chose No. This created a logical inconsistency between the discrete prediction and the stated probability.

### Evidence
- **ReAct+Reflection (Run 2, 20q):** 21% of timepoints had `predicted ≠ argmax(agent_prob_yes, agent_prob_no)`
- **Zero-Shot (Run 3, 939q):** 34% of timepoints had the same inconsistency
- All inconsistencies occurred in records with `confidence ≤ 0.5` (the model chose low-confidence options)

### Surprising sub-finding
When `predicted` (from "Selected Option:" text) and `argmax(agent_prob)` disagree, the **literal predicted is more accurate** (63.8% vs 36.2% in conflict cases). The model's default-to-No behavior under uncertainty is a better heuristic than flipping the choice, given the strong NO skew in the question distribution (79% No in Run 3).

### Implication for paper
Prompt design must specify whether the probability field means "P(Yes)" or "certainty in selected option." We changed to `P(Yes):` in Prompt v3 with an explicit consistency constraint. All future runs will use v3. This inconsistency cannot affect v3 results by construction.

### Caveat
The 63.8% literal-wins rate is specific to this dataset's NO skew. On a balanced dataset the advantage would likely disappear. Do not generalize beyond this distribution.

---

## Finding 2 — Strong NO Prediction Bias

### What was found
Both agent modes exhibited a systematic bias toward predicting No:

| Mode | Bias (predicted_yes_rate − gt_yes_rate) |
|------|-----------------------------------------|
| ReAct+Reflection (Run 2) | −27.4% |
| Zero-Shot (Run 3) | −13.4% |

ReAct+Reflection showed a stronger NO bias than zero-shot, even though its ground truth distribution was nearly balanced (49.4% Yes).

### Mechanism hypothesis
When the Reddit database contains no relevant posts for a question, the agent defaults to No. This is an **absence-of-evidence fallacy** — lack of evidence for Yes is treated as evidence for No. The Reflection block was designed to counter this by explicitly prompting the agent to distinguish "didn't find evidence" from "found evidence against." The fact that ReAct+Reflection shows *more* NO bias than ZS suggests this counter-measure was insufficient, or that the agent's tool searches were predominantly empty.

### Evidence
- Run 2 vs Run 3 NO bias gap: `|−27.4%| > |−13.4%|` despite Run 2 having a balanced GT
- Reflection prompt text explicitly warns: *"Be honest about absence-of-evidence vs evidence-of-absence."*
- Tool call rates for Run 2 averaged ~6–12 calls/timepoint; many returned thin or empty results

### Implication for paper
NO bias is a primary confound in accuracy comparisons. **Balanced Accuracy and MCC are required metrics** — raw accuracy is misleading on the imbalanced Run 3 subset (79% No). The paper must report both.

---

## Finding 3 — Market vs Agent Calibration Gap

### What was found
The Polymarket crowd (prediction market prices) is dramatically better calibrated than the LLM agent:

| Metric | Agent (ReAct+Refl) | Market |
|--------|---------------------|--------|
| Brier Score | 0.281 | 0.044 |
| Brier Skill Score (vs market) | −5.39 | — |

A Brier Skill Score of −5.39 means the agent's predictions are 5.39× worse than the market on mean-squared error.

### Why this is expected (not a paper-breaking result)
The market prices in this dataset were measured **just before resolution** — they reflect near-certainty information accumulated over months of trading. The LLM agent faces each timepoint *as if the question is still open*, with only Reddit posts up to that date. An agent using 2-week-old Reddit posts cannot match a crowd that has been updating prices daily with real money at stake.

### What the comparison actually tests
The paper's contribution is not "can an LLM beat a prediction market?" It is: **does giving the LLM a retrieval database improve calibration over zero-shot?** The market baseline should be framed as context, not as the primary comparison target.

### Time-series decomposition (not yet computed)
The market Brier Score of 0.044 is the **average across all timepoints**, including late ones where the market is very confident. Early timepoints (e.g., 4 weeks before resolution) likely have higher market Brier Scores. If the LLM agent is evaluated at early timepoints, the gap may be smaller. This analysis has not been computed.

### Implication for paper
- Primary comparison: ReAct vs Zero-Shot (same questions, same timepoints)
- Secondary context: vs market baseline with the caveat above
- Need to add per-timepoint Brier Score breakdown (distance to resolution date) before submission

---

## Finding 4 — Reflection Correlates with Accuracy (+16.8 pp)

### What was found
Within Run 2 (ReAct+Reflection, 20 questions), timepoints where a Reflection block appeared in the output were more accurate than those without:

| Condition | Accuracy |
|-----------|----------|
| With reflection block | ~76% (estimated) |
| Without reflection block | ~59% (estimated) |
| Difference | +16.8 pp |

### Caveat: causality is unclear
This correlation may reflect:
1. The reflection block itself improving reasoning (causal)
2. Easier questions producing more confident agents who both reflect more and answer correctly (selection bias)
3. Questions with more available Reddit evidence triggering more tool calls (and thus more reflections at the every-3-calls cadence), with better evidence also producing better accuracy

### Evidence quality
Run 2 has only 20 questions (233 timepoints). The confidence interval on this finding is approximately ±22 percentage points. This finding is **suggestive only** — not statistically conclusive at this sample size.

### Implication for paper
Run 5 (ReAct+Reflection, 939 questions) is required to confirm or refute this finding. Do not report the +16.8 pp number as a conclusion until Run 5 is complete.

---

## Finding 5 — Majority Vote Beats Per-Timepoint Accuracy

### What was found
When aggregating predictions across all timepoints for the same question (majority vote), question-level accuracy exceeded per-timepoint accuracy:

| Aggregation | Accuracy |
|-------------|----------|
| Per timepoint (Run 2) | 60.3% |
| Majority vote per question | ~65% (estimated) |

### Mechanism
Independent predictions at different timepoints for the same question are not i.i.d. — they share the same ground truth. But they use different cutoff dates and thus different Reddit evidence. When multiple independent predictions agree, this reduces per-prediction noise. Majority vote exploits this structure.

### Implication for paper
The time-series setup (multiple timepoints per question) provides a natural ensemble. The paper should compare: (1) per-timepoint accuracy, (2) majority vote accuracy, and (3) accuracy vs temporal distance to resolution. This decomposition is a key novelty of the time-series evaluation framework.

---

## Finding 6 — Speed vs Quality Tradeoff

### What was found
The three modes have very different latency profiles:

| Mode | Avg latency | Relative speed |
|------|-------------|----------------|
| Zero-Shot | 8.7s/timepoint | 1× (baseline) |
| ReAct (no reflection) | ~50–100s/timepoint (estimated) | ~10×+ |
| ReAct+Reflection | 329.4s/timepoint | 38× |

Zero-shot with `<think>` enabled (default) averages ~2000 token thinking block per call. ReAct+Reflection averages multiple tool calls (6–12) per timepoint with thinking enabled on each call.

### Speed improvement available — two approaches

**1. Disable thinking (`--no_thinking`):**  
Disabling `<think>` via `enable_thinking=False` eliminates the thinking overhead. Based on token counting (thinking blocks ≈ 2000 tokens, output ≈ 500 tokens), this should give ~4–5× speedup. Whether this degrades prediction quality is not yet measured.

**2. Parallel execution (`--parallel N`, implemented 2026-04-08):**  
The original pipeline sent one LLM request at a time, leaving the GPU idle between calls. With `--parallel 8`, 8 questions run concurrently and vLLM's continuous batching processes them as a single GPU batch. This gives ~4-8× throughput improvement with no quality impact.

| Mode | Sequential (old) | Parallel 8 (new) | Speedup |
|------|-----------------|-------------------|---------|
| ZS, 939q, question-by-question | ~2.3 hrs | ~17 min | ~8× |
| ReAct, 939q, question-by-question | ~21 hrs | ~2.6 hrs | ~8× |
| ZS, 939q, time-series (9,400 tp) | ~22 hrs | ~3 hrs | ~7× |

**Note:** `--parallel 1` (default) preserves the original sequential behavior. Parallel mode is purely additive — no existing code paths changed.

### Implication for paper
The latency numbers must be reported to contextualize the system's practical feasibility. With parallel execution, a full 7-setup experiment on 939 questions is feasible within ~1-2 days on a single GPU, down from ~7+ days sequential.

---

## Summary Table

| Finding | Runs with Evidence | Statistical Strength | Paper-Ready? |
|---------|--------------------|----------------------|--------------|
| 1. Prob/choice inconsistency (21–34%) | Run 2, Run 3 | Strong (large N in Run 3) | ✅ Yes |
| 2. NO prediction bias | Run 2, Run 3, Run 5, Run 6 | Strong | ✅ Yes (with caveat) |
| 3. Market calibration gap (BSS −5.39) | Run 2 only | Moderate (20q) | ⚠️ Frame carefully |
| 4. Reflection +16.8 pp | Run 2 only | Weak (20q, ±22%) | ❌ Needs thinking ON runs |
| 5. Majority vote improvement | Run 2 only | Weak (20q) | ❌ Needs thinking ON runs |
| 6. Speed vs quality tradeoff | Run 2, Run 3 | Strong (latency) | ✅ Latency numbers only |
| 7. SIGALRM unreliable for LLM timeouts | Diagnostic tests | Definitive | ✅ Engineering (reproducibility section) |
| 8. Graceful timeout preserves data | Implemented + validated (Run 6) | Strong (739/739 valid) | ✅ Report with sensitivity analysis |
| 9. Thinking + parallel = quality degradation | 12q ReAct parallel 8 | Partially invalidated (hybrid was root cause) | ⚠️ Re-evaluate with vector engine |
| 10. Vector vs hybrid: 325× speed difference | Bottleneck test | Definitive | ✅ Use vector for all runs |
| 11. End-to-end speed matrix | Speed tests | In progress | ⚠️ Awaiting thinking ON test |
| 14. ZS vs ReAct (no_thinking, 1000q) | Run 5 vs Run 6 | Strong (N=1000, 3 tests) | ⚠️ Proof-of-concept only (no_thinking) |
| 15. Graceful timeout accuracy = clean accuracy | Run 6 | Strong (72.4% vs 71.0%) | ✅ Validates graceful timeout design |

---

## Finding 7 — Timeout Architecture: SIGALRM Unreliable, Timer+Event Required

### What was found
Python's `signal.alarm()` (SIGALRM) is unreliable for timing out LLM API calls. In testing, a 120-second SIGALRM timeout failed to fire — the main thread completed in 311.7s with no error. The signal was delivered but the exception raised by the handler was swallowed by C-extension code (httpx/SSL network I/O).

### Evidence
- `test_run_single.py` Test 1: `timeout_sec=120`, completed in 311.7s, `error: None` (SIGALRM didn't work)
- `test_run_single.py` Test 1 after fix: `timeout_sec=120`, completed in 233.3s, `error: TIMEOUT` (timer worked)
- `test_run_single.py` Test 2 after fix: `timeout_sec=300`, completed in 29.7s, `error: None` (worker thread normal completion)

### Fix applied
Replaced SIGALRM with `threading.Timer` that sets a `threading.Event` after the budget expires. `run_react()` checks the event at 4 points per iteration (loop top, after LLM call, after HTTP error, before each tool call). Per-call HTTP timeout reduced from 120s to 90s so no single call can consume the entire budget.

### Implication for paper
This is an engineering finding, not a research finding. However, it's relevant to the Reproducibility section — other researchers implementing multi-turn agent loops with timeouts should use timer+event, not SIGALRM, for reliable timeout enforcement in Python.

---

## Finding 8 — Graceful Timeout Preserves Data

### What was found
Without graceful timeout, any question exceeding its time budget produces `error: TIMEOUT` with no usable prediction — lost data. With graceful timeout, the agent makes one final LLM call (no tools, 1024 tokens) to produce a `<prediction>` from whatever evidence it gathered. These predictions are tagged `GRACEFUL_TIMEOUT` for separate analysis.

### Implication for paper
- **Data completeness:** Near-zero lost predictions (only questions where even the 30s final call fails get `TIMEOUT`)
- **Sensitivity analysis:** Report metrics with and without `GRACEFUL_TIMEOUT` predictions to show they don't distort results
- **Standard practice:** This approach is consistent with agent benchmarks (SWE-bench, GAIA) that give agents a "wrap up" phase

---

## Finding 9 — Thinking ON + Parallel Batching Causes Effective Quality Degradation

### What was found
With `--parallel 8` and thinking enabled, GPU contention makes each LLM call take ~150-200s (vs ~50s with single request). The 300s timeout fires after only 1-2 tool calls. In `results_s2_20260408_190423.jsonl` (12 questions): ALL 12 hit GRACEFUL_TIMEOUT, average 431.6s latency, 1-2 tool calls each.

This means thinking ON + high parallelism **actively degrades quality** — the agent barely researches before being forced to conclude. The deep reasoning from `<think>` blocks is wasted because there's almost no evidence to reason about.

### Evidence
| Condition | Avg latency | Avg tool calls | Timeout rate |
|-----------|-------------|----------------|--------------|
| Thinking ON, parallel 8 | 431.6s | 1.5 | 100% (12/12) |
| Thinking OFF, parallel 8 | TBD (running) | Expected 5-8 | Expected <10% |

### Implication for paper
- **For large runs:** Use `--no_thinking --parallel 8` — more tool calls with simpler reasoning is better than fewer tool calls with deep reasoning that gets cut off
- **For quality comparison:** Run 50-100 questions with thinking ON at `--parallel 2` (lower GPU contention) and compare against thinking OFF results
- **Report both conditions:** The thinking ON vs OFF comparison is itself a research finding about the interaction between chain-of-thought, tool use, and computational budgets

---

## Finding 10 — Vector vs Hybrid Engine: 325× Speed Difference

### What was found
The `search_database` tool defaulted to `engine="hybrid"` (semantic + keyword). This was the root cause of ALL timeout issues across sessions — taking ~260s per query. Switching to `engine="vector"` (pure semantic) reduced query time to **0.8s** — a 325× improvement.

### Evidence
| Engine | Query time | ReAct per question (sequential, no_thinking) | Tool calls in 300s |
|--------|-----------|----------------------------------------------|-------------------|
| hybrid | 259.5s | 285s (1 tool call, always timeout) | 1 |
| vector | 0.8s | 79s (completes normally) | 11 |

### Impact on prior findings
- **Finding 9 (thinking + parallel = degradation):** Partially invalidated. The primary bottleneck was hybrid engine, not thinking + GPU contention. Thinking + parallel MAY still cause contention — test pending (`test_thinking_speed.py`).
- **Finding 6 (speed):** All prior latency numbers were inflated by the hybrid engine. True ReAct latency with vector engine: ~79s/question (no_thinking, sequential).

### Implication for paper
- Default engine should be `vector` for all experiments
- If hybrid results are needed for comparison, run as a separate ablation
- The 260s hybrid time was a server-side issue (confirmed by DB team), not an inherent limitation of the search approach

---

## Finding 11 — End-to-End Speed Matrix (Vector Engine, April 8)

### Validated test data

| Config | Per question | Tool calls | Error rate | 1000q estimate |
|--------|------------|------------|-----------|---------------|
| no_thinking, sequential | 79s | 11 | 0% (completes) | 22 hrs |
| no_thinking, parallel 8 | 33s throughput | 5-6 | 87% GRACEFUL_TIMEOUT | **9.2 hrs** |
| thinking ON, sequential | 89s | 3 | 0% (completes) | 24.8 hrs |
| thinking ON, parallel 8 | 65s throughput | 3-4 | 50% GRACEFUL_TIMEOUT + errors | 18 hrs |

### Key observations
1. **Thinking ON produces fewer but deeper tool calls** (3 vs 11) — the model reasons more per call and decides to stop earlier.
2. **Parallel 8 causes GPU contention in both modes** — but no_thinking degrades more gracefully (5-6 tools + graceful timeout) vs thinking ON (3-4 tools + some hard errors).
3. **No_thinking parallel 8 is the fastest usable config** at 9.2 hrs for 1000q. Agent still makes 5-6 tool calls with real evidence.
4. **Thinking ON parallel 8 has a bug** — Q6 hit `'NoneType' object is not subscriptable` with 0 tool calls. Needs investigation.
5. **For final paper results:** thinking ON sequential (89s, 3 tools, clean completion) or thinking ON parallel 4 (untested, estimated ~12 hrs).

### Recommended run strategy
| Purpose | Config | Estimated time |
|---------|--------|---------------|
| Proof of concept (overnight) | `--no_thinking --parallel 8` | 9.2 hrs |
| ZS baseline | `--setup 1 --parallel 8` | 17 min |
| Final paper results | thinking ON, `--parallel 4` | ~12 hrs (estimated) |

---

## Finding 12 — S1 Zero-Shot Baseline (1000 Questions, Prompt v4, April 8)

### Results

| Metric | Value |
|--------|-------|
| Questions | 1000 (981 valid, 19 null/token_limit) |
| Accuracy | 74.0% |
| Balanced Accuracy | 55.2% |
| MCC | 0.151 |
| Prediction Bias | -15.6% (strong NO bias) |
| Agent Brier Score | 0.200 |
| Market Brier Score | 0.055 |
| Brier Skill Score | -2.63 (market beats agent) |
| Prob Correlation (r) | 0.246 |
| Directional Agreement | 77.8% |
| Mean Abs Deviation | 0.301 |
| Avg Latency | 11.2s |

### Key observations
1. **High raw accuracy (74%) is misleading** — driven by NO bias matching the 74.6% NO base rate in ground truth. Balanced accuracy (55.2%) is barely above random.
2. **Market crushes the agent** — BSS of -2.63 means agent error is 3.6× the market's. Expected: the market has real money and information aggregation.
3. **Weak but positive correlation** (r=0.246) with market — the agent's probability estimates track the market direction 77.8% of the time but with much less confidence.
4. **NO bias persists in v4 prompt** — predicted Yes rate (9.8%) vs ground truth Yes rate (25.4%). The model defaults to "No" under uncertainty.

### Implication for paper
This is the baseline to beat. The key question is whether ReAct (S2) with tool retrieval can close the gap — especially on Brier Score and balanced accuracy. A BSS improvement from -2.63 toward 0 would show retrieval adds value.

---

## Finding 13 — Field Naming Inconsistency (`human_prob` vs `market_prob`)

### What was found
The codebase used two different field names for the same data (Polymarket price = market-implied probability):
- Time-series loader: `human_prob_yes`/`human_prob_no` ("human" as in human traders vs AI agent)
- Q-by-q loader: `market_prob_yes`/`market_prob_no`
- Eval script: expected `human_prob_yes` only

This caused the eval's MARKET COMPARISON section to silently skip for all q-by-q results (no Brier Skill Score computed).

Additionally, `_extract_market_probs()` filtered prices by cutoff date, but for 300/1000 questions the cutoff predated all price history, returning null.

### Fix
Standardized to `market_prob_yes`/`market_prob_no` everywhere. Added fallback to last available price. Both are the Polymarket price — the crowd/market's implied probability for each outcome.

### Implication for paper
Always use `market_prob_yes`/`market_prob_no` in code and paper. "Market probability" is clearer than "human probability" for readers.

---

## Finding 14 — ZS vs ReAct: No Significant Difference (no_thinking, 1000q)

### What was found
Overnight runs comparing S1 (Zero-Shot) vs S2 (ReAct, `--no_thinking --parallel 8`) on 1000 questions:

| Metric | S1 (ZS) | S2 (ReAct) | Δ |
|--------|---------|------------|---|
| Accuracy (excl. nulls) | 75.0% | 72.0% | −3.0% |
| Balanced Accuracy | 56.3% | 53.3% | −3.0% |
| MCC | 0.191 | 0.095 | −0.096 |
| Brier Score | 0.198 | 0.207 | +0.009 |
| BSS (vs market) | −2.45 | −2.70 | −0.25 |
| Hard errors | 17 (1.7%) | 20 (2.0%) | +3 |
| Avg latency | 12.9s | 324.0s | +311s |

Statistical tests (all non-significant):
- Brier bootstrap: p=0.37, 95% CI [−0.021, +0.008]
- McNemar accuracy: p=0.065
- Wilcoxon signed-rank: p=0.87

### Key observations
1. **ZS slightly outperforms ReAct on every metric** — but no difference is statistically significant
2. **ReAct is 25× slower** (324s vs 12.9s) for no accuracy gain
3. **Both have strong NO bias** (~9-10% predicted Yes vs 25.4% ground truth)
4. **Both far worse than market** (BSS −2.45 / −2.70)

### Caveats — this is NOT the final paper comparison
- S2 used `--no_thinking` — reasoning happens but is not displayed, and thinking tokens are not generated. Thinking ON may change results.
- 73.9% of S2 predictions came from graceful timeout recovery — the agent consistently needed more time than budget allowed at parallel 8
- Final paper results must use thinking ON with `--parallel 4`

### Implication for paper
This proof-of-concept suggests retrieval may not help for this task/model combination, but the `--no_thinking` + high graceful timeout rate means we cannot conclude yet. The thinking ON comparison is essential.

---

## Finding 15 — Graceful Timeout Predictions Are Equally Accurate

### What was found
In Run 6 (S2 ReAct), 739/1000 questions hit GRACEFUL_TIMEOUT. All 739 produced valid predictions:

| Completion type | Count | Accuracy |
|----------------|-------|----------|
| Clean (no error) | 241 | 71.0% |
| Graceful timeout | 739 | 72.4% |
| NoneType (no prediction) | 20 | — |

### Why this matters
The graceful timeout handler (final LLM call to extract prediction from partial evidence) produces predictions that are statistically indistinguishable from clean completions. This validates the design decision to recover predictions on timeout rather than discarding them.

### Implication for paper
- Graceful timeout predictions should be included in all analyses (not excluded as "errors")
- Report accuracy split as a sensitivity analysis to show they don't distort results
- The eval pipeline now correctly classifies these (was previously misreported as 75.9% "error rate")

---

## Finding 16 — Retrieval Introduces Confident Noise (Discordant Case Analysis)

### What was found
JSONL-level comparison of S1 (ZS) vs S2 (ReAct) on 1000 matched questions:

| Category | Count |
|----------|-------|
| Both correct | 676 |
| S1-only correct (retrieval HURT) | 83 |
| S2-only correct (retrieval HELPED) | 56 |
| Both wrong | 185 |

Net: retrieval flipped 27 more questions from right to wrong than from wrong to right.

Prediction flips:
- S1=No → S2=Yes: 75 flips, only **32.0% correct** (24/75). When Reddit convinced the agent to switch to Yes, it was wrong 68% of the time.
- S1=Yes → S2=No: 64 flips, **50.0% correct** (32/64). When retrieval killed a ZS Yes instinct, it was a coin flip — no signal, just noise.

S2 is more extreme: 178 predictions at p=0.0 or p=1.0 vs S1's 66. S2 avg confidence is higher (75.8% vs 70.5%) but less calibrated — overconfident when wrong (71.4% conf vs S1's 66.0%).

### Mechanism
The agent treats Reddit user opinions as factual evidence:
- Q43 (Senator Eichorn guilty?): S1 correctly predicts No (0.1). S2 finds comments about arrest/charges, interprets "charged" as "guilty," predicts Yes at 0.95. GT=No.
- Q35 (Trump deport 2M+?): S1 correctly predicts No (0.1). S2 finds political commentary about deportation *plans*, interprets aspirational policy as likely outcome, predicts Yes at 0.7. GT=No.
- Q50 (Atalanta advance?): S2 finds no relevant results, predicts No at 0.0. S1 uses parametric sports knowledge, predicts Yes at 0.6. GT=Yes.

The absence-of-evidence fallacy worsens with tools: S2 has 85 instances vs S1's 79. The agent replaces parametric reasoning with "my search returned nothing → No."

### Implication for paper
This is a key negative result: naive RAG over social media introduces confident noise that overrides calibrated uncertainty. The prompt needs source reliability instructions to distinguish Reddit opinions from factual reports. This finding motivates the reflection (S3) and self-critique (S5-S7) setups — can structured metacognition mitigate the noise problem?

---

## Finding 17 — Doc_id Format Mismatch Bug (10% Tool Budget Wasted)

### What was found
`search_database` returns compound `doc_id` format: `"2025-02:comment:mc806oy"`. The drilldown tools (`get_comment_core_info`, `get_post_core_info`) expect plain IDs: `"mc806oy"`.

The agent passes the compound format 25.3% of the time (345/1366 drilldown calls). Every one fails with `{"error": "Comment not found"}`.

| Category | Count | % of not-found |
|----------|-------|---------------|
| Wrong format (compound doc_id) | 345 | 81.8% |
| Legitimate not-found (ID in vector index, missing from SQL) | 77 | 18.2% |
| **Total wasted drilldown calls** | **422** | — |

Total tool calls: 4,146. Wasted: 422 (10.2%). With only 4.2 avg tool calls per question, ~0.42 calls/question are burned on dead-end lookups.

### Fix options
(a) Ask D agent team to accept compound doc_id format in lookup endpoints
(b) Strip prefix in `tools.py` client-side: extract ID after last `:`
(c) Ask D agent team to remove dangling IDs from vector index (fixes remaining 77)
(d) Change search results to return `object_id` more prominently

### Implication for paper
Must fix before final runs. This is an implementation bug, not a research finding — but it degrades S2 performance. Fixing recovers ~10% of tool budget, giving the agent more research capacity per question.

---

## Finding 18 — Tool Usage Distribution

### What was found
S2 tool call distribution across 1000 questions (4,146 total calls):

| Tool | Calls | % |
|------|-------|---|
| search_database | 2,737 | 66.0% |
| get_comment_core_info | 983 | 23.7% |
| get_post_core_info | 383 | 9.2% |
| get_post_comments_list | 22 | 0.5% |
| get_author_history_list | 21 | 0.5% |

- 598 questions (61%) used search_database ONLY — never drilled down into results
- 382 questions (39%) used mixed tools (search + drilldown)
- Mixed-tool accuracy: 73.3% vs search-only: 71.2% (+2.1pp, slight benefit from drilling down)
- `get_post_comments_list` and `get_author_history_list` are essentially unused

### Accuracy by tool call count
| Tool calls | Accuracy |
|-----------|----------|
| 1-3 | 72.1% (484 questions) |
| 4-6 | 69.9% (339 questions) |
| 7+ | 76.4% (157 questions) |

More tool calls doesn't linearly improve accuracy — 4-6 calls is actually the worst bucket.

### Implication for paper
The agent's tool strategy is shallow: mostly search → maybe one drilldown → conclude. Thread context and author history tools are almost never used despite being available. This suggests the prompt's "adaptive tool strategy" instructions aren't effective under time pressure, or the agent doesn't see value in deep exploration.

---

## Summary Table

| Finding | Runs with Evidence | Statistical Strength | Paper-Ready? |
|---------|--------------------|----------------------|--------------|
| 1. Prob/choice inconsistency (21–34%) | Run 2, Run 3 | Strong (large N in Run 3) | ✅ Yes |
| 2. NO prediction bias | Run 2, Run 3, Run 5, Run 6 | Strong | ✅ Yes (with caveat) |
| 3. Market calibration gap (BSS −5.39) | Run 2 only | Moderate (20q) | ⚠️ Frame carefully |
| 4. Reflection +16.8 pp | Run 2 only | Weak (20q, ±22%) | ❌ Needs thinking ON runs |
| 5. Majority vote improvement | Run 2 only | Weak (20q) | ❌ Needs thinking ON runs |
| 6. Speed vs quality tradeoff | Run 2, Run 3 | Strong (latency) | ✅ Latency numbers only |
| 7. SIGALRM unreliable for LLM timeouts | Diagnostic tests | Definitive | ✅ Engineering (reproducibility section) |
| 8. Graceful timeout preserves data | Implemented + validated (Run 6) | Strong (739/739 valid) | ✅ Report with sensitivity analysis |
| 9. Thinking + parallel = quality degradation | 12q ReAct parallel 8 | Partially invalidated (hybrid was root cause) | ⚠️ Re-evaluate with vector engine |
| 10. Vector vs hybrid: 325× speed difference | Bottleneck test | Definitive | ✅ Use vector for all runs |
| 11. End-to-end speed matrix | Speed tests | In progress | ⚠️ Awaiting thinking ON test |
| 14. ZS vs ReAct (no_thinking, 1000q) | Run 5 vs Run 6 | Strong (N=1000, 3 tests) | ⚠️ Proof-of-concept only (no_thinking) |
| 15. Graceful timeout accuracy = clean accuracy | Run 6 | Strong (72.4% vs 71.0%) | ✅ Validates graceful timeout design |
| 16. Retrieval introduces confident noise | Run 5 vs Run 6 JSONL | Strong (N=1000, case-level) | ✅ Key negative result |
| 17. Doc_id format mismatch bug | Run 6 JSONL | Definitive (345/422 not-found) | ⚠️ Fix before final runs |
| 18. Tool usage distribution (shallow strategy) | Run 6 JSONL | Strong (4,146 calls) | ✅ Report in error analysis |

---

## Open Questions for Paper

1. ~~Does retrieval (ReAct) improve calibration over zero-shot?~~ → Preliminary answer: No (no_thinking). Retrieval introduces noise (Finding 16). Needs thinking ON confirmation.
2. Does reflection further improve over plain ReAct? (Requires S3 run)
3. Does disabling `<think>` degrade quality? (Can compare Run 5 vs a thinking OFF ZS run)
4. At what temporal distance from resolution does LLM accuracy plateau? (Requires time-series decomposition of existing results)
5. Are there question categories where retrieval helps more? (Blocked by empty `topic` field — see DATASET.md §3)
6. What fraction of questions have zero relevant Reddit content? (Requires Text2SQL query over the full dataset)
7. Can prompt debiasing (NO bias + source reliability) improve S2 over S1? (Requires prompt change + re-run)
8. Does fixing the doc_id bug + recovering 10% tool budget change the S1 vs S2 result? (Requires bug fix + re-run)
