# Experiments Log
**Last updated:** 2026-04-09 (session 5)

This is the single source of truth mapping every result file to the exact conditions used to produce it.
Before reporting any number in the paper, verify it maps to a row here.

---

## Result Files

### Run 1 — Plain ReAct, Time-Series
| Field | Value |
|-------|-------|
| **File** | `agent/results/results_ts_XXXXXX.jsonl` |
| **Mode** | ReAct (no reflection, no zero-shot) |
| **Date run** | 2026-04-03 (inferred from timestamps inside file) |
| **Questions** | ~22 (265 timepoints total) |
| **Timepoints** | 265 |
| **Dataset** | `polymarket_binary_yesno.jsonl` — first N questions |
| **Prompt version** | **v2** (commit `5caad98`) — `Confidence:` output format |
| **max_tokens** | 4096 |
| **max_iterations** | 20 |
| **timeout_sec** | 300 |
| **thinking** | True (default) |
| **max_model_len** | 32768 |
| **Key metrics** | *(not yet evaluated — run `eval_forecasting.py` on this file)* |
| **Notes** | File name was placeholder; auto-naming was not set for this run |

---

### Run 2 — ReAct + Reflection, Time-Series
| Field | Value |
|-------|-------|
| **File** | `agent/results/results_r_ts_20260402_234954.jsonl` |
| **Mode** | ReAct + Reflection (`--reflection --timeseries`) |
| **Date run** | Started 2026-04-02 23:49 UTC; records timestamped 2026-04-03 07:06–07:39 UTC |
| **Questions** | 20 |
| **Timepoints** | 233 (avg 11.7 per question) |
| **Dataset** | `polymarket_binary_yesno.jsonl` — first 20 questions |
| **Dataset date range** | Timepoints from 2026-02-04 to 2026-02-25 |
| **GT distribution** | Yes=115 (49.4%), No=118 (50.6%) — **balanced** |
| **Prompt version** | **v2** (commit `5caad98`) — `Confidence:` output format |
| **max_tokens** | 4096 |
| **max_iterations** | 20 |
| **timeout_sec** | 300 |
| **thinking** | True (default) |
| **max_model_len** | 32768 |
| **Accuracy** | 60.3% (excl. nulls) |
| **Balanced Accuracy** | 60.1% |
| **MCC** | 0.244 |
| **Brier Score** | 0.281 (agent) vs 0.044 (market) |
| **Brier Skill Score** | −5.39 |
| **Avg latency** | 329.4s/timepoint |
| **Errors** | 3 (1× context_overflow, 2× token_limit) |
| **Null predictions** | 1 |
| **Notes** | Only 20 questions — **too few for statistical conclusions**. See §Statistical Significance. |

---

### Run 3 — Zero-Shot Baseline, Time-Series
| Field | Value |
|-------|-------|
| **File** | `agent/results/results_zs_ts_20260402_235856.jsonl` |
| **Mode** | Zero-shot (`--zero_shot --timeseries`) |
| **Date run** | Started 2026-04-02 23:58 UTC; records from 2026-04-03 07:02–07:42 UTC |
| **Questions** | 939 |
| **Timepoints** | 9,033 (avg 9.6 per question) |
| **Dataset** | `polymarket_binary_yesno.jsonl` — first 939 questions |
| **Dataset date range** | Timepoints from 2026-02-04 to 2026-02-27 |
| **GT distribution** | Yes=1,894 (21.0%), No=7,139 (79.0%) — **strongly imbalanced** |
| **Prompt version** | **v2** (commit `5caad98`) — `Confidence:` output format |
| **max_tokens** | 4096 |
| **tool_call_count** | Always 0 (zero-shot) |
| **thinking** | True (default) |
| **max_model_len** | 32768 |
| **Accuracy** | 77.3% (excl. nulls) |
| **Balanced Accuracy** | 54.1% |
| **MCC** | 0.125 |
| **Brier Score** | 0.263 (agent) |
| **Prediction bias** | −13.4% (strong NO bias) |
| **Avg latency** | 8.7s/timepoint |
| **Errors** | 3 (all token_limit) |
| **Null predictions** | 3 |
| **Notes** | Most statistically credible run (939 questions). High raw accuracy is misleading due to NO bias + imbalanced GT. |

---

### Run 4 — Zero-Shot Baseline, Q-by-Q (Prompt v4, 1000 questions)
| Field | Value |
|-------|-------|
| **File** | `agent/results/results_s1_20260408_160848.jsonl` |
| **Mode** | Zero-shot (`--setup 1`), question-by-question |
| **Date run** | 2026-04-08 16:08 |
| **Questions** | 1000 (981 valid, 19 null/token_limit) |
| **Dataset** | `polymarket_binary_yesno.jsonl` — first 1000 questions |
| **GT distribution** | Yes=245 (25.4%), No=736 (74.6%) — **imbalanced** |
| **Prompt version** | **v4** (XML `<prediction>` format, current HEAD) |
| **max_tokens** | 8192 |
| **thinking** | True (default) |
| **max_model_len** | 32768 |
| **Accuracy** | 74.0% (excl. nulls) |
| **Balanced Accuracy** | 55.2% |
| **MCC** | 0.151 |
| **Agent Brier Score** | 0.200 |
| **Market Brier Score** | 0.055 |
| **Brier Skill Score** | −2.63 |
| **Prediction bias** | −15.6% (strong NO bias) |
| **Prob Correlation (r)** | 0.246 |
| **Directional Agreement** | 77.8% |
| **Avg latency** | 11.2s/question |
| **Errors** | 19 (all token_limit) |
| **Null predictions** | 19 |
| **Notes** | First v4 prompt run. Market probs backfilled (300 originally null due to cutoff/history date mismatch — fixed in loader). S1 baseline for RQ1 comparison with S2. |

---

### Run 5 — Zero-Shot Baseline, Q-by-Q (Prompt v4, 1000 questions, overnight)
| Field | Value |
|-------|-------|
| **File** | `agent/results/results_s1_20260408_215821.jsonl` |
| **Mode** | Zero-shot (`--setup 1`), question-by-question |
| **Date run** | 2026-04-08 21:58 – 2026-04-08 22:25 (overnight via `run_overnight.sh`) |
| **Questions** | 1000 (983 valid, 17 token_limit) |
| **Dataset** | `polymarket_binary_yesno.jsonl` — first 1000 questions |
| **GT distribution** | Yes=249 (25.4%), No=734 (74.6%) — **imbalanced** |
| **Prompt version** | **v4** (XML `<prediction>` format) |
| **max_tokens** | 8192 |
| **thinking** | True |
| **parallel** | 8 |
| **max_model_len** | 32768 |
| **GPU** | GPU 1 (port 8001, 0.8 memory) |
| **Accuracy** | 75.0% (excl. nulls), 73.7% (all) |
| **Balanced Accuracy** | 56.3% |
| **MCC** | 0.191 |
| **Agent Brier Score** | 0.198 |
| **Market Brier Score** | 0.057 |
| **Brier Skill Score** | −2.45 |
| **Prediction bias** | −16.3% (strong NO bias) |
| **Prob Correlation (r)** | 0.247 |
| **Directional Agreement** | 78.5% |
| **Avg latency** | 12.9s |
| **Hard errors** | 17 (all token_limit, no prediction produced) |
| **Eval output** | `eval/eval_s1/` (metrics.json, metrics.csv, report.txt) |
| **Notes** | Replacement for Run 4 (same setup, fresh run on overnight GPU 1). |

---

### Run 6 — ReAct, Q-by-Q (Prompt v4, 1000 questions, overnight, no_thinking)
| Field | Value |
|-------|-------|
| **File** | `agent/results/results_s2_20260408_215822.jsonl` |
| **Mode** | ReAct (`--setup 2 --no_thinking`), question-by-question |
| **Date run** | 2026-04-08 21:58 – 2026-04-09 09:15 (overnight via `run_overnight.sh`) |
| **Questions** | 1000 (980 valid, 20 NoneType errors) |
| **Dataset** | `polymarket_binary_yesno.jsonl` — first 1000 questions |
| **GT distribution** | Yes=249 (25.4%), No=731 (74.6%) — **imbalanced** |
| **Prompt version** | **v4** (XML `<prediction>` format) |
| **max_tokens** | 8192 |
| **thinking** | False (`--no_thinking`) |
| **parallel** | 8 |
| **max_model_len** | 32768 |
| **GPU** | GPU 0 (dedicated vLLM instance, port 8000) |
| **Accuracy** | 72.0% (excl. nulls), 70.6% (all) |
| **Balanced Accuracy** | 53.3% |
| **MCC** | 0.095 |
| **Agent Brier Score** | 0.207 |
| **Market Brier Score** | 0.056 |
| **Brier Skill Score** | −2.70 |
| **Prediction bias** | −15.1% (strong NO bias) |
| **Prob Correlation (r)** | 0.174 |
| **Directional Agreement** | 76.0% |
| **Avg latency** | 324.0s |
| **Avg tool calls** | 4.2 (max 19) |
| **Hard errors** | 20 (NoneType — no prediction produced) |
| **Graceful timeouts** | 739 (73.9% — all produced valid predictions, accuracy 72.4%) |
| **Eval output** | `eval/eval_s2/` (metrics.json, metrics.csv, report.txt) |
| **Comparison** | `eval/comparison_s1_vs_s2.json` (vs Run 5) |
| **Notes** | Proof-of-concept run with `--no_thinking`. NOT final paper data — thinking ON required for final results. Graceful timeout rate high (73.9%) due to GPU contention at parallel 8. |

---

### Run 5 vs Run 6 — Statistical Comparison (RQ1 preliminary)
| Test | Δ (S1 − S2) | p-value | Significant? |
|------|-------------|---------|-------------|
| Brier (bootstrap) | −0.007 (S1 better) | 0.37 | No |
| Accuracy (McNemar) | +3.1% (S1 better) | 0.065 | No |
| Brier (Wilcoxon) | — | 0.87 | No |

**Preliminary RQ1 answer:** Retrieval does not significantly improve over zero-shot in this `--no_thinking` configuration. Both setups show strong NO bias. Final comparison requires thinking ON runs.

---

## ⚠️ Comparability Warning

**Run 2 (ReAct+Reflection, 20 questions) and Run 3 (ZS, 939 questions) are NOT directly comparable:**
- Different question subsets (first 20 vs first 939 of the same file)
- Different GT distributions (49.4% vs 21.0% Yes) — accuracy numbers cannot be compared naively
- Run 2 used only 20 questions — wide confidence intervals (~±22% accuracy)

**To produce comparable results:** Run ReAct on the same 939 questions used for ZS. Until then, do not report ReAct vs ZS accuracy comparisons as conclusions.

---

## ⚠️ Prompt Version Warning

All three existing result files (Runs 1-3) used **Prompt v2** (`Confidence:` format, git `5caad98`).

Future runs use **Prompt v4** (XML `<prediction>` format, current HEAD). v3 (`P(Yes):` format) was superseded by v4 before any runs were produced.

**v2 and v4 results are not directly comparable** because:
- v2: `confidence` = P(predicted option); `predicted` derived from explicit "Selected Option:" text
- v4: XML `<prediction><yes>0.72</yes><no>0.28</no></prediction>`; `predicted` derived from `yes > 0.5`
- 21–34% of v2 records have `predicted ≠ argmax(agent_prob)` (inconsistency, see FINDINGS.md)
- v4 eliminates this inconsistency by construction

**Reporting guidance:** Tag every reported number with its prompt version. See `paper/PROMPTS_VERSIONED.md` for full version history.

---

## Planned Runs — 7-Setup Framework (Prompt v4, 2026-04-07)

**All previous planned runs (4-7) are superseded by the 7-setup framework below.**

### Two Evaluation Modes

**Question-by-question (no `--timeseries`):** One prediction per question using market close date as cutoff. Faster to run, good for overall accuracy/calibration metrics. Carry-forward setups (4-7) have no cross-timepoint context in this mode (only one prediction per question).

**Time-series (`--timeseries`):** Multiple predictions per question at historical price timepoints. Required for temporal analysis and carry-forward evaluation. Slower but enables temporal metrics and belief revision analysis.

**Recommendation:** Start with question-by-question for validation and initial metrics. Run time-series for the full paper experiments where temporal analysis is needed.

### Phase 1 — Validation (20 questions each, question-by-question)
| Run | Setup | CLI | Purpose |
|-----|-------|-----|---------|
| V1 | 1 (ZS) | `--setup 1 --max_questions 20 --parallel 8` | Validate XML parsing |
| V2 | 2 (ReAct) | `--setup 2 --max_questions 20 --parallel 8` | Validate XML + tool use |
| V3 | 3 (ReAct+Refl) | `--setup 3 --max_questions 20 --parallel 8` | Validate reflection extraction |
| V4 | 4 (Concl Carry) | `--setup 4 --max_questions 20 --timeseries --parallel 8` | Validate carry-forward (needs timeseries) |
| V5 | 5 (Crit Carry) | `--setup 5 --max_questions 20 --timeseries --parallel 8` | Validate self-critique carry (needs timeseries) |
| V6 | 6 (Both Carry) | `--setup 6 --max_questions 20 --timeseries --parallel 8` | Validate both carry types (needs timeseries) |
| V7 | 7 (Full) | `--setup 7 --max_questions 20 --timeseries --parallel 8` | Validate all components together (needs timeseries) |

**Note:** Setups 4-7 require `--timeseries` for carry-forward to function (they need multiple timepoints per question). Setups 1-3 can be validated in question-by-question mode for faster turnaround.

### Phase 2 — Pilot (100 questions each)
Purpose: Preliminary metrics, verify statistical pipeline, estimate full-run time.

```bash
# Question-by-question (setups 1-3, fast)
for s in 1 2 3; do
  python agent/evals.py --setup $s --max_questions 100 --parallel 8
done

# Time-series (all setups, for temporal + carry-forward metrics)
for s in 1 2 3 4 5 6 7; do
  python agent/evals.py --timeseries --setup $s --max_questions 100 --parallel 8
done
```

### Phase 3 — Full Experiment (939 questions each)
| Run | Setup | Questions | Mode | Est. Timepoints | Purpose |
|-----|-------|-----------|------|-----------------|---------|
| F1 | 1 (ZS) | 939 | Both | ~9,400 (TS) | Zero-shot baseline |
| F2 | 2 (ReAct) | 939 | Both | ~9,400 (TS) | Tool retrieval baseline |
| F3 | 3 (ReAct+Refl) | 939 | Both | ~9,400 (TS) | Iteration reflection ablation |
| F4 | 4 (Concl Carry) | 939 | Time-series | ~9,400 | Conclusion carry ablation |
| F5 | 5 (Crit Carry) | 939 | Time-series | ~9,400 | Self-critique carry ablation |
| F6 | 6 (Both Carry) | 939 | Time-series | ~9,400 | Combined carry ablation |
| F7 | 7 (Full) | 939 | Time-series | ~9,400 | All components combined |

### Phase 4 — Analysis
```bash
# Pairwise comparisons (key pairs from 7 setups)
python eval/eval_forecasting.py results_s1.jsonl --compare results_s2.jsonl  # RQ1: tools help?
python eval/eval_forecasting.py results_s2.jsonl --compare results_s3.jsonl  # RQ2: reflection helps?
python eval/eval_forecasting.py results_s2.jsonl --compare results_s4.jsonl  # RQ3a: conclusion carry
python eval/eval_forecasting.py results_s2.jsonl --compare results_s5.jsonl  # RQ3b: critique carry
python eval/eval_forecasting.py results_s4.jsonl --compare results_s5.jsonl  # RQ4: conclusion vs critique
python eval/eval_forecasting.py results_s6.jsonl --compare results_s7.jsonl  # RQ5: synergy
```

### Estimated Runtime (with `--parallel 8`)

| Mode | Setup | Thinking | 939 Questions | Notes |
|------|-------|----------|---------------|-------|
| Question-by-question | 1 (ZS) | ON | ~17 min | Single LLM call per question |
| Question-by-question | 2-7 (ReAct) | **OFF** | **~1-2 hrs** | 5-8 tool calls per question, full research loop |
| Question-by-question | 2-7 (ReAct) | ON | ~15 hrs | GPU contention → 1-2 tool calls, most questions timeout |
| Time-series | 1 (ZS) | ON | ~3 hrs | ~9,400 timepoints, single call each |
| Time-series | 2-7 (ReAct) | **OFF** | ~6-12 hrs | ~9,400 timepoints, multi-turn each |
| Time-series | 2-7 (ReAct) | ON | ~60+ hrs | Impractical without multiple GPUs |

**IMPORTANT (April 8, session 3):** All prior runtime estimates were invalidated by the discovery that the hybrid search engine was taking 260s/query (Error 10). With vector engine (0.8s/query), the LLM GPU contention is the only remaining bottleneck.

**Validated speed data (vector engine, April 8 session 3):**
| Config | Per question | Tool calls | Completes? | 1000q estimate |
|--------|------------|------------|-----------|---------------|
| no_thinking, sequential | 79s | 11 | Yes (0% timeout) | 22 hrs |
| no_thinking, parallel 8 | 33s throughput | 5-6 | 87% graceful timeout | **9.2 hrs** |
| thinking ON, sequential | 89s | 3 | Yes (0% timeout) | 24.8 hrs |
| thinking ON, parallel 8 | 65s throughput | 3-4 | 50% graceful timeout + errors | 18 hrs |

**Recommended configs:**
- **Proof of concept / overnight:** `--no_thinking --parallel 8` → 9.2 hrs, agent makes 5-6 tool calls per question
- **Final paper results:** thinking ON, `--parallel 4` → ~12 hrs estimated (untested, interpolated)
- **ZS baseline:** `--setup 1 --parallel 8` → 17 min (no tools, thinking mode irrelevant)

**Previous sequential runtime for comparison:** ZS timeseries 939q took ~22 hrs; ReAct+Reflection 20q took ~21 hrs (both with hybrid engine).

### Error handling in results

| Error tag | Meaning | Prediction available? |
|-----------|---------|----------------------|
| `null` | Normal completion | Yes |
| `GRACEFUL_TIMEOUT` | Time budget exceeded, final prediction extracted | Yes (from partial evidence) |
| `TIMEOUT` | Time budget exceeded, final extraction also failed | No |
| `TOKEN_LIMIT` | Context window exhausted mid-generation | Partial (may have prediction in trace) |
