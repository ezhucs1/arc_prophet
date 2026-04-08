# Experiments Log
**Last updated:** 2026-04-04

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

## ⚠️ Comparability Warning

**Run 2 (ReAct+Reflection, 20 questions) and Run 3 (ZS, 939 questions) are NOT directly comparable:**
- Different question subsets (first 20 vs first 939 of the same file)
- Different GT distributions (49.4% vs 21.0% Yes) — accuracy numbers cannot be compared naively
- Run 2 used only 20 questions — wide confidence intervals (~±22% accuracy)

**To produce comparable results:** Run ReAct on the same 939 questions used for ZS. Until then, do not report ReAct vs ZS accuracy comparisons as conclusions.

---

## ⚠️ Prompt Version Warning

All three existing result files used **Prompt v2** (`Confidence:` format, git `5caad98`).

Future runs use **Prompt v3** (`P(Yes):` format, current HEAD).

**v2 and v3 results are not directly comparable** because:
- v2: `confidence` = P(predicted option); `predicted` derived from explicit "Selected Option:" text
- v3: `P(Yes)` = direct probability of Yes; `predicted = Yes if P(Yes) > 0.5`
- 21–34% of v2 records have `predicted ≠ argmax(agent_prob)` (inconsistency, see FINDINGS.md)
- v3 eliminates this inconsistency by construction

**Reporting guidance:** Tag every reported number with its prompt version.

---

## Planned Runs — 7-Setup Framework (Prompt v4, 2026-04-07)

**All previous planned runs (4-7) are superseded by the 7-setup framework below.**

### Phase 1 — Validation (20 questions each)
| Run | Setup | CLI | Purpose |
|-----|-------|-----|---------|
| V1 | 1 (ZS) | `--setup 1 --max_questions 20` | Validate XML parsing |
| V2 | 2 (ReAct) | `--setup 2 --max_questions 20` | Validate XML + tool use |
| V3 | 3 (ReAct+Refl) | `--setup 3 --max_questions 20` | Validate reflection extraction |
| V4 | 4 (Concl Carry) | `--setup 4 --max_questions 20` | Validate carry-forward |
| V5 | 5 (Crit Carry) | `--setup 5 --max_questions 20` | Validate self-critique carry |
| V6 | 6 (Both Carry) | `--setup 6 --max_questions 20` | Validate both carry types |
| V7 | 7 (Full) | `--setup 7 --max_questions 20` | Validate all components together |

### Phase 2 — Pilot (100 questions each)
Purpose: Preliminary metrics, verify statistical pipeline, estimate full-run time.

### Phase 3 — Full Experiment (939 questions each)
| Run | Setup | Questions | Timepoints (est.) | Purpose |
|-----|-------|-----------|--------------------|---------|
| F1 | 1 (ZS) | 939 | ~9,400 | Zero-shot baseline |
| F2 | 2 (ReAct) | 939 | ~9,400 | Tool retrieval baseline |
| F3 | 3 (ReAct+Refl) | 939 | ~9,400 | Iteration reflection ablation |
| F4 | 4 (Concl Carry) | 939 | ~9,400 | Conclusion carry ablation |
| F5 | 5 (Crit Carry) | 939 | ~9,400 | Self-critique carry ablation |
| F6 | 6 (Both Carry) | 939 | ~9,400 | Combined carry ablation |
| F7 | 7 (Full) | 939 | ~9,400 | All components combined |

### Phase 4 — Analysis
```bash
# Pairwise comparisons (21 pairs from 7 setups)
python eval/eval_forecasting.py results_s1.jsonl --compare results_s2.jsonl  # RQ1
python eval/eval_forecasting.py results_s2.jsonl --compare results_s3.jsonl  # RQ2
python eval/eval_forecasting.py results_s2.jsonl --compare results_s4.jsonl  # RQ3a
python eval/eval_forecasting.py results_s2.jsonl --compare results_s5.jsonl  # RQ3b
python eval/eval_forecasting.py results_s4.jsonl --compare results_s5.jsonl  # RQ4
python eval/eval_forecasting.py results_s6.jsonl --compare results_s7.jsonl  # RQ5
```
