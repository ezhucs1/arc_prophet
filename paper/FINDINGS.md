# Research Findings
**Last updated:** 2026-04-04

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

### Speed improvement available
Disabling `<think>` via `enable_thinking=False` eliminates the thinking overhead. Based on token counting (thinking blocks ≈ 2000 tokens, output ≈ 500 tokens), this should give ~4–5× speedup. The `--no_thinking` flag was added to support this ablation (Run 6 in planned experiments).

### Quality vs speed tradeoff unknown
Whether disabling thinking degrades prediction quality is not yet measured. This is the explicit purpose of Run 6.

### Implication for paper
The latency numbers must be reported to contextualize the system's practical feasibility. At 329s/timepoint × 9,033 timepoints = ~830 GPU-hours for a full ReAct+Reflection run. This is a meaningful resource constraint for the paper's scalability discussion.

---

## Summary Table

| Finding | Runs with Evidence | Statistical Strength | Paper-Ready? |
|---------|--------------------|----------------------|--------------|
| 1. Prob/choice inconsistency (21–34%) | Run 2, Run 3 | Strong (large N in Run 3) | ✅ Yes |
| 2. NO prediction bias | Run 2, Run 3 | Strong | ✅ Yes (with caveat) |
| 3. Market calibration gap (BSS −5.39) | Run 2 only | Moderate (20q) | ⚠️ Frame carefully |
| 4. Reflection +16.8 pp | Run 2 only | Weak (20q, ±22%) | ❌ Needs Run 5 |
| 5. Majority vote improvement | Run 2 only | Weak (20q) | ❌ Needs Run 5 |
| 6. Speed vs quality tradeoff | Run 2, Run 3 | Strong (latency) | ✅ Latency numbers only |

---

## Open Questions for Paper

1. Does retrieval (ReAct) improve calibration over zero-shot on the same 939-question set? (Requires Run 4)
2. Does reflection further improve over plain ReAct? (Requires Runs 4 + 5)
3. Does disabling `<think>` degrade quality? (Requires Run 6)
4. At what temporal distance from resolution does LLM accuracy plateau? (Requires time-series decomposition of existing results)
5. Are there question categories where retrieval helps more? (Blocked by empty `topic` field — see DATASET.md §3)
6. What fraction of questions have zero relevant Reddit content? (Requires Text2SQL query over the full dataset)
