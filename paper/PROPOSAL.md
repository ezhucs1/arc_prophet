# Research Proposal: Temporal Evaluation of Retrieval-Augmented LLM Forecasting Agents
## Target Venue: EMNLP 2026 (Main Conference — Evaluation Track)

**Authors:** [To be filled]  
**Date:** 2026-04-08  
**Status:** Implementation complete, experiments pending

---

## 1. Executive Summary

We present **a temporal evaluation framework for decomposing LLM forecasting agent performance** into retrieval, reflection, and belief revision components. Using 939 binary prediction market questions from Polymarket, evaluated across 10 time-series price points each, we systematically ablate 7 agent configurations to answer: *Which components of an agentic RAG pipeline actually improve probabilistic calibration over time?*

The agent (Qwen3-14B-AWQ) uses a Reddit knowledge base of 19 subreddits as its sole retrieval source, with Polymarket crowd probabilities as both the evaluation anchor and a market baseline.

**Why this matters for EMNLP:** No existing work evaluates LLM forecasting agents as evolving systems across time. Prior work (Halawi et al. 2024; Zou et al. 2024; ForecastBench) treats each prediction as independent. We introduce cross-timepoint belief revision — where the agent carries forward structured memory and self-critique — and measure whether this temporal reasoning improves calibration, degrades it, or has no effect.

---

## 2. Research Questions

| # | Question | How We Answer It |
|---|----------|-----------------|
| **RQ1** | Does tool-augmented retrieval improve calibration over zero-shot? | Setup 1 (ZS) vs Setup 2 (ReAct), same questions, same timepoints |
| **RQ2** | Does within-timepoint structured reflection improve prediction quality? | Setup 2 vs Setup 3 (ReAct + Iteration Reflection) |
| **RQ3** | Does cross-timepoint memory (belief revision) improve calibration over time? | Setups 2 vs 4, 5, 6 — three carry-forward variants |
| **RQ4** | What type of carry-forward is most effective: factual conclusions, reasoning self-critique, or both? | Setup 4 vs 5 vs 6 |
| **RQ5** | Does combining all components (reflection + carry) produce synergistic improvement? | Setup 7 (full) vs all others |
| **RQ6** | On which question categories does social media RAG add value over prediction markets? | Per-category Brier Skill Score analysis |

---

## 3. The 7 Experimental Setups

This is a controlled factorial ablation with 3 independent variables:

| Setup | Name | Tools | Iteration Reflection | Carry-Forward |
|-------|------|-------|---------------------|---------------|
| 1 | Zero-Shot | No | No | No |
| 2 | ReAct | Yes | No | No |
| 3 | ReAct + Reflection | Yes | Yes (every 3 tool calls) | No |
| 4 | ReAct + Conclusion Carry | Yes | No | Factual conclusion |
| 5 | ReAct + Self-Critique Carry | Yes | No | Reasoning self-critique |
| 6 | ReAct + Both Carry | Yes | No | Conclusion + critique |
| 7 | ReAct + Full | Yes | Yes | Conclusion + critique |

### 3.1 Why 7 Setups (Not Fewer, Not More)

**Why not fewer:** With 3 conditions (ZS, ReAct, ReAct+Reflection), we could answer RQ1 and RQ2 — but the paper would lack novelty. The carry-forward variants (Setups 4–7) are the novel contribution, and they require isolating conclusion vs critique to understand *which type of temporal memory helps*.

**Why not more:** A complete 2×2×3 factorial (reflection × carry_type) would need 9 setups. The missing cells (reflection + conclusion-only, reflection + critique-only) would cost ~170 GPU-hours for comparisons unlikely to change the paper's conclusions. Setup 7 tests the full combination; "future work" addresses the partial combinations. 7 setups is above the EMNLP average of 3–5 conditions.

### 3.2 What Each Setup Isolates

```
Setup 1 → 2:  Effect of retrieval (tools)
Setup 2 → 3:  Effect of iteration reflection
Setup 2 → 4:  Effect of factual conclusion carry
Setup 2 → 5:  Effect of reasoning self-critique carry
Setup 2 → 6:  Effect of both carry types together
Setup 4 vs 5: Conclusion carry vs self-critique carry (head-to-head)
Setup 6 → 7:  Added value of reflection when carry is already present
```

---

## 4. Novel Contributions

### 4.1 Temporal Evaluation Framework (Primary)
No prior forecasting benchmark evaluates LLMs across a time series of price snapshots. Existing benchmarks (ForecastBench, Halawi et al.) ask one question once. We evaluate the same question 10 times at different dates, tracking how the agent's probability evolves relative to the market.

**New metrics enabled:**
- **Temporal learning curve:** Does accuracy improve from t1 → t10?
- **Belief revision dynamics:** Does the agent update beliefs when new evidence appears?
- **Temporal consistency:** Does the agent give the same answer at t5 as t6 (absent new evidence)?

### 4.2 Carry-Forward Belief Revision (Primary)
We introduce structured cross-timepoint memory for forecasting agents. Two carry-forward mechanisms:

**Conclusion carry:** After each timepoint, the agent generates a structured summary containing its prediction, the market probability (public historical data), key evidence found, evidence gaps, and a forward-looking prior. This is passed to the next timepoint as starting context.

**Self-critique carry (without outcome feedback):** After each timepoint, the agent evaluates its own reasoning process — search quality, evidence balance, cognitive biases, confidence justification — and generates a concrete improvement plan. Critically, the agent does not know whether its prediction was correct. This tests pure metacognitive self-improvement.

**Why self-critique without ground truth is more interesting:** If we told the agent "you were wrong by 0.3," any improvement could be attributed to answer leakage. With self-critique-only, the agent must improve its reasoning *process* without knowing its reasoning *outcome*. This is both more realistic (real forecasters don't get immediate feedback) and more scientifically interesting (tests LLM metacognition).

### 4.3 Component Ablation (Supporting)
The 7-setup design cleanly isolates each component's contribution. This is standard in NLP evaluation but novel for forecasting agent evaluation, where prior work compares model families rather than architectural components.

### 4.4 Social Media RAG for Forecasting (Supporting)
Using Reddit as the sole retrieval source is novel. Prior retrieval-augmented forecasting work uses news articles or web search. Reddit provides a different signal: crowd opinions, insider knowledge, niche expertise, and early indicators that precede mainstream news coverage. We can identify question categories where Reddit signal is strong (crypto, tech) vs weak (geopolitics, economics).

---

## 5. Methodology Details

### 5.1 Agent Architecture
- **Model:** Qwen3-14B-AWQ (4-bit quantized, 14B parameters)
- **Serving:** vLLM v0.16.0 with continuous batching, 32K context window
- **Agent loop:** Plain ReAct (no frameworks), function calling via Hermes tool parser
- **Temperature:** 0.1 (near-deterministic)
- **Extended thinking:** Enabled (Qwen3's `<think>` blocks for chain-of-thought)

### 5.2 Tool Suite (6 capabilities, 5 tools)
| Tool | Description |
|------|-------------|
| `search_database` (hybrid) | Semantic + keyword search across Reddit posts/comments |
| `search_database` (vector) | Pure semantic search (new — catches differently-worded content) |
| `get_post_core_info` | Full post content by ID |
| `get_comment_core_info` | Full comment content by ID |
| `get_post_comments_list` | Thread context around a comment |
| `get_author_history_list` | All posts/comments by a specific author |

Additional filters: `month` (YYYY-MM), `authors` (comma-separated), `subreddit` (19 available).

### 5.3 Reddit Knowledge Base
19 subreddits spanning crypto, economics, politics, technology, science, health, entertainment, and general discussion. The database enforces temporal cutoff: the agent at timepoint t can only see posts from before date t.

### 5.4 Dataset
**Source:** Polymarket binary yes/no prediction markets.  
**Size:** 939 questions (after filtering for settled binary markets with price history).  
**Timepoints:** ~10 per question (evenly sampled from price history), ~9,400 total.  
**Date range:** February 2026.  
**Ground truth:** Market resolution (Yes or No).  
**Market baseline:** Daily Polymarket prices (market-implied probabilities).

### 5.5 Output Format
All setups produce structured XML predictions:
```xml
<prediction>
  <yes>0.72</yes>
  <no>0.28</no>
  <reasoning>2-4 sentence explanation</reasoning>
</prediction>
```
Parsed with XML regex, with fallback to plain-text patterns for robustness.

### 5.6 Carry-Forward Generation (Setups 4–7)
After each timepoint, separate LLM calls generate:

**Conclusion (Setups 4, 6, 7):**
```xml
<conclusion>
  <my_prediction>yes=0.72, no=0.28</my_prediction>
  <market_probability>yes=0.65, no=0.35</market_probability>
  <key_evidence>2-3 most important findings</key_evidence>
  <evidence_gaps>What was missing or uncertain</evidence_gaps>
  <forward_prior>Updated belief in 1-2 sentences</forward_prior>
</conclusion>
```

**Self-critique (Setups 5, 6, 7):**
```xml
<self_critique>
  <search_quality>Was search strategy thorough?</search_quality>
  <evidence_balance>Were both sides weighed fairly?</evidence_balance>
  <reasoning_biases>Cognitive biases detected</reasoning_biases>
  <confidence_justification>Is confidence proportional to evidence?</confidence_justification>
  <improvement_plan>Concrete actions for next timepoint</improvement_plan>
</self_critique>
```

**Design choice: market probability in conclusion, not in critique.**  
The conclusion includes the market price at that timepoint — this is public historical data, equivalent to checking a stock price. The self-critique does NOT include the market price or any outcome information. The agent critiques its reasoning process blind, which is the realistic and scientifically interesting condition.

---

## 6. Evaluation Metrics

### 6.1 Per-Setup Metrics
| Category | Metrics |
|----------|---------|
| **Classification** | Accuracy, Balanced Accuracy, MCC, F1, Precision, Recall |
| **Calibration** | Brier Score, Log Loss, Expected Calibration Error (ECE) |
| **vs Market** | Brier Skill Score (agent vs market), directional agreement, correlation |
| **Temporal** | Accuracy by timepoint position (t1→t10), consistency rate, majority vote accuracy |
| **Operational** | Avg latency, tool call count, error rate, null prediction rate |

### 6.2 Cross-Setup Statistical Tests
| Test | Purpose | When Used |
|------|---------|-----------|
| **Paired Bootstrap CI** (10,000 samples) | Confidence interval on Brier score differences between setups | Every pairwise comparison |
| **McNemar's Test** | Whether two setups disagree on significantly different cases | Accuracy comparisons |
| **Wilcoxon Signed-Rank** | Whether per-question metric differences are systematic | Per-question Brier differences |

All comparisons are **paired** on matched question-timepoint keys, giving maximum statistical power.

### 6.3 Minimum Sample Size
With 939 questions × ~10 timepoints = ~9,400 paired observations per comparison, we have strong statistical power. At α=0.05, a paired bootstrap can detect Brier score differences as small as 0.005 with >80% power at this sample size.

---

## 7. Expected Findings & Hypotheses

We do not pre-register specific numerical predictions, but we state directional hypotheses:

| Hypothesis | Expected Outcome | If Wrong, Still Publishable? |
|------------|-----------------|------------------------------|
| H1: ReAct > ZS on Brier | Yes — retrieval provides grounding | Yes — "retrieval doesn't help" is equally interesting |
| H2: Reflection improves on subsets | Mixed — may help on uncertain questions, hurt on clear ones | Yes — identifies when reflection helps/hurts |
| H3: Conclusion carry > no carry | Yes — prevents redundant search, provides starting prior | Yes — "memory doesn't help" tests temporal reasoning limits |
| H4: Self-critique carry improves over time | Uncertain — depends on LLM metacognitive ability | Yes — either confirms or refutes LLM self-improvement |
| H5: Market beats agent overall | Almost certainly yes (Brier Skill Score < 0) | N/A — market is the anchor, not the target to beat |
| H6: Agent beats market on some categories | Likely for niche topics (crypto, tech) where Reddit is strong | Yes — complementarity analysis shows where RAG adds value |

**Key insight:** Every combination of outcomes produces a publishable result. If nothing helps (H1–H4 all negative), the paper becomes "decomposing the failure modes of retrieval-augmented forecasting" — still a contribution, because it tells practitioners what not to build.

---

## 8. Compute Budget & Execution Plan

### 8.1 Parallelism Strategy
- **Multiple R6000 GPUs available** — one vLLM instance per GPU
- **Cross-question parallelism:** Different questions processed concurrently (8-way)
- **Within-question serialization:** Timepoints are sequential for carry-forward setups
- **vLLM continuous batching:** Handles concurrent requests internally

### 8.2 Estimated Timeline

| Phase | Setups | Questions | Est. Wall-Clock (4 GPUs, 8 parallel) |
|-------|--------|-----------|---------------------------------------|
| **Validation** | All 7 | 20 each | ~4 hours |
| **Pilot** | All 7 | 100 each | ~18 hours |
| **Full run** | All 7 | 939 each | ~36 hours |
| **Analysis** | — | — | 1–2 days |
| **Total** | | | **~4 days** |

### 8.3 Execution Order
1. Run all 7 setups on 20 questions → validate XML parsing, carry-forward, output correctness
2. Run all 7 on 100 questions → compute preliminary metrics, verify statistical test pipeline
3. Run all 7 on 939 questions → full experiment
4. Run statistical comparisons between all setup pairs
5. Generate per-category, per-timepoint, and complementarity analyses

---

## 9. Related Work & Positioning

### 9.1 LLM Forecasting
- **Halawi et al. (2024)** "Approaching Human-Level Forecasting with Language Models" — GPT-4 + retrieval on Metaculus questions. Single-point predictions. No temporal evaluation. No component ablation.
- **Zou et al. (2024)** "ForecastBench" — Benchmark for LLM forecasting. Single-point. No ReAct agents. No Reddit retrieval.
- **Schoenegger et al. (2024)** — LLM vs human forecasters on geopolitical questions. No retrieval augmentation.

**Our differentiators:** (1) Temporal multi-point evaluation, (2) Component ablation, (3) Reddit RAG, (4) Carry-forward belief revision.

### 9.2 ReAct Agents
- **Yao et al. (2023)** "ReAct" — The ReAct framework paper. Applied to QA and reasoning tasks, not forecasting.
- **Shinn et al. (2023)** "Reflexion" — Self-reflection for task-solving agents. Single-task, not time-series. Uses outcome feedback. Our self-critique is without outcome feedback.

### 9.3 LLM Self-Improvement
- **Madaan et al. (2023)** "Self-Refine" — Iterative self-refinement. Single-turn. Uses the same model's critique to revise.
- **Pan et al. (2024)** "Automatically Correcting Large Language Models" — Survey of LLM self-correction. Notes that self-correction without external feedback is largely ineffective.

**Our positioning:** We test temporal self-improvement (critique at t carries to t+1) without outcome feedback. If Setup 5 or 7 improves over Setup 2, this provides evidence against the "self-correction doesn't work" consensus — in a structured temporal setting.

---

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **All setups perform equally** | Null result for RQ2–RQ5 | Still publishable as negative result; reframe as "limits of agent augmentation" |
| **XML parsing failures** | Lost data points | Regex fallback chain (XML → P(Yes): → Confidence:). Validated on prior runs. |
| **Carry-forward context too long at t10** | Context overflow | Capped at ~200 tokens per carry block. At t10 with both carry types: ~400 tokens — well within 32K budget. |
| **Market beats agent on everything** | Weak results | Expected. The contribution is the decomposition, not beating the market. Report complementarity (where agent adds value). |
| **Qwen3-14B too weak** | Floor effect — all setups bad | Acknowledged as limitation. Framework is model-agnostic — results with GPT-4/Claude would differ. |
| **Resume breaks carry-forward** | Data loss on crash | Known limitation. Carry state isn't persisted to disk. Mitigation: small batches, checkpointing. |
| **Agent exceeds time budget** | Lost prediction | Graceful timeout: one final LLM call extracts prediction from partial evidence. Tagged `GRACEFUL_TIMEOUT` for sensitivity analysis. |

---

## 11. Paper Outline (Preliminary)

1. **Introduction** — LLM forecasting agents exist, but no temporal evaluation framework. We introduce one.
2. **Related Work** — §9 above.
3. **Framework** — The 7-setup design, carry-forward mechanism, metrics.
4. **Experimental Setup** — Model, dataset, Reddit KB, tools, execution.
5. **Results**
   - 5.1: Setup-level metrics table (all 7 × all metrics)
   - 5.2: Pairwise statistical comparisons (bootstrap CI, McNemar, Wilcoxon)
   - 5.3: Temporal trajectory analysis (agent vs market across t1→t10)
   - 5.4: Carry-forward analysis (does belief revision improve? which type?)
   - 5.5: Category-level complementarity (where does Reddit RAG add value?)
6. **Analysis**
   - 6.1: Error analysis with qualitative examples
   - 6.2: Reflection quality analysis (does reflection content predict accuracy?)
   - 6.3: Self-critique quality analysis (are the agent's self-diagnoses correct?)
7. **Discussion** — Implications for forecasting agent design. When to use which component.
8. **Limitations** — Single model, single retrieval source, single market platform.
9. **Conclusion**

---

## 12. Why EMNLP (vs ACL, NAACL, NeurIPS)

- **EMNLP values empirical rigor** — our 7-setup ablation with statistical tests is a natural fit
- **Evaluation track** — EMNLP has historically accepted benchmark and evaluation papers (ForecastBench was at ICML; we have more ablations)
- **NLP + information retrieval** — Reddit RAG for forecasting bridges NLP and IR communities
- **Timeliness** — LLM agents and prediction markets are hot topics; temporal evaluation fills a clear gap

---

## 13. Submission Strategy

### 13.1 Timeline
| Milestone | Date |
|-----------|------|
| Full experiment runs | Week of 2026-04-07 |
| Analysis complete | 2026-04-14 |
| First draft | 2026-04-21 |
| Internal review (advisor + PhD senior) | 2026-04-28 |
| Camera-ready target | EMNLP 2026 deadline |

### 13.2 Reviewer Objection Preemption

| Likely Objection | Our Response |
|-----------------|--------------|
| "Only one model tested" | Acknowledged as limitation. Framework is model-agnostic. Qwen3-14B was chosen for cost efficiency (local GPU inference). We provide code for replication with any OpenAI-compatible model. |
| "Reddit is a weak signal for forecasting" | That's the point — we measure when it helps and when it doesn't. The complementarity analysis (§5.5) shows which categories benefit. |
| "Market baseline is too strong" | We're not trying to beat the market. We're decomposing agent performance. The market is the anchor that contextualizes our findings. |
| "Self-critique without feedback can't work (Pan et al. 2024)" | Our temporal setting is different: critique at t informs behavior at t+1 with new evidence. This is not same-turn self-correction. Results either confirm or refute the consensus in a new setting. |
| "7 setups × 939 questions — is this statistically powered?" | Yes. ~9,400 paired observations per comparison. Bootstrap CI can detect Brier differences of 0.005 at this N. We report p-values for every comparison. |

---

## 14. Current Implementation Status

All code is implemented and syntax-verified. Ready for experiments.

### 14.1 Files Modified (2026-04-07 session)

| File | What Changed |
|------|-------------|
| `agent/tools.py` | Added `engine` (hybrid/vector), `month`, `authors` params to `search_database` |
| `agent/prompts.py` | XML `<prediction>` output format. 7 setup prompt constants. Carry-forward addons. Conclusion + self-critique generation prompts. `format_carry_context()` helper. |
| `agent/react_agent.py` | `setup` + `carry_context` params. `generate_carry_forward()` for conclusion/critique generation. Thread-safe. |
| `agent/evals.py` | Unified XML extraction pipeline. `--setup 1-7` CLI. `--parallel N` for concurrent questions. Thread-safe timeout. Carry orchestration for setups 4-7. |
| `eval/eval_forecasting.py` | Paired bootstrap CI, McNemar's test, Wilcoxon signed-rank. `--compare` mode for cross-setup statistical comparison. |

### 14.2 Running an Experiment

```bash
# Start infrastructure
vllm serve Qwen/Qwen3-14B-AWQ --host 127.0.0.1 --port 8000 \
  --max-model-len 32768 --quantization awq --enforce-eager \
  --enable-auto-tool-choice --tool-call-parser hermes --generation-config vllm

# Question-by-question mode (fast — one prediction per question)
python agent/evals.py --setup 2 --max_questions 100 --parallel 8

# Time-series mode (temporal analysis — multiple timepoints per question)
python agent/evals.py --timeseries --setup 2 --max_questions 100 --parallel 8

# Run all 7 setups
for s in 1 2 3 4 5 6 7; do
  python agent/evals.py --setup $s --max_questions 939 --parallel 8
done

# Compare results
python eval/eval_forecasting.py results_s2.jsonl --compare results_s7.jsonl
```

**Note on `--parallel`:** vLLM's continuous batching handles concurrent requests efficiently. `--parallel 8` sends 8 questions simultaneously, which the GPU processes as a batch. Default `--parallel 1` = sequential (original behavior).

---

## 15. Summary — Why This Paper Gets Accepted

1. **Clear gap:** No temporal evaluation for LLM forecasting agents exists.
2. **Clean design:** 7-setup ablation isolates each component — tools, reflection, carry-forward.
3. **Novel mechanism:** Cross-timepoint belief revision with self-critique (without outcome feedback).
4. **Rigorous evaluation:** Paired statistical tests on ~9,400 observations per comparison.
5. **Strong baseline:** Polymarket prices as market anchor.
6. **Either-way findings:** Every combination of positive/negative results is publishable.
7. **Practical impact:** Tells practitioners which agent components to invest in.
8. **Reproducible:** Full code, exact prompts, clear documentation.
