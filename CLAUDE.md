# ARC Project — Claude Code Instructions

## What This Project Is
ARC is a research eval system for an EMNLP paper. A ReAct LLM agent (Qwen3-14B-AWQ) uses Reddit retrieval to forecast binary Polymarket questions. Evaluated in time-series mode across 7 experimental setups with 3 ablation variables: tools, reflection, and carry-forward belief revision.

## How to Pick Up Where We Left Off
1. Read `paper/PROPOSAL.md` — the full research proposal with all design decisions, methodology, and submission strategy.
2. Read `paper/CHECKPOINT.md` §10-11 — known issues and decision status (all resolved).
3. The `paper/EXPERIMENTS.md` file maps every result file to its exact run conditions + planned runs.
4. The `paper/FINDINGS.md` file documents empirical findings from prior runs (v2 prompt, pre-framework).
5. All research docs live in `paper/` — do NOT create duplicates at the root.

## What Was Done Last (2026-04-07)
- **Designed 7-setup experiment framework** — factorial ablation of tools × reflection × carry-forward
- **Implemented all 7 setup prompts** in `agent/prompts.py` with XML `<prediction>` output format
- **Added carry-forward system** — conclusion + self-critique generation, cross-timepoint context passing
- **Added tool capabilities** — vector search, month filter, authors filter in `agent/tools.py`
- **Added parallel execution** — `--parallel N` for concurrent question processing in `agent/evals.py`
- **Added statistical tests** — bootstrap CI, McNemar, Wilcoxon in `eval/eval_forecasting.py --compare`
- **Updated all documentation** — CHECKPOINT.md, EXPERIMENTS.md, DECISIONS.md, PROMPTS_VERSIONED.md
- **Created `paper/PROPOSAL.md`** — full research proposal for professor/advisor review

## What's Next
- **Validation run:** All 7 setups on 20 questions to verify XML parsing and carry-forward work end-to-end
- **Pilot run:** All 7 setups on 100 questions for preliminary metrics
- **Full run:** All 7 setups on 939 questions (the paper experiments)

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
