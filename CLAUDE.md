# ARC Project — Claude Code Instructions

## What This Project Is
ARC is a research eval system for an EMNLP analysis paper. A ReAct LLM agent (V2.1) uses Reddit retrieval to forecast binary Polymarket questions. The pipeline is model-agnostic — any OpenAI-compatible LLM endpoint works.

## Current State (2026-04-16)
- **Paper direction:** analysis paper. V2.1 is the reference agent pipeline; LLM is the variable.
- **Bayes-med code removed** from source files, archived in `_archive/` for future use.
- **Archive:** `_archive/INDEX.md` maps everything that was removed — paper docs, Bayes code, analysis artifacts, scripts, tests.
- **Prior session logs:** `_archive/paper/SESSION_11.md` (latest), `_archive/paper/GRAVEYARD.md` (dead ends).
- **Brainstorm doc:** `_archive/paper/BRAINSTORM_20260416.md` — post-meeting direction pivot, core table design, open questions.

## Key Files
- `agent/react_agent.py` — V2.1 ReAct loop (`run_react_v2`)
- `agent/evals.py` — eval harness (parallel, timeout, JSONL output)
- `agent/prompts.py` — system prompts for 7 setups
- `agent/tools.py` — 5 Reddit tools (vector search only)
- `database/loader.py` — dataset loader
- `eval/eval_forecasting.py` — metrics + paired statistical tests

## How to Run
```bash
# V2.1, setup 2, no thinking, 8 parallel workers
python3 agent/evals.py --setup 2 --v2 --no_thinking \
    --max_questions 1000 --parallel 8 --timeout 120

# Compare two runs
python3 eval/eval_forecasting.py A.jsonl --compare B.jsonl --output compare.txt
```

## Important Rules
- **Explain before executing.** Always present proposed changes for user review before modifying code.
- **Do not run experiments** without confirming vLLM (port 8000) and D-agent (port 61001) are up.
- **Workflow:** Claude writes scripts + reads outputs; user pastes commands on the lab server.
- **Long runs:** wrap in `nohup bash <script> > logs/<name>.log 2>&1 &`.
- **Vector engine is hardcoded** — hybrid takes 260s/query, vector takes 0.8s.
- **The .venv has a broken python symlink.** Use system `python3`.

## Critical Config
- **Model:** Any OpenAI-compatible endpoint via vLLM (currently Qwen3-4B on GPU 2, port 8000)
- **D-agent:** Text2SQL vector engine on port 61001
- **Default timeout:** 120s per question
- **Context window:** 32768 tokens
