"""
Benchmark evaluation for the ReAct agent on the Polymarket dataset.

Usage:
    python agent/evals.py --max_questions 10
    python agent/evals.py --max_questions 100
    python agent/evals.py --results_file results/results_XXXXXX.jsonl   # resume
    python agent/evals.py --summarize results/results_XXXXXX.jsonl
"""

import sys
import argparse
import json
import re
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path when this file is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from database.loader import load_polymarket, load_polymarket_timeseries
from agent.react_agent import run_react, run_react_v2, generate_carry_forward

# Set by --v2 CLI flag; when True, run_single uses run_react_v2 (structured-round driver).
USE_V2 = False
from agent.prompts import (
    ZERO_SHOT_SYSTEM_PROMPT, SETUP_PROMPTS,
    SETUP_IS_ZERO_SHOT, SETUP_HAS_TOOLS,
    SETUP_USES_CONCLUSION, SETUP_USES_CRITIQUE,
    CONCLUSION_GENERATION_PROMPT, SELF_CRITIQUE_GENERATION_PROMPT,
    format_carry_context,
)

DATASET_PATH = Path(__file__).parent.parent / "database" / "polymarket_binary_yesno.jsonl"
RESULTS_DIR  = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_TIMEOUT_SEC = 120


# ── Answer / confidence extraction ───────────────────────────────────────────

def _strip_think(text: str) -> str:
    """
    Remove <think>...</think> blocks from text.
    If the opening tag has no matching closing tag (truncated generation),
    strip everything from <think> to end-of-string so partial thinking
    content cannot pollute extraction.
    """
    # Remove complete blocks first
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove any unclosed block (truncated mid-generation)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def extract_prediction_xml(text: str) -> Optional[dict]:
    """
    Parse the last <prediction> XML block → dict with keys 'yes', 'no', 'reasoning'.
    Returns None if no valid block found.
    """
    if not text:
        return None
    clean = _strip_think(text)
    # Find all <prediction> blocks, take the last one
    blocks = re.findall(r"<prediction>(.*?)</prediction>", clean, re.DOTALL)
    if not blocks:
        return None
    block = blocks[-1]
    yes_m = re.search(r"<yes>\s*([\d.]+)\s*</yes>", block)
    no_m  = re.search(r"<no>\s*([\d.]+)\s*</no>", block)
    reas_m = re.search(r"<reasoning>(.*?)</reasoning>", block, re.DOTALL)
    if not yes_m or not no_m:
        return None
    try:
        yes_val = round(max(0.0, min(1.0, float(yes_m.group(1)))), 4)
        no_val  = round(max(0.0, min(1.0, float(no_m.group(1)))), 4)
    except ValueError:
        return None
    return {
        "yes": yes_val,
        "no": no_val,
        "reasoning": reas_m.group(1).strip() if reas_m else None,
    }


def extract_selected_option(text: str, options: list[str]) -> Optional[str]:
    """
    Priority:
      1. Explicit 'Selected Option: X' label
      2. First option string found verbatim in clean text
    Strips <think> blocks (including unterminated ones) before matching.
    """
    if not text:
        return None
    clean = _strip_think(text)
    if not clean:
        return None
    # Take the last "Selected Option:" match — earlier ones may be from tool results
    # in the full trace.
    matches = list(re.finditer(r"Selected Option[:\s]+(.+?)(?:\n|$)", clean, re.IGNORECASE))
    if matches:
        candidate = matches[-1].group(1).strip().rstrip(".,;")
        for opt in options:
            if opt.lower() == candidate.lower():
                return opt
    for opt in options:
        if re.search(re.escape(opt), clean, re.IGNORECASE):
            return opt
    return None


def extract_reflection(text: str) -> Optional[str]:
    """
    Extract the last Reflection block from the agent output.
    Supports both new XML <reflection> format and legacy plain-text format.
    Searches visible text first; falls back to inside <think> if not found.
    Qwen3 sometimes puts the reflection block inside <think> despite instructions.
    """
    if not text:
        return None
    clean = _strip_think(text)

    # New XML format: <reflection>...</reflection>
    xml_blocks = re.findall(r"<reflection>(.*?)</reflection>", clean, re.DOTALL)
    if xml_blocks:
        return xml_blocks[-1].strip()

    # Legacy plain-text format: "Reflection: ..."
    _pattern = r"Reflection:(.*?)(?=\n(?:Reflection:|Reasoning:|Final Answer:|Selected Option:)|$)"
    matches = list(re.finditer(_pattern, clean, re.DOTALL | re.IGNORECASE))
    if matches:
        return matches[-1].group(1).strip()
    # Fallback: inside <think> block
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        # Try XML first inside think
        xml_blocks = re.findall(r"<reflection>(.*?)</reflection>", think_match.group(1), re.DOTALL)
        if xml_blocks:
            return xml_blocks[-1].strip()
        matches = list(re.finditer(_pattern, think_match.group(1), re.DOTALL | re.IGNORECASE))
        if matches:
            return matches[-1].group(1).strip()
    return None


def extract_self_critique(text: str) -> Optional[str]:
    """
    Extract Self-critique from the agent output.
    Supports both new XML <self_critique> format and legacy plain-text format.
    """
    if not text:
        return None
    clean = _strip_think(text)

    # New XML format: <self_critique>...</self_critique>
    xml_blocks = re.findall(r"<self_critique>(.*?)</self_critique>", clean, re.DOTALL)
    if xml_blocks:
        return xml_blocks[-1].strip()

    # Legacy plain-text format: "Self-critique: ..."
    m = re.search(r"Self-critique[:\s]+(.+?)(?=\nFinal Answer:|\nSelected Option:|$)",
                  clean, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def extract_prob_yes(text: str) -> Optional[float]:
    """
    Parse 'P(Yes): 0.75' → float in [0.0, 1.0] or None.

    P(Yes) is the canonical probability output: the probability that Yes is the
    correct answer.  Selected Option is derived from it (Yes if P(Yes) > 0.5).

    Falls back to the old 'Confidence:' label for backward compatibility with
    results produced before the prompt format change.
    """
    if not text:
        return None
    clean = _strip_think(text)

    # New format: P(Yes): 0.75
    # Use findall + take last match — the final answer is appended last in the
    # full trace, so earlier matches may come from tool results or reflections.
    matches = re.findall(r"P\s*\(\s*[Yy]es\s*\)\s*[:\s]+([\d.]+)", clean)
    if matches:
        try:
            return round(max(0.0, min(1.0, float(matches[-1]))), 4)
        except ValueError:
            pass

    # Legacy fallback: Confidence: 0.75  (old format, confidence-in-prediction)
    # In old records this was P(predicted option), NOT P(Yes), so we return None
    # to let the caller handle old records separately via extract_confidence_legacy.
    return None


def extract_confidence_legacy(text: str) -> Optional[float]:
    """
    Parse old 'Confidence: 0.75' label → float in [0.0, 1.0] or None.
    This was P(predicted option), not P(Yes).  Used only for backward-compat
    when reading pre-format-change result files.
    """
    if not text:
        return None
    clean = _strip_think(text)
    matches = re.findall(r"Confidence[:\s]+([\d.]+)", clean, re.IGNORECASE)
    if matches:
        try:
            return round(max(0.0, min(1.0, float(matches[-1]))), 4)
        except ValueError:
            pass
    return None


def extract_agent_prediction(text: str, options: list[str]) -> dict:
    """
    Unified extraction: try XML <prediction> first, then P(Yes):, then legacy Confidence:.

    Returns dict with keys:
      predicted:      str or None  ("Yes" / "No")
      agent_prob_yes: float or None
      agent_prob_no:  float or None
      confidence:     float or None (P(predicted option), for readability)
      reasoning:      str or None
    """
    result = {"predicted": None, "agent_prob_yes": None, "agent_prob_no": None,
              "confidence": None, "reasoning": None}

    # Priority 1: XML <prediction> block
    xml = extract_prediction_xml(text)
    if xml and xml["yes"] is not None:
        result["agent_prob_yes"] = xml["yes"]
        result["agent_prob_no"] = xml["no"]
        result["predicted"] = "Yes" if xml["yes"] > 0.5 else "No"
        result["confidence"] = xml["yes"] if result["predicted"] == "Yes" else xml["no"]
        result["reasoning"] = xml.get("reasoning")
        return result

    # Priority 2: P(Yes): label
    prob_yes = extract_prob_yes(text)
    if prob_yes is not None:
        result["agent_prob_yes"] = prob_yes
        result["agent_prob_no"] = round(1.0 - prob_yes, 4)
        result["predicted"] = "Yes" if prob_yes > 0.5 else "No"
        result["confidence"] = prob_yes if result["predicted"] == "Yes" else result["agent_prob_no"]
        return result

    # Priority 3: Legacy — Selected Option + Confidence
    predicted = extract_selected_option(text, options)
    legacy_conf = extract_confidence_legacy(text)
    result["predicted"] = predicted
    result["confidence"] = legacy_conf
    if legacy_conf is not None and predicted == "Yes":
        result["agent_prob_yes"] = legacy_conf
        result["agent_prob_no"] = round(1.0 - legacy_conf, 4)
    elif legacy_conf is not None and predicted == "No":
        result["agent_prob_no"] = legacy_conf
        result["agent_prob_yes"] = round(1.0 - legacy_conf, 4)
    return result


# ── Timeout helper ────────────────────────────────────────────────────────────


# ── Per-question runners ──────────────────────────────────────────────────────

def run_single(question: str, options: list[str], cutoff_time: str,
               timeout_sec: int = DEFAULT_TIMEOUT_SEC,
               reflection: bool = False,
               thinking: bool = True,
               setup: int | None = None,
               carry_context: str = "",
               description: str = "") -> dict:
    """
    Run one ReAct question with timeout.

    Uses threading.Timer + cancel_event for reliable timeout in ALL threads.
    SIGALRM is unreliable (swallowed by C extensions during HTTP I/O).
    No nested ThreadPoolExecutor (causes deadlock).
    """
    cancel = threading.Event()
    timer = threading.Timer(timeout_sec, cancel.set)
    timer.daemon = True
    timer.start()
    start = time.time()
    try:
        _runner = run_react_v2 if USE_V2 else run_react
        result = _runner(question, options, cutoff_time,
                         reflection=reflection, thinking=thinking,
                         setup=setup, carry_context=carry_context,
                         cancel_event=cancel, timeout_sec=timeout_sec,
                         description=description)
        return result
    except Exception as e:
        return {"final_answer": "", "tool_call_count": 0, "iteration_count": 0,
                "latency_sec": round(time.time() - start, 2), "error": str(e)}
    finally:
        timer.cancel()


def _run_zero_shot_impl(question: str, options: list[str], cutoff_time: str,
                        timeout_sec: int = 90, description: str = "") -> dict:
    """Inner zero-shot logic with per-call HTTP timeout."""
    from shared.llm import get_client, get_model
    start = time.time()
    client = get_client()
    model  = get_model()
    options_str = ", ".join(options)
    user_content = f"Question: {question}\n"
    if description:
        user_content += f"Description: {description}\n"
    user_content += (
        f"Options: {options_str}\n"
        f"Cutoff Date: {cutoff_time}\n\n"
        f"You MUST select exactly one of the provided options as your final answer."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ZERO_SHOT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_tokens=8192,
        timeout=float(timeout_sec),
    )
    choice = response.choices[0]
    error  = "TOKEN_LIMIT" if choice.finish_reason == "length" else None
    return {
        "final_answer":    choice.message.content or "",
        "tool_call_count": 0,
        "iteration_count": 1,
        "latency_sec":     round(time.time() - start, 2),
        "error":           error,
    }


def run_zero_shot(question: str, options: list[str], cutoff_time: str,
                  timeout_sec: int = DEFAULT_TIMEOUT_SEC, description: str = "") -> dict:
    """Single LLM call with no tools — pure zero-shot baseline. Thread-safe."""
    try:
        return _run_zero_shot_impl(question, options, cutoff_time, timeout_sec=timeout_sec, description=description)
    except Exception as e:
        return {"final_answer": "", "tool_call_count": 0, "iteration_count": 0, "latency_sec": 0.0, "error": str(e)}


# ── Resume / append helpers ───────────────────────────────────────────────────

def load_completed(results_file: Path) -> set[str]:
    done = set()
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    done.add(str(rec["question_id"]))
                except Exception:
                    pass
    return done


_write_lock = threading.Lock()

def append_record(results_file: Path, record: dict) -> None:
    with _write_lock:
        with open(results_file, "a") as f:
            f.write(json.dumps(record) + "\n")


# ── Summary ───────────────────────────────────────────────────────────────────

def summarize(results_file: Path) -> None:
    records = []
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    if not records:
        print("No records found.")
        return

    total    = len(records)
    correct  = sum(1 for r in records if r.get("correct") is True)
    errors   = sum(1 for r in records if r.get("error") and r["error"] != "TIMEOUT")
    timeouts = sum(1 for r in records if r.get("error") == "TIMEOUT")
    answered = total - errors - timeouts
    accuracy = correct / answered if answered > 0 else 0.0
    avg_tools = sum(r.get("tool_call_count", 0) for r in records) / total
    avg_lat   = sum(r.get("latency_sec", 0) for r in records) / total

    has_conf  = [r for r in records if r.get("confidence") is not None]
    conf_info = f"  Avg confidence: {sum(r['confidence'] for r in has_conf)/len(has_conf):.3f}" if has_conf else ""

    print(f"\n{'='*55}")
    print(f"Results: {results_file.name}")
    print(f"{'='*55}")
    print(f"  Total questions : {total}")
    print(f"  Answered        : {answered}")
    print(f"  Correct         : {correct}")
    print(f"  Accuracy        : {accuracy:.1%}")
    print(f"  Errors          : {errors}")
    print(f"  Timeouts        : {timeouts}")
    print(f"  Avg tool calls  : {avg_tools:.1f}")
    print(f"  Avg latency     : {avg_lat:.1f}s")
    if conf_info:
        print(conf_info)
    print(f"{'='*55}\n")


# ── Time-series resume helper ─────────────────────────────────────────────────

def load_completed_timeseries(results_file: Path) -> set[str]:
    """Return set of 'question_id:timepoint_date' keys already in results file."""
    done = set()
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    qid = str(rec["question_id"])
                    tdate = rec.get("timepoint_date", "")
                    if tdate:
                        done.add(f"{qid}:{tdate}")
                except Exception:
                    pass
    return done


# ── Per-question worker (used by both sequential and parallel modes) ──────────

def _process_question_timepoints(
    item: dict,
    question_id: str,
    q_idx: int,
    total_questions: int,
    setup: int,
    is_zs: bool,
    needs_conclusion: bool,
    needs_critique: bool,
    timeout_sec: int,
    thinking: bool,
    results_file: Path,
    done: set,
) -> int:
    """
    Process all timepoints for a single question. Returns count of completed timepoints.
    Thread-safe: uses _write_lock for file I/O, done set is read-only per question.
    """
    n_timepoints = len(item["timepoints"])
    last_conclusion = ""
    last_critique = ""
    completed = 0

    for t_idx, tp in enumerate(item["timepoints"]):
        resume_key = f"{question_id}:{tp['date']}"
        if resume_key in done:
            last_conclusion = ""
            last_critique = ""
            completed += 1
            continue

        mode_tag = f"S{setup}"
        print(f"[Q{q_idx+1}/{total_questions} T{t_idx+1}/{n_timepoints}] [{mode_tag}] "
              f"{item['question'][:70]}...  cutoff={tp['date']}")

        # Build carry context for setups 4-7
        carry_ctx = ""
        if t_idx > 0:
            carry_ctx = format_carry_context(
                conclusion=last_conclusion if needs_conclusion else "",
                critique=last_critique if needs_critique else "",
            )

        # Run prediction (no SIGALRM in threads — timeout handled by LLM client)
        if is_zs:
            result = run_zero_shot(item["question"], item["options"], tp["date"], timeout_sec,
                                   description=item.get("description", ""))
        else:
            result = run_single(item["question"], item["options"], tp["date"],
                                timeout_sec, thinking=thinking,
                                setup=setup, carry_context=carry_ctx,
                                description=item.get("description", ""))

        # Unified extraction
        pred = extract_agent_prediction(result["final_answer"], item["options"])
        predicted      = pred["predicted"]
        agent_prob_yes = pred["agent_prob_yes"]
        agent_prob_no  = pred["agent_prob_no"]
        confidence     = pred["confidence"]
        refl_text      = extract_reflection(result["final_answer"])
        self_critique  = extract_self_critique(result["final_answer"])

        correct = (predicted == item["ground_truth"]) if predicted is not None else None

        status = "✓" if correct is True else ("✗" if correct is False else "?")
        if result["error"]:
            status = f"ERR({result['error'][:20]})"
        py_str   = f"{agent_prob_yes:.2f}" if agent_prob_yes is not None else "n/a"
        hyes_str = f"{tp['market_prob_yes']:.2f}" if tp["market_prob_yes"] is not None else "n/a"
        print(f"  → {predicted!r:6}  GT:{item['ground_truth']!r}  {status}  "
              f"agent_yes:{py_str}  human_yes:{hyes_str}  "
              f"tools:{result['tool_call_count']}  {result['latency_sec']}s")

        # Generate carry-forward for setups 4-7
        carry_conclusion = ""
        carry_critique = ""
        if (needs_conclusion or needs_critique) and agent_prob_yes is not None:
            if needs_conclusion:
                carry_conclusion = generate_carry_forward(
                    CONCLUSION_GENERATION_PROMPT,
                    question_text=item["question"],
                    n=t_idx + 1,
                    total=n_timepoints,
                    date_context=tp["date"],
                    predicted_yes=agent_prob_yes,
                    predicted_no=agent_prob_no,
                    market_yes=tp["market_prob_yes"],
                    market_no=tp["market_prob_no"],
                    thinking=thinking,
                )
                last_conclusion = carry_conclusion

            if needs_critique:
                carry_critique = generate_carry_forward(
                    SELF_CRITIQUE_GENERATION_PROMPT,
                    question_text=item["question"],
                    n=t_idx + 1,
                    total=n_timepoints,
                    date_context=tp["date"],
                    predicted_yes=agent_prob_yes,
                    predicted_no=agent_prob_no,
                    thinking=thinking,
                )
                last_critique = carry_critique

        append_record(results_file, {
            "question_id":       question_id,
            "setup":             setup,
            "timepoint_index":   t_idx + 1,
            "timepoint_date":    tp["date"],
            "total_timepoints":  n_timepoints,
            "question":          item["question"],
            "description":       item.get("description", ""),
            "options":           item["options"],
            "ground_truth":      item["ground_truth"],
            "close_date":        item["close_date"],
            "topic":             item.get("topic", ""),
            "market_prob_yes":    tp["market_prob_yes"],
            "market_prob_no":     tp["market_prob_no"],
            "predicted":         predicted,
            "correct":           correct,
            "confidence":        confidence,
            "agent_prob_yes":    agent_prob_yes,
            "agent_prob_no":     agent_prob_no,
            "reflection":        refl_text,
            "self_critique":     self_critique,
            "carry_conclusion":  carry_conclusion if carry_conclusion else None,
            "carry_critique":    carry_critique if carry_critique else None,
            "tool_call_count":   result["tool_call_count"],
            "iteration_count":   result.get("iteration_count"),
            "latency_sec":       result["latency_sec"],
            "error":             result["error"],
            "timestamp":         datetime.now(timezone.utc).isoformat(),
            "final_answer":      result["final_answer"],
        })
        completed += 1

    return completed


# ── Time-series benchmark ─────────────────────────────────────────────────────

def benchmark_timeseries(
    max_questions: Optional[int],
    results_file: Path,
    timeout_sec: int,
    max_timepoints: Optional[int],
    dataset_path: Path,
    reflection: bool = False,
    zero_shot: bool = False,
    thinking: bool = True,
    setup: int | None = None,
    parallel: int = 1,
) -> None:
    """
    Run the agent (or zero-shot LLM) at every historical price timepoint per question.

    Args:
        parallel: Number of questions to process concurrently (default 1 = sequential).
                  Each question's timepoints are always sequential (required for carry-forward).
                  Different questions are independent and can run in parallel.
    """
    # Resolve setup from legacy flags if not explicitly provided
    if setup is None:
        if zero_shot:
            setup = 1
        elif reflection:
            setup = 3
        else:
            setup = 2

    is_zs = setup in SETUP_IS_ZERO_SHOT
    needs_conclusion = setup in SETUP_USES_CONCLUSION
    needs_critique = setup in SETUP_USES_CRITIQUE

    _SETUP_NAMES = {
        1: "Zero-Shot", 2: "ReAct", 3: "ReAct+Reflection",
        4: "ReAct+Conclusion", 5: "ReAct+Critique",
        6: "ReAct+Concl+Crit", 7: "ReAct+Full",
    }

    dataset = load_polymarket_timeseries(dataset_path, max_questions, max_timepoints)
    done    = load_completed_timeseries(results_file)

    total_timepoints = sum(len(q["timepoints"]) for q in dataset)
    done_count       = len(done)

    print(f"\n{'='*55}")
    print(f"Setup      : {setup} — {_SETUP_NAMES.get(setup, '?')}")
    print(f"Parallel   : {parallel} question(s) at a time")
    print(f"Dataset    : {len(dataset)} questions")
    print(f"Timepoints : {total_timepoints} total  (max_timepoints={max_timepoints or 'all'})")
    print(f"Done       : {done_count}  |  Remaining: {total_timepoints - done_count}")
    print(f"Output     : {results_file}")
    print(f"Timeout    : {timeout_sec}s/timepoint")
    print(f"{'='*55}\n")

    worker_kwargs = dict(
        setup=setup, is_zs=is_zs,
        needs_conclusion=needs_conclusion, needs_critique=needs_critique,
        timeout_sec=timeout_sec, thinking=thinking,
        results_file=results_file, done=done,
    )

    if parallel <= 1:
        # Sequential mode (original behavior)
        for i, item in enumerate(dataset):
            _process_question_timepoints(
                item=item, question_id=str(i + 1),
                q_idx=i, total_questions=len(dataset),
                **worker_kwargs,
            )
    else:
        # Parallel mode: process N questions concurrently
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {}
            for i, item in enumerate(dataset):
                f = pool.submit(
                    _process_question_timepoints,
                    item=item, question_id=str(i + 1),
                    q_idx=i, total_questions=len(dataset),
                    **worker_kwargs,
                )
                futures[f] = i
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as exc:
                    print(f"  [ERROR] Question {futures[f]+1} failed: {exc}")

    print(f"\nDone. Results saved to: {results_file}")
    summarize(results_file)


# ── Per-question worker (question-by-question mode) ──────────────────────────

def _process_single_question(
    item: dict,
    question_id: str,
    q_idx: int,
    total_questions: int,
    setup: int,
    is_zs: bool,
    timeout_sec: int,
    thinking: bool,
    results_file: Path,
    done: set,
) -> None:
    """Process a single question (no timepoints). Thread-safe."""
    if question_id in done:
        return

    _SETUP_NAMES = {
        1: "ZS", 2: "ReAct", 3: "ReAct+Refl",
        4: "ReAct+Concl", 5: "ReAct+Crit",
        6: "ReAct+C+C", 7: "ReAct+Full",
    }
    mode_tag = f"S{setup}({_SETUP_NAMES.get(setup, '?')})"
    print(f"[Q{q_idx+1}/{total_questions}] [{mode_tag}] {item['question'][:75]}...")

    if is_zs:
        result = run_zero_shot(item["question"], item["options"],
                               item["cutoff_date"], timeout_sec,
                               description=item.get("description", ""))
    else:
        result = run_single(item["question"], item["options"],
                            item["cutoff_date"], timeout_sec,
                            thinking=thinking, setup=setup,
                            description=item.get("description", ""))

    # Unified extraction: XML <prediction> → P(Yes): → legacy Confidence:
    pred = extract_agent_prediction(result["final_answer"], item["options"])
    predicted      = pred["predicted"]
    agent_prob_yes = pred["agent_prob_yes"]
    agent_prob_no  = pred["agent_prob_no"]
    confidence     = pred["confidence"]
    refl_text      = extract_reflection(result["final_answer"])
    self_critique  = extract_self_critique(result["final_answer"])

    correct = (predicted == item["ground_truth"]) if predicted is not None else None

    status = "✓" if correct is True else ("✗" if correct is False else "?")
    if result["error"]:
        status = f"ERR({result['error'][:25]})"
    py_str = f"{agent_prob_yes:.2f}" if agent_prob_yes is not None else "n/a"
    print(f"  → {predicted!r:6}  GT:{item['ground_truth']!r}  {status}  "
          f"P(Yes):{py_str}  tools:{result['tool_call_count']}  {result['latency_sec']}s")

    append_record(results_file, {
        "question_id":     question_id,
        "setup":           setup,
        "question":        item["question"],
        "description":     item.get("description", ""),
        "options":         item["options"],
        "ground_truth":    item["ground_truth"],
        "cutoff_date":     item["cutoff_date"],
        "topic":           item.get("topic", ""),
        "predicted":       predicted,
        "correct":         correct,
        "confidence":      confidence,
        "agent_prob_yes":  agent_prob_yes,
        "agent_prob_no":   agent_prob_no,
        "market_prob_yes": item.get("market_prob_yes"),
        "market_prob_no":  item.get("market_prob_no"),
        "reflection":      refl_text,
        "self_critique":   self_critique,
        "tool_call_count": result["tool_call_count"],
        "iteration_count": result.get("iteration_count"),
        "latency_sec":     result["latency_sec"],
        "error":           result["error"],
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "final_answer":    result["final_answer"],
    })


# ── Main benchmark loop ───────────────────────────────────────────────────────

def benchmark(max_questions: Optional[int], results_file: Path, timeout_sec: int,
              reflection: bool = False, zero_shot: bool = False, thinking: bool = True,
              setup: int | None = None, parallel: int = 1) -> None:
    """
    Run the agent (or zero-shot LLM) once per question (no timepoints).

    Args:
        parallel: Number of questions to process concurrently (default 1 = sequential).
                  vLLM continuous batching handles concurrent requests efficiently.
    """
    # Resolve setup from legacy flags if not explicitly provided
    if setup is None:
        if zero_shot:
            setup = 1
        elif reflection:
            setup = 3
        else:
            setup = 2

    is_zs = setup in SETUP_IS_ZERO_SHOT

    _SETUP_NAMES = {
        1: "Zero-Shot", 2: "ReAct", 3: "ReAct+Reflection",
        4: "ReAct+Conclusion", 5: "ReAct+Critique",
        6: "ReAct+Concl+Crit", 7: "ReAct+Full",
    }

    dataset = load_polymarket(DATASET_PATH, max_questions)
    done    = load_completed(results_file)

    print(f"\n{'='*55}")
    print(f"Setup   : {setup} — {_SETUP_NAMES.get(setup, '?')}")
    print(f"Parallel: {parallel} question(s) at a time")
    print(f"Dataset : {len(dataset)} questions")
    print(f"Done    : {len(done)}  |  Remaining: {len(dataset) - len(done)}")
    print(f"Output  : {results_file}")
    print(f"Timeout : {timeout_sec}s/question")
    print(f"{'='*55}\n")

    worker_kwargs = dict(
        setup=setup, is_zs=is_zs,
        timeout_sec=timeout_sec, thinking=thinking,
        results_file=results_file, done=done,
    )

    if parallel <= 1:
        # Sequential mode (original behavior)
        for i, item in enumerate(dataset):
            _process_single_question(
                item=item, question_id=str(i + 1),
                q_idx=i, total_questions=len(dataset),
                **worker_kwargs,
            )
    else:
        # Parallel mode: process N questions concurrently
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {}
            for i, item in enumerate(dataset):
                f = pool.submit(
                    _process_single_question,
                    item=item, question_id=str(i + 1),
                    q_idx=i, total_questions=len(dataset),
                    **worker_kwargs,
                )
                futures[f] = i
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as exc:
                    print(f"  [ERROR] Question {futures[f]+1} failed: {exc}")

    print(f"\nDone. Results saved to: {results_file}")
    summarize(results_file)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate forecasting agent on Polymarket dataset")
    parser.add_argument("--max_questions",  type=int, default=None,
                        help="Limit number of questions (default: all)")
    parser.add_argument("--results_file",   type=str, default=None,
                        help="Existing JSONL to resume from")
    parser.add_argument("--timeout",        type=int, default=DEFAULT_TIMEOUT_SEC,
                        help=f"Per-timepoint timeout in seconds (default {DEFAULT_TIMEOUT_SEC})")
    parser.add_argument("--summarize",      type=str, default=None, metavar="FILE",
                        help="Print summary from an existing results file and exit")
    parser.add_argument("--timeseries",     action="store_true",
                        help="Run at every historical price timepoint per question")
    parser.add_argument("--max_timepoints", type=int, default=None,
                        help="Max evenly-spaced timepoints per question (default: all)")
    parser.add_argument("--dataset",        type=str, default=None,
                        help="Path to dataset JSONL (default: polymarket_binary_yesno.jsonl)")
    parser.add_argument("--reflection",     action="store_true",
                        help="Enable structured reflection in ReAct prompt (ablation condition)")
    parser.add_argument("--zero_shot",      action="store_true",
                        help="Zero-shot baseline: single LLM call, no tools or retrieval")
    parser.add_argument("--setup",          type=int, default=None, choices=range(1, 8),
                        help="Setup number 1-7. Overrides --zero_shot and --reflection. "
                             "1=ZS, 2=ReAct, 3=ReAct+Reflection, 4=ReAct+Conclusion, "
                             "5=ReAct+Critique, 6=ReAct+Concl+Crit, 7=ReAct+Full")
    parser.add_argument("--parallel",       type=int, default=1,
                        help="Number of questions to process concurrently (default 1). "
                             "Each question's timepoints remain sequential for carry-forward. "
                             "Use with vLLM continuous batching for best throughput.")
    parser.add_argument("--no_thinking",    action="store_true",
                        help="Disable Qwen3 extended thinking (<think> blocks) for ~4-5x faster "
                             "inference. Reduces per-call latency from ~80s to ~15s at some "
                             "quality cost. Has no effect on --zero_shot mode.")
    parser.add_argument("--v2",             action="store_true",
                        help="Use structured-round driver (run_react_v2) — NOTHINK V2.1 is "
                             "the recommended config. Pairs well with --no_thinking.")
    args = parser.parse_args()
    if args.v2:
        globals()["USE_V2"] = True

    dataset_path = Path(args.dataset) if args.dataset else DATASET_PATH

    # Resolve setup number
    setup = args.setup
    if setup is None:
        if args.zero_shot:
            setup = 1
        elif args.reflection:
            setup = 3
        # else: benchmark_timeseries defaults to 2

    # Build result file prefix from active modes
    if not args.results_file:
        parts = ["results"]
        if setup is not None:
            parts.append(f"s{setup}")
        else:
            if args.zero_shot:
                parts.append("zs")
            if args.reflection:
                parts.append("r")
        if args.timeseries:
            parts.append("ts")
        prefix = "_".join(parts)
        auto_file = RESULTS_DIR / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    else:
        auto_file = None

    thinking = not args.no_thinking

    if args.summarize:
        summarize(Path(args.summarize))
    elif args.timeseries:
        results_file = Path(args.results_file) if args.results_file else auto_file
        benchmark_timeseries(args.max_questions, results_file, args.timeout,
                             args.max_timepoints, dataset_path,
                             reflection=args.reflection, zero_shot=args.zero_shot,
                             thinking=thinking, setup=setup,
                             parallel=args.parallel)
    else:
        results_file = Path(args.results_file) if args.results_file else auto_file
        benchmark(args.max_questions, results_file, args.timeout,
                  reflection=args.reflection, zero_shot=args.zero_shot,
                  thinking=thinking, setup=setup, parallel=args.parallel)
