"""
Benchmark evaluation for the V2.1 ReAct agent on the Polymarket dataset.

Usage:
    python agent/evals.py --max_questions 10
    python agent/evals.py --max_questions 100 --parallel 4
    python agent/evals.py --results_file results/results_XXXXXX.jsonl   # resume
    python agent/evals.py --summarize results/results_XXXXXX.jsonl
"""

import sys
import argparse
import json
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path when this file is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from database.loader import load_polymarket, load_polymarket_timeseries
from agent.react_agent import run_react_v2

DATASET_PATH = Path(__file__).parent.parent / "database" / "polymarket_binary_yesno.jsonl"
RESULTS_DIR  = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_TIMEOUT_SEC = 120


# ── Answer / confidence extraction ───────────────────────────────────────────

def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def extract_prediction_xml(text: str) -> Optional[dict]:
    """Parse the last <prediction> XML block."""
    if not text:
        return None
    clean = _strip_think(text)
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
    """Extract selected option from text."""
    if not text:
        return None
    clean = _strip_think(text)
    if not clean:
        return None
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


def extract_prob_yes(text: str) -> Optional[float]:
    """Parse 'P(Yes): 0.75' from text."""
    if not text:
        return None
    clean = _strip_think(text)
    matches = re.findall(r"P\s*\(\s*[Yy]es\s*\)\s*[:\s]+([\d.]+)", clean)
    if matches:
        try:
            return round(max(0.0, min(1.0, float(matches[-1]))), 4)
        except ValueError:
            pass
    return None


def extract_confidence_legacy(text: str) -> Optional[float]:
    """Parse old 'Confidence: 0.75' label for backward compatibility."""
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
    """Unified extraction: XML <prediction> -> P(Yes): -> legacy Confidence:."""
    result = {"predicted": None, "agent_prob_yes": None, "agent_prob_no": None,
              "confidence": None, "reasoning": None}

    xml = extract_prediction_xml(text)
    if xml and xml["yes"] is not None:
        result["agent_prob_yes"] = xml["yes"]
        result["agent_prob_no"] = xml["no"]
        result["predicted"] = "Yes" if xml["yes"] > 0.5 else "No"
        result["confidence"] = xml["yes"] if result["predicted"] == "Yes" else xml["no"]
        result["reasoning"] = xml.get("reasoning")
        return result

    prob_yes = extract_prob_yes(text)
    if prob_yes is not None:
        result["agent_prob_yes"] = prob_yes
        result["agent_prob_no"] = round(1.0 - prob_yes, 4)
        result["predicted"] = "Yes" if prob_yes > 0.5 else "No"
        result["confidence"] = prob_yes if result["predicted"] == "Yes" else result["agent_prob_no"]
        return result

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


# ── Per-question runner ──────────────────────────────────────────────────────

def run_single(question: str, options: list[str], cutoff_time: str,
               timeout_sec: int = DEFAULT_TIMEOUT_SEC,
               thinking: bool = True,
               description: str = "") -> dict:
    """Run one V2.1 ReAct question with timeout."""
    cancel = threading.Event()
    timer = threading.Timer(timeout_sec, cancel.set)
    timer.daemon = True
    timer.start()
    start = time.time()
    try:
        result = run_react_v2(question, options, cutoff_time,
                              thinking=thinking,
                              cancel_event=cancel, timeout_sec=timeout_sec,
                              description=description)
        return result
    except Exception as e:
        return {"final_answer": "", "tool_call_count": 0, "iteration_count": 0,
                "latency_sec": round(time.time() - start, 2), "error": str(e)}
    finally:
        timer.cancel()


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


# ── Per-question worker (time-series mode) ──────────────────────────────────

def _process_question_timepoints(
    item: dict,
    question_id: str,
    q_idx: int,
    total_questions: int,
    timeout_sec: int,
    thinking: bool,
    results_file: Path,
    done: set,
) -> int:
    """Process all timepoints for a single question. Thread-safe."""
    n_timepoints = len(item["timepoints"])
    completed = 0

    for t_idx, tp in enumerate(item["timepoints"]):
        resume_key = f"{question_id}:{tp['date']}"
        if resume_key in done:
            completed += 1
            continue

        print(f"[Q{q_idx+1}/{total_questions} T{t_idx+1}/{n_timepoints}] "
              f"{item['question'][:70]}...  cutoff={tp['date']}")

        result = run_single(item["question"], item["options"], tp["date"],
                            timeout_sec, thinking=thinking,
                            description=item.get("description", ""))

        pred = extract_agent_prediction(result["final_answer"], item["options"])
        predicted      = pred["predicted"]
        agent_prob_yes = pred["agent_prob_yes"]
        agent_prob_no  = pred["agent_prob_no"]
        confidence     = pred["confidence"]

        correct = (predicted == item["ground_truth"]) if predicted is not None else None

        status = "✓" if correct is True else ("✗" if correct is False else "?")
        if result["error"]:
            status = f"ERR({result['error'][:20]})"
        py_str   = f"{agent_prob_yes:.2f}" if agent_prob_yes is not None else "n/a"
        hyes_str = f"{tp['market_prob_yes']:.2f}" if tp["market_prob_yes"] is not None else "n/a"
        print(f"  → {predicted!r:6}  GT:{item['ground_truth']!r}  {status}  "
              f"agent_yes:{py_str}  human_yes:{hyes_str}  "
              f"tools:{result['tool_call_count']}  {result['latency_sec']}s")

        append_record(results_file, {
            "question_id":       question_id,
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
    thinking: bool = True,
    parallel: int = 1,
) -> None:
    """Run the V2.1 agent at every historical price timepoint per question."""
    dataset = load_polymarket_timeseries(dataset_path, max_questions, max_timepoints)
    done    = load_completed_timeseries(results_file)

    total_timepoints = sum(len(q["timepoints"]) for q in dataset)
    done_count       = len(done)

    print(f"\n{'='*55}")
    print(f"Mode       : V2.1 ReAct (time-series)")
    print(f"Parallel   : {parallel} question(s) at a time")
    print(f"Dataset    : {len(dataset)} questions")
    print(f"Timepoints : {total_timepoints} total  (max_timepoints={max_timepoints or 'all'})")
    print(f"Done       : {done_count}  |  Remaining: {total_timepoints - done_count}")
    print(f"Output     : {results_file}")
    print(f"Timeout    : {timeout_sec}s/timepoint")
    print(f"{'='*55}\n")

    worker_kwargs = dict(
        timeout_sec=timeout_sec, thinking=thinking,
        results_file=results_file, done=done,
    )

    if parallel <= 1:
        for i, item in enumerate(dataset):
            _process_question_timepoints(
                item=item, question_id=str(i + 1),
                q_idx=i, total_questions=len(dataset),
                **worker_kwargs,
            )
    else:
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
    timeout_sec: int,
    thinking: bool,
    results_file: Path,
    done: set,
) -> None:
    """Process a single question (no timepoints). Thread-safe."""
    if question_id in done:
        return

    print(f"[Q{q_idx+1}/{total_questions}] {item['question'][:75]}...")

    result = run_single(item["question"], item["options"],
                        item["cutoff_date"], timeout_sec,
                        thinking=thinking,
                        description=item.get("description", ""))

    pred = extract_agent_prediction(result["final_answer"], item["options"])
    predicted      = pred["predicted"]
    agent_prob_yes = pred["agent_prob_yes"]
    agent_prob_no  = pred["agent_prob_no"]
    confidence     = pred["confidence"]

    correct = (predicted == item["ground_truth"]) if predicted is not None else None

    status = "✓" if correct is True else ("✗" if correct is False else "?")
    if result["error"]:
        status = f"ERR({result['error'][:25]})"
    py_str = f"{agent_prob_yes:.2f}" if agent_prob_yes is not None else "n/a"
    print(f"  → {predicted!r:6}  GT:{item['ground_truth']!r}  {status}  "
          f"P(Yes):{py_str}  tools:{result['tool_call_count']}  {result['latency_sec']}s")

    append_record(results_file, {
        "question_id":     question_id,
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
        "tool_call_count": result["tool_call_count"],
        "iteration_count": result.get("iteration_count"),
        "latency_sec":     result["latency_sec"],
        "error":           result["error"],
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "final_answer":    result["final_answer"],
    })


# ── Main benchmark loop ───────────────────────────────────────────────────────

def benchmark(max_questions: Optional[int], results_file: Path, timeout_sec: int,
              thinking: bool = True, parallel: int = 1) -> None:
    """Run the V2.1 agent once per question (no timepoints)."""
    dataset = load_polymarket(DATASET_PATH, max_questions)
    done    = load_completed(results_file)

    print(f"\n{'='*55}")
    print(f"Mode    : V2.1 ReAct")
    print(f"Parallel: {parallel} question(s) at a time")
    print(f"Dataset : {len(dataset)} questions")
    print(f"Done    : {len(done)}  |  Remaining: {len(dataset) - len(done)}")
    print(f"Output  : {results_file}")
    print(f"Timeout : {timeout_sec}s/question")
    print(f"{'='*55}\n")

    worker_kwargs = dict(
        timeout_sec=timeout_sec, thinking=thinking,
        results_file=results_file, done=done,
    )

    if parallel <= 1:
        for i, item in enumerate(dataset):
            _process_single_question(
                item=item, question_id=str(i + 1),
                q_idx=i, total_questions=len(dataset),
                **worker_kwargs,
            )
    else:
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
    parser = argparse.ArgumentParser(description="Evaluate V2.1 ReAct agent on Polymarket dataset")
    parser.add_argument("--max_questions",  type=int, default=None,
                        help="Limit number of questions (default: all)")
    parser.add_argument("--results_file",   type=str, default=None,
                        help="Existing JSONL to resume from")
    parser.add_argument("--timeout",        type=int, default=DEFAULT_TIMEOUT_SEC,
                        help=f"Per-question timeout in seconds (default {DEFAULT_TIMEOUT_SEC})")
    parser.add_argument("--summarize",      type=str, default=None, metavar="FILE",
                        help="Print summary from an existing results file and exit")
    parser.add_argument("--timeseries",     action="store_true",
                        help="Run at every historical price timepoint per question")
    parser.add_argument("--max_timepoints", type=int, default=None,
                        help="Max evenly-spaced timepoints per question (default: all)")
    parser.add_argument("--dataset",        type=str, default=None,
                        help="Path to dataset JSONL (default: polymarket_binary_yesno.jsonl)")
    parser.add_argument("--parallel",       type=int, default=1,
                        help="Number of questions to process concurrently (default 1).")
    parser.add_argument("--no_thinking",    action="store_true",
                        help="Disable extended thinking (<think> blocks).")
    args = parser.parse_args()

    dataset_path = Path(args.dataset) if args.dataset else DATASET_PATH
    thinking = not args.no_thinking

    if not args.results_file:
        parts = ["results", "v21"]
        if args.timeseries:
            parts.append("ts")
        prefix = "_".join(parts)
        auto_file = RESULTS_DIR / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    else:
        auto_file = None

    if args.summarize:
        summarize(Path(args.summarize))
    elif args.timeseries:
        results_file = Path(args.results_file) if args.results_file else auto_file
        benchmark_timeseries(args.max_questions, results_file, args.timeout,
                             args.max_timepoints, dataset_path,
                             thinking=thinking, parallel=args.parallel)
    else:
        results_file = Path(args.results_file) if args.results_file else auto_file
        benchmark(args.max_questions, results_file, args.timeout,
                  thinking=thinking, parallel=args.parallel)
