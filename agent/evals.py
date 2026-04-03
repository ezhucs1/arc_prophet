"""
Benchmark evaluation for the ReAct agent on the Polymarket dataset.

Usage:
    python agent/evals.py --max_questions 10
    python agent/evals.py --max_questions 100
    python agent/evals.py --results_file agent/results/results_XXXXXX.jsonl   # resume
    python agent/evals.py --summarize agent/results/results_XXXXXX.jsonl
"""

import sys
import argparse
import json
import re
import signal
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path when this file is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from database.loader import load_polymarket, load_polymarket_timeseries
from agent.react_agent import run_react

DATASET_PATH = Path(__file__).parent.parent / "database" / "polymarket_binary_yesno.jsonl"
RESULTS_DIR  = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_TIMEOUT_SEC = 300


# ── Answer / confidence extraction ───────────────────────────────────────────

def extract_selected_option(text: str, options: list[str]) -> Optional[str]:
    """
    Priority:
      1. Explicit 'Selected Option: X' label
      2. First option string found verbatim in clean text
    Strips <think> blocks before matching.
    """
    if not text:
        return None
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip() or text
    m = re.search(r"Selected Option[:\s]+(.+?)(?:\n|$)", clean, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip().rstrip(".,;")
        for opt in options:
            if opt.lower() == candidate.lower():
                return opt
    for opt in options:
        if re.search(re.escape(opt), clean, re.IGNORECASE):
            return opt
    return None


def extract_reflection(text: str) -> Optional[str]:
    """Extract the last Reflection block from the agent output, if present."""
    if not text:
        return None
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Grab the last occurrence — most recent reflection is most informative
    matches = list(re.finditer(r"Reflection:(.*?)(?=\n(?:Reflection:|Reasoning:|Final Answer:|Selected Option:|$))",
                               clean, re.DOTALL | re.IGNORECASE))
    if matches:
        return matches[-1].group(1).strip()
    return None


def extract_self_critique(text: str) -> Optional[str]:
    """Extract Self-critique line from the final answer block."""
    if not text:
        return None
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"Self-critique[:\s]+(.+?)(?=\nFinal Answer:|\nSelected Option:|$)",
                  clean, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def extract_confidence(text: str) -> Optional[float]:
    """Parse 'Confidence: 0.75' → float in [0.0, 1.0] or None."""
    if not text:
        return None
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"Confidence[:\s]+([\d.]+)", clean, re.IGNORECASE)
    if m:
        try:
            return round(max(0.0, min(1.0, float(m.group(1)))), 4)
        except ValueError:
            pass
    return None


# ── Timeout helper ────────────────────────────────────────────────────────────

class _Timeout(Exception):
    pass

def _alarm_handler(signum, frame):
    raise _Timeout()


# ── Per-question runner ───────────────────────────────────────────────────────

def run_single(question: str, options: list[str], cutoff_time: str,
               timeout_sec: int = DEFAULT_TIMEOUT_SEC) -> dict:
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_sec)
    try:
        return run_react(question, options, cutoff_time)
    except _Timeout:
        return {"final_answer": "", "tool_call_count": 0,
                "latency_sec": float(timeout_sec), "error": "TIMEOUT"}
    except Exception as e:
        return {"final_answer": "", "tool_call_count": 0, "latency_sec": 0.0, "error": str(e)}
    finally:
        signal.alarm(0)


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


def append_record(results_file: Path, record: dict) -> None:
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


# ── Time-series benchmark ─────────────────────────────────────────────────────

def benchmark_timeseries(
    max_questions: Optional[int],
    results_file: Path,
    timeout_sec: int,
    max_timepoints: Optional[int],
    dataset_path: Path,
) -> None:
    """
    Run the agent at every historical price timepoint per question.

    For each (question, timepoint):
      - Agent receives timepoint_date as cutoff — NOT the close/resolution date.
      - Output record includes agent probs AND human (market) probs at that timepoint.

    This produces N records per question (one per timepoint), forming a time series
    that can be compared against the market's evolving probability.
    """
    dataset = load_polymarket_timeseries(dataset_path, max_questions, max_timepoints)
    done    = load_completed_timeseries(results_file)

    total_timepoints = sum(len(q["timepoints"]) for q in dataset)
    done_count       = len(done)

    print(f"\n{'='*55}")
    print(f"Dataset    : {len(dataset)} questions")
    print(f"Timepoints : {total_timepoints} total  (max_timepoints={max_timepoints or 'all'})")
    print(f"Done       : {done_count}  |  Remaining: {total_timepoints - done_count}")
    print(f"Output     : {results_file}")
    print(f"Timeout    : {timeout_sec}s/timepoint")
    print(f"{'='*55}\n")

    completed_timepoints = 0
    for i, item in enumerate(dataset):
        question_id  = str(i + 1)
        n_timepoints = len(item["timepoints"])

        for t_idx, tp in enumerate(item["timepoints"]):
            resume_key = f"{question_id}:{tp['date']}"
            if resume_key in done:
                completed_timepoints += 1
                continue

            print(f"[Q{i+1}/{len(dataset)} T{t_idx+1}/{n_timepoints}] "
                  f"{item['question'][:70]}...  cutoff={tp['date']}")

            # Agent receives the timepoint date — close_date is intentionally withheld
            result = run_single(item["question"], item["options"], tp["date"], timeout_sec)

            predicted     = extract_selected_option(result["final_answer"], item["options"])
            confidence    = extract_confidence(result["final_answer"])
            reflection    = extract_reflection(result["final_answer"])
            self_critique = extract_self_critique(result["final_answer"])
            correct       = (predicted == item["ground_truth"]) if predicted is not None else None

            if confidence is not None and predicted == "Yes":
                agent_prob_yes = confidence
                agent_prob_no  = round(1.0 - confidence, 4)
            elif confidence is not None and predicted == "No":
                agent_prob_no  = confidence
                agent_prob_yes = round(1.0 - confidence, 4)
            else:
                agent_prob_yes = agent_prob_no = None

            status   = "✓" if correct is True else ("✗" if correct is False else "?")
            if result["error"]:
                status = f"ERR({result['error'][:20]})"
            conf_str = f"{confidence:.2f}" if confidence is not None else "n/a"
            hyes_str = f"{tp['human_prob_yes']:.2f}" if tp["human_prob_yes"] is not None else "n/a"
            print(f"  → {predicted!r:6}  GT:{item['ground_truth']!r}  {status}  "
                  f"agent_yes:{conf_str if predicted=='Yes' else (f'{agent_prob_yes:.2f}' if agent_prob_yes is not None else 'n/a')}  "
                  f"human_yes:{hyes_str}  "
                  f"tools:{result['tool_call_count']}  {result['latency_sec']}s")

            append_record(results_file, {
                "question_id":       question_id,
                "timepoint_index":   t_idx + 1,
                "timepoint_date":    tp["date"],
                "total_timepoints":  n_timepoints,
                "question":          item["question"],
                "options":           item["options"],
                "ground_truth":      item["ground_truth"],
                "close_date":        item["close_date"],
                "topic":             item.get("topic", ""),
                "human_prob_yes":    tp["human_prob_yes"],
                "human_prob_no":     tp["human_prob_no"],
                "predicted":         predicted,
                "correct":           correct,
                "confidence":        confidence,
                "agent_prob_yes":    agent_prob_yes,
                "agent_prob_no":     agent_prob_no,
                "reflection":        reflection,
                "self_critique":     self_critique,
                "tool_call_count":   result["tool_call_count"],
                "latency_sec":       result["latency_sec"],
                "error":             result["error"],
                "timestamp":         datetime.utcnow().isoformat(),
                "final_answer":      result["final_answer"],
            })
            done.add(resume_key)
            completed_timepoints += 1

    print(f"\nDone. Results saved to: {results_file}")
    summarize(results_file)


# ── Main benchmark loop ───────────────────────────────────────────────────────

def benchmark(max_questions: Optional[int], results_file: Path, timeout_sec: int) -> None:
    dataset = load_polymarket(DATASET_PATH, max_questions)
    done    = load_completed(results_file)

    print(f"\n{'='*55}")
    print(f"Dataset : {len(dataset)} questions")
    print(f"Done    : {len(done)}  |  Remaining: {len(dataset) - len(done)}")
    print(f"Output  : {results_file}")
    print(f"Timeout : {timeout_sec}s/question")
    print(f"{'='*55}\n")

    for i, item in enumerate(dataset):
        question_id = str(i + 1)
        if question_id in done:
            continue

        print(f"[Q{i+1}/{len(dataset)}] {item['question'][:80]}...")

        result = run_single(item["question"], item["options"],
                            item["cutoff_date"], timeout_sec)

        predicted    = extract_selected_option(result["final_answer"], item["options"])
        confidence   = extract_confidence(result["final_answer"])
        reflection   = extract_reflection(result["final_answer"])
        self_critique= extract_self_critique(result["final_answer"])
        correct      = (predicted == item["ground_truth"]) if predicted is not None else None

        if confidence is not None and predicted == "Yes":
            agent_prob_yes, agent_prob_no = confidence, round(1.0 - confidence, 4)
        elif confidence is not None and predicted == "No":
            agent_prob_no, agent_prob_yes = confidence, round(1.0 - confidence, 4)
        else:
            agent_prob_yes = agent_prob_no = None

        status   = "✓" if correct is True else ("✗" if correct is False else "?")
        if result["error"]:
            status = f"ERR({result['error'][:25]})"
        conf_str = f"{confidence:.2f}" if confidence is not None else "n/a"
        print(f"  → {predicted!r:6}  GT:{item['ground_truth']!r}  {status}  "
              f"conf:{conf_str}  tools:{result['tool_call_count']}  {result['latency_sec']}s")

        append_record(results_file, {
            "question_id":     question_id,
            "question":        item["question"],
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
            "reflection":      reflection,
            "self_critique":   self_critique,
            "tool_call_count": result["tool_call_count"],
            "latency_sec":     result["latency_sec"],
            "error":           result["error"],
            "timestamp":       datetime.utcnow().isoformat(),
            "final_answer":    result["final_answer"],
        })
        done.add(question_id)

    print(f"\nDone. Results saved to: {results_file}")
    summarize(results_file)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ReAct agent on Polymarket dataset")
    parser.add_argument("--max_questions",  type=int, default=None,
                        help="Limit number of questions (default: all)")
    parser.add_argument("--results_file",   type=str, default=None,
                        help="Existing JSONL to resume from")
    parser.add_argument("--timeout",        type=int, default=DEFAULT_TIMEOUT_SEC,
                        help=f"Per-timepoint timeout in seconds (default {DEFAULT_TIMEOUT_SEC})")
    parser.add_argument("--summarize",      type=str, default=None, metavar="FILE",
                        help="Print summary from an existing results file and exit")
    parser.add_argument("--timeseries",     action="store_true",
                        help="Run agent at every historical price timepoint per question")
    parser.add_argument("--max_timepoints", type=int, default=None,
                        help="Max evenly-spaced timepoints per question (default: all). "
                             "Recommended: omit on first run to collect full data.")
    parser.add_argument("--dataset",        type=str, default=None,
                        help="Path to dataset JSONL (default: polymarket_binary_yesno.jsonl)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset) if args.dataset else DATASET_PATH

    if args.summarize:
        summarize(Path(args.summarize))
    elif args.timeseries:
        results_file = (
            Path(args.results_file) if args.results_file
            else RESULTS_DIR / f"results_ts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        benchmark_timeseries(args.max_questions, results_file, args.timeout,
                             args.max_timepoints, dataset_path)
    else:
        results_file = (
            Path(args.results_file) if args.results_file
            else RESULTS_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        benchmark(args.max_questions, results_file, args.timeout)
