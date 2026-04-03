"""
Interactive single-query entry point.

Usage:
    python run.py --query "Will Bitcoin hit 100k by Feb 2026?" --cutoff_time 2026-02-01
    python run.py --query "Will inflation drop below 3%?" --cutoff_time 2026-01-01 --options Yes No
"""

import argparse
import re
from datetime import date
from dotenv import load_dotenv

load_dotenv()

from agent.react_agent import run_react


def run_agent(query: str, cutoff_time: str = None, options: list[str] = None):
    if not cutoff_time:
        cutoff_time = date.today().isoformat()
    if not options:
        options = ["Yes", "No"]

    print(f"\n[Query]   : {query}")
    print(f"[Options] : {', '.join(options)}")
    print(f"[Cutoff]  : {cutoff_time}\n" + "-" * 55)

    result = run_react(query, options, cutoff_time)
    final  = result["final_answer"] or ""

    think_match = re.search(r"<think>(.*?)</think>", final, re.DOTALL)
    visible     = re.sub(r"<think>.*?</think>", "", final, flags=re.DOTALL).strip()

    if think_match:
        think_text = think_match.group(1).strip()
        if think_text:
            print(f"[Reasoning]:\n{think_text}\n" + "-" * 55)

    print(f"\n[Answer]:\n{visible}")
    print(f"\n[Stats] tools={result['tool_call_count']}  "
          f"latency={result['latency_sec']}s  "
          f"error={result['error']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query",       required=True, help="The question to answer")
    parser.add_argument("--cutoff_time", default=None,  help="Data cutoff YYYY-MM-DD (default: today)")
    parser.add_argument("--options",     nargs="+",     default=["Yes", "No"])
    args = parser.parse_args()

    run_agent(args.query, args.cutoff_time, args.options)
