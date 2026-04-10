"""Test the new ReAct prompt on a few questions.

Tests hypothesis decomposition, evidence chaining, fact vs opinion,
and probability tracking. Run on lab server with vLLM + Text2SQL up.

Usage: python3 test_react_v2.py [--port 8000] [--thinking]
"""
import sys, os, time, json, argparse
sys.path.insert(0, ".")

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8000, help="vLLM port")
parser.add_argument("--thinking", action="store_true", help="Enable thinking (default: off for speed)")
args = parser.parse_args()

# Override vLLM port if needed
os.environ["VLLM_API_BASE"] = f"http://127.0.0.1:{args.port}/v1"

from agent.react_agent import run_react
from agent.evals import extract_agent_prediction

# Test questions — chosen to exercise specific failure modes
TEST_QUESTIONS = [
    {
        "id": "Q43",
        "question": "Will Senator Eichorn be found guilty of soliciting a child?",
        "options": ["Yes", "No"],
        "cutoff_date": "2025-12-31",
        "ground_truth": "No",
        "why": "Tests fact vs opinion: 'charged' != 'guilty'. Old agent got P(Yes)=0.95, should be low.",
    },
    {
        "id": "Q35",
        "question": "Will Trump deport 2,000,000 or more people?",
        "options": ["Yes", "No"],
        "cutoff_date": "2025-12-31",
        "ground_truth": "No",
        "why": "Tests plans vs outcomes: Reddit discusses deportation *plans*, not actual numbers.",
    },
    {
        "id": "BTC",
        "question": "Will Bitcoin reach $100,000 by December 31, 2025?",
        "options": ["Yes", "No"],
        "cutoff_date": "2025-06-01",
        "ground_truth": "Yes",
        "why": "Tests Yes prediction: agent should not default to No. BTC hit 100k in late 2024.",
    },
]

print("=" * 65)
print(f"ReAct v2 Prompt Test — {len(TEST_QUESTIONS)} questions")
print(f"Port: {args.port} | Thinking: {'ON' if args.thinking else 'OFF'}")
print("=" * 65)

for tq in TEST_QUESTIONS:
    print(f"\n{'─' * 65}")
    print(f"[{tq['id']}] {tq['question']}")
    print(f"  GT: {tq['ground_truth']} | Test: {tq['why']}")
    print(f"{'─' * 65}")

    start = time.time()
    result = run_react(
        question=tq["question"],
        options=tq["options"],
        cutoff_time=tq["cutoff_date"],
        thinking=args.thinking,
        setup=2,
        timeout_sec=120,
    )
    elapsed = time.time() - start

    pred = extract_agent_prediction(result["final_answer"], tq["options"])

    correct = pred["predicted"] == tq["ground_truth"] if pred["predicted"] else None
    status = "CORRECT" if correct else ("WRONG" if correct is False else "NO PRED")

    print(f"\n  Result: {status}")
    print(f"  Predicted: {pred['predicted']}, P(Yes)={pred['agent_prob_yes']}")
    print(f"  Tool calls: {result['tool_call_count']} | Iterations: {result['iteration_count']}")
    print(f"  Latency: {elapsed:.1f}s | Error: {result['error']}")

    # Show the trace (abbreviated)
    trace = result["final_answer"]
    # Count unique search queries
    search_calls = []
    for line in trace.split("\n"):
        if "[TOOL CALL] search_database" in line:
            search_calls.append(line)

    print(f"\n  Search queries ({len(search_calls)}):")
    for sc in search_calls:
        # Extract just the query arg
        try:
            args_start = sc.index("(") + 1
            args_end = sc.rindex(")")
            call_args = json.loads(sc[args_start:args_end])
            print(f"    → \"{call_args.get('query', '?')}\" [r/{call_args.get('subreddit', 'all')}]")
        except Exception:
            print(f"    → {sc[:120]}")

    # Save full trace for review
    trace_file = f"test_react_v2_{tq['id']}.txt"
    with open(trace_file, "w") as f:
        f.write(f"Question: {tq['question']}\n")
        f.write(f"GT: {tq['ground_truth']}\n")
        f.write(f"Predicted: {pred['predicted']}, P(Yes)={pred['agent_prob_yes']}\n")
        f.write(f"Tool calls: {result['tool_call_count']}, Latency: {elapsed:.1f}s\n")
        f.write(f"Error: {result['error']}\n")
        f.write(f"\n{'=' * 65}\nFULL TRACE\n{'=' * 65}\n\n")
        f.write(trace)
    print(f"\n  Full trace saved to: {trace_file}")

print(f"\n{'=' * 65}")
print("Done. Review trace files for hypothesis decomposition and evidence chaining.")
print("=" * 65)
