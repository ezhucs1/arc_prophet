"""
ReAct agent — plain while-loop, no frameworks.

Loop: call LLM → if tool calls, execute them → repeat → return final answer.
"""

import sys
import json
import time
from pathlib import Path

# Ensure project root is on sys.path when this file is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.llm import get_client, get_model
from agent.tools import TOOL_MAP, TOOL_SCHEMAS
from agent.prompts import REACT_SYSTEM_PROMPT


def run_react(
    question: str,
    options: list[str],
    cutoff_time: str,
    max_iterations: int = 20,
) -> dict:
    """
    Run the ReAct loop for one question.

    Returns:
        final_answer    (str)        — raw LLM output including Reasoning/Selected Option/Confidence
        tool_call_count (int)        — number of tool calls made
        latency_sec     (float)      — wall-clock seconds
        error           (str | None) — error message if failed, else None
    """
    options_str = ", ".join(options)
    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Question: {question}\n"
            f"Options: {options_str}\n"
            f"Cutoff Date: {cutoff_time}\n\n"
            f"You MUST select exactly one of the provided options as your final answer."
        )},
    ]
    tool_call_count = 0
    start = time.time()

    try:
        client = get_client()
        model  = get_model()

        for _ in range(max_iterations):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                temperature=0.1,
                max_tokens=1024,
            )
            msg = response.choices[0].message

            assistant_turn = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant_turn["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_turn)

            if not msg.tool_calls:
                return {
                    "final_answer":    msg.content or "",
                    "tool_call_count": tool_call_count,
                    "latency_sec":     round(time.time() - start, 2),
                    "error":           None,
                }

            for tc in msg.tool_calls:
                tool_call_count += 1
                fn = TOOL_MAP.get(tc.function.name)
                if fn is None:
                    result = f"Unknown tool: {tc.function.name}"
                else:
                    try:
                        result = fn(**json.loads(tc.function.arguments))
                    except Exception as e:
                        result = f"TOOL ERROR: {e}"
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      str(result),
                })

        last_content = next(
            (m["content"] for m in reversed(messages) if m["role"] == "assistant"), ""
        )
        return {
            "final_answer":    last_content,
            "tool_call_count": tool_call_count,
            "latency_sec":     round(time.time() - start, 2),
            "error":           "Iteration limit reached",
        }

    except Exception as e:
        return {
            "final_answer":    "",
            "tool_call_count": tool_call_count,
            "latency_sec":     round(time.time() - start, 2),
            "error":           str(e),
        }
