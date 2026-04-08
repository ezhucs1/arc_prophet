"""
ReAct agent — plain while-loop, no frameworks.

Loop: call LLM → if tool calls, execute them → repeat → return final answer.
"""

import re
import sys
import json
import time
from pathlib import Path

# Ensure project root is on sys.path when this file is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.llm import get_client, get_model
from agent.tools import TOOL_MAP, TOOL_SCHEMAS
from agent.prompts import (
    REACT_SYSTEM_PROMPT, REACT_SYSTEM_PROMPT_REFLECTION,
    SETUP_PROMPTS, SETUP_IS_ZERO_SHOT, SETUP_HAS_TOOLS,
    CONCLUSION_GENERATION_PROMPT, SELF_CRITIQUE_GENERATION_PROMPT,
)

# Must match --max-model-len in the vLLM launch command.
# Used only to compute how many output tokens are safe to request given the
# current prompt size — no messages are injected, no tools are removed.
_CONTEXT_WINDOW = 32768
_CONTEXT_BUFFER = 100   # small safety margin so the API never fires a 400


def run_react(
    question: str,
    options: list[str],
    cutoff_time: str,
    max_iterations: int = 20,
    reflection: bool = False,
    thinking: bool = True,
    max_tokens: int = 8192,
    setup: int | None = None,
    carry_context: str = "",
) -> dict:
    """
    Run the ReAct loop for one question.

    The LLM decides freely: which tools to call, how many times, when to stop,
    what to conclude.  The only infrastructure adjustment made each iteration is
    capping max_tokens to however many output tokens can physically fit given the
    current prompt size (context_window - prompt_tokens - small_buffer).  This
    prevents a hard 400 API rejection without touching the LLM's decision space.

    Args:
        thinking:      Enable Qwen3 extended thinking (<think> blocks).
        max_tokens:    Upper bound on output tokens per call (default 8192).
        setup:         Setup number 1-7. If provided, overrides `reflection` flag.
                       If None, falls back to legacy behavior (reflection bool).
        carry_context: Pre-formatted carry-forward text (conclusion + critique from
                       prior timepoint). Inserted into the user prompt.

    Returns:
        final_answer    (str)        — raw LLM output
        tool_call_count (int)        — number of tool calls made
        latency_sec     (float)      — wall-clock seconds
        error           (str | None) — error message if failed, else None
    """
    # Resolve system prompt
    if setup is not None:
        system_prompt = SETUP_PROMPTS[setup]
    else:
        # Legacy path
        system_prompt = REACT_SYSTEM_PROMPT_REFLECTION if reflection else REACT_SYSTEM_PROMPT

    options_str = ", ".join(options)
    user_content = (
        f"Question: {question}\n"
        f"Options: {options_str}\n"
        f"Cutoff Date: {cutoff_time}"
    )
    if carry_context:
        user_content += f"\n{carry_context}"
    user_content += "\n\nYou MUST select exactly one of the provided options as your final answer."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    tool_call_count  = 0
    start            = time.time()
    last_prompt_toks = 0   # updated from response.usage after each successful call
    trace_parts: list[str] = []  # full reasoning trace across all iterations

    # vLLM extra_body for Qwen3 thinking control
    extra_body = {"chat_template_kwargs": {"enable_thinking": thinking}}

    try:
        client = get_client()
        model  = get_model()

        for _ in range(max_iterations):
            # How many output tokens can we safely request?
            # If last_prompt_toks is known, cap to what fits in the context window.
            # This is pure arithmetic — no messages added, no tools removed.
            if last_prompt_toks > 0:
                available = _CONTEXT_WINDOW - last_prompt_toks - _CONTEXT_BUFFER
                call_max_tokens = min(max_tokens, max(1, available))
            else:
                call_max_tokens = max_tokens

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                temperature=0.1,
                max_tokens=call_max_tokens,
                extra_body=extra_body,
            )

            # Record actual prompt size so the next iteration can compute its budget.
            if response.usage:
                last_prompt_toks = response.usage.prompt_tokens
            choice = response.choices[0]
            msg    = choice.message

            # Truncated generation: tool call JSON was cut off mid-stream,
            # causing hermes_tool_parser to silently drop the tool calls.
            if choice.finish_reason == "length":
                if msg.content:
                    trace_parts.append(msg.content)
                return {
                    "final_answer":    "\n\n".join(trace_parts),
                    "tool_call_count": tool_call_count,
                    "latency_sec":     round(time.time() - start, 2),
                    "error":           "TOKEN_LIMIT",
                }

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

            # Record assistant reasoning in the trace
            if msg.content:
                trace_parts.append(msg.content)

            if not msg.tool_calls:
                # Final answer is the full trace, not just the last message
                return {
                    "final_answer":    "\n\n".join(trace_parts),
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

                # Record tool call and result in the trace
                try:
                    args_pretty = json.loads(tc.function.arguments)
                except Exception:
                    args_pretty = tc.function.arguments
                result_str = str(result)
                # Truncate very long tool results in the trace to keep it readable
                if len(result_str) > 2000:
                    result_str = result_str[:2000] + f"\n... [truncated, {len(str(result))} chars total]"
                trace_parts.append(
                    f"[TOOL CALL] {tc.function.name}({json.dumps(args_pretty, ensure_ascii=False)})\n"
                    f"[TOOL RESULT]\n{result_str}"
                )

        return {
            "final_answer":    "\n\n".join(trace_parts),
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


def generate_carry_forward(
    prompt_template: str,
    question_text: str,
    n: int,
    total: int,
    date_context: str,
    predicted_yes: float,
    predicted_no: float,
    market_yes: float | None = None,
    market_no: float | None = None,
    thinking: bool = True,
) -> str:
    """
    Run a single LLM call to generate a carry-forward block (conclusion or critique).

    Args:
        prompt_template: CONCLUSION_GENERATION_PROMPT or SELF_CRITIQUE_GENERATION_PROMPT
        Other args: values to fill into the template.

    Returns:
        The generated text (should contain <conclusion> or <self_critique> XML).
    """
    fmt_kwargs = {
        "question_text": question_text,
        "n": n,
        "total": total,
        "date_context": date_context,
        "predicted_yes": f"{predicted_yes:.4f}",
        "predicted_no": f"{predicted_no:.4f}",
    }
    # Only conclusion prompt uses market values
    if market_yes is not None:
        fmt_kwargs["market_yes"] = f"{market_yes:.4f}"
        fmt_kwargs["market_no"] = f"{market_no:.4f}"

    prompt = prompt_template.format(**fmt_kwargs)

    extra_body = {"chat_template_kwargs": {"enable_thinking": thinking}}

    try:
        client = get_client()
        model = get_model()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
            extra_body=extra_body,
        )
        content = response.choices[0].message.content or ""
        # Strip <think> blocks from carry-forward output
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        content = re.sub(r"<think>.*", "", content, flags=re.DOTALL)
        return content.strip()
    except Exception as e:
        return f"[Carry-forward generation failed: {e}]"
