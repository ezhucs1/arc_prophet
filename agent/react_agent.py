"""
ReAct agent — plain while-loop, no frameworks.

Loop: call LLM → if tool calls, execute them → repeat → return final answer.
"""

import re
import sys
import json
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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
_CONTEXT_BUFFER = 512   # safety margin so the API never fires a 400


def _trim_search_result(raw: str) -> str:
    """Compact a search_database result for LLM context.

    Strips top-level echo (question, cutoff_time, table, engine, filters) and
    per-hit noise fields (doc_id, month, author, created_utc, created_at, title,
    distance).  Keeps only: text_preview, object_id, doc_type, subreddit.

    The raw result is still used for auto-drilldown parsing — this trimmed
    version is what goes into messages to save context tokens (~40% smaller).
    """
    try:
        data = json.loads(raw)
        hits = data.get("hits", [])
        compact = []
        for h in hits:
            compact.append({
                "object_id": h.get("object_id", ""),
                "doc_type": h.get("doc_type", ""),
                "subreddit": h.get("subreddit", ""),
                "text_preview": h.get("text_preview", ""),
            })
        return json.dumps({"hits": compact}, ensure_ascii=False)
    except Exception:
        return raw  # not JSON — pass through unchanged


def _parse_text_tool_calls(content: str):
    """Fallback parser: extract tool calls written as plain-text <tool_call> blocks.

    Qwen3 sometimes outputs tool calls as text instead of using the function-calling
    interface, especially after long <think> blocks. This parses them into synthetic
    ToolCall objects so the ReAct loop can execute them normally.

    Returns a list of synthetic tool call objects, or [] if none found.
    """
    from types import SimpleNamespace
    calls = []
    # Find each <tool_call> tag, then extract the JSON object using brace counting
    # (simple regex can't handle nested braces in "arguments": {...})
    for m in re.finditer(r'<tool_call>\s*', content):
        start = m.end()
        if start >= len(content) or content[start] != '{':
            continue
        # Brace-counting to find the matching closing brace
        depth = 0
        end = start
        for i in range(start, len(content)):
            if content[i] == '{':
                depth += 1
            elif content[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if depth != 0:
            continue  # unbalanced braces
        try:
            data = json.loads(content[start:end])
            name = data.get("name", "")
            args = data.get("arguments", {})
            if name and name in TOOL_MAP:
                tc = SimpleNamespace(
                    id=f"text_tc_{len(calls)}",
                    function=SimpleNamespace(
                        name=name,
                        arguments=json.dumps(args),
                    ),
                )
                calls.append(tc)
        except (json.JSONDecodeError, AttributeError):
            continue
    return calls


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
    cancel_event: threading.Event | None = None,
    timeout_sec: int | None = None,
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
        cancel_event:  If set, the loop exits early (used by thread-based timeout
                       to stop abandoned requests from blocking the GPU).
        timeout_sec:   Overall time budget for this question. If elapsed time
                       exceeds this, the loop exits with TIMEOUT error.

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
    iteration_count  = 0
    round_count      = 0   # tracks search rounds for forced multi-round
    seen_searches: set[tuple[str, str]] = set()  # (query, subreddit) dedup
    seen_drilldowns: set[str] = set()             # object_id dedup across iterations
    start            = time.time()
    last_prompt_toks = 0   # updated from response.usage after each successful call
    trace_parts: list[str] = []  # full reasoning trace across all iterations

    # vLLM extra_body for Qwen3 thinking control
    extra_body = {"chat_template_kwargs": {"enable_thinking": thinking}}

    def _timed_out() -> bool:
        """Check both cancel_event and wall-clock budget."""
        if cancel_event is not None and cancel_event.is_set():
            return True
        if timeout_sec is not None and (time.time() - start) >= timeout_sec:
            return True
        return False

    def _graceful_timeout(client, model, error_label: str = "GRACEFUL_TIMEOUT") -> dict:
        """Force one final LLM call to extract a prediction from whatever
        evidence was gathered so far.  Returns *error_label* (got a prediction)
        or TIMEOUT (even the final call failed)."""
        try:
            messages.append({
                "role": "user",
                "content": (
                    "TIME BUDGET EXCEEDED. Based on ALL evidence you have gathered so far, "
                    "produce your final prediction NOW. Do not call any more tools.\n\n"
                    "<prediction>\n"
                    "  <yes>{probability}</yes>\n"
                    "  <no>{probability}</no>\n"
                    "  <reasoning>{synthesize all evidence gathered}</reasoning>\n"
                    "</prediction>"
                ),
            })
            final = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
                extra_body=extra_body,
                timeout=30.0,
            )
            content = final.choices[0].message.content or ""
            if content:
                trace_parts.append(content)
            return {
                "final_answer":    "\n\n".join(trace_parts),
                "tool_call_count": tool_call_count,
                "iteration_count": iteration_count,
                "latency_sec":     round(time.time() - start, 2),
                "error":           error_label,
            }
        except Exception:
            return {
                "final_answer":    "\n\n".join(trace_parts),
                "tool_call_count": tool_call_count,
                "iteration_count": iteration_count,
                "latency_sec":     round(time.time() - start, 2),
                "error":           "TIMEOUT",
            }

    # Per-call HTTP timeout — must be shorter than overall budget so we get
    # a chance to check the budget between calls.  90s is generous for a
    # single vLLM generation; if it takes longer something is wrong.
    _HTTP_TIMEOUT = min(90.0, (timeout_sec or 90.0))

    try:
        client = get_client()
        model  = get_model()

        for _ in range(max_iterations):
            iteration_count += 1
            if _timed_out():
                return _graceful_timeout(client, model)

            # How many output tokens can we safely request?
            if last_prompt_toks > 0:
                available = _CONTEXT_WINDOW - last_prompt_toks - _CONTEXT_BUFFER
                call_max_tokens = min(max_tokens, max(1, available))
                # If context is nearly full, wrap up and predict from
                # whatever evidence we have so far.
                if available < 1024:
                    return _graceful_timeout(client, model, "CONTEXT_LIMIT")
            else:
                call_max_tokens = max_tokens

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    temperature=0.1,
                    max_tokens=call_max_tokens,
                    extra_body=extra_body,
                    timeout=_HTTP_TIMEOUT,
                )
            except Exception as http_err:
                # Context overflow (400) — wrap up and predict from gathered evidence
                if "400" in str(http_err) or "input_tokens" in str(http_err):
                    return _graceful_timeout(client, model, "CONTEXT_LIMIT")
                # HTTP timeout or network error — check if we've exceeded budget
                if _timed_out():
                    return _graceful_timeout(client, model)
                # Not a budget issue — propagate
                raise

            # Check budget immediately after the (potentially long) LLM call
            if _timed_out():
                return _graceful_timeout(client, model)

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
                    "iteration_count": iteration_count,
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

            # Fallback: if model wrote tool calls as plain text <tool_call> instead
            # of using function-calling, parse them and re-inject as real tool calls.
            if not msg.tool_calls and msg.content and "<tool_call>" in msg.content:
                parsed_tcs = _parse_text_tool_calls(msg.content)
                if parsed_tcs:
                    # Rewrite the assistant turn with synthetic tool_calls
                    msg.tool_calls = parsed_tcs
                    assistant_turn["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in parsed_tcs
                    ]
                    messages[-1] = assistant_turn  # replace the already-appended turn

            if not msg.tool_calls:
                # If model tried to predict before 3 search rounds, push it back
                if round_count < 3 and not _timed_out():
                    messages.append({
                        "role": "user",
                        "content": (
                            "You have not completed enough search rounds. "
                            "Do more searches to gather evidence before predicting."
                        ),
                    })
                    continue
                return {
                    "final_answer":    "\n\n".join(trace_parts),
                    "tool_call_count": tool_call_count,
                    "iteration_count": iteration_count,
                    "latency_sec":     round(time.time() - start, 2),
                    "error":           None,
                }

            # ── Execute all tool calls in parallel ──────────────────────
            def _exec_tool(tc):
                """Execute a single tool call. Returns (tc, result_str)."""
                fn = TOOL_MAP.get(tc.function.name)
                if fn is None:
                    return tc, f"Unknown tool: {tc.function.name}"
                try:
                    args = json.loads(tc.function.arguments)
                    if tc.function.name == "search_database":
                        key = (args.get("query", "").strip().lower(),
                               args.get("subreddit", "").strip().lower())
                        if key in seen_searches:
                            return tc, ("DUPLICATE SEARCH: You already ran this exact query. "
                                        "Review previous results and try a DIFFERENT search "
                                        "angle targeting a different hypothesis.")
                        seen_searches.add(key)
                    return tc, str(fn(**args))
                except Exception as e:
                    return tc, f"TOOL ERROR: {e}"

            def _auto_drilldown(search_result_str, cutoff_time):
                """Extract top IDs from search results and drill down automatically.
                Returns list of (tool_name, args_dict, result_str)."""
                drilldowns = []
                try:
                    data = json.loads(search_result_str)
                    hits = data.get("hits", [])[:3]  # top 3 results
                except Exception:
                    return drilldowns
                for hit in hits:
                    obj_id = hit.get("object_id", "")
                    doc_type = hit.get("doc_type", "")
                    if not obj_id or obj_id in seen_drilldowns:
                        continue
                    seen_drilldowns.add(obj_id)
                    if doc_type == "comment":
                        fn = TOOL_MAP.get("get_comment_core_info")
                        tool_name = "get_comment_core_info"
                        args = {"comment_id": obj_id, "cutoff_time": cutoff_time}
                    else:
                        fn = TOOL_MAP.get("get_post_core_info")
                        tool_name = "get_post_core_info"
                        args = {"post_id": obj_id, "cutoff_time": cutoff_time}
                    if fn:
                        try:
                            result = str(fn(**args))
                        except Exception as e:
                            result = f"TOOL ERROR: {e}"
                        drilldowns.append((tool_name, args, result))
                return drilldowns

            if _timed_out():
                return _graceful_timeout(client, model)

            # Phase 1: execute all LLM-requested tool calls in parallel
            results_map = {}  # tc.id → result_str
            with ThreadPoolExecutor(max_workers=len(msg.tool_calls)) as pool:
                futures = {pool.submit(_exec_tool, tc): tc for tc in msg.tool_calls}
                for f in as_completed(futures):
                    tc_done, res = f.result()
                    results_map[tc_done.id] = res

            # Phase 2: auto-drilldown on search results (parallel)
            auto_drilldown_results = []  # list of (tool_name, args, result)
            search_tcs = [tc for tc in msg.tool_calls
                          if tc.function.name == "search_database"
                          and "DUPLICATE" not in results_map.get(tc.id, "")]
            if search_tcs:
                # Extract cutoff_time from first search call
                try:
                    first_args = json.loads(search_tcs[0].function.arguments)
                    cutoff_for_drill = first_args.get("cutoff_time", "")
                except Exception:
                    cutoff_for_drill = ""

                if cutoff_for_drill:
                    with ThreadPoolExecutor(max_workers=len(search_tcs)) as pool:
                        drill_futures = {
                            pool.submit(_auto_drilldown, results_map[tc.id], cutoff_for_drill): tc
                            for tc in search_tcs
                        }
                        for f in as_completed(drill_futures):
                            auto_drilldown_results.extend(f.result())

            # Append LLM-requested tool results to messages and trace
            for tc in msg.tool_calls:
                tool_call_count += 1
                result_str = results_map[tc.id]
                # Trim search results for LLM context (raw kept for auto-drilldown above)
                msg_str = _trim_search_result(result_str) if tc.function.name == "search_database" else result_str
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      msg_str,
                })
                try:
                    args_pretty = json.loads(tc.function.arguments)
                except Exception:
                    args_pretty = tc.function.arguments
                trace_parts.append(
                    f"[TOOL CALL] {tc.function.name}({json.dumps(args_pretty, ensure_ascii=False)})\n"
                    f"[TOOL RESULT]\n{result_str}"
                )

            # Append auto-drilldown results as a summary block in the next user message
            if auto_drilldown_results:
                drill_summary_parts = []
                for tool_name, args, result in auto_drilldown_results:
                    tool_call_count += 1
                    drill_summary_parts.append(
                        f"[AUTO-DRILLDOWN] {tool_name}({json.dumps(args, ensure_ascii=False)})\n{result}"
                    )
                    trace_parts.append(
                        f"[AUTO-DRILLDOWN] {tool_name}({json.dumps(args, ensure_ascii=False)})\n"
                        f"[TOOL RESULT]\n{result}"
                    )
                drill_text = "\n\n".join(drill_summary_parts)
                messages.append({
                    "role": "user",
                    "content": (
                        f"Here are the full texts for the top results from your searches "
                        f"(auto-retrieved, {len(auto_drilldown_results)} drilldowns):\n\n{drill_text}"
                    ),
                })

            # ── Forced multi-round: keep searching until round 3 ──────
            # Count rounds where the model actually issued search calls.
            has_searches = any(tc.function.name == "search_database" for tc in msg.tool_calls)
            if has_searches:
                round_count += 1
            if round_count < 3 and has_searches and not _timed_out():
                messages.append({
                    "role": "user",
                    "content": (
                        f"Round {round_count} of 3 complete. Review the evidence above. "
                        f"Identify NEW leads (names, dates, events, numbers) and do "
                        f"{'3-4' if round_count == 1 else '2-3'} follow-up searches "
                        f"targeting gaps in your evidence. Do NOT predict yet."
                    ),
                })

        return {
            "final_answer":    "\n\n".join(trace_parts),
            "tool_call_count": tool_call_count,
            "iteration_count": iteration_count,
            "latency_sec":     round(time.time() - start, 2),
            "error":           "Iteration limit reached",
        }

    except Exception as e:
        return {
            "final_answer":    "",
            "tool_call_count": tool_call_count,
            "iteration_count": iteration_count,
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
