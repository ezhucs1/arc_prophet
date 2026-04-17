"""
ReAct agent — plain while-loop, no frameworks.

Loop: call LLM → if tool calls, execute them → repeat → return final answer.
"""

import os
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
    REACT_SYSTEM_PROMPT, REACT_SYSTEM_PROMPT_REFLECTION, REACT_SYSTEM_PROMPT_V2,
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
    max_iterations: int = 8,
    reflection: bool = False,
    thinking: bool = True,
    max_tokens: int = 1500,
    setup: int | None = None,
    carry_context: str = "",
    cancel_event: threading.Event | None = None,
    timeout_sec: int | None = None,
    description: str = "",
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
    user_content = f"Question: {question}\n"
    if description:
        user_content += f"Description: {description}\n"
    user_content += (
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
    similar_reject_count = 0  # total TOO SIMILAR rejections; after 5, let through
    seen_drilldowns: set[str] = set()             # object_id dedup across iterations
    start            = time.time()
    last_prompt_toks = 0   # updated from response.usage after each successful call
    trace_parts: list[str] = []  # full reasoning trace across all iterations

    # vLLM extra_body for Qwen3 thinking control
    # thinking_budget caps <think> block length per turn — prevents
    # rambling self-talk in smaller models while preserving hypothesis planning.
    # Passed via chat_template_kwargs (Qwen3 chat template feature).
    _thinking_kwargs = {"enable_thinking": thinking}
    if thinking:
        _thinking_kwargs["thinking_budget"] = 300
    extra_body = {
        "chat_template_kwargs": _thinking_kwargs,
    }

    def _timed_out() -> bool:
        """Check both cancel_event and wall-clock budget."""
        if cancel_event is not None and cancel_event.is_set():
            return True
        if timeout_sec is not None and (time.time() - start) >= timeout_sec:
            return True
        return False

    # Regex for guided generation: forces valid <prediction> XML output.
    _PREDICTION_REGEX = (
        r"<prediction>\s*"
        r"<yes>(0(\.\d{1,4})?|1(\.0{1,4})?)</yes>\s*"
        r"<no>(0(\.\d{1,4})?|1(\.0{1,4})?)</no>\s*"
        r"<reasoning>[^<]{10,500}</reasoning>\s*"
        r"</prediction>"
    )

    def _guided_final_prediction(client, model, messages, extra_body,
                                  trace_parts, tool_call_count, iteration_count, start,
                                  error_label=None) -> dict:
        """Force a structured final prediction using guided generation.

        Sends one last LLM call with guided_regex to guarantee parseable
        <prediction> XML output regardless of model size.
        """
        try:
            messages.append({
                "role": "user",
                "content": (
                    "Based on ALL evidence gathered, produce your final prediction NOW. "
                    "Do not call any more tools. Output ONLY the <prediction> XML block."
                ),
            })
            # Disable thinking for the final structured call — we only
            # need the XML output, not more reasoning.
            final_extra = {
                "chat_template_kwargs": {"enable_thinking": False},
                "guided_regex": _PREDICTION_REGEX,
            }
            final = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=512,
                extra_body=final_extra,
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
            # Guided generation failed — fall back to unguided
            return {
                "final_answer":    "\n\n".join(trace_parts),
                "tool_call_count": tool_call_count,
                "iteration_count": iteration_count,
                "latency_sec":     round(time.time() - start, 2),
                "error":           error_label or "GUIDED_PARSE_FAIL",
            }

    def _final_prediction(client, model, error_label=None) -> dict:
        """Produce the final <prediction> XML via guided-regex."""
        return _guided_final_prediction(
            client, model, messages, extra_body,
            trace_parts, tool_call_count, iteration_count, start,
            error_label=error_label,
        )

    def _graceful_timeout(client, model, error_label: str = "GRACEFUL_TIMEOUT") -> dict:
        """Force one final LLM call to extract a prediction from whatever
        evidence was gathered so far, using guided generation for reliable
        XML output.  Returns *error_label* (got a prediction) or TIMEOUT
        (even the final call failed)."""
        return _final_prediction(client, model, error_label=error_label)

    # Per-call HTTP timeout — must be shorter than overall budget so we get
    # a chance to check the budget between calls.  90s is generous for a
    # single vLLM generation; if it takes longer something is wrong.
    _HTTP_TIMEOUT = min(15.0, (timeout_sec or 15.0))

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

            # Keep full assistant content (including <think>) in message
            # history so the model retains its evidence synthesis between
            # rounds.  Per-turn size is capped by thinking_budget: 300.
            raw_content = msg.content or ""

            assistant_turn = {"role": "assistant", "content": raw_content}
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

            # Record full assistant reasoning (with thinking) in the trace
            if raw_content:
                trace_parts.append(raw_content)

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
                # Force a structured final prediction via guided generation.
                # This guarantees parseable XML regardless of model size.
                return _final_prediction(client, model)

            # ── Execute all tool calls in parallel ──────────────────────
            def _is_near_duplicate(query: str) -> bool:
                """Check if query shares >60% of words with any previous query."""
                words_new = set(query.strip().lower().split())
                if not words_new:
                    return False
                for prev_query, _ in seen_searches:
                    words_prev = set(prev_query.split())
                    if not words_prev:
                        continue
                    overlap = len(words_new & words_prev) / min(len(words_new), len(words_prev))
                    if overlap > 0.6:
                        return True
                return False

            def _exec_tool(tc):
                """Execute a single tool call. Returns (tc, result_str)."""
                nonlocal similar_reject_count
                fn = TOOL_MAP.get(tc.function.name)
                if fn is None:
                    return tc, f"Unknown tool: {tc.function.name}"
                try:
                    args = json.loads(tc.function.arguments)
                    if tc.function.name == "search_database":
                        query = args.get("query", "").strip().lower()
                        sub = args.get("subreddit", "").strip().lower()
                        key = (query, sub)
                        if key in seen_searches:
                            return tc, ("DUPLICATE SEARCH: You already ran this exact query. "
                                        "Try a DIFFERENT search angle with different keywords.")
                        if _is_near_duplicate(query):
                            similar_reject_count += 1
                            if similar_reject_count <= 2:
                                # Build list of already-used words for guidance
                                used_words = set()
                                for prev_q, _ in seen_searches:
                                    used_words.update(prev_q.split())
                                return tc, (
                                    f"TOO SIMILAR to a previous search (>60% word overlap). "
                                    f"Words already covered: {', '.join(sorted(used_words)[:12])}. "
                                    f"Check your GAPS list — search for those unsolved questions "
                                    f"first using completely different keywords. If all gaps are "
                                    f"addressed, investigate a new angle: different actors, "
                                    f"mechanisms, obstacles, or historical precedents."
                                )
                            # After 2 rejections, let searches through to prevent
                            # infinite loops — a redundant search (0.8s) is cheaper
                            # than another wasted iteration (~5-10s).
                        seen_searches.add(key)
                        # Round 1: force all-subreddit search (strip subreddit)
                        if round_count == 0 and sub:
                            args.pop("subreddit", None)
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
                n_next = '3-4' if round_count == 1 else '2-3'
                messages.append({
                    "role": "user",
                    "content": (
                        f"Round {round_count} of 3 complete. Before searching, you MUST "
                        f"write these 4 lines as visible text (NOT inside <think>):\n\n"
                        f"FOUND: [2-3 key facts discovered this round]\n"
                        f"GAPS: [2-3 specific unanswered questions]\n"
                        f"P(Yes): {{previous}} → {{updated}} because {{one reason}}\n"
                        f"NEXT: [{n_next} search queries targeting the gaps above — "
                        f"each MUST use DIFFERENT keywords than all previous searches]\n\n"
                        f"Then issue your {n_next} searches. Do NOT predict yet."
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


def run_react_v2(
    question: str,
    options: list[str],
    cutoff_time: str,
    max_rounds: int | None = None,
    thinking: bool = True,
    setup: int | None = 2,
    carry_context: str = "",
    cancel_event: threading.Event | None = None,
    timeout_sec: int | None = None,
    description: str = "",
    **_kwargs,
) -> dict:
    """Xiao-style structured-round driver.

    Each round: LLM emits <state>/<plan>/<queries> -> we parse & run queries
    in parallel + auto-drilldown -> LLM emits <rethink> -> termination check.
    Terminates on empty gaps, stable p_yes, or max_rounds. Final prediction
    via guided regex.
    """
    options_str = ", ".join(options)
    user_content = f"Question: {question}\n"
    if description:
        user_content += f"Description: {description}\n"
    user_content += f"Options: {options_str}\nCutoff Date: {cutoff_time}"
    if carry_context:
        user_content += f"\n{carry_context}"
    user_content += "\n\nBegin round 1."

    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT_V2},
        {"role": "user", "content": user_content},
    ]

    # NOTE: thinking_budget is silently ignored by this vLLM version; rely on
    # max_tokens as the hard cap. Call A gets enough room for <think> + structured.
    _thinking_kwargs = {"enable_thinking": thinking}
    extra_body = {"chat_template_kwargs": _thinking_kwargs}
    # V2.1 (default) / v21plus / v23 variant selection via V2_VARIANT env var
    _variant = os.getenv("V2_VARIANT", "v21").lower()
    if _variant == "v23":
        drilldown_top_k = 5
        if max_rounds is None:
            max_rounds = 5
    elif _variant == "v21plus":
        # V2.1 with wider fan-out (6-10 queries) + one extra round.
        # Drilldown stays at 3 to avoid the opinion-volume regression V2.3 hit.
        drilldown_top_k = 3
        if max_rounds is None:
            max_rounds = 5
    elif _variant == "v21pp":
        # V2.1 + symmetric sourced-evidence calibration rule (prompt-only change).
        # Same runtime config as V2.1 — isolated test of the calibration hypothesis.
        drilldown_top_k = 3
        if max_rounds is None:
            max_rounds = 4
    elif _variant == "v21ppp":
        # V2.1 + auto author_history on top-1 hit per search (driver-only change).
        # Adds cheap metadata to test tool-count lever without opinion-flooding.
        drilldown_top_k = 3
        if max_rounds is None:
            max_rounds = 4
    else:
        drilldown_top_k = 3
        if max_rounds is None:
            max_rounds = 4
    call_a_max = 1800 if thinking else 1200
    call_b_max = 600
    new_hits_per_round: list[int] = []

    tool_call_count = 0
    trace_parts: list[str] = []
    seen_searches: set[tuple[str, str]] = set()
    seen_drilldowns: set[str] = set()
    seen_authors: set[str] = set()  # v21ppp: dedup author_history lookups across question
    p_yes_history: list[float] = []
    start = time.time()

    _PREDICTION_REGEX = (
        r"<prediction>\s*<yes>(0(\.\d{1,4})?|1(\.0{1,4})?)</yes>\s*"
        r"<no>(0(\.\d{1,4})?|1(\.0{1,4})?)</no>\s*"
        r"<reasoning>[^<]{10,500}</reasoning>\s*</prediction>"
    )

    def _timed_out() -> bool:
        if cancel_event is not None and cancel_event.is_set():
            return True
        if timeout_sec is not None and (time.time() - start) >= timeout_sec:
            return True
        return False

    client = get_client()
    model = get_model()

    def _final_prediction_regex(error_label: str | None = None) -> dict:
        msgs_final = messages + [{
            "role": "user",
            "content": "All rounds complete. Emit the final <prediction> block now based on all gathered evidence.",
        }]
        final_extra = {
            "chat_template_kwargs": {"enable_thinking": False},
            "guided_regex": _PREDICTION_REGEX,
        }
        try:
            final = client.chat.completions.create(
                model=model, messages=msgs_final,
                temperature=0.1, max_tokens=512,
                extra_body=final_extra, timeout=30.0,
            )
            content = final.choices[0].message.content or ""
            if content:
                trace_parts.append(content)
            return {
                "final_answer":    "\n\n".join(trace_parts),
                "tool_call_count": tool_call_count,
                "iteration_count": len(p_yes_history),
                "latency_sec":     round(time.time() - start, 2),
                "error":           error_label,
            }
        except Exception as e:
            return {
                "final_answer":    "\n\n".join(trace_parts),
                "tool_call_count": tool_call_count,
                "iteration_count": len(p_yes_history),
                "latency_sec":     round(time.time() - start, 2),
                "error":           error_label or f"FINAL_FAIL: {e}",
            }

    def _final_prediction(error_label: str | None = None) -> dict:
        """Produce the final <prediction> XML via guided-regex."""
        return _final_prediction_regex(error_label)

    try:
        for round_n in range(1, max_rounds + 1):
            if _timed_out():
                return _final_prediction("TIMEOUT")

            # ── Call A: emit <state>/<plan>/<queries> ──
            try:
                resp_a = client.chat.completions.create(
                    model=model, messages=messages,
                    temperature=0.1, max_tokens=call_a_max,
                    stop=["</queries>"],
                    extra_body=extra_body, timeout=45.0,
                )
            except Exception as e:
                if "400" in str(e) or "context" in str(e).lower():
                    return _final_prediction("CONTEXT_LIMIT")
                raise

            # Defensive: vLLM under saturation / degraded state can return a
            # response whose .choices is None or empty. Treat that as a soft
            # failure and drop into the final prediction path rather than
            # crashing with 'NoneType'/'IndexError' and losing the whole run.
            choices_a = getattr(resp_a, "choices", None) or []
            if not choices_a:
                trace_parts.append(f"[DEGRADED_RESPONSE] Round {round_n} call A: empty choices; ending rounds.")
                return _final_prediction("DEGRADED_RESPONSE")
            content_a = (choices_a[0].message.content or "") + "\n</queries>"
            messages.append({"role": "assistant", "content": content_a})
            trace_parts.append(f"=== ROUND {round_n} ===\n{content_a}")

            # Parse <queries>; fall back to any {"q":...} JSON lines anywhere
            # (rescues runs where <think> ate the budget and <queries> never opened)
            queries: list[dict] = []
            q_match = re.search(r"<queries>\s*(.*?)\s*</queries>", content_a, re.DOTALL)
            scan_region = q_match.group(1) if q_match else content_a
            for raw in re.findall(r'\{\s*"q"\s*:.*?\}', scan_region, re.DOTALL):
                try:
                    queries.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
            if not q_match and queries:
                trace_parts.append(f"[FALLBACK] <queries> tag missing; recovered {len(queries)} JSON queries from content")
            if not queries:
                trace_parts.append("[VOID ROUND] no valid queries parsed — terminating")
                break

            # Intra-round fuzzy dedup only: >75% word overlap with a query already
            # accepted THIS round is dropped. Inter-round dupes are handled by the
            # exact-match seen_searches check inside _exec_query.
            def _near_dup(q: str, against: list[str]) -> bool:
                wn = set(q.lower().split())
                if not wn:
                    return True
                for prev in against:
                    wp = set(prev.lower().split())
                    if not wp:
                        continue
                    if len(wn & wp) / min(len(wn), len(wp)) > 0.75:
                        return True
                return False

            accepted: list[dict] = []
            accepted_texts: list[str] = []
            dropped = 0
            for q in queries:
                qt = (q.get("q") or "").strip()
                if not qt or _near_dup(qt, accepted_texts):
                    dropped += 1
                    continue
                accepted.append(q)
                accepted_texts.append(qt.lower())
            if dropped:
                trace_parts.append(f"[DEDUP] dropped {dropped} near-duplicate queries this round")
            queries = accepted
            if not queries:
                trace_parts.append("[VOID ROUND] all queries were near-duplicates — terminating")
                break

            # Extract gaps + p_yes for termination
            gaps_empty = False
            gaps_m = re.search(r"<gaps>(.*?)</gaps>", content_a, re.DOTALL)
            if gaps_m:
                gtxt = gaps_m.group(1).strip().lower()
                stripped = re.sub(r"[\-\s]", "", gtxt)
                if not stripped or stripped == "none":
                    gaps_empty = True
            pyes_m = re.search(r"<p_yes>\s*([\d.]+)\s*</p_yes>", content_a)
            if pyes_m:
                try:
                    p_yes_history.append(float(pyes_m.group(1)))
                except ValueError:
                    pass

            # ── Execute queries in parallel ──
            def _exec_query(q: dict):
                qtext = (q.get("q") or "").strip()
                sub = (q.get("subreddit") or "").strip()
                args: dict = {"query": qtext, "cutoff_time": cutoff_time}
                # Round 1: drop subreddit to cast wide (consistent with v1)
                if sub and round_n > 1:
                    args["subreddit"] = sub
                key = (qtext.lower(), args.get("subreddit", "").lower())
                if key in seen_searches:
                    return args, "DUPLICATE SEARCH — use different keywords next round."
                seen_searches.add(key)
                try:
                    return args, str(TOOL_MAP["search_database"](**args))
                except Exception as e:
                    return args, f"TOOL ERROR: {e}"

            if _timed_out():
                return _final_prediction("TIMEOUT")
            with ThreadPoolExecutor(max_workers=max(1, len(queries))) as pool:
                q_results = list(pool.map(_exec_query, queries))
            tool_call_count += len(q_results)

            # ── Auto-drilldown top-K per successful search ──
            drilldowns: list[tuple[str, dict, str]] = []
            round_new_hits = 0
            for args, result_str in q_results:
                try:
                    hits = json.loads(result_str).get("hits", [])[:drilldown_top_k]
                except Exception:
                    hits = []
                for hit in hits:
                    oid = hit.get("object_id", "")
                    if not oid or oid in seen_drilldowns:
                        continue
                    seen_drilldowns.add(oid)
                    round_new_hits += 1
                    if hit.get("doc_type") == "comment":
                        tn = "get_comment_core_info"
                        d_args = {"comment_id": oid, "cutoff_time": cutoff_time}
                    else:
                        tn = "get_post_core_info"
                        d_args = {"post_id": oid, "cutoff_time": cutoff_time}
                    try:
                        drilldowns.append((tn, d_args, str(TOOL_MAP[tn](**d_args))))
                    except Exception as e:
                        drilldowns.append((tn, d_args, f"TOOL ERROR: {e}"))
            tool_call_count += len(drilldowns)

            # ── v21ppp: author_history on top-1 hit per search (cheap metadata) ──
            author_lookups: list[tuple[str, dict, str]] = []
            if _variant == "v21ppp":
                for args, result_str in q_results:
                    try:
                        hits = json.loads(result_str).get("hits", [])
                    except Exception:
                        hits = []
                    for hit in hits:
                        author = (hit.get("author") or "").strip()
                        if not author or author in ("[deleted]", "[removed]"):
                            continue
                        if author in seen_authors:
                            break
                        seen_authors.add(author)
                        a_args = {"author_id": author, "cutoff_time": cutoff_time,
                                  "max_posts": 5, "max_comments": 5}
                        try:
                            author_lookups.append(
                                ("get_author_history_list", a_args,
                                 str(TOOL_MAP["get_author_history_list"](**a_args)))
                            )
                        except Exception as e:
                            author_lookups.append(
                                ("get_author_history_list", a_args, f"TOOL ERROR: {e}")
                            )
                        break  # only top-1-with-author per search
                tool_call_count += len(author_lookups)

            # ── Build results message for round ──
            results_blocks_msg = []
            for args, res in q_results:
                trimmed = _trim_search_result(res)
                results_blocks_msg.append(
                    f"[TOOL CALL] search_database({json.dumps(args, ensure_ascii=False)})\n[TOOL RESULT]\n{trimmed}"
                )
                trace_parts.append(
                    f"[TOOL CALL] search_database({json.dumps(args, ensure_ascii=False)})\n[TOOL RESULT]\n{res}"
                )
            for tn, d_args, res in drilldowns:
                results_blocks_msg.append(
                    f"[AUTO-DRILLDOWN] {tn}({json.dumps(d_args, ensure_ascii=False)})\n{res}"
                )
                trace_parts.append(
                    f"[AUTO-DRILLDOWN] {tn}({json.dumps(d_args, ensure_ascii=False)})\n[TOOL RESULT]\n{res}"
                )
            for tn, a_args, res in author_lookups:
                results_blocks_msg.append(
                    f"[AUTO-METADATA] {tn}({json.dumps(a_args, ensure_ascii=False)})\n{res}"
                )
                trace_parts.append(
                    f"[AUTO-METADATA] {tn}({json.dumps(a_args, ensure_ascii=False)})\n[TOOL RESULT]\n{res}"
                )

            is_last = (round_n == max_rounds)
            nudge = (
                "This is the FINAL round. Emit <rethink> (3 sentences max), then the final <prediction> block."
                if is_last else
                "Emit <rethink> (3 sentences max) integrating this evidence. "
                "Then either emit the next round's <state>/<plan>/<queries>, OR the final <prediction> if gaps are resolved."
            )
            messages.append({
                "role": "user",
                "content": f"Round {round_n} tool results:\n\n" + "\n\n".join(results_blocks_msg) + f"\n\n{nudge}",
            })

            # ── Call B: <rethink> ──
            if _timed_out():
                return _final_prediction("TIMEOUT")
            try:
                resp_b = client.chat.completions.create(
                    model=model, messages=messages,
                    temperature=0.1, max_tokens=call_b_max,
                    stop=["</rethink>"],
                    extra_body=extra_body, timeout=30.0,
                )
            except Exception as e:
                if "400" in str(e) or "context" in str(e).lower():
                    return _final_prediction("CONTEXT_LIMIT")
                raise

            choices_b = getattr(resp_b, "choices", None) or []
            if not choices_b:
                trace_parts.append(f"[DEGRADED_RESPONSE] Round {round_n} call B: empty choices; ending rounds.")
                return _final_prediction("DEGRADED_RESPONSE")
            content_b = (choices_b[0].message.content or "") + "\n</rethink>"
            messages.append({"role": "assistant", "content": content_b})
            trace_parts.append(content_b)

            # ── Termination ──
            new_hits_per_round.append(round_new_hits)
            if gaps_empty:
                trace_parts.append("[STOP] gaps empty")
                break
            if round_n >= 2 and round_new_hits == 0:
                trace_parts.append("[STOP] zero new evidence this round")
                break

        return _final_prediction()

    except Exception as e:
        return {
            "final_answer":    "\n\n".join(trace_parts),
            "tool_call_count": tool_call_count,
            "iteration_count": len(p_yes_history),
            "latency_sec":     round(time.time() - start, 2),
            "error":           f"V2_CRASH: {e}",
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
