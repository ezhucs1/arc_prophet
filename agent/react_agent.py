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
from agent.prompts import REACT_SYSTEM_PROMPT_V2

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


def run_react_v2(
    question: str,
    options: list[str],
    cutoff_time: str,
    max_rounds: int | None = None,
    thinking: bool = True,
    cancel_event: threading.Event | None = None,
    timeout_sec: int | None = None,
    description: str = "",
    **_kwargs,
) -> dict:
    """V2.1 structured-round driver.

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
    user_content += "\n\nBegin round 1."

    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT_V2},
        {"role": "user", "content": user_content},
    ]

    # NOTE: thinking_budget is silently ignored by this vLLM version; rely on
    # max_tokens as the hard cap. Call A gets enough room for <think> + structured.
    _thinking_kwargs = {"enable_thinking": thinking}
    extra_body = {"chat_template_kwargs": _thinking_kwargs}
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


