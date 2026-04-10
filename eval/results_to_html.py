#!/usr/bin/env python3
"""
Convert JSONL result files to a human-readable HTML report.

Usage:
    python eval/results_to_html.py agent/results/results_r_ts_20260404_070127.jsonl
    python eval/results_to_html.py agent/results/results_r_ts_20260404_070127.jsonl -o report.html
    python eval/results_to_html.py agent/results/*.jsonl                  # multiple files
"""

import argparse
import html
import json
import re
import sys
from pathlib import Path


# ── Trace parsing ────────────────────────────────────────────────────────────

def _parse_trace_segments(text: str) -> list[dict]:
    """
    Parse final_answer into ordered segments.

    New format (full trace): contains [TOOL CALL] / [TOOL RESULT] markers
    interspersed with assistant messages (which may contain <think> blocks).

    Old format (last-message only): a single assistant message with optional
    <think> block and visible output.

    Returns list of dicts with keys:
        type: "thinking" | "assistant" | "tool_call" | "tool_result"
        content: str
    """
    if not text:
        return []

    # Split on [TOOL CALL] and [TOOL RESULT] markers
    # Pattern: split keeping the delimiter
    parts = re.split(r"(\[TOOL CALL\] |\[TOOL RESULT\]\n?)", text)

    segments = []
    i = 0
    while i < len(parts):
        part = parts[i]

        if part.startswith("[TOOL CALL] "):
            # Next part is the tool call content
            content = parts[i + 1] if i + 1 < len(parts) else ""
            segments.append({"type": "tool_call", "content": content.strip()})
            i += 2
            continue

        if part.startswith("[TOOL RESULT]"):
            content = parts[i + 1] if i + 1 < len(parts) else ""
            segments.append({"type": "tool_result", "content": content.strip()})
            i += 2
            continue

        # Assistant text (may contain <think> blocks)
        if part.strip():
            _parse_assistant_text(part.strip(), segments)
        i += 1

    return segments


def _parse_assistant_text(text: str, segments: list[dict]) -> None:
    """Split an assistant message into thinking and visible segments."""
    # Process <think> blocks and visible text in order
    pos = 0
    for m in re.finditer(r"<think>(.*?)</think>", text, re.DOTALL):
        # Visible text before this <think> block
        before = text[pos:m.start()].strip()
        if before:
            segments.append({"type": "assistant", "content": before})
        segments.append({"type": "thinking", "content": m.group(1).strip()})
        pos = m.end()

    # Check for unclosed <think> at end
    unclosed = re.search(r"<think>(.*)\Z", text[pos:], re.DOTALL)
    if unclosed:
        before = text[pos:pos + unclosed.start()].strip()
        if before:
            segments.append({"type": "assistant", "content": before})
        segments.append({"type": "thinking", "content": unclosed.group(1).strip()})
    else:
        remaining = text[pos:].strip()
        if remaining:
            segments.append({"type": "assistant", "content": remaining})


# ── HTML formatting ──────────────────────────────────────────────────────────

def _format_assistant_text(text: str) -> str:
    """Escape HTML and highlight known structured labels."""
    escaped = html.escape(text)
    for label in ("Reflection:", "Self-critique:", "Reasoning:",
                  "Final Answer:", "Selected Option:", r"P\(Yes\):", "Confidence:"):
        escaped = re.sub(
            rf"^({label})",
            rf'<span class="label">\1</span>',
            escaped,
            flags=re.MULTILINE,
        )
    for sub in (r"Evidence for (?:Yes|No):", "Gaps / uncertainties:",
                r"Current P\(Yes\):", "Current belief:", "Next action:"):
        escaped = re.sub(
            rf"^(\s*{sub})",
            rf'<span class="sublabel">\1</span>',
            escaped,
            flags=re.MULTILINE,
        )
    return escaped


def _render_segments(segments: list[dict]) -> str:
    """Render parsed segments into HTML sections."""
    if not segments:
        return ""

    parts = []
    tool_step = 0

    for seg in segments:
        if seg["type"] == "thinking":
            parts.append(
                f'<details class="seg seg-think">'
                f'<summary>Thinking</summary>'
                f'<pre class="content think-content">{html.escape(seg["content"])}</pre>'
                f'</details>'
            )
        elif seg["type"] == "assistant":
            parts.append(
                f'<div class="seg seg-assistant">'
                f'<pre class="content">{_format_assistant_text(seg["content"])}</pre>'
                f'</div>'
            )
        elif seg["type"] == "tool_call":
            tool_step += 1
            parts.append(
                f'<details class="seg seg-tool">'
                f'<summary>Tool Call #{tool_step}: {html.escape(seg["content"][:120])}</summary>'
                f'<pre class="content tool-call-content">{html.escape(seg["content"])}</pre>'
                f'</details>'
            )
        elif seg["type"] == "tool_result":
            parts.append(
                f'<details class="seg seg-tool">'
                f'<summary>Tool Result #{tool_step} ({len(seg["content"])} chars)</summary>'
                f'<pre class="content tool-result-content">{html.escape(seg["content"])}</pre>'
                f'</details>'
            )

    return "\n".join(parts)


def record_to_html(rec: dict, idx: int) -> str:
    """Render one JSONL record as an HTML card."""
    qid = rec.get("question_id", "?")
    tp_idx = rec.get("timepoint_index")
    tp_date = rec.get("timepoint_date") or rec.get("cutoff_date", "")
    question = html.escape(rec.get("question", ""))
    gt = rec.get("ground_truth", "?")
    predicted = rec.get("predicted")
    correct = rec.get("correct")
    error = rec.get("error")
    prob_yes = rec.get("agent_prob_yes")
    human_yes = rec.get("market_prob_yes")
    tool_count = rec.get("tool_call_count", 0)
    latency = rec.get("latency_sec", 0)
    close_date = rec.get("close_date", "")

    # Status badge
    if error:
        badge = '<span class="badge badge-error">ERROR</span>'
    elif correct is True:
        badge = '<span class="badge badge-correct">CORRECT</span>'
    elif correct is False:
        badge = '<span class="badge badge-wrong">WRONG</span>'
    else:
        badge = '<span class="badge badge-null">NULL</span>'

    # Title line
    tp_part = f" &middot; T{tp_idx}" if tp_idx is not None else ""
    title = f"Q{qid}{tp_part} &middot; {tp_date}"

    # Metadata row
    pred_str = predicted or "—"
    prob_str = f"{prob_yes:.2f}" if prob_yes is not None else "—"
    human_str = f"{human_yes:.2f}" if human_yes is not None else "—"

    # Parse and render trace
    segments = _parse_trace_segments(rec.get("final_answer", ""))
    body_html = _render_segments(segments)

    # Error display
    error_html = ""
    if error:
        error_html = f'<div class="seg error-section"><strong>Error:</strong> {html.escape(str(error))}</div>'

    return f"""
    <div class="card" id="rec-{idx}">
      <div class="card-header">
        <div class="card-title">{title} {badge}</div>
        <div class="card-question">{question}</div>
        <div class="meta-row">
          <span>GT: <strong>{gt}</strong></span>
          <span>Predicted: <strong>{pred_str}</strong></span>
          <span>P(Yes): <strong>{prob_str}</strong></span>
          <span>Human P(Yes): <strong>{human_str}</strong></span>
          <span>Tools: <strong>{tool_count}</strong></span>
          <span>Latency: <strong>{latency:.1f}s</strong></span>
          {"<span>Close: <strong>" + close_date + "</strong></span>" if close_date else ""}
        </div>
      </div>
      <div class="card-body">
        {error_html}
        {body_html}
      </div>
    </div>"""


def build_html(records: list[dict], source_name: str) -> str:
    """Build complete HTML document from records."""
    total = len(records)
    n_correct = sum(1 for r in records if r.get("correct") is True)
    n_wrong = sum(1 for r in records if r.get("correct") is False)
    n_error = sum(1 for r in records if r.get("error"))
    n_null = total - n_correct - n_wrong - n_error
    answered = n_correct + n_wrong
    accuracy = n_correct / answered if answered else 0
    questions = len({r.get("question_id") for r in records})

    cards = "\n".join(record_to_html(r, i) for i, r in enumerate(records))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Results — {html.escape(source_name)}</title>
<style>
  :root {{
    --bg: #f8f9fa; --card-bg: #fff; --border: #dee2e6;
    --text: #212529; --muted: #6c757d; --green: #198754;
    --red: #dc3545; --orange: #fd7e14; --blue: #0d6efd; --gray: #6c757d;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5;
    padding: 1rem; max-width: 1100px; margin: 0 auto;
  }}
  h1 {{ font-size: 1.4rem; margin-bottom: .25rem; }}
  .file-name {{ color: var(--muted); font-size: .85rem; margin-bottom: .75rem; }}
  .summary {{
    display: flex; flex-wrap: wrap; gap: .5rem 1.5rem;
    padding: .75rem 1rem; background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 6px; margin-bottom: 1rem; font-size: .9rem;
  }}
  .summary span {{ white-space: nowrap; }}
  .controls {{
    display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: 1rem; align-items: center;
  }}
  .controls input, .controls select {{
    padding: .35rem .6rem; border: 1px solid var(--border); border-radius: 4px;
    font-size: .85rem;
  }}
  .controls input {{ flex: 1; min-width: 180px; max-width: 350px; }}
  .card {{
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 6px; margin-bottom: .75rem; overflow: hidden;
  }}
  .card.hidden {{ display: none; }}
  .card-header {{ padding: .6rem .9rem; border-bottom: 1px solid var(--border); }}
  .card-title {{ font-weight: 600; font-size: .95rem; display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; }}
  .card-question {{ color: var(--muted); font-size: .85rem; margin-top: .2rem; }}
  .meta-row {{
    display: flex; flex-wrap: wrap; gap: .3rem 1.2rem;
    font-size: .8rem; color: var(--muted); margin-top: .35rem;
  }}
  .badge {{
    display: inline-block; padding: .1rem .45rem; border-radius: 3px;
    font-size: .75rem; font-weight: 600; color: #fff;
  }}
  .badge-correct {{ background: var(--green); }}
  .badge-wrong {{ background: var(--red); }}
  .badge-error {{ background: var(--orange); }}
  .badge-null {{ background: var(--gray); }}
  .card-body {{ padding: .5rem .9rem .7rem; }}
  .seg {{ margin-bottom: .4rem; }}
  .seg summary {{
    font-size: .8rem; font-weight: 600; color: var(--muted);
    cursor: pointer; padding: .2rem 0;
  }}
  .seg-think summary {{ color: #e67700; }}
  .seg-tool summary {{ color: #5f3dc4; }}
  .content {{
    white-space: pre-wrap; word-break: break-word;
    font-family: "SF Mono", "Cascadia Code", "Fira Code", monospace;
    font-size: .82rem; line-height: 1.6; padding: .5rem .7rem;
    background: #f1f3f5; border-radius: 4px; max-height: 600px; overflow-y: auto;
  }}
  .think-content {{ background: #fff8e1; }}
  .tool-call-content {{ background: #f3f0ff; }}
  .tool-result-content {{ background: #e8f5e9; }}
  .error-section {{ color: var(--red); font-size: .85rem; padding: .3rem 0; }}
  .label {{ color: var(--blue); font-weight: 700; }}
  .sublabel {{ color: #7048e8; font-weight: 600; }}
  details > summary {{ list-style: none; }}
  details > summary::before {{ content: "\\25b6 "; font-size: .7rem; }}
  details[open] > summary::before {{ content: "\\25bc "; }}
</style>
</head>
<body>

<h1>ARC Results Viewer</h1>
<div class="file-name">{html.escape(source_name)} &mdash; {total} records, {questions} questions</div>

<div class="summary">
  <span>Total: <strong>{total}</strong></span>
  <span>Correct: <strong style="color:var(--green)">{n_correct}</strong></span>
  <span>Wrong: <strong style="color:var(--red)">{n_wrong}</strong></span>
  <span>Errors: <strong style="color:var(--orange)">{n_error}</strong></span>
  <span>Null: <strong>{n_null}</strong></span>
  <span>Accuracy: <strong>{accuracy:.1%}</strong></span>
</div>

<div class="controls">
  <input type="text" id="search" placeholder="Search questions...">
  <select id="filter">
    <option value="all">All</option>
    <option value="correct">Correct</option>
    <option value="wrong">Wrong</option>
    <option value="error">Errors</option>
    <option value="null">Null</option>
  </select>
</div>

<div id="cards">
{cards}
</div>

<script>
const cards = document.querySelectorAll('.card');
const search = document.getElementById('search');
const filter = document.getElementById('filter');

function applyFilters() {{
  const q = search.value.toLowerCase();
  const f = filter.value;
  cards.forEach(c => {{
    const text = c.textContent.toLowerCase();
    const matchQ = !q || text.includes(q);
    let matchF = true;
    if (f === 'correct') matchF = c.querySelector('.badge-correct');
    else if (f === 'wrong') matchF = c.querySelector('.badge-wrong');
    else if (f === 'error') matchF = c.querySelector('.badge-error');
    else if (f === 'null') matchF = c.querySelector('.badge-null');
    c.classList.toggle('hidden', !(matchQ && matchF));
  }});
}}
search.addEventListener('input', applyFilters);
filter.addEventListener('change', applyFilters);
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL result files to human-readable HTML")
    parser.add_argument("files", nargs="+", help="JSONL result file(s)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output HTML path (default: <input>.html)")
    args = parser.parse_args()

    for filepath in args.files:
        path = Path(filepath)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            continue

        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        if not records:
            print(f"No records in {path}", file=sys.stderr)
            continue

        out_path = Path(args.output) if args.output else path.with_suffix(".html")
        html_content = build_html(records, path.name)
        out_path.write_text(html_content)
        print(f"{path.name}: {len(records)} records -> {out_path}")


if __name__ == "__main__":
    main()
