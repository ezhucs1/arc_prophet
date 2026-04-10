#!/usr/bin/env python3
"""
Generate side-by-side HTML comparison of S1 (ZS) vs S2 (ReAct) examples.

Usage:
    python eval/comparison_to_html.py
    python eval/comparison_to_html.py -o eval/comparison_examples.html
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Reuse trace parsing from results_to_html
sys.path.insert(0, str(Path(__file__).parent))
from results_to_html import _parse_trace_segments, _format_assistant_text
import html as html_mod


def _render_segments_open(segments: list[dict]) -> str:
    """Render parsed segments into HTML — all expanded (open) by default."""
    if not segments:
        return ""

    parts = []
    tool_step = 0

    for seg in segments:
        if seg["type"] == "thinking":
            parts.append(
                f'<details open class="seg seg-think">'
                f'<summary>Thinking</summary>'
                f'<pre class="content think-content">{html_mod.escape(seg["content"])}</pre>'
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
                f'<details open class="seg seg-tool">'
                f'<summary>Tool Call #{tool_step}: {html_mod.escape(seg["content"][:120])}</summary>'
                f'<pre class="content tool-call-content">{html_mod.escape(seg["content"])}</pre>'
                f'</details>'
            )
        elif seg["type"] == "tool_result":
            parts.append(
                f'<details open class="seg seg-tool">'
                f'<summary>Tool Result #{tool_step} ({len(seg["content"])} chars)</summary>'
                f'<pre class="content tool-result-content">{html_mod.escape(seg["content"])}</pre>'
                f'</details>'
            )

    return "\n".join(parts)


S1_FILE = "agent/results/results_s1_20260408_215821.jsonl"
S2_FILE = "agent/results/results_s2_20260408_215822.jsonl"

# Selected question IDs and categories
SELECTED = [
    ("43", "Retrieval HURT — Reddit opinions treated as facts"),
    ("35", "Retrieval HURT — Plans/aspirations treated as outcomes"),
    ("50", "Retrieval HURT — Absence of evidence fallacy"),
    ("22", "Retrieval HELPED — ZS hallucinated, retrieval corrected"),
    ("28", "Retrieval HELPED — Retrieval found relevant evidence"),
    ("60", "Retrieval HELPED — Retrieval found specific info"),
    ("3",  "Both Correct — Easy No question"),
    ("2",  "Both Wrong — Hard Yes question (geopolitics)"),
    ("37", "Both Wrong — Shows doc_id bug (6/8 tool calls NOT FOUND)"),
]


def load_jsonl(path):
    records = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line.strip())
            records[r["question_id"]] = r
    return records


def render_record_card(rec, label, idx):
    """Render one record as an HTML card, reusing existing trace parser."""
    qid = rec.get("question_id", "?")
    predicted = rec.get("predicted", "—")
    correct = rec.get("correct")
    error = rec.get("error")
    prob_yes = rec.get("agent_prob_yes")
    market_yes = rec.get("market_prob_yes")
    tool_count = rec.get("tool_call_count", 0)
    latency = rec.get("latency_sec", 0)
    gt = rec.get("ground_truth", "?")

    if error and predicted is None:
        badge = '<span class="badge badge-error">ERROR</span>'
    elif correct is True:
        badge = '<span class="badge badge-correct">CORRECT</span>'
    elif correct is False:
        badge = '<span class="badge badge-wrong">WRONG</span>'
    else:
        badge = '<span class="badge badge-null">NULL</span>'

    prob_str = f"{prob_yes:.2f}" if prob_yes is not None else "—"
    market_str = f"{market_yes:.2f}" if market_yes is not None else "—"
    error_str = f" | Error: {error}" if error else ""

    segments = _parse_trace_segments(rec.get("final_answer", ""))
    # Strip <prediction> block from the last segment so it doesn't appear
    # buried inside a collapsed tool result — we render it separately below
    if segments:
        last = segments[-1]
        pred_match = re.search(r"\s*<prediction>.*?</prediction>\s*$", last["content"], re.DOTALL)
        if pred_match:
            last["content"] = last["content"][:pred_match.start()]
    body_html = _render_segments_open(segments)

    # Extract final reasoning from <prediction><reasoning>...</reasoning></prediction>
    fa = rec.get("final_answer", "")
    reasoning_html = ""
    m = re.search(r"<reasoning>(.*?)</reasoning>", fa, re.DOTALL)
    if m:
        reasoning_text = m.group(1).strip()
        prob_m = re.search(r"<yes>([\d.]+)</yes>\s*<no>([\d.]+)</no>", fa)
        prob_line = ""
        if prob_m:
            prob_line = (f'<div style="font-size:.8rem;color:var(--muted);margin-bottom:.3rem;">'
                         f'P(Yes)={prob_m.group(1)}, P(No)={prob_m.group(2)}</div>')
        reasoning_html = (
            f'<div class="seg" style="margin-top:.5rem;padding:.5rem .7rem;'
            f'background:#e3f2fd;border-radius:4px;border-left:3px solid var(--blue);">'
            f'<div style="font-size:.78rem;font-weight:700;color:var(--blue);margin-bottom:.2rem;">'
            f'Final Reasoning</div>'
            f'{prob_line}'
            f'<div style="font-size:.82rem;line-height:1.5;">'
            f'{html_mod.escape(reasoning_text)}</div>'
            f'</div>'
        )

    return f"""
    <div class="card-half">
      <div class="card-half-header">
        <strong>{html_mod.escape(label)}</strong> {badge}
      </div>
      <div class="meta-row">
        <span>Predicted: <strong>{predicted}</strong></span>
        <span>P(Yes): <strong>{prob_str}</strong></span>
        <span>Tools: <strong>{tool_count}</strong></span>
        <span>Latency: <strong>{latency:.1f}s</strong></span>
        <span>{error_str}</span>
      </div>
      <div class="card-body">
        {body_html}
        {reasoning_html}
      </div>
    </div>"""


def build_comparison_html(s1_records, s2_records, selected):
    pairs_html = []

    for qid, category in selected:
        r1 = s1_records.get(qid)
        r2 = s2_records.get(qid)
        if not r1 or not r2:
            continue

        question = html_mod.escape(r1.get("question", ""))
        gt = r1.get("ground_truth", "?")
        market = r1.get("market_prob_yes")
        market_str = f"{market:.4f}" if market is not None else "—"

        left = render_record_card(r1, "S1 — Zero-Shot (thinking ON)", 0)
        right = render_record_card(r2, "S2 — ReAct + Tools (no_thinking)", 1)

        pairs_html.append(f"""
    <div class="pair">
      <div class="pair-header">
        <div class="pair-category">{html_mod.escape(category)}</div>
        <div class="pair-question">Q{qid}: {question}</div>
        <div class="pair-meta">Ground Truth: <strong>{gt}</strong> &nbsp;|&nbsp; Market P(Yes): <strong>{market_str}</strong></div>
      </div>
      <div class="pair-cards">
        {left}
        {right}
      </div>
    </div>""")

    # Aggregate stats
    all_qids = set(s1_records.keys()) & set(s2_records.keys())
    both_correct = s1_only = s2_only = both_wrong = 0
    for qid in all_qids:
        r1, r2 = s1_records[qid], s2_records[qid]
        if r1.get("predicted") is None or r2.get("predicted") is None:
            continue
        c1, c2 = r1["correct"], r2["correct"]
        if c1 and c2: both_correct += 1
        elif c1 and not c2: s1_only += 1
        elif c2 and not c1: s2_only += 1
        else: both_wrong += 1

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>S1 vs S2 — Side-by-Side Examples</title>
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
    padding: 1.5rem; max-width: 1400px; margin: 0 auto;
  }}
  h1 {{ font-size: 1.5rem; margin-bottom: .25rem; }}
  h2 {{ font-size: 1.15rem; margin: 1.5rem 0 .5rem; color: var(--muted); }}
  .subtitle {{ color: var(--muted); font-size: .9rem; margin-bottom: .5rem; }}
  .note {{
    background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px;
    padding: .6rem 1rem; font-size: .85rem; margin-bottom: 1rem;
  }}
  .summary {{
    display: flex; flex-wrap: wrap; gap: .5rem 1.5rem;
    padding: .75rem 1rem; background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 6px; margin-bottom: 1.5rem; font-size: .88rem;
  }}
  .summary span {{ white-space: nowrap; }}

  .pair {{
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 8px; margin-bottom: 1.5rem; overflow: hidden;
  }}
  .pair-header {{
    padding: .7rem 1rem; border-bottom: 2px solid var(--border);
    background: #f1f3f5;
  }}
  .pair-category {{
    font-size: .8rem; font-weight: 700; text-transform: uppercase;
    color: var(--blue); margin-bottom: .15rem;
  }}
  .pair-question {{ font-size: 1rem; font-weight: 600; }}
  .pair-meta {{ font-size: .82rem; color: var(--muted); margin-top: .2rem; }}
  .pair-cards {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 0;
  }}
  .card-half {{
    padding: .6rem .9rem .8rem; border-right: 1px solid var(--border);
    min-width: 0;
  }}
  .card-half:last-child {{ border-right: none; }}
  .card-half-header {{
    font-size: .9rem; margin-bottom: .3rem;
    display: flex; align-items: center; gap: .5rem;
  }}
  .meta-row {{
    display: flex; flex-wrap: wrap; gap: .25rem .9rem;
    font-size: .78rem; color: var(--muted); margin-bottom: .4rem;
  }}
  .badge {{
    display: inline-block; padding: .1rem .4rem; border-radius: 3px;
    font-size: .72rem; font-weight: 600; color: #fff;
  }}
  .badge-correct {{ background: var(--green); }}
  .badge-wrong {{ background: var(--red); }}
  .badge-error {{ background: var(--orange); }}
  .badge-null {{ background: var(--gray); }}
  .card-body {{ font-size: .85rem; }}
  .seg {{ margin-bottom: .3rem; }}
  .seg summary {{
    font-size: .78rem; font-weight: 600; color: var(--muted);
    cursor: pointer; padding: .15rem 0;
  }}
  .seg-think summary {{ color: #e67700; }}
  .seg-tool summary {{ color: #5f3dc4; }}
  .content {{
    white-space: pre-wrap; word-break: break-word;
    font-family: "SF Mono", "Cascadia Code", "Fira Code", monospace;
    font-size: .78rem; line-height: 1.55; padding: .4rem .6rem;
    background: #f1f3f5; border-radius: 4px;
  }}
  .think-content {{ background: #fff8e1; }}
  .tool-call-content {{ background: #f3f0ff; }}
  .tool-result-content {{ background: #e8f5e9; }}
  .label {{ color: var(--blue); font-weight: 700; }}
  .sublabel {{ color: #7048e8; font-weight: 600; }}
  details > summary {{ list-style: none; }}
  details > summary::before {{ content: "\\25b6 "; font-size: .65rem; }}
  details[open] > summary::before {{ content: "\\25bc "; }}

  .stats-table {{
    border-collapse: collapse; font-size: .85rem; margin-bottom: 1rem;
  }}
  .stats-table th, .stats-table td {{
    padding: .35rem .8rem; border: 1px solid var(--border); text-align: left;
  }}
  .stats-table th {{ background: #f1f3f5; font-weight: 600; }}
  .stats-table td:nth-child(2), .stats-table td:nth-child(3) {{ text-align: right; }}
  .delta-neg {{ color: var(--red); }}
  .delta-pos {{ color: var(--green); }}

  @media (max-width: 900px) {{
    .pair-cards {{ grid-template-columns: 1fr; }}
    .card-half {{ border-right: none; border-bottom: 1px solid var(--border); }}
  }}
</style>
</head>
<body>

<h1>S1 (Zero-Shot) vs S2 (ReAct) — Side-by-Side Examples</h1>
<div class="subtitle">P/D Agent Forecasting System &mdash; Qwen3-14B-AWQ via vLLM &mdash; 1,000 Polymarket questions</div>

<div class="note">
  <strong>Note:</strong> S1 uses thinking ON (explicit &lt;think&gt; reasoning blocks visible).
  S2 was run with <code>--no_thinking</code> for compute efficiency &mdash; the model still reasons
  internally but does not emit &lt;think&gt; blocks. S2 shows the tool call &amp; retrieval process instead.
</div>

<h2>Aggregate Results</h2>
<table class="stats-table">
  <tr><th>Metric</th><th>S1 (Zero-Shot)</th><th>S2 (ReAct)</th><th>Delta</th></tr>
  <tr><td>Accuracy (excl. nulls)</td><td>75.0%</td><td>72.0%</td><td class="delta-neg">-3.0pp</td></tr>
  <tr><td>Balanced Accuracy</td><td>56.3%</td><td>53.3%</td><td class="delta-neg">-3.0pp</td></tr>
  <tr><td>MCC</td><td>0.191</td><td>0.095</td><td class="delta-neg">-0.096</td></tr>
  <tr><td>Brier Score</td><td>0.198</td><td>0.207</td><td class="delta-neg">+0.009</td></tr>
  <tr><td>BSS (vs market)</td><td>-2.45</td><td>-2.70</td><td class="delta-neg">-0.25</td></tr>
  <tr><td>Avg Latency</td><td>12.9s</td><td>324.0s</td><td>—</td></tr>
  <tr><td>Graceful Timeout Rate</td><td>0%</td><td>73.9%</td><td>—</td></tr>
</table>

<div class="summary">
  <span>Discordant pairs: <strong>{s1_only + s2_only}</strong> / 1000</span>
  <span>S1-only correct: <strong style="color:var(--blue)">{s1_only}</strong> (retrieval hurt)</span>
  <span>S2-only correct: <strong style="color:var(--green)">{s2_only}</strong> (retrieval helped)</span>
  <span>Both correct: <strong>{both_correct}</strong></span>
  <span>Both wrong: <strong style="color:var(--red)">{both_wrong}</strong></span>
  <span>Statistical tests: Brier p=0.37, McNemar p=0.065, Wilcoxon p=0.87 (all non-significant)</span>
</div>

<h2>Selected Examples ({len(selected)} questions)</h2>

{"".join(pairs_html)}

</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate S1 vs S2 comparison HTML")
    parser.add_argument("-o", "--output", default="eval/comparison_examples.html",
                        help="Output HTML path")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    s1 = load_jsonl(root / S1_FILE)
    s2 = load_jsonl(root / S2_FILE)

    html_content = build_comparison_html(s1, s2, SELECTED)
    out = Path(args.output)
    out.write_text(html_content)
    print(f"Written {len(SELECTED)} examples to {out}")


if __name__ == "__main__":
    main()
