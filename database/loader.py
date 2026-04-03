"""
Loads Polymarket binary yes/no questions from the JSONL database files.

Each JSONL entry has:
  - prediction.title             → question text
  - prediction.outcomes          → [{"label": "Yes"}, {"label": "No"}]
  - prediction.close_iso_utc     → market close/resolution date (used as cutoff)
  - prediction.source_payload    → nested dict with:
      .outcomes       → JSON string of outcome labels e.g. '["Yes", "No"]'
      .outcomePrices  → JSON string of settled prices e.g. '["1", "0"]' (winner = "1")
      .category       → topic string
  - history_prices               → daily probability snapshots per outcome
"""

import json
from pathlib import Path
from typing import Optional

DATABASE_DIR = Path(__file__).parent
DEFAULT_PATH = DATABASE_DIR / "polymarket_binary_yesno.jsonl"


def _derive_ground_truth(source_payload: dict) -> Optional[str]:
    """
    Derive Yes/No ground truth from source_payload.outcomePrices + source_payload.outcomes.
    The outcome with settled price "1" is the winner.
    """
    try:
        labels = json.loads(source_payload.get("outcomes", "[]"))
        prices = json.loads(source_payload.get("outcomePrices", "[]"))
    except (json.JSONDecodeError, TypeError):
        return None

    for label, price in zip(labels, prices):
        if str(price).strip() == "1":
            if label in ("Yes", "No"):
                return label
    return None


def _extract_market_probs(history_prices: list, cutoff_date: str) -> tuple[Optional[float], Optional[float]]:
    """
    Return (market_prob_yes, market_prob_no) at the closest date at or before cutoff_date.
    Each is the market-implied probability (0.0–1.0) for that outcome.
    """
    best: dict[str, tuple[str, float]] = {}  # outcome_label → (best_date, price)
    for entry in history_prices:
        label = entry.get("outcome_label")
        if label not in ("Yes", "No"):
            continue
        d = entry.get("date_utc", "")[:10]
        if not d or d > cutoff_date:
            continue
        prev_date = best.get(label, ("", None))[0]
        if d >= prev_date:
            try:
                best[label] = (d, round(float(entry["price"]), 4))
            except (ValueError, KeyError):
                pass
    yes = best.get("Yes", (None, None))[1]
    no  = best.get("No",  (None, None))[1]
    return yes, no


def _extract_timeseries(history_prices: list, close_date: str) -> list[dict]:
    """
    Build a sorted list of daily timepoints from history_prices.

    Each entry: {date, human_prob_yes, human_prob_no}

    The close_date is excluded — passing the resolution date as the agent's
    cutoff would leak the outcome (news/posts from that day reveal the result).
    """
    by_date: dict[str, dict[str, float]] = {}
    for entry in history_prices:
        label = entry.get("outcome_label")
        if label not in ("Yes", "No"):
            continue
        d = entry.get("date_utc", "")[:10]
        if not d or d >= close_date:          # exclude close date and beyond
            continue
        try:
            price = round(float(entry["price"]), 4)
        except (ValueError, KeyError):
            continue
        if d not in by_date:
            by_date[d] = {}
        # keep last-seen price for the date (entries are already time-ordered)
        by_date[d][label] = price

    timepoints = []
    for d in sorted(by_date):
        probs = by_date[d]
        yes = probs.get("Yes")
        no  = probs.get("No")
        if yes is None and no is None:
            continue
        timepoints.append({
            "date":           d,
            "human_prob_yes": yes,
            "human_prob_no":  no,
        })
    return timepoints


def load_polymarket_timeseries(
    path: Path = DEFAULT_PATH,
    max_items: int = None,
    max_timepoints: int = None,
) -> list[dict]:
    """
    Load Polymarket questions with full time-series price history.

    Returns a list of question dicts, each with a ``timepoints`` key:
      [{"date": "YYYY-MM-DD", "human_prob_yes": float, "human_prob_no": float}, ...]

    Timepoints are evenly sampled when max_timepoints is set (default: all).
    The market close date is excluded from timepoints to prevent data leakage.

    Each dict also includes: question, options, ground_truth, close_date, topic.
    """
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            pred = entry.get("prediction", {})
            title = (pred.get("title") or "").strip()
            if not title:
                continue

            close = pred.get("close_iso_utc", "")
            if not close:
                continue
            close_date = close[:10]

            gt = (pred.get("result") or "").strip()
            if gt not in ("Yes", "No"):
                source = pred.get("source_payload") or {}
                gt = _derive_ground_truth(source) or ""
            if gt not in ("Yes", "No"):
                continue

            history = entry.get("history_prices", [])
            timepoints = _extract_timeseries(history, close_date)
            if not timepoints:
                continue

            # Evenly-spaced sampling when max_timepoints is set
            if max_timepoints and len(timepoints) > max_timepoints:
                step = (len(timepoints) - 1) / (max_timepoints - 1)
                timepoints = [timepoints[round(i * step)] for i in range(max_timepoints)]

            source = pred.get("source_payload") or {}
            topic = source.get("category", "") if isinstance(source, dict) else ""

            results.append({
                "question":   title,
                "options":    ["Yes", "No"],
                "ground_truth": gt,
                "close_date": close_date,
                "topic":      topic,
                "timepoints": timepoints,
            })

            if max_items and len(results) >= max_items:
                break

    return results


def load_polymarket(path: Path = DEFAULT_PATH, max_items: int = None) -> list[dict]:
    """
    Load questions from a Polymarket JSONL file.

    Returns a list of dicts with keys:
      question, options, ground_truth, cutoff_date, market_prob_yes, topic
    """
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            pred = entry.get("prediction", {})
            title = (pred.get("title") or "").strip()
            if not title:
                continue

            close = pred.get("close_iso_utc", "")
            if not close:
                continue
            cutoff_date = close[:10]  # "YYYY-MM-DD"

            # Ground truth: prefer prediction.result, fall back to outcomePrices
            gt = (pred.get("result") or "").strip()
            if gt not in ("Yes", "No"):
                source = pred.get("source_payload") or {}
                gt = _derive_ground_truth(source) or ""
            if gt not in ("Yes", "No"):
                continue

            history = entry.get("history_prices", [])
            market_prob_yes, market_prob_no = _extract_market_probs(history, cutoff_date)

            source = pred.get("source_payload") or {}
            topic = source.get("category", "") if isinstance(source, dict) else ""

            results.append({
                "question":        title,
                "options":         ["Yes", "No"],
                "ground_truth":    gt,
                "cutoff_date":     cutoff_date,
                "market_prob_yes": market_prob_yes,
                "market_prob_no":  market_prob_no,
                "topic":           topic,
            })

            if max_items and len(results) >= max_items:
                break

    return results
