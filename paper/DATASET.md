# Dataset Documentation
**Last updated:** 2026-04-04

---

## 1. Polymarket Dataset

### Source
Polymarket is a decentralized prediction market platform. Binary yes/no markets each have:
- A question and resolution criteria
- A close/resolution date
- Daily price snapshots (`history_prices`) representing market-implied probabilities
- A settled outcome (Yes = price 1.0, No = price 0.0)

⚠️ **The data collection API/method is not yet documented.** See REPRODUCIBILITY.md §6.

### Files
| File | Size | Description | Used in experiments |
|------|------|-------------|---------------------|
| `polymarket_binary_yesno.jsonl` | ~361 MB | All binary markets | Runs 1, 2, 3 |
| `polymarket_binary_weekly_plus.jsonl` | ~212 MB | Markets open ≥ 1 week | Not yet used |
| `polymarket_binary_monthly_plus.jsonl` | ~47 MB | Markets open ≥ 1 month | Not yet used |
| `polymarket_binary_yearly_plus.jsonl` | ~382 KB | Markets open ≥ 1 year | Not yet used |

All files are gitignored (too large). They must be obtained separately.

### Filtering (applied by `database/loader.py`)
Records are included only if:
1. `ground_truth ∈ {Yes, No}` — derived from `outcomePrices` (settled price = 1) or `prediction.result` field
2. At least one `history_prices` entry exists before the close date
3. `close_iso_utc` is present

### Ground Truth Derivation
```python
# From source_payload.outcomePrices (JSON string) + source_payload.outcomes (JSON string)
# The outcome with settled price "1" is the winner.
labels = json.loads(source_payload["outcomes"])   # e.g. ["Yes", "No"]
prices = json.loads(source_payload["outcomePrices"])  # e.g. ["1", "0"]
ground_truth = labels[prices.index("1")]  # → "Yes"
```
Falls back to `prediction.result` field when `outcomePrices` is unavailable.

---

## 2. Time-Series Construction

### Timepoint Extraction
For each question, the full price history is converted into a sorted list of daily timepoints:
- Each `history_prices` entry with a distinct date becomes one timepoint
- **The close/resolution date is excluded** to prevent data leakage
- Dates after the close date are also excluded

### Data Leakage Prevention
The agent receives **only** the `timepoint_date` as its `cutoff_time`. The `close_date` is:
- Stored in the result record for analysis
- **Never sent to the agent**

This ensures the agent cannot search for news about the outcome on the resolution day (e.g., "DoorDash Q4 2025 results announced").

### Evenly-Spaced Sampling
When `--max_timepoints N` is set, timepoints are sampled evenly:
```python
step = (len(timepoints) - 1) / (max_timepoints - 1)
timepoints = [timepoints[round(i * step)] for i in range(max_timepoints)]
```
This preserves the temporal distribution while capping the number of agent calls per question.

---

## 3. Dataset Statistics (Observed in Experiments)

### Run 2 — ReAct+Reflection subset (20 questions)
| Metric | Value |
|--------|-------|
| Questions | 20 |
| Total timepoints | 233 |
| Avg timepoints/question | 11.65 |
| Min timepoints/question | *(not recorded)* |
| Max timepoints/question | *(not recorded)* |
| Date range (timepoints) | 2026-02-04 to 2026-02-25 |
| Ground truth Yes | 115 (49.4%) |
| Ground truth No | 118 (50.6%) |
| Topic distribution | ⚠️ topic field is empty for all records |

### Run 3 — Zero-Shot subset (939 questions)
| Metric | Value |
|--------|-------|
| Questions | 939 |
| Total timepoints | 9,033 |
| Avg timepoints/question | 9.62 |
| Date range (timepoints) | 2026-02-04 to 2026-02-27 |
| Ground truth Yes | 1,894 (21.0%) |
| Ground truth No | 7,139 (79.0%) |
| Topic distribution | ⚠️ topic field is empty for all records |

### ⚠️ Important: Dataset Imbalance
The 939-question ZS subset is **strongly imbalanced** (79% No). This heavily inflates raw accuracy for a NO-biased predictor. All accuracy comparisons must account for this. Use **Balanced Accuracy** or **MCC** as primary metrics, not raw accuracy.

### ⚠️ Important: Subsets Are Not the Same
- Run 2 (20q) and Run 3 (939q) both start from the beginning of `polymarket_binary_yesno.jsonl`
- Run 2's 20 questions are a subset of Run 3's 939 questions
- The 20-question subset is nearly balanced (49.4% Yes) while the full 939 is not (21% Yes)
- This difference is likely due to topic distribution among the first 20 questions

### ⚠️ Topic Field Empty
The `topic` field is populated from `source_payload.category` but is empty in all observed records. This prevents topic-level analysis in `eval_forecasting.py`. **Action required:** Investigate whether `category` is available in the dataset file or must be retrieved separately.

---

## 4. Human (Market) Probability

For each timepoint, `human_prob_yes` and `human_prob_no` are the Polymarket prices at that date:
- Prices are extracted from `history_prices` as the closest entry at or before the timepoint date
- Prices are the market-implied probabilities (sum ≈ 1.0 but may not be exactly 1 due to fees)
- These are used as the **market baseline** in Brier Score comparisons

The market's Brier Score (0.044 on the ReAct+Reflection subset) reflects that prediction markets are very well-calibrated close to resolution. Early timepoints likely have higher market Brier Scores — this time-series decomposition has not yet been computed.

---

## 5. Questions for Paper

These must be answered before submission:

1. What is the total question count in `polymarket_binary_yesno.jsonl` after filtering?
2. What is the date range of all available markets?
3. What Polymarket categories are represented?
4. Was any additional filtering applied (e.g., minimum trading volume, minimum price history length)?
5. What is the date range of the Reddit knowledge base, and does it cover the question timepoints?
6. Are there questions where the Reddit database has no relevant content? What fraction?
