#!/usr/bin/env python3
"""
P/D Agent Forecasting Evaluation Suite
======================================

Comprehensive evaluation metrics for binary prediction tasks.
Designed for Polymarket-style Yes/No forecasting evaluation.

Features:
- All standard classification metrics (Accuracy, F1, Precision, Recall)
- Imbalanced-data metrics (MCC, Cohen's Kappa, Balanced Accuracy)
- Probability/calibration metrics (Brier, Log Loss, ECE) - when confidence available
- Topic-level breakdown
- Operational metrics (latency, tool calls, errors)
- Export to JSON, CSV, or formatted report

Usage:
    python eval_forecasting.py results.jsonl
    python eval_forecasting.py results.jsonl --output report.json
    python eval_forecasting.py results.jsonl --format csv --output metrics.csv
    python eval_forecasting.py results.jsonl --format all --output-dir ./eval_results/

Author: Generated for P/D Multi-Agent Forecasting System
"""

import json
import argparse
import sys
import os
import re
import math
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path


# =============================================================================
# DATA CLASSES FOR METRICS
# =============================================================================

@dataclass
class ConfusionMatrix:
    """Raw confusion matrix counts."""
    tp: int = 0  # True Positive: predicted Yes, actual Yes
    tn: int = 0  # True Negative: predicted No, actual No
    fp: int = 0  # False Positive: predicted Yes, actual No
    fn: int = 0  # False Negative: predicted No, actual Yes
    
    @property
    def total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn
    
    @property
    def positive_actual(self) -> int:
        """Total actual Yes cases."""
        return self.tp + self.fn
    
    @property
    def negative_actual(self) -> int:
        """Total actual No cases."""
        return self.tn + self.fp
    
    @property
    def positive_predicted(self) -> int:
        """Total predicted Yes cases."""
        return self.tp + self.fp
    
    @property
    def negative_predicted(self) -> int:
        """Total predicted No cases."""
        return self.tn + self.fn


@dataclass
class ClassMetrics:
    """Metrics for a single class (Yes or No)."""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    support: int = 0  # Number of actual cases for this class


@dataclass
class BinaryClassificationMetrics:
    """Complete binary classification metrics."""
    # Sample counts
    total_samples: int = 0
    valid_predictions: int = 0
    null_predictions: int = 0
    
    # Confusion matrix
    confusion_matrix: ConfusionMatrix = field(default_factory=ConfusionMatrix)
    
    # Basic metrics
    accuracy: float = 0.0
    accuracy_excluding_nulls: float = 0.0
    
    # Per-class metrics
    yes_class: ClassMetrics = field(default_factory=ClassMetrics)
    no_class: ClassMetrics = field(default_factory=ClassMetrics)
    
    # Aggregate F1 scores
    f1_macro: float = 0.0
    f1_micro: float = 0.0
    f1_weighted: float = 0.0
    
    # Imbalanced-data metrics
    matthews_correlation_coefficient: float = 0.0
    cohens_kappa: float = 0.0
    balanced_accuracy: float = 0.0
    
    # Bias analysis
    yes_rate_ground_truth: float = 0.0
    yes_rate_predicted: float = 0.0
    prediction_bias: float = 0.0
    
    # Specificity and sensitivity (aliases)
    sensitivity: float = 0.0  # Same as recall_yes (TPR)
    specificity: float = 0.0  # TNR


@dataclass
class CalibrationMetrics:
    """Probability calibration metrics (requires confidence scores)."""
    available: bool = False
    n_samples_with_confidence: int = 0
    
    # Core calibration metrics
    brier_score: Optional[float] = None
    log_loss: Optional[float] = None
    
    # Expected Calibration Error (binned)
    expected_calibration_error: Optional[float] = None
    max_calibration_error: Optional[float] = None
    
    # Confidence analysis
    avg_confidence: Optional[float] = None
    avg_confidence_when_correct: Optional[float] = None
    avg_confidence_when_wrong: Optional[float] = None
    overconfidence_rate: Optional[float] = None  # Wrong but confidence > 0.7
    underconfidence_rate: Optional[float] = None  # Correct but confidence < 0.5
    
    # Calibration curve data (for plotting)
    calibration_bins: Optional[List[Dict[str, float]]] = None


@dataclass
class OperationalMetrics:
    """Operational and efficiency metrics."""
    # Latency
    avg_latency_sec: float = 0.0
    median_latency_sec: float = 0.0
    std_latency_sec: float = 0.0
    min_latency_sec: float = 0.0
    max_latency_sec: float = 0.0
    p90_latency_sec: float = 0.0
    p95_latency_sec: float = 0.0
    p99_latency_sec: float = 0.0

    # Tool usage
    avg_tool_calls: float = 0.0
    median_tool_calls: float = 0.0
    max_tool_calls: int = 0
    zero_tool_call_rate: float = 0.0

    # Errors
    total_errors: int = 0
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    token_limit_rate: float = 0.0   # finish_reason=length or TOKEN_LIMIT
    context_overflow_rate: float = 0.0  # 400 context window exceeded

    # Error breakdown
    error_types: Dict[str, int] = field(default_factory=dict)


@dataclass
class TimeSeriesMetrics:
    """Metrics for time-series evaluation (multiple timepoints per question)."""
    available: bool = False
    n_questions: int = 0
    n_total_timepoints: int = 0
    avg_timepoints_per_question: float = 0.0

    # Accuracy at different positions in the timeline
    accuracy_first_timepoint: Optional[float] = None   # earliest available date
    accuracy_last_timepoint: Optional[float] = None    # closest to resolution
    accuracy_early_half: Optional[float] = None        # timepoint_index <= median
    accuracy_late_half: Optional[float] = None         # timepoint_index > median

    # Per-question consistency: does the agent agree with itself across timepoints?
    consistency_rate: float = 0.0        # fraction of questions where all timepoints agree
    majority_vote_accuracy: float = 0.0  # accuracy using per-question majority vote


@dataclass
class MarketComparisonMetrics:
    """Compare agent probability estimates against prediction market (human) probabilities."""
    available: bool = False
    n_samples: int = 0

    # Brier scores on P(Yes) — lower is better, perfect = 0, random = 0.25
    agent_brier_score: Optional[float] = None
    market_brier_score: Optional[float] = None
    brier_skill_score: Optional[float] = None  # 1 - (agent_BS / market_BS); >0 means agent beats market

    # Probability alignment
    prob_correlation: Optional[float] = None       # Pearson r(agent_prob_yes, human_prob_yes)
    mean_absolute_deviation: Optional[float] = None  # avg |agent_prob_yes - human_prob_yes|

    # Directional agreement: both sides of 0.5 threshold
    directional_agreement_rate: Optional[float] = None  # agent and market agree on which side of 50%


@dataclass
class ReflectionQualityMetrics:
    """Metrics about reflection and self-critique content in outputs."""
    available: bool = False
    n_total: int = 0
    n_with_reflection: int = 0
    n_with_self_critique: int = 0
    reflection_rate: float = 0.0
    self_critique_rate: float = 0.0

    # Accuracy split: does having a reflection or self-critique correlate with better accuracy?
    accuracy_with_reflection: Optional[float] = None
    accuracy_without_reflection: Optional[float] = None
    accuracy_with_self_critique: Optional[float] = None
    accuracy_without_self_critique: Optional[float] = None


@dataclass
class TopicMetrics:
    """Metrics for a specific topic/category."""
    topic: str = ""
    n_samples: int = 0
    n_valid: int = 0
    n_correct: int = 0
    n_null: int = 0
    
    accuracy: float = 0.0
    f1_yes: float = 0.0
    f1_no: float = 0.0
    f1_macro: float = 0.0
    
    # Confusion matrix for topic
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    # Metadata
    timestamp: str = ""
    input_file: str = ""
    total_questions: int = 0

    # Core metrics
    classification: BinaryClassificationMetrics = field(default_factory=BinaryClassificationMetrics)
    calibration: CalibrationMetrics = field(default_factory=CalibrationMetrics)
    operational: OperationalMetrics = field(default_factory=OperationalMetrics)

    # Time-series specific (populated when timepoint fields are present)
    timeseries: TimeSeriesMetrics = field(default_factory=TimeSeriesMetrics)

    # Market comparison (populated when human_prob_yes fields are present)
    market_comparison: MarketComparisonMetrics = field(default_factory=MarketComparisonMetrics)

    # Reflection quality (populated when reflection/self_critique fields are present)
    reflection_quality: ReflectionQualityMetrics = field(default_factory=ReflectionQualityMetrics)

    # Breakdown by topic
    by_topic: Dict[str, TopicMetrics] = field(default_factory=dict)

    # Error analysis
    common_error_patterns: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# METRIC COMPUTATION FUNCTIONS
# =============================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default on zero denominator."""
    return numerator / denominator if denominator > 0 else default


def compute_confusion_matrix(data: List[Dict]) -> Tuple[ConfusionMatrix, int, int]:
    """
    Compute confusion matrix from prediction data.
    
    Returns:
        (confusion_matrix, valid_count, null_count)
    """
    cm = ConfusionMatrix()
    null_count = 0
    
    for record in data:
        gt = record.get('ground_truth')
        pred = record.get('predicted')
        
        if pred is None:
            null_count += 1
            continue
        
        if gt == 'Yes' and pred == 'Yes':
            cm.tp += 1
        elif gt == 'No' and pred == 'No':
            cm.tn += 1
        elif gt == 'No' and pred == 'Yes':
            cm.fp += 1
        elif gt == 'Yes' and pred == 'No':
            cm.fn += 1
    
    valid_count = cm.total
    return cm, valid_count, null_count


def compute_classification_metrics(data: List[Dict]) -> BinaryClassificationMetrics:
    """Compute all binary classification metrics."""
    metrics = BinaryClassificationMetrics()
    metrics.total_samples = len(data)
    
    # Compute confusion matrix
    cm, valid, null = compute_confusion_matrix(data)
    metrics.confusion_matrix = cm
    metrics.valid_predictions = valid
    metrics.null_predictions = null
    
    if valid == 0:
        return metrics
    
    # Basic accuracy
    correct = cm.tp + cm.tn
    metrics.accuracy = safe_divide(correct, metrics.total_samples)
    metrics.accuracy_excluding_nulls = safe_divide(correct, valid)
    
    # Yes class metrics
    metrics.yes_class.precision = safe_divide(cm.tp, cm.tp + cm.fp)
    metrics.yes_class.recall = safe_divide(cm.tp, cm.tp + cm.fn)
    metrics.yes_class.support = cm.tp + cm.fn
    if metrics.yes_class.precision + metrics.yes_class.recall > 0:
        metrics.yes_class.f1 = 2 * metrics.yes_class.precision * metrics.yes_class.recall / \
                               (metrics.yes_class.precision + metrics.yes_class.recall)
    
    # No class metrics
    metrics.no_class.precision = safe_divide(cm.tn, cm.tn + cm.fn)
    metrics.no_class.recall = safe_divide(cm.tn, cm.tn + cm.fp)
    metrics.no_class.support = cm.tn + cm.fp
    if metrics.no_class.precision + metrics.no_class.recall > 0:
        metrics.no_class.f1 = 2 * metrics.no_class.precision * metrics.no_class.recall / \
                              (metrics.no_class.precision + metrics.no_class.recall)
    
    # F1 scores
    metrics.f1_macro = (metrics.yes_class.f1 + metrics.no_class.f1) / 2
    metrics.f1_micro = metrics.accuracy_excluding_nulls  # Same as accuracy for binary
    
    # Weighted F1
    total_support = metrics.yes_class.support + metrics.no_class.support
    if total_support > 0:
        metrics.f1_weighted = (
            metrics.yes_class.f1 * metrics.yes_class.support +
            metrics.no_class.f1 * metrics.no_class.support
        ) / total_support
    
    # Matthews Correlation Coefficient
    mcc_num = (cm.tp * cm.tn) - (cm.fp * cm.fn)
    mcc_denom = math.sqrt(
        (cm.tp + cm.fp) * (cm.tp + cm.fn) * (cm.tn + cm.fp) * (cm.tn + cm.fn)
    )
    metrics.matthews_correlation_coefficient = safe_divide(mcc_num, mcc_denom)
    
    # Sensitivity and Specificity
    metrics.sensitivity = metrics.yes_class.recall  # TPR
    metrics.specificity = safe_divide(cm.tn, cm.tn + cm.fp)  # TNR
    
    # Balanced Accuracy
    metrics.balanced_accuracy = (metrics.sensitivity + metrics.specificity) / 2
    
    # Cohen's Kappa
    p_observed = safe_divide(correct, valid)
    p_yes_gt = safe_divide(cm.tp + cm.fn, valid)
    p_no_gt = safe_divide(cm.tn + cm.fp, valid)
    p_yes_pred = safe_divide(cm.tp + cm.fp, valid)
    p_no_pred = safe_divide(cm.tn + cm.fn, valid)
    p_expected = (p_yes_gt * p_yes_pred) + (p_no_gt * p_no_pred)
    metrics.cohens_kappa = safe_divide(p_observed - p_expected, 1 - p_expected)
    
    # Bias analysis
    gt_yes = sum(1 for d in data if d.get('ground_truth') == 'Yes')
    pred_yes = sum(1 for d in data if d.get('predicted') == 'Yes')
    metrics.yes_rate_ground_truth = safe_divide(gt_yes, metrics.total_samples)
    metrics.yes_rate_predicted = safe_divide(pred_yes, valid)
    metrics.prediction_bias = metrics.yes_rate_predicted - metrics.yes_rate_ground_truth
    
    return metrics


def extract_prob_yes_from_record(record: Dict) -> Optional[float]:
    """
    Extract P(Yes) — the probability that Yes is the correct answer — from a record.

    Priority:
    1. agent_prob_yes field (pre-computed, always P(Yes))
    2. P(Yes): label in final_answer text (new prompt format)
    3. Legacy: Confidence field + predicted direction (old prompt format)
       - if predicted=Yes: P(Yes) = confidence
       - if predicted=No:  P(Yes) = 1 - confidence
    """
    # 1. Pre-computed field — most reliable
    if record.get('agent_prob_yes') is not None:
        return float(record['agent_prob_yes'])

    final_answer = record.get('final_answer', '')
    pred         = record.get('predicted')

    # 2. New format: P(Yes): 0.75
    if final_answer:
        m = re.search(r'P\s*\(\s*[Yy]es\s*\)\s*[:\s]+([\d.]+)', final_answer)
        if m:
            val = float(m.group(1))
            if val > 1:
                val /= 100
            return max(0.0, min(1.0, val))

    # 3. Legacy: Confidence label (P(predicted option))
    if final_answer and pred is not None:
        m = re.search(r'[Cc]onfidence[:\s]+([\d.]+)', final_answer)
        if m:
            conf = float(m.group(1))
            if conf > 1:
                conf /= 100
            conf = max(0.0, min(1.0, conf))
            return conf if pred == 'Yes' else (1.0 - conf)

    return None


def compute_calibration_metrics(data: List[Dict], n_bins: int = 10) -> CalibrationMetrics:
    """Compute probability calibration metrics.

    Uses agent_prob_yes directly when available (preferred — already converted
    to P(Yes) by evals.py).  Falls back to deriving P(Yes) from the confidence
    field + predicted direction.
    """
    metrics = CalibrationMetrics()

    records_with_conf = []
    for record in data:
        gt = record.get('ground_truth')
        pred = record.get('predicted')

        if pred is None or gt is None:
            continue

        prob_yes = extract_prob_yes_from_record(record)
        if prob_yes is None:
            continue

        prob_yes = float(prob_yes)
        actual  = 1 if gt == 'Yes' else 0
        correct = (pred == gt)
        # agent confidence in its own prediction (not necessarily P(Yes))
        conf_in_pred = prob_yes if pred == 'Yes' else (1.0 - prob_yes)

        records_with_conf.append({
            'prob_yes':   prob_yes,
            'actual':     actual,
            'correct':    correct,
            'confidence': conf_in_pred,
        })
    
    metrics.n_samples_with_confidence = len(records_with_conf)
    
    if len(records_with_conf) < 5:
        return metrics
    
    metrics.available = True
    n = len(records_with_conf)
    
    # Brier Score
    metrics.brier_score = sum(
        (r['prob_yes'] - r['actual'])**2 for r in records_with_conf
    ) / n
    
    # Log Loss
    eps = 1e-15
    log_loss = 0
    for r in records_with_conf:
        p = max(min(r['prob_yes'], 1 - eps), eps)
        y = r['actual']
        log_loss -= (y * math.log(p) + (1 - y) * math.log(1 - p))
    metrics.log_loss = log_loss / n
    
    # Calibration Error (binned)
    bins = [[] for _ in range(n_bins)]
    for r in records_with_conf:
        bin_idx = min(int(r['prob_yes'] * n_bins), n_bins - 1)
        bins[bin_idx].append(r)
    
    ece = 0
    mce = 0
    calibration_bins = []
    
    for i, bin_records in enumerate(bins):
        if bin_records:
            avg_conf = sum(r['prob_yes'] for r in bin_records) / len(bin_records)
            avg_actual = sum(r['actual'] for r in bin_records) / len(bin_records)
            bin_error = abs(avg_conf - avg_actual)
            ece += len(bin_records) * bin_error
            mce = max(mce, bin_error)
            
            calibration_bins.append({
                'bin_start': i / n_bins,
                'bin_end': (i + 1) / n_bins,
                'avg_predicted_prob': avg_conf,
                'avg_actual_outcome': avg_actual,
                'n_samples': len(bin_records),
                'calibration_error': bin_error
            })
    
    metrics.expected_calibration_error = ece / n
    metrics.max_calibration_error = mce
    metrics.calibration_bins = calibration_bins
    
    # Confidence analysis
    metrics.avg_confidence = sum(r['confidence'] for r in records_with_conf) / n
    
    correct_records = [r for r in records_with_conf if r['correct']]
    wrong_records = [r for r in records_with_conf if not r['correct']]
    
    if correct_records:
        metrics.avg_confidence_when_correct = sum(
            r['confidence'] for r in correct_records
        ) / len(correct_records)
    
    if wrong_records:
        metrics.avg_confidence_when_wrong = sum(
            r['confidence'] for r in wrong_records
        ) / len(wrong_records)
        metrics.overconfidence_rate = sum(
            1 for r in wrong_records if r['confidence'] > 0.7
        ) / len(wrong_records)
    
    if correct_records:
        metrics.underconfidence_rate = sum(
            1 for r in correct_records if r['confidence'] < 0.5
        ) / len(correct_records)
    
    return metrics


def compute_operational_metrics(data: List[Dict]) -> OperationalMetrics:
    """Compute operational and efficiency metrics."""
    metrics = OperationalMetrics()
    
    # Extract latencies
    latencies = [
        d.get('latency_sec', 0) 
        for d in data 
        if d.get('latency_sec') is not None and d.get('latency_sec') > 0
    ]
    
    if latencies:
        latencies_sorted = sorted(latencies)
        n = len(latencies)
        
        metrics.avg_latency_sec = sum(latencies) / n
        metrics.median_latency_sec = latencies_sorted[n // 2]
        metrics.min_latency_sec = latencies_sorted[0]
        metrics.max_latency_sec = latencies_sorted[-1]
        
        # Standard deviation
        mean = metrics.avg_latency_sec
        variance = sum((x - mean) ** 2 for x in latencies) / n
        metrics.std_latency_sec = math.sqrt(variance)
        
        # Percentiles
        metrics.p90_latency_sec = latencies_sorted[int(n * 0.90)]
        metrics.p95_latency_sec = latencies_sorted[int(n * 0.95)]
        metrics.p99_latency_sec = latencies_sorted[min(int(n * 0.99), n - 1)]
    
    # Tool usage
    tool_calls = [d.get('tool_call_count', 0) for d in data]
    if tool_calls:
        tool_calls_sorted = sorted(tool_calls)
        metrics.avg_tool_calls = sum(tool_calls) / len(tool_calls)
        metrics.median_tool_calls = tool_calls_sorted[len(tool_calls) // 2]
        metrics.max_tool_calls = max(tool_calls)
        metrics.zero_tool_call_rate = sum(1 for t in tool_calls if t == 0) / len(tool_calls)
    
    # Errors
    errors = [d for d in data if d.get('error')]
    metrics.total_errors = len(errors)
    metrics.error_rate = safe_divide(len(errors), len(data))

    # Classify error types — order matters (most specific first)
    error_types: Dict[str, int] = defaultdict(int)
    timeout_count       = 0
    token_limit_count   = 0
    context_overflow_count = 0

    for d in errors:
        raw = str(d.get('error', ''))
        msg = raw.lower()

        if raw == 'TOKEN_LIMIT' or raw == 'token_limit':
            # finish_reason == "length": generation was cut off mid-stream
            error_types['token_limit'] += 1
            token_limit_count += 1
        elif raw == 'TIMEOUT' or msg == 'timeout':
            error_types['timeout'] += 1
            timeout_count += 1
        elif raw == 'Iteration limit reached' or 'iteration limit' in msg:
            error_types['iteration_limit'] += 1
        elif raw in ('CONTEXT_LIMIT', 'context_limit') or \
                'input_tokens' in msg or 'context length' in msg or 'maximum input length' in msg:
            # CONTEXT_LIMIT = guard triggered cleanly; 400 BadRequest = guard missed
            error_types['context_overflow'] += 1
            context_overflow_count += 1
        elif 'recursion' in msg:
            error_types['recursion_limit'] += 1
        elif 'rate' in msg:
            error_types['rate_limit'] += 1
        else:
            error_types['other'] += 1

    metrics.timeout_rate        = safe_divide(timeout_count, len(data))
    metrics.token_limit_rate    = safe_divide(token_limit_count, len(data))
    metrics.context_overflow_rate = safe_divide(context_overflow_count, len(data))
    metrics.error_types = dict(error_types)
    
    return metrics


def compute_topic_metrics(data: List[Dict]) -> Dict[str, TopicMetrics]:
    """Compute metrics broken down by topic."""
    by_topic = defaultdict(list)
    for record in data:
        topic = record.get('topic', 'unknown')
        by_topic[topic].append(record)
    
    topic_metrics = {}
    
    for topic, records in by_topic.items():
        tm = TopicMetrics(topic=topic)
        tm.n_samples = len(records)
        
        # Compute confusion matrix for topic
        for record in records:
            gt = record.get('ground_truth')
            pred = record.get('predicted')
            
            if pred is None:
                tm.n_null += 1
                continue
            
            tm.n_valid += 1
            
            if gt == 'Yes' and pred == 'Yes':
                tm.tp += 1
                tm.n_correct += 1
            elif gt == 'No' and pred == 'No':
                tm.tn += 1
                tm.n_correct += 1
            elif gt == 'No' and pred == 'Yes':
                tm.fp += 1
            elif gt == 'Yes' and pred == 'No':
                tm.fn += 1
        
        # Compute metrics
        tm.accuracy = safe_divide(tm.n_correct, tm.n_valid)
        
        # F1 for Yes
        precision_yes = safe_divide(tm.tp, tm.tp + tm.fp)
        recall_yes = safe_divide(tm.tp, tm.tp + tm.fn)
        if precision_yes + recall_yes > 0:
            tm.f1_yes = 2 * precision_yes * recall_yes / (precision_yes + recall_yes)
        
        # F1 for No
        precision_no = safe_divide(tm.tn, tm.tn + tm.fn)
        recall_no = safe_divide(tm.tn, tm.tn + tm.fp)
        if precision_no + recall_no > 0:
            tm.f1_no = 2 * precision_no * recall_no / (precision_no + recall_no)
        
        tm.f1_macro = (tm.f1_yes + tm.f1_no) / 2
        
        topic_metrics[topic] = tm
    
    return topic_metrics


def analyze_error_patterns(data: List[Dict]) -> Dict[str, int]:
    """Analyze common error patterns in predictions."""
    patterns = defaultdict(int)
    
    for record in data:
        if record.get('correct') == False:
            final_answer = record.get('final_answer', '').lower()
            
            # Absence of evidence pattern
            if any(phrase in final_answer for phrase in [
                'no evidence', 'absence of', 'lack of', 'not found',
                'no results', 'no posts', 'no confirmation', 'no data'
            ]):
                patterns['absence_of_evidence_fallacy'] += 1
            
            # Negation confusion
            question = record.get('question', '').lower()
            if 'not ' in question or "won't" in question or "will not" in question:
                patterns['negation_in_question'] += 1
            
            # High tool call count (possible thrashing)
            if record.get('tool_call_count', 0) > 15:
                patterns['excessive_tool_calls'] += 1
            
            # Zero tool calls (possible truncation)
            if record.get('tool_call_count', 0) == 0:
                if '<tool_call>' in record.get('final_answer', ''):
                    patterns['truncated_tool_calls'] += 1
                else:
                    patterns['no_tool_calls'] += 1
        
        # Null predictions
        if record.get('predicted') is None:
            if record.get('error'):
                patterns['null_due_to_error'] += 1
            elif '<tool_call>' in record.get('final_answer', ''):
                patterns['null_due_to_truncation'] += 1
            else:
                patterns['null_unknown_cause'] += 1
    
    return dict(patterns)


def _pearson_r(xs: List[float], ys: List[float]) -> Optional[float]:
    """Pearson correlation coefficient, or None if degenerate."""
    n = len(xs)
    if n < 3:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov  = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if std_x == 0 or std_y == 0:
        return None
    return cov / (std_x * std_y)


def compute_timeseries_metrics(data: List[Dict]) -> TimeSeriesMetrics:
    """Compute metrics specific to time-series evaluation.

    Detects time-series records by the presence of 'timepoint_index' field.
    Groups records by question_id to evaluate within-question consistency
    and majority-vote accuracy.
    """
    metrics = TimeSeriesMetrics()

    ts_records = [r for r in data if r.get('timepoint_index') is not None]
    if not ts_records:
        return metrics

    metrics.available = True
    metrics.n_total_timepoints = len(ts_records)

    # Group by question_id
    by_question: Dict[str, List[Dict]] = defaultdict(list)
    for r in ts_records:
        by_question[str(r['question_id'])].append(r)

    metrics.n_questions = len(by_question)
    metrics.avg_timepoints_per_question = metrics.n_total_timepoints / metrics.n_questions

    # Sort each question's timepoints by index
    for qid in by_question:
        by_question[qid].sort(key=lambda r: r.get('timepoint_index', 0))

    # Collect first and last timepoint records
    first_recs = [recs[0]  for recs in by_question.values() if recs]
    last_recs  = [recs[-1] for recs in by_question.values() if recs]

    def _acc(recs: List[Dict]) -> Optional[float]:
        valid = [r for r in recs if r.get('predicted') is not None and r.get('correct') is not None]
        if not valid:
            return None
        return sum(1 for r in valid if r['correct']) / len(valid)

    metrics.accuracy_first_timepoint = _acc(first_recs)
    metrics.accuracy_last_timepoint  = _acc(last_recs)

    # Early / late halves by timepoint_index relative to total_timepoints
    early, late = [], []
    for r in ts_records:
        idx   = r.get('timepoint_index', 1)
        total = r.get('total_timepoints', 1)
        mid   = total / 2.0
        if idx <= mid:
            early.append(r)
        else:
            late.append(r)

    metrics.accuracy_early_half = _acc(early) if early else None
    metrics.accuracy_late_half  = _acc(late)  if late  else None

    # Consistency: all timepoints for a question predict the same option
    consistent = 0
    for recs in by_question.values():
        preds = [r.get('predicted') for r in recs if r.get('predicted') is not None]
        if preds and len(set(preds)) == 1:
            consistent += 1

    metrics.consistency_rate = safe_divide(consistent, metrics.n_questions)

    # Majority-vote accuracy: most common prediction per question vs ground truth
    mv_correct = 0
    mv_total   = 0
    for recs in by_question.values():
        gt = recs[0].get('ground_truth')
        if gt is None:
            continue
        preds = [r.get('predicted') for r in recs if r.get('predicted') is not None]
        if not preds:
            continue
        # majority vote
        vote: Dict[str, int] = defaultdict(int)
        for p in preds:
            vote[p] += 1
        majority = max(vote, key=lambda k: vote[k])
        mv_total += 1
        if majority == gt:
            mv_correct += 1

    metrics.majority_vote_accuracy = safe_divide(mv_correct, mv_total)

    return metrics


def compute_market_comparison_metrics(data: List[Dict]) -> MarketComparisonMetrics:
    """Compare agent P(Yes) estimates against prediction market probabilities.

    Requires both 'agent_prob_yes' and 'human_prob_yes' fields.
    Only records where both are non-null and ground_truth is known are used.
    """
    metrics = MarketComparisonMetrics()

    valid = [
        r for r in data
        if extract_prob_yes_from_record(r) is not None
        and r.get('human_prob_yes') is not None
        and r.get('ground_truth') is not None
    ]
    if len(valid) < 5:
        return metrics

    metrics.available = True
    metrics.n_samples = len(valid)

    agent_probs  = [extract_prob_yes_from_record(r) for r in valid]
    market_probs = [float(r['human_prob_yes'])      for r in valid]
    actuals      = [1 if r['ground_truth'] == 'Yes' else 0 for r in valid]

    n = len(valid)

    # Brier scores: BS = mean (p_yes - actual)^2
    agent_bs  = sum((a - y) ** 2 for a, y in zip(agent_probs,  actuals)) / n
    market_bs = sum((m - y) ** 2 for m, y in zip(market_probs, actuals)) / n
    metrics.agent_brier_score  = round(agent_bs,  4)
    metrics.market_brier_score = round(market_bs, 4)

    # Brier Skill Score: 1 - (agent_BS / market_BS)
    # Positive = agent beats market; negative = agent worse than market
    if market_bs > 0:
        metrics.brier_skill_score = round(1.0 - agent_bs / market_bs, 4)

    # Pearson correlation between agent and market probability tracks
    metrics.prob_correlation = _pearson_r(agent_probs, market_probs)
    if metrics.prob_correlation is not None:
        metrics.prob_correlation = round(metrics.prob_correlation, 4)

    # Mean absolute deviation
    metrics.mean_absolute_deviation = round(
        sum(abs(a - m) for a, m in zip(agent_probs, market_probs)) / n, 4
    )

    # Directional agreement: both above 0.5 or both below 0.5
    agree = sum(
        1 for a, m in zip(agent_probs, market_probs)
        if (a > 0.5) == (m > 0.5)
    )
    metrics.directional_agreement_rate = round(safe_divide(agree, n), 4)

    return metrics


def compute_reflection_quality_metrics(data: List[Dict]) -> ReflectionQualityMetrics:
    """Analyse how reflection and self-critique correlate with prediction quality."""
    metrics = ReflectionQualityMetrics()

    # Only records where we have a definitive correct/incorrect label
    scored = [r for r in data if r.get('correct') is not None]
    if not scored:
        return metrics

    # Check if reflection fields are present in the dataset at all
    has_field = any('reflection' in r or 'self_critique' in r for r in data)
    if not has_field:
        return metrics

    metrics.available = True
    metrics.n_total = len(scored)

    with_refl   = [r for r in scored if r.get('reflection')]
    with_crit   = [r for r in scored if r.get('self_critique')]
    without_refl = [r for r in scored if not r.get('reflection')]
    without_crit = [r for r in scored if not r.get('self_critique')]

    metrics.n_with_reflection   = len(with_refl)
    metrics.n_with_self_critique = len(with_crit)
    metrics.reflection_rate     = round(safe_divide(len(with_refl), metrics.n_total), 4)
    metrics.self_critique_rate  = round(safe_divide(len(with_crit), metrics.n_total), 4)

    def _accuracy(recs: List[Dict]) -> Optional[float]:
        valid = [r for r in recs if r.get('predicted') is not None]
        if not valid:
            return None
        return round(sum(1 for r in valid if r['correct']) / len(valid), 4)

    if with_refl:
        metrics.accuracy_with_reflection    = _accuracy(with_refl)
    if without_refl:
        metrics.accuracy_without_reflection = _accuracy(without_refl)
    if with_crit:
        metrics.accuracy_with_self_critique    = _accuracy(with_crit)
    if without_crit:
        metrics.accuracy_without_self_critique = _accuracy(without_crit)

    return metrics


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def evaluate(data: List[Dict], input_file: str = "") -> EvaluationReport:
    """
    Run complete evaluation on prediction data.
    
    Args:
        data: List of prediction records
        input_file: Path to input file (for metadata)
    
    Returns:
        Complete EvaluationReport
    """
    report = EvaluationReport()
    report.timestamp = datetime.now().isoformat()
    report.input_file = input_file
    report.total_questions = len(data)
    
    # Compute all metrics
    report.classification    = compute_classification_metrics(data)
    report.calibration       = compute_calibration_metrics(data)
    report.operational       = compute_operational_metrics(data)
    report.timeseries        = compute_timeseries_metrics(data)
    report.market_comparison = compute_market_comparison_metrics(data)
    report.reflection_quality = compute_reflection_quality_metrics(data)
    report.by_topic          = compute_topic_metrics(data)
    report.common_error_patterns = analyze_error_patterns(data)

    return report


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def format_text_report(report: EvaluationReport) -> str:
    """Format evaluation report as readable text."""
    lines = []
    c = report.classification
    cm = c.confusion_matrix
    cal = report.calibration
    op = report.operational
    
    lines.append("=" * 70)
    lines.append("P/D AGENT FORECASTING EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {report.timestamp}")
    lines.append(f"Input: {report.input_file}")
    lines.append(f"Total questions: {report.total_questions}")
    lines.append("")
    
    # Confusion Matrix
    lines.append("CONFUSION MATRIX")
    lines.append("-" * 50)
    lines.append(f"                      Predicted")
    lines.append(f"                      Yes      No")
    lines.append(f"Actual Yes            {cm.tp:4d}    {cm.fn:4d}    (n={cm.positive_actual})")
    lines.append(f"Actual No             {cm.fp:4d}    {cm.tn:4d}    (n={cm.negative_actual})")
    lines.append(f"")
    lines.append(f"Valid predictions: {c.valid_predictions}")
    lines.append(f"Null predictions:  {c.null_predictions}")
    lines.append("")
    
    # Core Metrics
    lines.append("CLASSIFICATION METRICS")
    lines.append("-" * 50)
    lines.append(f"Accuracy (all):              {c.accuracy:.1%}")
    lines.append(f"Accuracy (excl. nulls):      {c.accuracy_excluding_nulls:.1%}")
    lines.append(f"Balanced Accuracy:           {c.balanced_accuracy:.1%}")
    lines.append("")
    
    # Per-class metrics table
    lines.append(f"{'Metric':<20} {'Yes':>12} {'No':>12}")
    lines.append(f"{'Precision':<20} {c.yes_class.precision:>12.3f} {c.no_class.precision:>12.3f}")
    lines.append(f"{'Recall':<20} {c.yes_class.recall:>12.3f} {c.no_class.recall:>12.3f}")
    lines.append(f"{'F1 Score':<20} {c.yes_class.f1:>12.3f} {c.no_class.f1:>12.3f}")
    lines.append(f"{'Support':<20} {c.yes_class.support:>12d} {c.no_class.support:>12d}")
    lines.append("")
    
    # Aggregate metrics
    lines.append("AGGREGATE METRICS")
    lines.append("-" * 50)
    lines.append(f"F1 Macro:                    {c.f1_macro:.4f}")
    lines.append(f"F1 Micro:                    {c.f1_micro:.4f}")
    lines.append(f"F1 Weighted:                 {c.f1_weighted:.4f}")
    lines.append(f"Matthews Correlation (MCC):  {c.matthews_correlation_coefficient:.4f}")
    lines.append(f"Cohen's Kappa:               {c.cohens_kappa:.4f}")
    lines.append("")
    
    # Bias analysis
    lines.append("BIAS ANALYSIS")
    lines.append("-" * 50)
    lines.append(f"Ground Truth Yes Rate:       {c.yes_rate_ground_truth:.1%}")
    lines.append(f"Predicted Yes Rate:          {c.yes_rate_predicted:.1%}")
    lines.append(f"Prediction Bias:             {c.prediction_bias:+.1%}")
    if c.prediction_bias < -0.1:
        lines.append("⚠️  Strong NO bias detected")
    elif c.prediction_bias > 0.1:
        lines.append("⚠️  Strong YES bias detected")
    lines.append("")
    
    # Calibration metrics
    lines.append("CALIBRATION METRICS")
    lines.append("-" * 50)
    if cal.available:
        lines.append(f"Samples with confidence:     {cal.n_samples_with_confidence}")
        lines.append(f"Brier Score:                 {cal.brier_score:.4f}  (lower is better)")
        lines.append(f"Log Loss:                    {cal.log_loss:.4f}  (lower is better)")
        lines.append(f"Expected Calibration Error:  {cal.expected_calibration_error:.4f}")
        lines.append(f"Max Calibration Error:       {cal.max_calibration_error:.4f}")
        lines.append(f"Avg Confidence:              {cal.avg_confidence:.1%}")
        if cal.avg_confidence_when_correct is not None:
            lines.append(f"Avg Confidence (correct):    {cal.avg_confidence_when_correct:.1%}")
        if cal.avg_confidence_when_wrong is not None:
            lines.append(f"Avg Confidence (wrong):      {cal.avg_confidence_when_wrong:.1%}")
        if cal.overconfidence_rate is not None:
            lines.append(f"Overconfidence Rate:         {cal.overconfidence_rate:.1%}")
    else:
        lines.append("Not available - add 'confidence' field to P agent output")
        lines.append("Format: 'Confidence: 0.75' in final_answer or 'confidence' JSON field")
    lines.append("")
    
    # Operational metrics
    lines.append("OPERATIONAL METRICS")
    lines.append("-" * 50)
    lines.append(f"Avg Latency:                 {op.avg_latency_sec:.1f}s")
    lines.append(f"Median Latency:              {op.median_latency_sec:.1f}s")
    lines.append(f"P95 Latency:                 {op.p95_latency_sec:.1f}s")
    lines.append(f"Max Latency:                 {op.max_latency_sec:.1f}s")
    lines.append(f"Avg Tool Calls:              {op.avg_tool_calls:.1f}")
    lines.append(f"Max Tool Calls:              {op.max_tool_calls}")
    lines.append(f"Zero Tool Call Rate:         {op.zero_tool_call_rate:.1%}")
    lines.append(f"Error Rate:                  {op.error_rate:.1%}  ({op.total_errors} total)")
    lines.append(f"Timeout Rate:                {op.timeout_rate:.1%}")
    lines.append(f"Token-Limit Rate:            {op.token_limit_rate:.1%}")
    lines.append(f"Context-Overflow Rate:       {op.context_overflow_rate:.1%}")
    if op.error_types:
        lines.append(f"Error breakdown:             {op.error_types}")
    lines.append("")

    # Time-series metrics
    ts = report.timeseries
    if ts.available:
        lines.append("TIME-SERIES METRICS")
        lines.append("-" * 50)
        lines.append(f"Questions:                   {ts.n_questions}")
        lines.append(f"Total timepoints:            {ts.n_total_timepoints}")
        lines.append(f"Avg timepoints/question:     {ts.avg_timepoints_per_question:.1f}")
        if ts.accuracy_first_timepoint is not None:
            lines.append(f"Accuracy (first timepoint):  {ts.accuracy_first_timepoint:.1%}")
        if ts.accuracy_last_timepoint is not None:
            lines.append(f"Accuracy (last timepoint):   {ts.accuracy_last_timepoint:.1%}")
        if ts.accuracy_early_half is not None:
            lines.append(f"Accuracy (early half):       {ts.accuracy_early_half:.1%}")
        if ts.accuracy_late_half is not None:
            lines.append(f"Accuracy (late half):        {ts.accuracy_late_half:.1%}")
        lines.append(f"Consistency rate:            {ts.consistency_rate:.1%}  "
                     f"(all timepoints agree per question)")
        lines.append(f"Majority-vote accuracy:      {ts.majority_vote_accuracy:.1%}  "
                     f"(per-question majority vote vs ground truth)")
        lines.append("")

    # Market comparison
    mc = report.market_comparison
    if mc.available:
        lines.append("MARKET COMPARISON")
        lines.append("-" * 50)
        lines.append(f"Samples with market probs:   {mc.n_samples}")
        lines.append(f"Agent Brier Score:           {mc.agent_brier_score:.4f}  "
                     f"(lower is better; random=0.25)")
        lines.append(f"Market Brier Score:          {mc.market_brier_score:.4f}")
        if mc.brier_skill_score is not None:
            sign = "+" if mc.brier_skill_score >= 0 else ""
            verdict = "agent beats market" if mc.brier_skill_score > 0 else "market beats agent"
            lines.append(f"Brier Skill Score:           {sign}{mc.brier_skill_score:.4f}  ({verdict})")
        if mc.prob_correlation is not None:
            lines.append(f"Prob. Correlation (r):       {mc.prob_correlation:.4f}  "
                         f"(agent_prob_yes vs human_prob_yes)")
        if mc.mean_absolute_deviation is not None:
            lines.append(f"Mean Abs. Deviation:         {mc.mean_absolute_deviation:.4f}  "
                         f"(avg |agent_p_yes - market_p_yes|)")
        if mc.directional_agreement_rate is not None:
            lines.append(f"Directional Agreement:       {mc.directional_agreement_rate:.1%}  "
                         f"(both same side of 50%)")
        lines.append("")

    # Reflection quality
    rq = report.reflection_quality
    if rq.available:
        lines.append("REFLECTION & SELF-CRITIQUE QUALITY")
        lines.append("-" * 50)
        lines.append(f"Reflection rate:             {rq.reflection_rate:.1%}  "
                     f"({rq.n_with_reflection}/{rq.n_total})")
        lines.append(f"Self-critique rate:          {rq.self_critique_rate:.1%}  "
                     f"({rq.n_with_self_critique}/{rq.n_total})")
        if rq.accuracy_with_reflection is not None and rq.accuracy_without_reflection is not None:
            delta = rq.accuracy_with_reflection - rq.accuracy_without_reflection
            sign  = "+" if delta >= 0 else ""
            lines.append(f"Acc. with reflection:        {rq.accuracy_with_reflection:.1%}  "
                         f"({sign}{delta:.1%} vs without)")
        if rq.accuracy_with_self_critique is not None and rq.accuracy_without_self_critique is not None:
            delta = rq.accuracy_with_self_critique - rq.accuracy_without_self_critique
            sign  = "+" if delta >= 0 else ""
            lines.append(f"Acc. with self-critique:     {rq.accuracy_with_self_critique:.1%}  "
                         f"({sign}{delta:.1%} vs without)")
        lines.append("")

    # By topic
    lines.append("METRICS BY TOPIC")
    lines.append("-" * 50)
    lines.append(f"{'Topic':<20} {'Acc':>8} {'F1-Yes':>8} {'F1-No':>8} {'F1-M':>8} {'N':>6}")
    for topic, tm in sorted(report.by_topic.items(), key=lambda x: x[1].n_samples, reverse=True):
        lines.append(
            f"{topic:<20} {tm.accuracy:>8.1%} {tm.f1_yes:>8.3f} {tm.f1_no:>8.3f} "
            f"{tm.f1_macro:>8.3f} {tm.n_samples:>6}"
        )
    lines.append("")

    # Error patterns
    if report.common_error_patterns:
        lines.append("ERROR PATTERN ANALYSIS")
        lines.append("-" * 50)
        for pattern, count in sorted(report.common_error_patterns.items(), key=lambda x: -x[1]):
            lines.append(f"{pattern:<35} {count:>5}")
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


def format_csv_report(report: EvaluationReport) -> str:
    """Format key metrics as CSV for easy import to spreadsheets."""
    lines = []
    c = report.classification
    
    # Header
    lines.append("metric,value,category")
    
    # Classification metrics
    lines.append(f"accuracy,{c.accuracy:.4f},classification")
    lines.append(f"accuracy_excl_null,{c.accuracy_excluding_nulls:.4f},classification")
    lines.append(f"balanced_accuracy,{c.balanced_accuracy:.4f},classification")
    lines.append(f"precision_yes,{c.yes_class.precision:.4f},classification")
    lines.append(f"recall_yes,{c.yes_class.recall:.4f},classification")
    lines.append(f"f1_yes,{c.yes_class.f1:.4f},classification")
    lines.append(f"precision_no,{c.no_class.precision:.4f},classification")
    lines.append(f"recall_no,{c.no_class.recall:.4f},classification")
    lines.append(f"f1_no,{c.no_class.f1:.4f},classification")
    lines.append(f"f1_macro,{c.f1_macro:.4f},classification")
    lines.append(f"f1_micro,{c.f1_micro:.4f},classification")
    lines.append(f"f1_weighted,{c.f1_weighted:.4f},classification")
    lines.append(f"mcc,{c.matthews_correlation_coefficient:.4f},classification")
    lines.append(f"cohens_kappa,{c.cohens_kappa:.4f},classification")
    lines.append(f"prediction_bias,{c.prediction_bias:.4f},bias")
    
    # Calibration metrics
    cal = report.calibration
    if cal.available:
        lines.append(f"brier_score,{cal.brier_score:.4f},calibration")
        lines.append(f"log_loss,{cal.log_loss:.4f},calibration")
        lines.append(f"ece,{cal.expected_calibration_error:.4f},calibration")
    
    # Operational metrics
    op = report.operational
    lines.append(f"avg_latency_sec,{op.avg_latency_sec:.2f},operational")
    lines.append(f"p95_latency_sec,{op.p95_latency_sec:.2f},operational")
    lines.append(f"avg_tool_calls,{op.avg_tool_calls:.2f},operational")
    lines.append(f"error_rate,{op.error_rate:.4f},operational")
    lines.append(f"token_limit_rate,{op.token_limit_rate:.4f},operational")
    lines.append(f"context_overflow_rate,{op.context_overflow_rate:.4f},operational")

    # Time-series metrics
    ts = report.timeseries
    if ts.available:
        lines.append(f"ts_n_questions,{ts.n_questions},timeseries")
        lines.append(f"ts_consistency_rate,{ts.consistency_rate:.4f},timeseries")
        lines.append(f"ts_majority_vote_accuracy,{ts.majority_vote_accuracy:.4f},timeseries")
        if ts.accuracy_first_timepoint is not None:
            lines.append(f"ts_accuracy_first,{ts.accuracy_first_timepoint:.4f},timeseries")
        if ts.accuracy_last_timepoint is not None:
            lines.append(f"ts_accuracy_last,{ts.accuracy_last_timepoint:.4f},timeseries")
        if ts.accuracy_early_half is not None:
            lines.append(f"ts_accuracy_early_half,{ts.accuracy_early_half:.4f},timeseries")
        if ts.accuracy_late_half is not None:
            lines.append(f"ts_accuracy_late_half,{ts.accuracy_late_half:.4f},timeseries")

    # Market comparison metrics
    mc = report.market_comparison
    if mc.available:
        lines.append(f"agent_brier_score,{mc.agent_brier_score:.4f},market_comparison")
        lines.append(f"market_brier_score,{mc.market_brier_score:.4f},market_comparison")
        if mc.brier_skill_score is not None:
            lines.append(f"brier_skill_score,{mc.brier_skill_score:.4f},market_comparison")
        if mc.prob_correlation is not None:
            lines.append(f"prob_correlation,{mc.prob_correlation:.4f},market_comparison")
        if mc.mean_absolute_deviation is not None:
            lines.append(f"mean_abs_deviation,{mc.mean_absolute_deviation:.4f},market_comparison")
        if mc.directional_agreement_rate is not None:
            lines.append(f"directional_agreement,{mc.directional_agreement_rate:.4f},market_comparison")

    # Reflection quality
    rq = report.reflection_quality
    if rq.available:
        lines.append(f"reflection_rate,{rq.reflection_rate:.4f},reflection")
        lines.append(f"self_critique_rate,{rq.self_critique_rate:.4f},reflection")
        if rq.accuracy_with_reflection is not None:
            lines.append(f"accuracy_with_reflection,{rq.accuracy_with_reflection:.4f},reflection")
        if rq.accuracy_without_reflection is not None:
            lines.append(f"accuracy_without_reflection,{rq.accuracy_without_reflection:.4f},reflection")
        if rq.accuracy_with_self_critique is not None:
            lines.append(f"accuracy_with_self_critique,{rq.accuracy_with_self_critique:.4f},reflection")
        if rq.accuracy_without_self_critique is not None:
            lines.append(f"accuracy_without_self_critique,{rq.accuracy_without_self_critique:.4f},reflection")

    return "\n".join(lines)


def report_to_dict(report: EvaluationReport) -> Dict[str, Any]:
    """Convert report to dictionary for JSON serialization."""
    def dataclass_to_dict(obj):
        if hasattr(obj, '__dataclass_fields__'):
            result = {}
            for field_name in obj.__dataclass_fields__:
                value = getattr(obj, field_name)
                result[field_name] = dataclass_to_dict(value)
            return result
        elif isinstance(obj, dict):
            return {k: dataclass_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [dataclass_to_dict(v) for v in obj]
        else:
            return obj
    
    return dataclass_to_dict(report)


# =============================================================================
# CROSS-SETUP STATISTICAL COMPARISON
# =============================================================================

def _load_predictions_by_key(filepath: str) -> Dict[str, Dict]:
    """Load JSONL, index by 'question_id:timepoint_date' for paired matching."""
    preds = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = str(rec.get("question_id", ""))
            tdate = rec.get("timepoint_date", rec.get("cutoff_date", ""))
            key = f"{qid}:{tdate}"
            preds[key] = rec
    return preds


def _paired_brier_scores(records_a: List[Dict], records_b: List[Dict]
                         ) -> Tuple[List[float], List[float]]:
    """Extract matched Brier scores from two lists of paired records."""
    brier_a, brier_b = [], []
    for ra, rb in zip(records_a, records_b):
        gt = 1.0 if ra["ground_truth"] == "Yes" else 0.0
        pa = ra.get("agent_prob_yes")
        pb = rb.get("agent_prob_yes")
        if pa is not None and pb is not None:
            brier_a.append((pa - gt) ** 2)
            brier_b.append((pb - gt) ** 2)
    return brier_a, brier_b


def bootstrap_ci(
    scores_a: List[float],
    scores_b: List[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
) -> Dict[str, float]:
    """
    Paired bootstrap confidence interval on mean(scores_a) - mean(scores_b).
    Returns: mean_diff, ci_lower, ci_upper, p_value (two-sided).
    """
    import random
    n = len(scores_a)
    if n == 0:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0, "n": 0}

    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    observed_diff = sum(diffs) / n

    boot_diffs = []
    for _ in range(n_bootstrap):
        sample = random.choices(diffs, k=n)
        boot_diffs.append(sum(sample) / n)
    boot_diffs.sort()

    alpha = 1 - ci
    lo_idx = int(n_bootstrap * alpha / 2)
    hi_idx = int(n_bootstrap * (1 - alpha / 2))
    ci_lower = boot_diffs[lo_idx]
    ci_upper = boot_diffs[min(hi_idx, n_bootstrap - 1)]

    # Two-sided p-value: proportion of bootstrap samples on the opposite side of zero
    if observed_diff >= 0:
        p_value = sum(1 for d in boot_diffs if d <= 0) / n_bootstrap
    else:
        p_value = sum(1 for d in boot_diffs if d >= 0) / n_bootstrap
    p_value = min(p_value * 2, 1.0)  # two-sided

    return {
        "mean_diff": round(observed_diff, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "p_value": round(p_value, 4),
        "n": n,
    }


def mcnemar_test(correct_a: List[bool], correct_b: List[bool]) -> Dict[str, float]:
    """
    McNemar's test for paired binary classification.
    Tests whether two setups disagree on significantly different cases.
    """
    # Count discordant pairs
    b_count = 0  # A correct, B wrong
    c_count = 0  # A wrong, B correct
    for a, b in zip(correct_a, correct_b):
        if a and not b:
            b_count += 1
        elif not a and b:
            c_count += 1

    n_discordant = b_count + c_count
    if n_discordant == 0:
        return {"chi2": 0.0, "p_value": 1.0, "n_discordant": 0,
                "a_only_correct": b_count, "b_only_correct": c_count}

    # McNemar chi-squared with continuity correction
    chi2 = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)

    # Approximate p-value from chi2 distribution (1 df)
    # Using survival function approximation
    p_value = math.exp(-chi2 / 2)  # conservative approximation for chi2(1)

    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 4),
        "n_discordant": n_discordant,
        "a_only_correct": b_count,
        "b_only_correct": c_count,
    }


def wilcoxon_signed_rank(diffs: List[float]) -> Dict[str, float]:
    """
    Wilcoxon signed-rank test on paired differences.
    Tests whether the median difference is significantly different from zero.
    Uses normal approximation for n > 20.
    """
    # Remove zeros
    nonzero = [(abs(d), 1 if d > 0 else -1) for d in diffs if d != 0]
    n = len(nonzero)
    if n == 0:
        return {"W": 0.0, "z": 0.0, "p_value": 1.0, "n": 0}

    # Rank by absolute value
    nonzero.sort(key=lambda x: x[0])
    ranks = list(range(1, n + 1))

    # Handle ties (average rank)
    i = 0
    while i < n:
        j = i + 1
        while j < n and nonzero[j][0] == nonzero[i][0]:
            j += 1
        if j > i + 1:
            avg_rank = sum(ranks[i:j]) / (j - i)
            for k in range(i, j):
                ranks[k] = avg_rank
        i = j

    # W+ = sum of ranks for positive differences
    w_plus = sum(ranks[i] for i in range(n) if nonzero[i][1] > 0)
    w_minus = sum(ranks[i] for i in range(n) if nonzero[i][1] < 0)
    W = min(w_plus, w_minus)

    # Normal approximation (n > 20)
    if n > 20:
        mean_w = n * (n + 1) / 4
        std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = (W - mean_w) / std_w if std_w > 0 else 0.0
        # Two-sided p-value from normal approximation
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    else:
        # For small n, report W statistic without p-value approximation
        z = 0.0
        p_value = None  # would need exact tables

    return {
        "W": round(W, 4),
        "z": round(z, 4),
        "p_value": round(p_value, 4) if p_value is not None else None,
        "n": n,
        "W_plus": round(w_plus, 4),
        "W_minus": round(w_minus, 4),
    }


def compare_setups(file_a: str, file_b: str, n_bootstrap: int = 10000) -> str:
    """
    Statistical comparison between two result files.
    Returns formatted text report with bootstrap CI, McNemar, and Wilcoxon tests.
    """
    preds_a = _load_predictions_by_key(file_a)
    preds_b = _load_predictions_by_key(file_b)

    # Find matched keys
    common_keys = sorted(set(preds_a.keys()) & set(preds_b.keys()))
    if not common_keys:
        return "ERROR: No matched question-timepoint pairs found between the two files."

    matched_a = [preds_a[k] for k in common_keys]
    matched_b = [preds_b[k] for k in common_keys]

    # Setup names from records
    name_a = file_a.split("/")[-1]
    name_b = file_b.split("/")[-1]
    setup_a = matched_a[0].get("setup", "?")
    setup_b = matched_b[0].get("setup", "?")

    lines = []
    lines.append("=" * 65)
    lines.append(f"STATISTICAL COMPARISON")
    lines.append(f"  A: {name_a}  (setup {setup_a})")
    lines.append(f"  B: {name_b}  (setup {setup_b})")
    lines.append(f"  Matched pairs: {len(common_keys)}")
    lines.append("=" * 65)

    # 1. Paired Bootstrap on Brier Score
    brier_a, brier_b = _paired_brier_scores(matched_a, matched_b)
    if brier_a:
        boot = bootstrap_ci(brier_a, brier_b, n_bootstrap=n_bootstrap)
        lines.append(f"\n--- Brier Score (Paired Bootstrap, n={boot['n']}) ---")
        lines.append(f"  Mean Brier A:  {sum(brier_a)/len(brier_a):.4f}")
        lines.append(f"  Mean Brier B:  {sum(brier_b)/len(brier_b):.4f}")
        lines.append(f"  Δ (A - B):     {boot['mean_diff']:+.4f}  "
                      f"95% CI [{boot['ci_lower']:+.4f}, {boot['ci_upper']:+.4f}]")
        lines.append(f"  p-value:       {boot['p_value']:.4f}"
                      + ("  *" if boot['p_value'] < 0.05 else "")
                      + ("  **" if boot['p_value'] < 0.01 else ""))
        if boot['mean_diff'] > 0:
            lines.append(f"  → B is better (lower Brier)")
        elif boot['mean_diff'] < 0:
            lines.append(f"  → A is better (lower Brier)")
    else:
        lines.append("\n--- Brier Score: insufficient probability data ---")

    # 2. McNemar's Test on Accuracy
    correct_a = [r.get("correct", False) is True for r in matched_a]
    correct_b = [r.get("correct", False) is True for r in matched_b]
    acc_a = sum(correct_a) / len(correct_a) if correct_a else 0
    acc_b = sum(correct_b) / len(correct_b) if correct_b else 0
    mcn = mcnemar_test(correct_a, correct_b)
    lines.append(f"\n--- Accuracy (McNemar's Test, n={len(common_keys)}) ---")
    lines.append(f"  Accuracy A:    {acc_a:.4f}")
    lines.append(f"  Accuracy B:    {acc_b:.4f}")
    lines.append(f"  Δ (A - B):     {acc_a - acc_b:+.4f}")
    lines.append(f"  A correct, B wrong: {mcn['a_only_correct']}")
    lines.append(f"  B correct, A wrong: {mcn['b_only_correct']}")
    lines.append(f"  χ²:            {mcn['chi2']:.4f}")
    lines.append(f"  p-value:       {mcn['p_value']:.4f}"
                  + ("  *" if mcn['p_value'] < 0.05 else "")
                  + ("  **" if mcn['p_value'] < 0.01 else ""))

    # 3. Wilcoxon Signed-Rank on per-question Brier differences
    if brier_a:
        brier_diffs = [a - b for a, b in zip(brier_a, brier_b)]
        wilc = wilcoxon_signed_rank(brier_diffs)
        lines.append(f"\n--- Brier Difference (Wilcoxon Signed-Rank, n={wilc['n']}) ---")
        lines.append(f"  W+:            {wilc['W_plus']:.1f}  (A worse)")
        lines.append(f"  W-:            {wilc['W_minus']:.1f}  (B worse)")
        lines.append(f"  W:             {wilc['W']:.1f}")
        if wilc['p_value'] is not None:
            lines.append(f"  z:             {wilc['z']:.4f}")
            lines.append(f"  p-value:       {wilc['p_value']:.4f}"
                          + ("  *" if wilc['p_value'] < 0.05 else "")
                          + ("  **" if wilc['p_value'] < 0.01 else ""))

    lines.append("\n" + "=" * 65)
    lines.append("Significance: * p<0.05  ** p<0.01")
    lines.append("Positive Δ Brier = A has higher (worse) Brier score")
    lines.append("=" * 65)

    return "\n".join(lines)


# =============================================================================
# MAIN CLI
# =============================================================================

def load_data(filepath: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate P/D Agent Forecasting Predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval_forecasting.py results.jsonl
  python eval_forecasting.py results.jsonl --output metrics.json
  python eval_forecasting.py results.jsonl --format csv --output metrics.csv
  python eval_forecasting.py results.jsonl --format all --output-dir ./eval_results/
        """
    )
    parser.add_argument('input_file', help='JSONL file with predictions')
    parser.add_argument('--compare', metavar='FILE_B',
                        help='Second JSONL file for statistical comparison (paired tests)')
    parser.add_argument('--bootstrap-n', type=int, default=10000,
                        help='Number of bootstrap samples for CI (default 10000)')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--output-dir', help='Output directory (for --format all)')
    parser.add_argument(
        '--format', '-f',
        choices=['text', 'json', 'csv', 'all'],
        default='text',
        help='Output format (default: text)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress console output (only write to file)'
    )
    
    args = parser.parse_args()

    # --- Compare mode ---
    if args.compare:
        report = compare_setups(args.input_file, args.compare,
                                n_bootstrap=args.bootstrap_n)
        if not args.quiet:
            print(report)
        if args.output:
            Path(args.output).write_text(report)
            if not args.quiet:
                print(f"\nComparison saved to {args.output}")
        sys.exit(0)

    # Load data
    try:
        data = load_data(args.input_file)
    except FileNotFoundError:
        print(f"Error: File not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not data:
        print("Error: No data found in input file", file=sys.stderr)
        sys.exit(1)
    
    # Run evaluation
    report = evaluate(data, args.input_file)
    
    # Output
    if args.format == 'all':
        # Output all formats to directory
        output_dir = Path(args.output_dir or './eval_results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Text report
        text_report = format_text_report(report)
        (output_dir / 'report.txt').write_text(text_report)
        
        # JSON
        json_data = report_to_dict(report)
        (output_dir / 'metrics.json').write_text(json.dumps(json_data, indent=2))
        
        # CSV
        csv_data = format_csv_report(report)
        (output_dir / 'metrics.csv').write_text(csv_data)
        
        if not args.quiet:
            print(text_report)
            print(f"\nResults saved to {output_dir}/")
    
    elif args.format == 'json':
        json_data = report_to_dict(report)
        json_str = json.dumps(json_data, indent=2)
        
        if args.output:
            Path(args.output).write_text(json_str)
            if not args.quiet:
                print(f"JSON saved to {args.output}")
        else:
            print(json_str)
    
    elif args.format == 'csv':
        csv_data = format_csv_report(report)
        
        if args.output:
            Path(args.output).write_text(csv_data)
            if not args.quiet:
                print(f"CSV saved to {args.output}")
        else:
            print(csv_data)
    
    else:  # text
        text_report = format_text_report(report)
        
        if not args.quiet:
            print(text_report)
        
        if args.output:
            Path(args.output).write_text(text_report)
            if not args.quiet:
                print(f"\nReport saved to {args.output}")


if __name__ == '__main__':
    main()
