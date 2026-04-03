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
    timeout_rate: float = 0.0  # Recursion limit errors
    token_overflow_rate: float = 0.0  # Context length errors
    
    # Error breakdown
    error_types: Dict[str, int] = field(default_factory=dict)


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


def extract_confidence(record: Dict) -> Optional[float]:
    """
    Extract confidence score from a record.
    
    Checks:
    1. Direct 'confidence' field
    2. Direct 'prob_yes' field
    3. Patterns in 'final_answer' text
    """
    # Direct fields
    if 'confidence' in record and record['confidence'] is not None:
        return float(record['confidence'])
    
    if 'prob_yes' in record and record['prob_yes'] is not None:
        return float(record['prob_yes'])
    
    if 'probability' in record and record['probability'] is not None:
        return float(record['probability'])
    
    # Extract from text
    final_answer = record.get('final_answer', '')
    if not final_answer:
        return None
    
    patterns = [
        r'[Cc]onfidence[:\s]+(\d+\.?\d*)%',
        r'[Cc]onfidence[:\s]+(\d+\.?\d*)',
        r'(\d+\.?\d*)%\s*confident',
        r'[Pp]\([Yy]es\)\s*[=:]\s*(\d+\.?\d*)',
        r'[Pp]robability[:\s]+(\d+\.?\d*)%',
        r'[Pp]robability[:\s]+(\d+\.?\d*)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, final_answer)
        if match:
            value = float(match.group(1))
            if value > 1:
                value = value / 100
            return max(0.0, min(1.0, value))
    
    return None


def compute_calibration_metrics(data: List[Dict], n_bins: int = 10) -> CalibrationMetrics:
    """Compute probability calibration metrics."""
    metrics = CalibrationMetrics()
    
    # Extract records with confidence scores
    records_with_conf = []
    for record in data:
        gt = record.get('ground_truth')
        pred = record.get('predicted')
        
        if pred is None or gt is None:
            continue
        
        conf = extract_confidence(record)
        if conf is None:
            continue
        
        # Convert to P(Yes)
        if pred == 'Yes':
            prob_yes = conf
        else:
            prob_yes = 1 - conf
        
        actual = 1 if gt == 'Yes' else 0
        correct = (pred == gt)
        
        records_with_conf.append({
            'prob_yes': prob_yes,
            'actual': actual,
            'correct': correct,
            'confidence': conf if pred == 'Yes' else (1 - conf)
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
    
    # Classify error types
    error_types = defaultdict(int)
    timeout_count = 0
    token_overflow_count = 0
    
    for d in errors:
        error_msg = str(d.get('error', '')).lower()
        
        if 'recursion' in error_msg or 'limit' in error_msg:
            error_types['recursion_limit'] += 1
            timeout_count += 1
        elif 'token' in error_msg or '15361' in error_msg or 'context' in error_msg:
            error_types['token_overflow'] += 1
            token_overflow_count += 1
        elif 'timeout' in error_msg:
            error_types['timeout'] += 1
            timeout_count += 1
        elif 'rate' in error_msg:
            error_types['rate_limit'] += 1
        else:
            error_types['other'] += 1
    
    metrics.timeout_rate = safe_divide(timeout_count, len(data))
    metrics.token_overflow_rate = safe_divide(token_overflow_count, len(data))
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
    report.classification = compute_classification_metrics(data)
    report.calibration = compute_calibration_metrics(data)
    report.operational = compute_operational_metrics(data)
    report.by_topic = compute_topic_metrics(data)
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
    lines.append(f"Error Rate:                  {op.error_rate:.1%}")
    lines.append(f"Timeout Rate:                {op.timeout_rate:.1%}")
    if op.error_types:
        lines.append(f"Error breakdown:             {op.error_types}")
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
