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
class ToolEfficiencyBucket:
    """Metrics for a specific tool call range."""
    bucket_name: str = ""
    min_tools: int = 0
    max_tools: int = 0
    n_samples: int = 0
    n_correct: int = 0
    n_wrong: int = 0
    n_null: int = 0
    accuracy: float = 0.0
    f1_yes: float = 0.0
    f1_macro: float = 0.0
    avg_latency_sec: float = 0.0
    
    # Confusion matrix for this bucket
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0


@dataclass
class ToolEfficiencyAnalysis:
    """Analysis of tool call count vs accuracy."""
    # Bucketed analysis
    buckets: List[ToolEfficiencyBucket] = field(default_factory=list)
    
    # Optimal tool call analysis
    optimal_tool_range_min: int = 0
    optimal_tool_range_max: int = 0
    optimal_range_accuracy: float = 0.0
    optimal_range_f1_macro: float = 0.0
    
    # Correlation analysis
    tool_accuracy_correlation: float = 0.0  # Pearson correlation
    tool_latency_correlation: float = 0.0
    
    # Recommended cap
    recommended_recursion_cap: int = 0
    cap_reasoning: str = ""
    
    # Diminishing returns analysis
    marginal_accuracy_by_tool: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class TimePerformanceBucket:
    """Metrics for a specific time range."""
    bucket_name: str = ""
    min_sec: float = 0.0
    max_sec: float = 0.0
    n_samples: int = 0
    n_correct: int = 0
    accuracy: float = 0.0
    f1_macro: float = 0.0
    avg_tool_calls: float = 0.0


@dataclass
class TimePerformanceAnalysis:
    """Analysis of time spent vs performance for research paper."""
    # Time buckets
    time_buckets: List[TimePerformanceBucket] = field(default_factory=list)
    
    # Efficiency metrics
    accuracy_per_second: float = 0.0
    correct_answers_per_minute: float = 0.0
    total_time_sec: float = 0.0
    
    # Optimal time analysis
    optimal_time_range_min_sec: float = 0.0
    optimal_time_range_max_sec: float = 0.0
    optimal_time_accuracy: float = 0.0
    
    # Time-accuracy tradeoff
    time_accuracy_correlation: float = 0.0
    
    # For paper: performance at different time budgets
    performance_at_time_budget: List[Dict[str, Any]] = field(default_factory=list)
    
    # Efficiency frontier (Pareto optimal points)
    pareto_frontier: List[Dict[str, float]] = field(default_factory=list)
    
    # Throughput analysis
    questions_per_minute: float = 0.0
    estimated_throughput_at_cap: Dict[int, float] = field(default_factory=dict)


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
    
    # NEW: Tool efficiency analysis
    tool_efficiency: ToolEfficiencyAnalysis = field(default_factory=ToolEfficiencyAnalysis)
    
    # NEW: Time-performance analysis
    time_performance: TimePerformanceAnalysis = field(default_factory=TimePerformanceAnalysis)


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


def compute_tool_efficiency_analysis(data: List[Dict]) -> ToolEfficiencyAnalysis:
    """
    Analyze the relationship between tool call count and accuracy.
    
    This helps determine the optimal recursion cap for the agent.
    """
    analysis = ToolEfficiencyAnalysis()
    
    # Define tool call buckets
    bucket_ranges = [
        (0, 0, "0 tools"),
        (1, 3, "1-3 tools"),
        (4, 6, "4-6 tools"),
        (7, 10, "7-10 tools"),
        (11, float('inf'), "11+ tools"),
    ]
    
    # Collect data for each bucket
    for min_t, max_t, name in bucket_ranges:
        bucket = ToolEfficiencyBucket(
            bucket_name=name,
            min_tools=min_t,
            max_tools=int(max_t) if max_t != float('inf') else 999
        )
        
        bucket_records = []
        for record in data:
            tool_count = record.get('tool_call_count', 0)
            if min_t <= tool_count <= max_t:
                bucket_records.append(record)
                bucket.n_samples += 1
                
                gt = record.get('ground_truth')
                pred = record.get('predicted')
                
                if pred is None:
                    bucket.n_null += 1
                elif pred == gt:
                    bucket.n_correct += 1
                    if gt == 'Yes':
                        bucket.tp += 1
                    else:
                        bucket.tn += 1
                else:
                    bucket.n_wrong += 1
                    if pred == 'Yes':
                        bucket.fp += 1
                    else:
                        bucket.fn += 1
        
        # Compute metrics for bucket
        valid = bucket.n_samples - bucket.n_null
        if valid > 0:
            bucket.accuracy = bucket.n_correct / valid
            
            # F1-Yes
            precision_yes = safe_divide(bucket.tp, bucket.tp + bucket.fp)
            recall_yes = safe_divide(bucket.tp, bucket.tp + bucket.fn)
            bucket.f1_yes = safe_divide(
                2 * precision_yes * recall_yes,
                precision_yes + recall_yes
            )
            
            # F1-No
            precision_no = safe_divide(bucket.tn, bucket.tn + bucket.fn)
            recall_no = safe_divide(bucket.tn, bucket.tn + bucket.fp)
            f1_no = safe_divide(
                2 * precision_no * recall_no,
                precision_no + recall_no
            )
            
            bucket.f1_macro = (bucket.f1_yes + f1_no) / 2
        
        # Average latency
        latencies = [r.get('latency_sec', 0) for r in bucket_records if r.get('latency_sec')]
        if latencies:
            bucket.avg_latency_sec = sum(latencies) / len(latencies)
        
        analysis.buckets.append(bucket)
    
    # Find optimal bucket (highest accuracy with reasonable sample size)
    valid_buckets = [b for b in analysis.buckets if b.n_samples >= 10]
    if valid_buckets:
        best_bucket = max(valid_buckets, key=lambda b: b.accuracy)
        analysis.optimal_tool_range_min = best_bucket.min_tools
        analysis.optimal_tool_range_max = best_bucket.max_tools
        analysis.optimal_range_accuracy = best_bucket.accuracy
        analysis.optimal_range_f1_macro = best_bucket.f1_macro
    
    # Compute correlation between tool count and correctness
    tool_counts = []
    correct_flags = []
    for record in data:
        if record.get('predicted') is not None:
            tool_counts.append(record.get('tool_call_count', 0))
            correct_flags.append(1 if record.get('correct') else 0)
    
    if len(tool_counts) > 2:
        # Pearson correlation
        n = len(tool_counts)
        mean_tools = sum(tool_counts) / n
        mean_correct = sum(correct_flags) / n
        
        numerator = sum((t - mean_tools) * (c - mean_correct) 
                       for t, c in zip(tool_counts, correct_flags))
        denom_tools = math.sqrt(sum((t - mean_tools)**2 for t in tool_counts))
        denom_correct = math.sqrt(sum((c - mean_correct)**2 for c in correct_flags))
        
        if denom_tools > 0 and denom_correct > 0:
            analysis.tool_accuracy_correlation = numerator / (denom_tools * denom_correct)
    
    # Compute marginal accuracy by tool count
    for i in range(0, 21):  # 0 to 20 tools
        records_at_i = [r for r in data if r.get('tool_call_count', 0) == i]
        if records_at_i:
            valid = [r for r in records_at_i if r.get('predicted') is not None]
            correct = sum(1 for r in valid if r.get('correct'))
            analysis.marginal_accuracy_by_tool.append({
                'tool_count': i,
                'n_samples': len(records_at_i),
                'n_valid': len(valid),
                'accuracy': correct / len(valid) if valid else 0.0
            })
    
    # Determine recommended cap
    # Find the point where accuracy drops significantly
    if analysis.marginal_accuracy_by_tool:
        peak_acc = 0
        peak_tool = 0
        for item in analysis.marginal_accuracy_by_tool:
            if item['n_samples'] >= 5 and item['accuracy'] > peak_acc:
                peak_acc = item['accuracy']
                peak_tool = item['tool_count']
        
        # Find where accuracy drops below 70% of peak
        threshold = peak_acc * 0.7
        recommended_cap = peak_tool
        for item in analysis.marginal_accuracy_by_tool:
            if item['tool_count'] > peak_tool and item['n_samples'] >= 5:
                if item['accuracy'] < threshold:
                    break
                recommended_cap = item['tool_count']
        
        # Add some buffer
        analysis.recommended_recursion_cap = min(recommended_cap + 2, 15)
        analysis.cap_reasoning = (
            f"Peak accuracy {peak_acc:.1%} at {peak_tool} tools. "
            f"Accuracy drops significantly after {recommended_cap} tools. "
            f"Recommended cap: {analysis.recommended_recursion_cap} (with buffer)."
        )
    
    return analysis


def compute_time_performance_analysis(data: List[Dict]) -> TimePerformanceAnalysis:
    """
    Analyze the relationship between time spent and performance.
    
    This provides data for research paper on efficiency-accuracy tradeoffs.
    """
    analysis = TimePerformanceAnalysis()
    
    # Filter records with latency
    records_with_latency = [r for r in data if r.get('latency_sec') is not None]
    
    if not records_with_latency:
        return analysis
    
    # Total time
    analysis.total_time_sec = sum(r['latency_sec'] for r in records_with_latency)
    
    # Define time buckets (in seconds)
    time_ranges = [
        (0, 10, "0-10s"),
        (10, 20, "10-20s"),
        (20, 30, "20-30s"),
        (30, 45, "30-45s"),
        (45, 60, "45-60s"),
        (60, 90, "60-90s"),
        (90, float('inf'), "90s+"),
    ]
    
    for min_t, max_t, name in time_ranges:
        bucket = TimePerformanceBucket(
            bucket_name=name,
            min_sec=min_t,
            max_sec=max_t if max_t != float('inf') else 9999
        )
        
        bucket_records = []
        for record in records_with_latency:
            latency = record.get('latency_sec', 0)
            if min_t <= latency < max_t:
                bucket_records.append(record)
                bucket.n_samples += 1
                
                if record.get('predicted') is not None and record.get('correct'):
                    bucket.n_correct += 1
        
        if bucket_records:
            valid = [r for r in bucket_records if r.get('predicted') is not None]
            if valid:
                bucket.accuracy = sum(1 for r in valid if r.get('correct')) / len(valid)
                
                # F1-Macro
                tp = sum(1 for r in valid if r.get('ground_truth') == 'Yes' and r.get('predicted') == 'Yes')
                tn = sum(1 for r in valid if r.get('ground_truth') == 'No' and r.get('predicted') == 'No')
                fp = sum(1 for r in valid if r.get('ground_truth') == 'No' and r.get('predicted') == 'Yes')
                fn = sum(1 for r in valid if r.get('ground_truth') == 'Yes' and r.get('predicted') == 'No')
                
                p_yes = safe_divide(tp, tp + fp)
                r_yes = safe_divide(tp, tp + fn)
                f1_yes = safe_divide(2 * p_yes * r_yes, p_yes + r_yes)
                
                p_no = safe_divide(tn, tn + fn)
                r_no = safe_divide(tn, tn + fp)
                f1_no = safe_divide(2 * p_no * r_no, p_no + r_no)
                
                bucket.f1_macro = (f1_yes + f1_no) / 2
            
            bucket.avg_tool_calls = sum(r.get('tool_call_count', 0) for r in bucket_records) / len(bucket_records)
        
        analysis.time_buckets.append(bucket)
    
    # Find optimal time range
    valid_buckets = [b for b in analysis.time_buckets if b.n_samples >= 10]
    if valid_buckets:
        best_bucket = max(valid_buckets, key=lambda b: b.accuracy)
        analysis.optimal_time_range_min_sec = best_bucket.min_sec
        analysis.optimal_time_range_max_sec = best_bucket.max_sec
        analysis.optimal_time_accuracy = best_bucket.accuracy
    
    # Efficiency metrics
    total_correct = sum(1 for r in records_with_latency 
                       if r.get('predicted') is not None and r.get('correct'))
    analysis.correct_answers_per_minute = (total_correct / analysis.total_time_sec) * 60
    
    valid_with_latency = [r for r in records_with_latency if r.get('predicted') is not None]
    if valid_with_latency:
        total_accuracy = sum(1 for r in valid_with_latency if r.get('correct')) / len(valid_with_latency)
        avg_latency = analysis.total_time_sec / len(records_with_latency)
        analysis.accuracy_per_second = total_accuracy / avg_latency if avg_latency > 0 else 0
    
    # Throughput
    analysis.questions_per_minute = (len(records_with_latency) / analysis.total_time_sec) * 60
    
    # Compute time-accuracy correlation
    latencies = []
    correct_flags = []
    for record in valid_with_latency:
        latencies.append(record.get('latency_sec', 0))
        correct_flags.append(1 if record.get('correct') else 0)
    
    if len(latencies) > 2:
        n = len(latencies)
        mean_lat = sum(latencies) / n
        mean_correct = sum(correct_flags) / n
        
        numerator = sum((l - mean_lat) * (c - mean_correct) 
                       for l, c in zip(latencies, correct_flags))
        denom_lat = math.sqrt(sum((l - mean_lat)**2 for l in latencies))
        denom_correct = math.sqrt(sum((c - mean_correct)**2 for c in correct_flags))
        
        if denom_lat > 0 and denom_correct > 0:
            analysis.time_accuracy_correlation = numerator / (denom_lat * denom_correct)
    
    # Performance at different time budgets (for paper)
    # "If we cut off all queries at X seconds, what accuracy would we get?"
    time_budgets = [15, 20, 30, 45, 60, 90, 120]
    for budget in time_budgets:
        records_in_budget = [r for r in valid_with_latency if r.get('latency_sec', 0) <= budget]
        records_over_budget = [r for r in valid_with_latency if r.get('latency_sec', 0) > budget]
        
        if records_in_budget:
            accuracy_in_budget = sum(1 for r in records_in_budget if r.get('correct')) / len(records_in_budget)
            
            # If we timeout over-budget queries (count as wrong)
            total_for_timeout = len(records_in_budget) + len(records_over_budget)
            correct_with_timeout = sum(1 for r in records_in_budget if r.get('correct'))
            accuracy_with_timeout = correct_with_timeout / total_for_timeout if total_for_timeout > 0 else 0
            
            analysis.performance_at_time_budget.append({
                'budget_sec': budget,
                'n_completed': len(records_in_budget),
                'n_timeout': len(records_over_budget),
                'pct_completed': len(records_in_budget) / len(valid_with_latency),
                'accuracy_completed': accuracy_in_budget,
                'accuracy_with_timeout_as_wrong': accuracy_with_timeout,
                'estimated_throughput_per_min': 60 / budget
            })
    
    # Estimate throughput at different recursion caps
    for cap in [3, 5, 7, 10, 15, 20]:
        records_at_cap = [r for r in records_with_latency if r.get('tool_call_count', 0) <= cap]
        if records_at_cap:
            avg_time_at_cap = sum(r['latency_sec'] for r in records_at_cap) / len(records_at_cap)
            analysis.estimated_throughput_at_cap[cap] = 60 / avg_time_at_cap if avg_time_at_cap > 0 else 0
    
    # Pareto frontier: accuracy vs throughput
    # Group by tool count and compute accuracy and throughput for each
    pareto_points = []
    for tool_count in range(0, 21):
        records_at_t = [r for r in valid_with_latency if r.get('tool_call_count', 0) == tool_count]
        if len(records_at_t) >= 5:
            acc = sum(1 for r in records_at_t if r.get('correct')) / len(records_at_t)
            avg_time = sum(r['latency_sec'] for r in records_at_t) / len(records_at_t)
            throughput = 60 / avg_time if avg_time > 0 else 0
            
            pareto_points.append({
                'tool_count': tool_count,
                'accuracy': acc,
                'throughput_per_min': throughput,
                'avg_latency_sec': avg_time,
                'n_samples': len(records_at_t)
            })
    
    # Filter to Pareto optimal (not dominated by any other point)
    analysis.pareto_frontier = []
    for point in pareto_points:
        is_dominated = False
        for other in pareto_points:
            if (other['accuracy'] >= point['accuracy'] and 
                other['throughput_per_min'] >= point['throughput_per_min'] and
                (other['accuracy'] > point['accuracy'] or 
                 other['throughput_per_min'] > point['throughput_per_min'])):
                is_dominated = True
                break
        if not is_dominated:
            analysis.pareto_frontier.append(point)
    
    analysis.pareto_frontier.sort(key=lambda x: x['throughput_per_min'], reverse=True)
    
    return analysis


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
    
    # NEW: Tool efficiency and time-performance analysis
    report.tool_efficiency = compute_tool_efficiency_analysis(data)
    report.time_performance = compute_time_performance_analysis(data)
    
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
    
    # Tool efficiency analysis
    te = report.tool_efficiency
    lines.append("TOOL CALL EFFICIENCY ANALYSIS")
    lines.append("-" * 50)
    lines.append(f"{'Tool Range':<15} {'N':>6} {'Acc':>8} {'F1-Yes':>8} {'F1-M':>8} {'Lat(s)':>8}")
    for bucket in te.buckets:
        if bucket.n_samples > 0:
            lines.append(
                f"{bucket.bucket_name:<15} {bucket.n_samples:>6} "
                f"{bucket.accuracy:>8.1%} {bucket.f1_yes:>8.3f} "
                f"{bucket.f1_macro:>8.3f} {bucket.avg_latency_sec:>8.1f}"
            )
    lines.append("")
    lines.append(f"Optimal tool range:              {te.optimal_tool_range_min}-{te.optimal_tool_range_max} tools")
    lines.append(f"Optimal range accuracy:          {te.optimal_range_accuracy:.1%}")
    lines.append(f"Tool-accuracy correlation:       {te.tool_accuracy_correlation:.3f}")
    lines.append(f"Recommended recursion cap:       {te.recommended_recursion_cap}")
    if te.cap_reasoning:
        lines.append(f"Reasoning: {te.cap_reasoning}")
    lines.append("")
    
    # Marginal accuracy by tool count
    lines.append("MARGINAL ACCURACY BY TOOL COUNT")
    lines.append("-" * 50)
    lines.append(f"{'Tools':>6} {'N':>6} {'Accuracy':>10}")
    for item in te.marginal_accuracy_by_tool:
        if item['n_samples'] >= 5:  # Only show significant buckets
            lines.append(f"{item['tool_count']:>6} {item['n_samples']:>6} {item['accuracy']:>10.1%}")
    lines.append("")
    
    # Time-performance analysis
    tp = report.time_performance
    lines.append("TIME-PERFORMANCE ANALYSIS")
    lines.append("-" * 50)
    lines.append(f"Total time:                      {tp.total_time_sec/60:.1f} minutes")
    lines.append(f"Questions per minute:            {tp.questions_per_minute:.2f}")
    lines.append(f"Correct answers per minute:      {tp.correct_answers_per_minute:.2f}")
    lines.append(f"Time-accuracy correlation:       {tp.time_accuracy_correlation:.3f}")
    lines.append("")
    
    lines.append(f"{'Time Range':<12} {'N':>6} {'Acc':>8} {'F1-M':>8} {'Avg Tools':>10}")
    for bucket in tp.time_buckets:
        if bucket.n_samples > 0:
            lines.append(
                f"{bucket.bucket_name:<12} {bucket.n_samples:>6} "
                f"{bucket.accuracy:>8.1%} {bucket.f1_macro:>8.3f} "
                f"{bucket.avg_tool_calls:>10.1f}"
            )
    lines.append("")
    
    # Performance at time budgets (for paper)
    if tp.performance_at_time_budget:
        lines.append("PERFORMANCE AT TIME BUDGETS (for research paper)")
        lines.append("-" * 50)
        lines.append(f"{'Budget':>8} {'Complete':>10} {'Timeout':>10} {'Acc(done)':>10} {'Acc(w/TO)':>10} {'QPM':>8}")
        for item in tp.performance_at_time_budget:
            lines.append(
                f"{item['budget_sec']:>6}s "
                f"{item['n_completed']:>10} {item['n_timeout']:>10} "
                f"{item['accuracy_completed']:>10.1%} "
                f"{item['accuracy_with_timeout_as_wrong']:>10.1%} "
                f"{item['estimated_throughput_per_min']:>8.1f}"
            )
        lines.append("")
    
    # Pareto frontier
    if tp.pareto_frontier:
        lines.append("PARETO FRONTIER (Accuracy vs Throughput)")
        lines.append("-" * 50)
        lines.append(f"{'Tools':>6} {'Accuracy':>10} {'Throughput':>12} {'Avg Lat':>10} {'N':>6}")
        for point in tp.pareto_frontier:
            lines.append(
                f"{point['tool_count']:>6} {point['accuracy']:>10.1%} "
                f"{point['throughput_per_min']:>10.2f}/min "
                f"{point['avg_latency_sec']:>8.1f}s {point['n_samples']:>6}"
            )
        lines.append("")
    
    # Throughput estimates at different caps
    if tp.estimated_throughput_at_cap:
        lines.append("ESTIMATED THROUGHPUT AT RECURSION CAPS")
        lines.append("-" * 50)
        for cap, throughput in sorted(tp.estimated_throughput_at_cap.items()):
            lines.append(f"Cap {cap:>2} tools: {throughput:.2f} questions/min")
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
    
    # Tool efficiency metrics
    te = report.tool_efficiency
    lines.append(f"optimal_tool_min,{te.optimal_tool_range_min},tool_efficiency")
    lines.append(f"optimal_tool_max,{te.optimal_tool_range_max},tool_efficiency")
    lines.append(f"optimal_tool_accuracy,{te.optimal_range_accuracy:.4f},tool_efficiency")
    lines.append(f"tool_accuracy_correlation,{te.tool_accuracy_correlation:.4f},tool_efficiency")
    lines.append(f"recommended_recursion_cap,{te.recommended_recursion_cap},tool_efficiency")
    
    # Time-performance metrics
    tp = report.time_performance
    lines.append(f"total_time_sec,{tp.total_time_sec:.2f},time_performance")
    lines.append(f"questions_per_minute,{tp.questions_per_minute:.4f},time_performance")
    lines.append(f"correct_per_minute,{tp.correct_answers_per_minute:.4f},time_performance")
    lines.append(f"time_accuracy_correlation,{tp.time_accuracy_correlation:.4f},time_performance")
    lines.append(f"optimal_time_min_sec,{tp.optimal_time_range_min_sec:.2f},time_performance")
    lines.append(f"optimal_time_max_sec,{tp.optimal_time_range_max_sec:.2f},time_performance")
    
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
