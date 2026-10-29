"""
Benchmark Evaluation Metrics for Unsupervised Weather Sensor Anomaly Detection
Computes standard classification metrics and breakdown across sensor fault types.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def evaluate_anomaly_detection(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, any]:
    """
    Evaluates detector performance against ground-truth anomaly tags.
    """
    if y_pred is None:
        y_pred = (y_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate metrics with zero-division handling
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(y_true, y_scores)
    except Exception:
        pr_auc = 0.0

    return {
        "precision": float(np.round(prec, 4)),
        "recall": float(np.round(rec, 4)),
        "f1_score": float(np.round(f1, 4)),
        "roc_auc": float(np.round(roc_auc, 4)),
        "pr_auc": float(np.round(pr_auc, 4)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "total_anomalies": int(np.sum(y_true)),
        "detected_anomalies": int(np.sum(y_pred)),
    }


def evaluate_by_fault_type(
    df_results: pd.DataFrame,
    ground_truth_col: str = "is_anomaly",
    fault_type_col: str = "anomaly_type",
    pred_col: str = "is_anomaly_pred",
) -> pd.DataFrame:
    """
    Evaluates detection recall and count breakdown across each distinct fault type.
    """
    breakdown = []
    unique_faults = [f for f in df_results[fault_type_col].unique() if f != "none"]

    for fault in unique_faults:
        subset = df_results[df_results[fault_type_col] == fault]
        total = len(subset)
        detected = int(subset[pred_col].sum())
        recall = detected / total if total > 0 else 0.0

        breakdown.append({
            "fault_type": fault,
            "total_occurrences": total,
            "detected_count": detected,
            "detection_recall": round(recall, 4),
            "missed_count": total - detected,
        })

    # Add Normal (Specificity check)
    normal_subset = df_results[df_results[ground_truth_col] == 0]
    total_normal = len(normal_subset)
    fp = int(normal_subset[pred_col].sum())
    specificity = (total_normal - fp) / total_normal if total_normal > 0 else 1.0

    breakdown.append({
        "fault_type": "nominal_normal_data",
        "total_occurrences": total_normal,
        "detected_count": total_normal - fp,
        "detection_recall": round(specificity, 4),  # Specificity for normal
        "missed_count": fp,  # False alarms
    })

    return pd.DataFrame(breakdown)
