"""
Custom evaluation metrics for fuel theft detection.
Provides specialized metrics for imbalanced classification.
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.metrics import (
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix
)


def pr_auc(scores: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculate Precision-Recall AUC.
    
    Args:
        scores: Predicted scores/probabilities
        y_true: True binary labels
    
    Returns:
        PR-AUC score (0.0 if single class)
    """
    if len(np.unique(y_true)) < 2:
        return 0.0
    
    return float(average_precision_score(y_true, scores))


def recall_at_fpr(
    scores: np.ndarray,
    y_true: np.ndarray,
    max_fpr: float = 0.05
) -> Tuple[float, float]:
    """
    Calculate recall at a specified false positive rate.
    
    Args:
        scores: Predicted scores/probabilities
        y_true: True binary labels
        max_fpr: Maximum false positive rate
    
    Returns:
        Tuple of (recall, threshold) at the specified FPR
    """
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.5
    
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    
    # Find indices where FPR <= max_fpr
    mask = fpr <= max_fpr
    
    if not mask.any():
        return 0.0, 0.5
    
    # Get maximum recall at allowed FPR
    idx = int(np.argmax(tpr[mask]))
    
    return float(tpr[mask][idx]), float(thresholds[mask][idx])


def precision_at_recall(
    scores: np.ndarray,
    y_true: np.ndarray,
    min_recall: float = 0.8
) -> Tuple[float, float]:
    """
    Calculate precision at a specified recall level.
    
    Args:
        scores: Predicted scores/probabilities
        y_true: True binary labels
        min_recall: Minimum recall level
    
    Returns:
        Tuple of (precision, threshold) at the specified recall
    """
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.5
    
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    
    # Find indices where recall >= min_recall
    mask = recall >= min_recall
    
    if not mask.any():
        return 0.0, 0.5
    
    # Get maximum precision at allowed recall
    idx = int(np.argmax(precision[mask]))
    
    # Handle threshold array being one element shorter
    threshold_idx = min(idx, len(thresholds) - 1)
    
    return float(precision[mask][idx]), float(thresholds[threshold_idx])


def threshold_at_fpr(
    scores: np.ndarray,
    y_true: np.ndarray,
    max_fpr: float = 0.05,
    min_neg_for_per_pattern: int = 20
) -> dict:
    """
    Find optimal threshold for a target FPR.
    Robust version with fallbacks for small groups.
    
    Args:
        scores: Predicted scores/probabilities
        y_true: True binary labels
        max_fpr: Target false positive rate
        min_neg_for_per_pattern: Minimum negatives for proportional FP calculation
    
    Returns:
        Dictionary with threshold, recall, precision, and actual FPR
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    
    if len(np.unique(y_true)) < 2:
        return dict(threshold=np.nan, recall=0.0, precision=0.0, fpr=0.0)
    
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    
    # Drop the leading inf threshold
    finite = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]
    
    # Determine allowed false positives
    n_neg = int((y_true == 0).sum())
    
    if n_neg == 0:
        return dict(threshold=np.nan, recall=1.0, precision=1.0, fpr=0.0)
    
    # For small groups, allow at least 1 FP
    if n_neg < min_neg_for_per_pattern:
        target_fp = 1
    else:
        target_fp = int(np.floor(max_fpr * n_neg))
    
    # Evaluate thresholds in descending order
    order = np.argsort(-thresholds)
    thresholds_sorted = thresholds[order]
    
    best = dict(threshold=np.nan, recall=0.0, precision=0.0, fpr=0.0)
    best_tpr = -1.0
    
    for threshold in thresholds_sorted:
        y_pred = (scores >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle degenerate case
            tn = int((y_pred == 0).sum() - (y_true == 1).sum())
            fp = int((y_pred == 1).sum())
            fn = int((y_true == 1).sum())
            tp = 0
        
        if fp <= target_fp:
            curr_tpr = tp / (tp + fn) if (tp + fn) else 0.0
            if curr_tpr > best_tpr:
                precision = tp / (tp + fp) if (tp + fp) else 0.0
                curr_fpr = fp / (fp + tn) if (fp + tn) else 0.0
                best = dict(
                    threshold=float(threshold),
                    recall=curr_tpr,
                    precision=precision,
                    fpr=curr_fpr
                )
                best_tpr = curr_tpr
    
    # Fallback: best within max_fpr
    if np.isnan(best["threshold"]):
        mask = fpr <= max_fpr
        if mask.any():
            idx = int(np.argmax(tpr[mask]))
            threshold = float(thresholds[mask][idx])
            y_pred = (scores >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            curr_fpr = fp / (fp + tn) if (fp + tn) else 0.0
            best = dict(
                threshold=threshold,
                recall=tpr[mask][idx],
                precision=precision,
                fpr=curr_fpr
            )
    
    return best


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: Optional[np.ndarray] = None
) -> dict:
    """
    Generate comprehensive classification metrics.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        scores: Optional predicted probabilities
    
    Returns:
        Dictionary with all metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle cases where only one class is predicted
        tn = int((y_pred == 0).sum())
        fp = int((y_pred == 1).sum() - (y_true == 1).sum())
        fn = int((y_true == 1).sum() - (y_pred == 1).sum())
        tp = int((y_pred == 1).sum() - fp)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    report = {
        'confusion_matrix': {
            'TP': int(tp),
            'FP': int(fp),
            'TN': int(tn),
            'FN': int(fn)
        },
        'metrics': {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'fpr': float(fpr)
        }
    }
    
    # Add PR-AUC if scores provided
    if scores is not None:
        report['metrics']['pr_auc'] = pr_auc(scores, y_true)
    
    return report