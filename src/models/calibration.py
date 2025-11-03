"""
Model Calibration Utilities
Purpose: Standalone calibration functions for probability calibration.
"""

import logging
from typing import Literal, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


def calibrate_classifier(
    base_estimator: BaseEstimator,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: Literal["sigmoid", "isotonic"] = "sigmoid",
    cv: int = 5,
    ensemble: bool = True,
) -> CalibratedClassifierCV:
    """
    Calibrate a classifier's probability estimates.
    
    Args:
        base_estimator: Fitted base classifier
        X_val: Validation features
        y_val: Validation labels
        method: Calibration method ('sigmoid' or 'isotonic')
        cv: Number of cross-validation folds (if ensemble=True)
        ensemble: If True, use CV ensemble; if False, use single calibrator
    
    Returns:
        Calibrated classifier
    
    Examples:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> rf = RandomForestClassifier().fit(X_train, y_train)
        >>> rf_cal = calibrate_classifier(rf, X_val, y_val, method='sigmoid')
        >>> proba = rf_cal.predict_proba(X_test)
    """
    
    if method not in ("sigmoid", "isotonic"):
        raise ValueError(f"Invalid calibration method: {method}. Use 'sigmoid' or 'isotonic'")
    
    logger.info(f"Calibrating classifier using {method} method (cv={cv}, ensemble={ensemble})")
    
    # Check if already fitted
    if not hasattr(base_estimator, "predict_proba"):
        logger.warning("Base estimator does not have predict_proba - calibration may fail")
    
    try:
        calibrated = CalibratedClassifierCV(
            base_estimator=base_estimator,
            method=method,
            cv=cv if ensemble else "prefit",
            ensemble=ensemble,
        )
        
        if ensemble:
            # CV ensemble: refit on validation data
            calibrated.fit(X_val, y_val)
        else:
            # Single calibrator: use prefit estimator
            calibrated.fit(X_val, y_val)
        
        logger.info("✓ Calibration complete")
        return calibrated
        
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        raise


def check_calibration_quality(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Assess calibration quality using reliability diagram statistics.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        n_bins: Number of bins for reliability diagram
    
    Returns:
        Dictionary with calibration metrics:
        - bin_centers: Mean predicted probability per bin
        - bin_accuracies: Actual positive rate per bin
        - bin_counts: Number of samples per bin
        - ece: Expected Calibration Error
        - mce: Maximum Calibration Error
    """
    
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba, bin_edges[1:-1])
    
    bin_centers = []
    bin_accuracies = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_centers.append(y_proba[mask].mean())
            bin_accuracies.append(y_true[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_centers.append(np.nan)
            bin_accuracies.append(np.nan)
            bin_counts.append(0)
    
    bin_centers = np.array(bin_centers)
    bin_accuracies = np.array(bin_accuracies)
    bin_counts = np.array(bin_counts)
    
    # Expected Calibration Error (weighted average)
    valid_mask = ~np.isnan(bin_accuracies)
    if valid_mask.sum() > 0:
        weights = bin_counts[valid_mask] / bin_counts[valid_mask].sum()
        ece = np.sum(weights * np.abs(bin_centers[valid_mask] - bin_accuracies[valid_mask]))
        mce = np.max(np.abs(bin_centers[valid_mask] - bin_accuracies[valid_mask]))
    else:
        ece = np.nan
        mce = np.nan
    
    return {
        'bin_centers': bin_centers,
        'bin_accuracies': bin_accuracies,
        'bin_counts': bin_counts,
        'ece': float(ece),
        'mce': float(mce),
    }


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    save_path: Optional[str] = None,
):
    """
    Plot reliability diagram (calibration curve).
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        n_bins: Number of bins
        title: Plot title
        save_path: If provided, save plot to this path
    """
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available - cannot plot calibration curve")
        return
    
    cal_stats = check_calibration_quality(y_true, y_proba, n_bins=n_bins)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    # Plot actual calibration
    valid_mask = ~np.isnan(cal_stats['bin_accuracies'])
    ax.plot(
        cal_stats['bin_centers'][valid_mask],
        cal_stats['bin_accuracies'][valid_mask],
        'o-',
        label='Model calibration',
    )
    
    # Add bin size as point size
    sizes = cal_stats['bin_counts'][valid_mask] / cal_stats['bin_counts'][valid_mask].max() * 200
    ax.scatter(
        cal_stats['bin_centers'][valid_mask],
        cal_stats['bin_accuracies'][valid_mask],
        s=sizes,
        alpha=0.3,
        color='blue',
    )
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f"{title}\nECE: {cal_stats['ece']:.4f}, MCE: {cal_stats['mce']:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved calibration curve to {save_path}")
    else:
        plt.show()
    
    plt.close()


def temperature_scaling(
    logits: np.ndarray,
    y_true: np.ndarray,
    temperature: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    """
    Apply temperature scaling to logits for calibration.
    
    Temperature scaling is a simple post-processing method that divides
    logits by a learned temperature parameter before applying softmax.
    
    Args:
        logits: Raw model outputs (before softmax)
        y_true: True labels
        temperature: If provided, use this temperature. Otherwise, optimize.
    
    Returns:
        Tuple of (calibrated_probabilities, optimal_temperature)
    """
    
    from scipy.optimize import minimize
    from scipy.special import softmax
    
    def nll_loss(temp):
        """Negative log-likelihood loss for temperature optimization."""
        scaled_logits = logits / temp
        probs = softmax(scaled_logits, axis=1)
        
        # Avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        
        # Cross-entropy loss
        n_samples = len(y_true)
        loss = -np.sum(np.log(probs[range(n_samples), y_true])) / n_samples
        return loss
    
    if temperature is None:
        # Optimize temperature
        result = minimize(nll_loss, x0=1.0, method='Nelder-Mead')
        temperature = float(result.x[0])
        logger.info(f"Optimal temperature: {temperature:.4f}")
    
    # Apply temperature scaling
    scaled_logits = logits / temperature
    calibrated_probs = softmax(scaled_logits, axis=1)
    
    return calibrated_probs, temperature