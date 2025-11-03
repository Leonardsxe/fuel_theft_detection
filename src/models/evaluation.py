"""
Model Evaluation
Purpose: Comprehensive model evaluation with metrics, visualizations, and reports.

Design Principles:
- Single Responsibility: Each function has one clear purpose
- DRY: Shared helper functions for common operations
- Explicit configuration: Type-safe ModelConfig
- Separation of concerns: Metrics vs visualization vs reporting
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple, Optional, Any, List
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from src.config.settings import ModelConfig
from src.utils.metrics import pr_auc, threshold_at_fpr, classification_report_dict

logger = logging.getLogger(__name__)


# ========================================
# Helper Functions (DRY Principle)
# ========================================

def _get_scores(model: Any, X: Any) -> np.ndarray:
    """
    Extract probability-like scores from any model.
    
    Args:
        model: Trained model (sklearn-compatible)
        X: Feature matrix
    
    Returns:
        Array of scores in [0, 1]
    
    Note:
        Tries predict_proba first, falls back to normalized decision_function
    """
    # Try predict_proba (most models)
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception as e:
            logger.debug(f"predict_proba failed: {e}")
    
    # Fallback: decision_function with normalization
    if hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X)  # type: ignore[attr-defined]
            scores = np.asarray(scores, dtype=float)
            
            # Normalize to [0, 1]
            mn, mx = np.min(scores), np.max(scores)
            if mx - mn < 1e-12:
                logger.warning("Constant decision scores - returning zeros")
                return np.zeros_like(scores)
            
            return (scores - mn) / (mx - mn)
        except Exception as e:
            logger.error(f"decision_function failed: {e}")
    
    # Last resort: should not happen with proper models
    raise AttributeError(
        f"Model {type(model).__name__} has no predict_proba or decision_function"
    )


def _feature_names_from_preprocessor(model: Any) -> Optional[List[str]]:
    """
    Extract transformed feature names from pipeline's ColumnTransformer.
    
    Args:
        model: Pipeline with 'preprocess' step
    
    Returns:
        List of feature names, or None if unavailable
    """
    try:
        # Navigate pipeline structure
        pre = model.named_steps.get("preprocess")
        if pre is None:
            return None
        
        # Get feature names from transformer
        fn = pre.get_feature_names_out()
        return list(fn)
    
    except Exception as e:
        logger.debug(f"Could not extract feature names from preprocessor: {e}")
        return None


def _feature_importances(model: Any) -> Optional[np.ndarray]:
    """
    Extract feature importances from various model types.
    
    Args:
        model: Trained model
    
    Returns:
        Array of importances, or None if unavailable
    
    Note:
        Handles tree models, linear models, and calibrated wrappers
    """
    try:
        # Try to unwrap pipeline
        clf = model.named_steps.get("clf", model) if hasattr(model, "named_steps") else model
    except Exception:
        clf = model
    
    # Try various wrapped/nested structures
    candidates = [
        getattr(clf, "base_estimator", None),
        getattr(clf, "estimator", None),
        getattr(clf, "best_estimator_", None),
        clf,
    ]
    
    for candidate in candidates:
        if candidate is None:
            continue
        
        # Tree-based models
        if hasattr(candidate, "feature_importances_"):
            return np.asarray(candidate.feature_importances_)
        
        # Linear models (use absolute coefficients)
        if hasattr(candidate, "coef_"):
            coef = np.asarray(candidate.coef_)
            # Flatten if multi-dimensional
            coef = coef.ravel() if coef.ndim > 1 else coef
            return np.abs(coef)  # Use absolute values for importance
    
    return None


def _save_pr_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    model_name: str,
    output_path: Path,
    pr_auc_value: float,
) -> None:
    """
    Save Precision-Recall curve plot.
    
    Args:
        y_true: True labels
        scores: Predicted scores
        model_name: Model name for title
        output_path: Where to save plot
        pr_auc_value: PR-AUC value for display
    """
    try:
        precision, recall, _ = precision_recall_curve(y_true, scores)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, linewidth=2, label=f'PR-AUC = {pr_auc_value:.3f}')
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curve — {model_name}', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        fig.savefig(output_path, dpi=160, bbox_inches='tight')
        plt.close(fig)
        
        logger.debug(f"Saved PR curve: {output_path}")
    
    except Exception as e:
        logger.warning(f"Failed to save PR curve for {model_name}: {e}")


# ========================================
# Model Evaluator Class
# ========================================

class ModelEvaluator:
    """
    Comprehensive model evaluation with metrics and visualizations.
    
    Design Principles:
    - Single Responsibility: Evaluation and reporting only
    - Explicit configuration: Type-safe settings
    - DRY: Shared helper functions
    
    Attributes:
        target_fpr: Target false positive rate for threshold selection
        save_pr_curves: Whether to generate PR curve plots
        save_feature_importance: Whether to save feature importance
        output_dir: Directory for saving artifacts
    
    Example:
        >>> evaluator = ModelEvaluator.from_model_config(config)
        >>> comparison, importances = evaluator.evaluate_multiple(
        ...     models, X_test, y_test, feature_names
        ... )
    """
    
    def __init__(
        self,
        target_fpr: float = 0.05,
        save_pr_curves: bool = True,
        save_feature_importance: bool = True,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize evaluator with configuration.
        
        Args:
            target_fpr: Target FPR for threshold selection (default: 5%)
            save_pr_curves: Whether to save PR curve plots
            save_feature_importance: Whether to save feature importances
            output_dir: Output directory for artifacts
        
        Raises:
            ValueError: If target_fpr not in (0, 1)
        """
        if not 0 < target_fpr < 1:
            raise ValueError(f"target_fpr must be in (0, 1), got {target_fpr}")
        
        self.target_fpr = float(target_fpr)
        self.save_pr_curves = bool(save_pr_curves)
        self.save_feature_importance = bool(save_feature_importance)
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Evaluation output directory: {self.output_dir}")
    
    @classmethod
    def from_model_config(
        cls,
        model_cfg: ModelConfig,
        output_dir: Optional[Path] = None
    ) -> "ModelEvaluator":
        """
        Create evaluator from ModelConfig.
        
        Args:
            model_cfg: Model configuration object
            output_dir: Optional output directory
        
        Returns:
            Configured ModelEvaluator instance
        """
        ev = getattr(model_cfg, "evaluation", None)
        
        # Extract config values with defaults
        target = getattr(ev, "target_fpr", 0.05) if ev else 0.05
        save_pr = getattr(ev, "save_pr_curves", True) if ev else True
        save_fi = getattr(ev, "save_feature_importance", True) if ev else True
        
        return cls(
            target_fpr=target,
            save_pr_curves=save_pr,
            save_feature_importance=save_fi,
            output_dir=output_dir
        )
    
    def evaluate_single(
        self,
        model: Any,
        X_test: Any,
        y_test: np.ndarray,
        model_name: str = "model",
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name for identification
            feature_names: Optional feature names
        
        Returns:
            Dictionary with metrics and artifacts
        """
        # Get predictions
        scores = _get_scores(model, X_test)
        
        # Calculate PR-AUC
        pr_auc_val = pr_auc(scores, y_test)
        
        # Find threshold at target FPR
        threshold_info = threshold_at_fpr(scores, y_test, max_fpr=self.target_fpr)
        
        # Get classification metrics at threshold
        y_pred = (scores >= threshold_info['threshold']).astype(int)
        class_report = classification_report_dict(y_test, y_pred)
        
        results = {
            'model': model_name,
            'pr_auc': pr_auc_val,
            'threshold': threshold_info.get('threshold', np.nan),
            'recall': threshold_info.get('recall', 0.0),
            'precision': threshold_info.get('precision', 0.0),
            'fpr': threshold_info.get('fpr', 0.0),
            **class_report,
        }
        
        # Optional: Save PR curve
        if self.save_pr_curves and self.output_dir:
            output_path = self.output_dir / f"pr_curve_{model_name}.png"
            _save_pr_curve(y_test, scores, model_name, output_path, pr_auc_val)
        
        # Optional: Extract feature importance
        if self.save_feature_importance:
            fi = _feature_importances(model)
            if fi is not None:
                # Align with feature names
                transformed_names = _feature_names_from_preprocessor(model) or feature_names
                if transformed_names is None or len(transformed_names) != len(fi):
                    transformed_names = [f"f{i}" for i in range(len(fi))]
                
                results['feature_importance'] = pd.DataFrame({
                    'feature': transformed_names,
                    'importance': fi
                }).sort_values('importance', ascending=False)
        
        return results
    
    def evaluate_multiple(
        self,
        models: Dict[str, Any],
        X_test: Any,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Evaluate multiple models and compare.
        
        Args:
            models: Dictionary of {name: model}
            X_test: Test features
            y_test: Test labels
            feature_names: Optional feature names
        
        Returns:
            Tuple of (comparison_df, feature_importances_dict)
        """
        if not models:
            logger.warning("No models provided for evaluation")
            return pd.DataFrame(), {}
        
        logger.info(f"Evaluating {len(models)} models...")
        
        rows = []
        fi_dict: Dict[str, pd.DataFrame] = {}
        
        for name, model in models.items():
            try:
                logger.info(f"  Evaluating {name}...")
                
                results = self.evaluate_single(
                    model=model,
                    X_test=X_test,
                    y_test=y_test,
                    model_name=name,
                    feature_names=feature_names,
                )
                
                # Extract feature importance if available
                if 'feature_importance' in results:
                    fi_dict[name] = results.pop('feature_importance')
                
                rows.append(results)
            
            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {e}")
                continue
        
        # Create comparison DataFrame
        if not rows:
            logger.error("No models successfully evaluated")
            return pd.DataFrame(), {}
        
        comparison = pd.DataFrame(rows).sort_values('pr_auc', ascending=False)
        
        # Save artifacts
        if self.output_dir:
            # Save comparison metrics
            comparison.to_csv(self.output_dir / "model_comparison.csv", index=False)
            logger.info(f"Saved comparison metrics: {self.output_dir / 'model_comparison.csv'}")
            
            # Save feature importances
            if self.save_feature_importance and fi_dict:
                fi_dir = self.output_dir / "feature_importance"
                fi_dir.mkdir(parents=True, exist_ok=True)
                
                for model_name, df_fi in fi_dict.items():
                    fi_path = fi_dir / f"{model_name}.csv"
                    df_fi.to_csv(fi_path, index=False)
                
                logger.info(f"Saved feature importances: {fi_dir}")
        
        logger.info("✓ Evaluation complete")
        
        return comparison, fi_dict
    
    def generate_report(
        self,
        comparison: pd.DataFrame,
        top_n: int = 100,
    ) -> pd.DataFrame:
        """
        Generate summary report from comparison results.
        
        Args:
            comparison: DataFrame from evaluate_multiple
            top_n: Number of top models to include
        
        Returns:
            Report DataFrame
        
        Note:
            Currently returns comparison (can be extended for custom reporting)
        """
        if comparison.empty:
            logger.warning("Empty comparison DataFrame - no report to generate")
            return comparison
        
        # Could add additional summary statistics here
        report = comparison.head(top_n).copy()
        
        # Log summary
        if not report.empty:
            best = report.iloc[0]
            logger.info(f"\nBest model: {best['model']}")
            logger.info(f"  PR-AUC: {best['pr_auc']:.4f}")
            logger.info(f"  Precision @ {self.target_fpr*100:.0f}% FPR: {best['precision']:.4f}")
            logger.info(f"  Recall @ {self.target_fpr*100:.0f}% FPR: {best['recall']:.4f}")
        
        return report


# ========================================
# Convenience Functions
# ========================================

def quick_evaluate(
    model: Any,
    X_test: Any,
    y_test: np.ndarray,
    target_fpr: float = 0.05,
) -> Dict[str, float]:
    """
    Quick evaluation of a single model (no saving artifacts).
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        target_fpr: Target FPR for threshold
    
    Returns:
        Dictionary with key metrics
    
    Example:
        >>> metrics = quick_evaluate(model, X_test, y_test)
        >>> print(f"PR-AUC: {metrics['pr_auc']:.3f}")
    """
    evaluator = ModelEvaluator(
        target_fpr=target_fpr,
        save_pr_curves=False,
        save_feature_importance=False,
        output_dir=None,
    )
    
    results = evaluator.evaluate_single(model, X_test, y_test)
    
    # Return only numeric metrics
    return {
        'pr_auc': results['pr_auc'],
        'threshold': results['threshold'],
        'precision': results['precision'],
        'recall': results['recall'],
        'fpr': results['fpr'],
    }