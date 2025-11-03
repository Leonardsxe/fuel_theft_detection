"""
Pattern-Specific Model Training
Purpose: Train specialized models for each detection pattern with adaptive FPR targets.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, average_precision_score, roc_curve

from src.config.settings import ModelConfig, PatternModelConfig
from src.models.training import (
    train_random_forest,
    train_logistic_regression,
    train_xgboost,
    train_lightgbm,
    _dedupe_columns,
)
from src.utils.metrics import pr_auc

logger = logging.getLogger(__name__)


class PatternSpecificTrainer:
    """Train specialized models for each detection pattern."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.pattern_models: Dict[str, Pipeline] = {}
        self.pattern_thresholds: Dict[str, float] = {}
        self.pattern_results: List[Dict] = []
    
    def train_pattern_models(
        self,
        events_df: pd.DataFrame,
        X: pd.DataFrame,
        y: np.ndarray,
        train_mask: np.ndarray,
        test_mask: np.ndarray,
        preprocess: ColumnTransformer,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, float], pd.DataFrame]:
        """
        Train specialized models for each pattern with pattern-specific thresholds.
        
        Args:
            events_df: Events DataFrame with 'pattern' column
            X: Feature matrix
            y: Labels
            train_mask: Boolean mask for training events
            test_mask: Boolean mask for test events
            preprocess: Fitted ColumnTransformer
        
        Returns:
            Tuple of (pattern_models, pattern_thresholds, results_df)
        """
        
        X = _dedupe_columns(X)
        
        # Get pattern configurations
        pattern_configs = self.config.pattern_models
        
        for pattern in events_df["pattern"].unique():
            pattern_str = str(pattern)
            
            # Get pattern-specific config (with defaults)
            pattern_cfg = pattern_configs.get(
                pattern_str,
                PatternModelConfig(model_type="random_forest", target_fpr=0.05)
            )
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Training model for pattern: {pattern_str}")
            logger.info(f"Model type: {pattern_cfg.model_type}")
            logger.info(f"Target FPR: {pattern_cfg.target_fpr * 100:.0f}%")
            logger.info(f"{'='*60}")
            
            # Filter to this pattern
            pattern_mask_train = train_mask & (events_df["pattern"] == pattern).values
            pattern_mask_test = test_mask & (events_df["pattern"] == pattern).values
            
            n_train = pattern_mask_train.sum()
            n_pos_train = y[pattern_mask_train].sum()
            n_test = pattern_mask_test.sum()
            n_pos_test = y[pattern_mask_test].sum()
            
            logger.info(f"Train: {n_train} events ({n_pos_train} positive, {n_pos_train/max(n_train,1)*100:.1f}%)")
            logger.info(f"Test: {n_test} events ({n_pos_test} positive, {n_pos_test/max(n_test,1)*100:.1f}%)")
            
            # Skip if insufficient data
            if n_train < 30 or n_pos_train < 5:
                logger.warning(f"⚠️  Skipping {pattern_str}: insufficient training data")
                logger.info(f"   Using baseline model instead for this pattern")
                continue
            
            # Warn about extreme imbalance
            test_prevalence = n_pos_test / max(n_test, 1)
            if test_prevalence > 0.7 or test_prevalence < 0.05:
                logger.warning(
                    f"⚠️  Warning: Extreme test set imbalance ({test_prevalence*100:.1f}% positive)"
                )
                logger.warning(f"   Pattern-specific threshold may be unreliable")
            
            # Get pattern-specific data
            X_pat_train = X.loc[pattern_mask_train]
            y_pat_train = y[pattern_mask_train]
            X_pat_test = X.loc[pattern_mask_test]
            y_pat_test = y[pattern_mask_test]
            
            # Train model based on pattern config
            try:
                model = self._train_pattern_model(
                    pattern_cfg.model_type,
                    pattern_str,
                    preprocess,
                    X_pat_train,
                    y_pat_train,
                )
                
                if model is None:
                    continue
                
                logger.info(f"✓ Training complete for {pattern_str}")
                
            except Exception as e:
                logger.error(f"⚠️  Training failed for {pattern_str}: {e}")
                continue
            
            # Evaluate and find threshold
            try:
                scores_test = model.predict_proba(X_pat_test)[:, 1]
                
                pr_auc_val = pr_auc(scores_test, y_pat_test)
                
                threshold = self._find_threshold_at_fpr(
                    scores_test,
                    y_pat_test,
                    target_fpr=pattern_cfg.target_fpr,
                )
                
                # Calculate metrics at threshold
                metrics = self._calculate_metrics(scores_test, y_pat_test, threshold)
                
                logger.info(f"\nResults:")
                logger.info(f"  PR-AUC: {pr_auc_val:.4f}")
                logger.info(f"  Threshold: {threshold:.4f}")
                logger.info(f"  Precision: {metrics['precision']:.4f}")
                logger.info(f"  Recall: {metrics['recall']:.4f}")
                logger.info(f"  Actual FPR: {metrics['fpr']:.4f} (target: {pattern_cfg.target_fpr:.4f})")
                
                # Store results
                self.pattern_models[pattern_str] = model
                self.pattern_thresholds[pattern_str] = threshold
                
                self.pattern_results.append({
                    'pattern': pattern_str,
                    'model_type': pattern_cfg.model_type,
                    'n_train': n_train,
                    'n_pos_train': int(n_pos_train),
                    'n_test': n_test,
                    'n_pos_test': int(n_pos_test),
                    'pr_auc': pr_auc_val,
                    'threshold': threshold,
                    'target_fpr': pattern_cfg.target_fpr,
                    'actual_fpr': metrics['fpr'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                })
                
            except Exception as e:
                logger.error(f"⚠️  Evaluation failed for {pattern_str}: {e}")
                continue
        
        results_df = pd.DataFrame(self.pattern_results)
        
        return self.pattern_models, self.pattern_thresholds, results_df
    
    def _train_pattern_model(
        self,
        model_type: str,
        pattern_name: str,
        preprocess: ColumnTransformer,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
    ) -> Optional[Pipeline]:
        """Train a specific model type for a pattern."""
        
        model_type = model_type.lower()
        
        logger.info(f"Training {model_type} model...")
        
        try:
            if model_type == "random_forest":
                return train_random_forest(preprocess, X_train, y_train, self.config)
            
            elif model_type == "logistic_regression":
                return train_logistic_regression(preprocess, X_train, y_train, self.config)
            
            elif model_type == "xgboost":
                # Determine which XGBoost config to use based on pattern name
                which_pattern = "extended" if "extended" in pattern_name.lower() else "short"
                return train_xgboost(preprocess, X_train, y_train, self.config, which=which_pattern)
            
            elif model_type == "lightgbm":
                return train_lightgbm(preprocess, X_train, y_train, self.config)
            
            else:
                logger.error(f"Unknown model type: {model_type}")
                return None
                
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return None
    
    def _find_threshold_at_fpr(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        target_fpr: float = 0.05,
    ) -> float:
        """Find threshold that achieves target FPR."""
        
        if len(np.unique(y_true)) < 2:
            return 0.5
        
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        
        # Remove infinite thresholds
        finite_mask = np.isfinite(thresholds)
        fpr = fpr[finite_mask]
        tpr = tpr[finite_mask]
        thresholds = thresholds[finite_mask]
        
        if len(thresholds) == 0:
            logger.warning("No valid thresholds found, using median score")
            return float(np.median(scores))
        
        # Find threshold closest to target FPR
        idx = np.argmin(np.abs(fpr - target_fpr))
        threshold = float(thresholds[idx])
        
        # Ensure threshold is within score range
        threshold = np.clip(threshold, scores.min(), scores.max())
        
        return threshold
    
    def _calculate_metrics(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        threshold: float,
    ) -> Dict[str, float]:
        """Calculate classification metrics at given threshold."""
        
        y_pred = (scores >= threshold).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle degenerate case
            tn = int((y_pred == 0).sum())
            fp = int((y_pred == 1).sum() - (y_true == 1).sum())
            fn = int((y_true == 1).sum() - (y_pred == 1).sum())
            tp = int((y_pred == 1).sum() - fp)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'fpr': float(fpr),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
        }
    
    def save_results(self, output_dir: Path) -> None:
        """Save pattern-specific models and results."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        import joblib
        for pattern, model in self.pattern_models.items():
            model_path = output_dir / f"pattern_model_{pattern}.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved pattern model: {model_path}")
        
        # Save thresholds
        import json
        threshold_path = output_dir / "pattern_thresholds.json"
        with open(threshold_path, 'w') as f:
            json.dump(self.pattern_thresholds, f, indent=2)
        logger.info(f"Saved pattern thresholds: {threshold_path}")
        
        # Save results
        if self.pattern_results:
            results_df = pd.DataFrame(self.pattern_results)
            results_path = output_dir / "pattern_specific_results.csv"
            results_df.to_csv(results_path, index=False)
            logger.info(f"Saved pattern results: {results_path}")