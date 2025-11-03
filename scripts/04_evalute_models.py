"""
Script 04: Evaluate Models

Comprehensive evaluation of trained models with:
- Overall metrics at target FPR
- Per-pattern performance analysis
- Confusion matrices
- PR curves (overall and per-pattern)
- Feature importance analysis
- Top predictions for review
- Calibration quality assessment

Usage:
    python scripts/04_evaluate_models.py
"""

import sys
from pathlib import Path
from datetime import datetime
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from src.config.loader import load_config, validate_config
from src.utils.logging_config import setup_logging, log_section
from src.utils.metrics import pr_auc, threshold_at_fpr, classification_report_dict
from src.models.calibration import check_calibration_quality, plot_calibration_curve

logger = logging.getLogger(__name__)


def load_trained_models(models_dir: Path) -> dict:
    """Load all trained models from directory."""
    
    models = {}
    model_files = {
        'logreg_cal': 'logreg_cal.pkl',
        'rf_cal': 'rf_cal.pkl',
        'xgb': 'xgb.pkl',
        'lgbm': 'lgbm.pkl',
        'iso_forest': 'iso_forest.pkl',
    }
    
    for name, filename in model_files.items():
        path = models_dir / filename
        if path.exists():
            try:
                models[name] = joblib.load(path)
                logger.info(f"✓ Loaded {name} from {path}")
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")
        else:
            logger.warning(f"Model file not found: {path}")
    
    return models


def load_test_data(events_path: Path, raw_path: Path, train_ratio: float = 0.80) -> tuple:
    """Load and prepare test data."""
    
    from src.utils.timezone import ensure_series_utc
    from src.utils.splitting import create_temporal_split_on_raw_data, map_events_to_split
    
    # Load raw data for split
    raw_df = pd.read_csv(raw_path)
    raw_df["timestamp"] = ensure_series_utc(raw_df["timestamp"])
    
    # Create split
    raw_train_mask = create_temporal_split_on_raw_data(raw_df, train_ratio=train_ratio)
    
    # Load events
    events_df = pd.read_csv(events_path)
    events_df["start_time"] = pd.to_datetime(events_df["start_time"], utc=True)
    events_df["end_time"] = pd.to_datetime(events_df["end_time"], utc=True)
    
    # Map events to split
    events_df = map_events_to_split(events_df, raw_df, raw_train_mask)
    
    return events_df, raw_df, raw_train_mask


def prepare_features_and_labels(events_df: pd.DataFrame) -> tuple:
    """Extract features and labels from events."""
    
    # Feature columns (must match training)
    numeric_cols = [c for c in [
        "drop_gal", "min_step_gal", "p95_abs_dfuel", "n_negative_steps",
        "duration_min", "cluster_count", "n_points", "pct_ign_on", "rate_gpm",
        "hod_sin", "hod_cos",
        "drop_gal_vehicle_zscore", "rate_gpm_vehicle_zscore",
        "min_step_gal_vehicle_zscore", "duration_min_vehicle_zscore",
        "drop_pct_of_avg", "drop_pct_of_max",
        "pre_event_distance_km", "pre_event_avg_speed",
        "pre_event_moving_pct", "pre_event_fuel_change",
        "speed_std", "speed_max", "movement_variability",
        "lat_std", "lon_std", "coord_range_km", "window_trip_km",
    ] if c in events_df.columns]
    
    binary_cols = [c for c in ["is_hotspot", "is_weekend", "is_night"] if c in events_df.columns]
    cat_cols = [c for c in ["pattern"] if c in events_df.columns]
    
    feature_cols = numeric_cols + binary_cols + cat_cols
    
    X = events_df[feature_cols].copy()
    y = events_df["y"].fillna(0).astype(int).values if "y" in events_df.columns else np.zeros(len(events_df), dtype=int)
    
    return X, y, feature_cols


def evaluate_model_at_threshold(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    target_fpr: float = 0.05,
) -> dict:
    """Evaluate single model and find optimal threshold."""
    
    # Get scores
    scores = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
    
    # Calculate PR-AUC
    pr_auc_val = pr_auc(scores, y_test)
    
    # Find threshold at target FPR
    threshold_info = threshold_at_fpr(scores, y_test, max_fpr=target_fpr)
    
    # Get classification report
    y_pred = (scores >= threshold_info['threshold']).astype(int)
    report = classification_report_dict(y_test, y_pred)
    
    return {
        'pr_auc': pr_auc_val,
        'threshold': threshold_info['threshold'],
        'scores': scores,
        'predictions': y_pred,
        **threshold_info,
        **report,
    }


def evaluate_per_pattern(
    model,
    events_df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    test_mask: np.ndarray,
    overall_threshold: float,
    target_fpr: float = 0.05,
) -> pd.DataFrame:
    """Evaluate model performance per pattern."""
    
    scores = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
    
    results = []
    
    for pattern in events_df.loc[test_mask, 'pattern'].unique():
        pattern_mask = (events_df.loc[test_mask, 'pattern'] == pattern).values
        
        if pattern_mask.sum() < 10:
            continue
        
        y_pat = y_test[pattern_mask]
        scores_pat = scores[pattern_mask]
        
        if len(np.unique(y_pat)) < 2:
            continue
        
        # Pattern-specific threshold
        threshold_info = threshold_at_fpr(scores_pat, y_pat, max_fpr=target_fpr)
        pattern_threshold = threshold_info['threshold']
        
        # Fallback to overall threshold if needed
        if not np.isfinite(pattern_threshold):
            pattern_threshold = overall_threshold
            used_fallback = True
        else:
            used_fallback = False
        
        # Calculate metrics
        y_pred_pat = (scores_pat >= pattern_threshold).astype(int)
        report = classification_report_dict(y_pat, y_pred_pat)
        
        results.append({
            'pattern': pattern,
            'n_total': int(len(y_pat)),
            'n_positive': int(y_pat.sum()),
            'pr_auc': pr_auc(scores_pat, y_pat),
            'threshold': float(pattern_threshold),
            'used_fallback': used_fallback,
            **threshold_info,
            **report,
        })
    
    return pd.DataFrame(results)


def plot_pr_curves(
    models: dict,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    output_dir: Path,
):
    """Generate PR curves for all models."""
    
    from sklearn.metrics import precision_recall_curve
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        scores = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
        
        if len(np.unique(y_test)) < 2:
            continue
        
        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr_auc_val = pr_auc(scores, y_test)
        
        plt.plot(recall, precision, label=f'{name} (AUC={pr_auc_val:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves - Overall')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    output_path = output_dir / 'pr_curves_overall.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved PR curves: {output_path}")


def plot_per_pattern_pr_curves(
    model,
    model_name: str,
    events_df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    test_mask: np.ndarray,
    output_dir: Path,
):
    """Generate per-pattern PR curves for best model."""
    
    from sklearn.metrics import precision_recall_curve
    
    scores = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
    
    patterns = events_df.loc[test_mask, 'pattern'].unique()
    n_patterns = len(patterns)
    
    if n_patterns == 0:
        return
    
    cols = min(3, n_patterns)
    rows = (n_patterns + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, pattern in enumerate(patterns):
        pattern_mask = (events_df.loc[test_mask, 'pattern'] == pattern).values
        
        if pattern_mask.sum() < 10:
            axes[idx].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'{pattern}')
            continue
        
        y_pat = y_test[pattern_mask]
        scores_pat = scores[pattern_mask]
        
        if len(np.unique(y_pat)) < 2:
            axes[idx].text(0.5, 0.5, 'Single class', ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'{pattern}')
            continue
        
        precision, recall, _ = precision_recall_curve(y_pat, scores_pat)
        pr_auc_val = pr_auc(scores_pat, y_pat)
        
        axes[idx].plot(recall, precision, 'b-')
        axes[idx].set_xlabel('Recall')
        axes[idx].set_ylabel('Precision')
        axes[idx].set_title(f'{pattern}\nAUC: {pr_auc_val:.3f}')
        axes[idx].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(patterns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path = output_dir / f'pr_curves_per_pattern_{model_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved per-pattern PR curves: {output_path}")


def evaluate_calibration(
    models: dict,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    output_dir: Path,
):
    """Assess calibration quality for all models."""
    
    calibration_results = []
    
    for name, model in models.items():
        scores = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
        
        cal_stats = check_calibration_quality(y_test, scores, n_bins=10)
        
        calibration_results.append({
            'model': name,
            'ece': cal_stats['ece'],
            'mce': cal_stats['mce'],
        })
        
        # Plot calibration curve
        plot_calibration_curve(
            y_test,
            scores,
            n_bins=10,
            title=f'Calibration Curve - {name}',
            save_path=str(output_dir / f'calibration_{name}.png'),
        )
    
    cal_df = pd.DataFrame(calibration_results)
    cal_df.to_csv(output_dir / 'calibration_metrics.csv', index=False)
    
    logger.info("Calibration assessment complete")
    return cal_df


def generate_top_predictions(
    models: dict,
    events_df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    test_mask: np.ndarray,
    top_n: int = 100,
) -> pd.DataFrame:
    """Generate top predictions for manual review."""
    
    # Use best model (by name convention)
    best_model_name = 'rf_cal' if 'rf_cal' in models else list(models.keys())[0]
    model = models[best_model_name]
    
    scores = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
    
    # Create results dataframe
    results = events_df.loc[test_mask].copy()
    results['predicted_proba'] = scores
    results['predicted_label'] = (scores >= 0.5).astype(int)
    
    # Sort by probability (descending)
    results = results.sort_values('predicted_proba', ascending=False)
    
    # Select relevant columns
    cols = [c for c in [
        'vehicle_id', 'start_time', 'end_time', 'duration_min',
        'drop_gal', 'pattern', 'is_hotspot', 'n_negative_steps',
        'y', 'predicted_proba', 'predicted_label'
    ] if c in results.columns]
    
    return results[cols].head(top_n)


def main() -> int:
    """Main execution."""
    
    # Setup logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    setup_logging(
        level="INFO",
        log_file=log_dir / "04_evaluate_models.log",
        log_to_console=True
    )
    
    log_section("FUEL THEFT DETECTION - MODEL EVALUATION")
    logger.info(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(
            detection_config_path=Path("config/detection_config.yaml"),
            model_config_path=Path("config/model_config.yaml"),
            path_config_path=Path("config/paths_config.yaml"),
        )
        validate_config(config)
        
        # Load trained models
        log_section("LOADING TRAINED MODELS")
        models_dir = Path(config.paths.data.models)
        models = load_trained_models(models_dir)
        
        if not models:
            logger.error("No trained models found!")
            return 1
        
        logger.info(f"Loaded {len(models)} models: {', '.join(models.keys())}")
        
        # Load test data
        log_section("LOADING TEST DATA")
        events_df, raw_df, raw_train_mask = load_test_data(
            config.paths.output.events_csv,
            config.paths.input.combined_csv,
            train_ratio=config.model.splitting.train_ratio,
        )
        
        test_mask = ~events_df['is_train'].values
        X, y, feature_names = prepare_features_and_labels(events_df)
        
        X_test = X.loc[test_mask]
        y_test = y[test_mask]
        
        logger.info(f"Test set: {len(X_test)} events, {int(y_test.sum())} positive ({y_test.mean()*100:.1f}%)")
        
        # Overall evaluation
        log_section("EVALUATING MODELS - OVERALL")
        
        overall_results = []
        target_fpr = config.model.evaluation.target_fpr
        
        for name, model in models.items():
            logger.info(f"\nEvaluating {name}...")
            try:
                result = evaluate_model_at_threshold(model, X_test, y_test, target_fpr=target_fpr)
                overall_results.append({
                    'model': name,
                    **result
                })
                logger.info(f"  PR-AUC: {result['pr_auc']:.4f}")
                logger.info(f"  Threshold: {result['threshold']:.4f}")
                logger.info(f"  Precision: {result['precision']:.4f}")
                logger.info(f"  Recall: {result['recall']:.4f}")
            except Exception as e:
                logger.error(f"Evaluation failed for {name}: {e}")
        
        overall_df = pd.DataFrame(overall_results)
        overall_df = overall_df.sort_values('pr_auc', ascending=False)
        
        # Save overall metrics
        reports_dir = Path(config.paths.data.reports)
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        overall_df.to_csv(reports_dir / 'evaluation_overall.csv', index=False)
        logger.info(f"Saved overall metrics: {reports_dir / 'evaluation_overall.csv'}")
        
        # Per-pattern evaluation (best model)
        log_section("EVALUATING MODELS - PER PATTERN")
        
        best_model_name = overall_df.iloc[0]['model']
        best_model = models[best_model_name]
        best_threshold = overall_df.iloc[0]['threshold']
        
        logger.info(f"Using best model: {best_model_name}")
        
        per_pattern_df = evaluate_per_pattern(
            best_model,
            events_df,
            X_test,
            y_test,
            test_mask,
            best_threshold,
            target_fpr=target_fpr,
        )
        
        if not per_pattern_df.empty:
            per_pattern_df.to_csv(reports_dir / 'evaluation_per_pattern.csv', index=False)
            logger.info(f"Saved per-pattern metrics: {reports_dir / 'evaluation_per_pattern.csv'}")
            
            logger.info("\nPer-pattern performance:")
            print(per_pattern_df[['pattern', 'n_total', 'pr_auc', 'precision', 'recall']].to_string(index=False))
        
        # Generate visualizations
        log_section("GENERATING VISUALIZATIONS")
        
        plot_pr_curves(models, X_test, y_test, reports_dir)
        plot_per_pattern_pr_curves(
            best_model,
            best_model_name,
            events_df,
            X_test,
            y_test,
            test_mask,
            reports_dir,
        )
        
        # Calibration assessment
        log_section("ASSESSING CALIBRATION")
        
        cal_df = evaluate_calibration(models, X_test, y_test, reports_dir)
        
        logger.info("\nCalibration metrics:")
        print(cal_df.to_string(index=False))
        
        # Top predictions
        log_section("GENERATING TOP PREDICTIONS")
        
        top_preds = generate_top_predictions(
            models,
            events_df,
            X_test,
            y_test,
            test_mask,
            top_n=config.model.evaluation.save_top_predictions,
        )
        
        top_preds.to_csv(reports_dir / 'top_predictions.csv', index=False)
        logger.info(f"Saved top predictions: {reports_dir / 'top_predictions.csv'}")
        
        # Summary
        log_section("EVALUATION SUMMARY")
        
        logger.info(f"\nBest Model: {best_model_name}")
        logger.info(f"  PR-AUC: {overall_df.iloc[0]['pr_auc']:.4f}")
        logger.info(f"  Precision @ {target_fpr*100:.0f}% FPR: {overall_df.iloc[0]['precision']:.4f}")
        logger.info(f"  Recall @ {target_fpr*100:.0f}% FPR: {overall_df.iloc[0]['recall']:.4f}")
        
        logger.info(f"\nAll results saved to: {reports_dir}")
        logger.info("✓ Evaluation complete")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())