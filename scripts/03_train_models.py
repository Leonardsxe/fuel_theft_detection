# scripts/03_train_models.py
"""
Script 03: Train Models (config-driven)

Trains fuel theft detection models using the modular architecture.
- Loads raw telemetry + detected events via config
- Creates time-aware split (using shared utils)
- Engineers features (temporal/spatial/behavioral + vehicle normalization)
- Trains/evaluates multiple models
- Optionally trains pattern-specific models
- Saves models and reports

Usage:
    python scripts/03_train_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
import logging
import warnings

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib

# === Config & logging ===
from src.config.loader import load_config, validate_config
from src.utils.logging_config import setup_logging, log_section

# === Shared utilities ===
from src.utils.timezone import ensure_series_utc
from src.utils.splitting import (
    create_temporal_split_on_raw_data,
    map_events_to_split,
    validate_no_leakage,
)

# === Features & models ===
from src.config.settings import FeatureConfig
from src.features.engineering import FeatureEngineer
from src.models.training import ModelTrainer, _dedupe_columns
from src.models.evaluation import ModelEvaluator
from src.models.pattern_models import PatternSpecificTrainer

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


# ---------------------------
# Path resolution
# ---------------------------
def _resolve_paths_from_config(cfg) -> dict:
    """Pull canonical paths from typed config with safe fallbacks."""
    paths = {}

    # Inputs (must exist)
    paths["combined_csv"] = getattr(cfg.paths.input, "combined_csv", Path("data/processed/combined_dataset.csv"))
    paths["events_csv"] = getattr(cfg.paths.output, "events_csv", Path("data/events/fuel_theft_events.csv"))

    # Canonical dirs for artifacts
    paths["models_dir"] = getattr(cfg.paths.data, "models", Path("data/models"))
    paths["reports_dir"] = getattr(cfg.paths.data, "reports", Path("data/reports"))

    # Canonical files for reports/metrics
    paths["overall_metrics_csv"] = getattr(cfg.paths.output, "overall_metrics", paths["reports_dir"] / "comprehensive_metrics_overall.csv")
    paths["feature_importance_csv"] = getattr(cfg.paths.output, "feature_importance", paths["reports_dir"] / "feature_importance.csv")

    # Convenience (created by this script)
    paths["feature_stats_csv"] = paths["reports_dir"] / "feature_statistics.csv"
    paths["per_model_fi_dir"] = paths["reports_dir"] / "feature_importances"

    return paths


def _setup_output_dirs(paths: dict) -> None:
    for p in [paths["models_dir"], paths["reports_dir"], paths["per_model_fi_dir"]]:
        Path(p).mkdir(parents=True, exist_ok=True)
    logger.info(f"Models dir:  {paths['models_dir']}")
    logger.info(f"Reports dir: {paths['reports_dir']}")


# ---------------------------
# Data loading
# ---------------------------
def _load_raw_and_events(paths: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Raw telemetry
    combined_csv = Path(paths["combined_csv"])
    if not combined_csv.exists():
        raise FileNotFoundError(f"Raw data not found: {combined_csv}")
    logger.info(f"Loading raw telemetry: {combined_csv}")
    raw_df = pd.read_csv(combined_csv)
    raw_df["timestamp"] = ensure_series_utc(raw_df["timestamp"])
    required = [
        "vehicle_id",
        "timestamp",
        "total_fuel_gal",
        "speed_kmh",
        "ignition",
        "stationary",
        "stationary_on",
        "ign_off",
        "dfuel",
    ]
    missing = [c for c in required if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Raw telemetry missing required columns: {missing}")
    logger.info(f"Loaded {len(raw_df):,} telemetry rows across {raw_df['vehicle_id'].nunique()} vehicles")
    logger.info(f"Date range: {raw_df['timestamp'].min()}  →  {raw_df['timestamp'].max()}")

    # Events
    events_csv = Path(paths["events_csv"])
    if not events_csv.exists():
        raise FileNotFoundError(f"Events not found: {events_csv} — run scripts/02_detect_events.py first.")
    logger.info(f"Loading detected events: {events_csv}")
    events_df = pd.read_csv(events_csv)
    events_df["start_time"] = pd.to_datetime(events_df["start_time"], utc=True, errors="coerce")
    events_df["end_time"] = pd.to_datetime(events_df["end_time"], utc=True, errors="coerce")
    logger.info(f"Loaded {len(events_df):,} detected events")

    if "pattern" in events_df.columns:
        counts = events_df["pattern"].value_counts()
        logger.info("Pattern distribution:\n" + counts.to_string())

    if "y" in events_df.columns:
        pos = int(pd.to_numeric(events_df["y"], errors="coerce").fillna(0).sum())
        logger.info(f"Labeled positives: {pos}")

    return raw_df, events_df


# ---------------------------
# Features
# ---------------------------
def _engineer_features(events_df: pd.DataFrame, raw_df: pd.DataFrame, event_train_mask: np.ndarray, cfg) -> pd.DataFrame:
    # Map config.model.features -> FeatureConfig dataclass
    fcfg = getattr(cfg.model, "features", None)

    feature_config = FeatureConfig(
        behavioral_lookback_hours=getattr(fcfg, "behavioral_lookback_hours", 2),
        enable_vehicle_normalization=getattr(fcfg, "enable_vehicle_normalization", True),
        enable_behavioral_features=getattr(fcfg, "enable_behavioral_features", True),
        enable_temporal_features=getattr(fcfg, "enable_temporal_features", True),
        enable_spatial_features=getattr(fcfg, "enable_spatial_features", True),
    )

    logger.info(
        "Feature configuration: "
        f"lookback={feature_config.behavioral_lookback_hours}h, "
        f"temporal={feature_config.enable_temporal_features}, "
        f"spatial={feature_config.enable_spatial_features}, "
        f"behavioral={feature_config.enable_behavioral_features}, "
        f"vehicle_norm={feature_config.enable_vehicle_normalization}"
    )

    fe = FeatureEngineer(feature_config)
    logger.info("Engineering features (fit on TRAIN events only for normalization)…")
    enriched = fe.fit_transform(events_df, raw_df, event_train_mask)

    # Quick descriptive stats (not model importances)
    stats = enriched.select_dtypes(include=[np.number]).describe().T.reset_index().rename(columns={"index": "feature"})
    stats_path = Path(cfg.paths.data.reports) / "feature_statistics.csv"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(stats_path, index=False)
    logger.info(f"Saved feature stats → {stats_path}")

    # Brief overview of generated feature families
    new_features = set(enriched.columns) - set(events_df.columns)
    tf = [f for f in new_features if any(k in f for k in ["hod_", "week", "night", "duration"])]
    sf = [f for f in new_features if any(k in f for k in ["lat_", "lon_", "coord_", "trip_", "window_trip"])]
    bf = [f for f in new_features if any(k in f for k in ["pre_event", "speed_", "parking"])]
    nf = [f for f in new_features if any(k in f for k in ["zscore", "pct_of"])]

    logger.info(f"New features: temporal={len(tf)}, spatial={len(sf)}, behavioral={len(bf)}, normalized={len(nf)}")
    return enriched


def _prepare_feature_matrix(events_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:

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
        "lat_std", "lon_std", "coord_range_km",
        "window_trip_km",
    ] if c in events_df.columns]

    binary_cols = [c for c in ["is_hotspot", "is_weekend", "is_night"] if c in events_df.columns]
    cat_cols = [c for c in ["pattern"] if c in events_df.columns]

    feature_names = numeric_cols + binary_cols + cat_cols

    X = events_df[feature_names].copy()
    X = _dedupe_columns(X)
    feature_names = list(X.columns)
    y = events_df["y"].fillna(0).astype(int).values if "y" in events_df.columns else np.zeros(len(events_df), dtype=int)

    missing = X.isnull().sum()
    if (missing > 0).any():
        logger.warning("Missing values detected:\n" + missing[missing > 0].to_string())
        logger.info("Downstream trainer will handle imputation (0/median) and encoding in its pipeline.")

    logger.info(f"Feature matrix: X={X.shape}, positives={int(y.sum())} ({y.mean()*100:.1f}%)")
    return X, y, feature_names


# ---------------------------
# Train / Evaluate / Save
# ---------------------------
def _train_models(X: pd.DataFrame, y: np.ndarray, event_train_mask: np.ndarray, cfg) -> tuple[dict, ModelTrainer]:
    trainer = ModelTrainer(cfg.model)
    logger.info("Training models…")
    X_tr = X.loc[event_train_mask]
    y_tr = y[event_train_mask]
    models = trainer.train_all(X_tr, y_tr)
    logger.info(f"Trained models: {', '.join(models.keys())}")
    return models, trainer


def _evaluate_models(
    models: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    event_test_mask: np.ndarray,
    feature_names: list[str],
    cfg,
    paths: dict
):
    evaluator = ModelEvaluator.from_model_config(cfg.model, output_dir=paths["reports_dir"])

    X_test = X.loc[event_test_mask]
    y_test = y[event_test_mask]
    logger.info(f"Test set: size={len(X_test):,}, positives={int(y_test.sum())} ({y_test.mean()*100:.1f}%)")

    comparison_df, fi_dict = evaluator.evaluate_multiple(models, X_test, y_test, feature_names)

    # Persist comparison metrics (overall)
    paths["overall_metrics_csv"].parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(paths["overall_metrics_csv"], index=False)
    logger.info(f"Saved overall metrics → {paths['overall_metrics_csv']}")

    # Persist feature importance (best model if available)
    if not comparison_df.empty and fi_dict:
        best_name = comparison_df.sort_values("pr_auc", ascending=False)["model"].iloc[0]
        # Save per-model feature importances as well
        for name, df_fi in fi_dict.items():
            out = paths["per_model_fi_dir"] / f"{name}.csv"
            df_fi.to_csv(out, index=False)
        if best_name in fi_dict:
            fi_dict[best_name].to_csv(paths["feature_importance_csv"], index=False)
            logger.info(f"Saved feature importance (best: {best_name}) → {paths['feature_importance_csv']}")

    # Pretty log
    if not comparison_df.empty:
        logger.info("Evaluation results (sorted by PR-AUC):")
        for row in comparison_df.sort_values("pr_auc", ascending=False).itertuples(index=False):
            logger.info(f"{row.model:>20s}  PR-AUC={row.pr_auc:.4f}  "
                        f"P@FPR={row.precision:.4f}  R@FPR={row.recall:.4f}  "
                        f"FPR={row.fpr:.4f}  thr={row.threshold:.4f}")

    return comparison_df, fi_dict


def _save_models(models: dict, paths: dict, cfg) -> None:
    """Save trained models to canonical artifact paths from config."""
    # Preferred artifact filenames from config
    artifact_targets = {
        "random_forest": getattr(cfg.paths.output, "random_forest_model", None),
        "logistic_regression": getattr(cfg.paths.output, "logistic_regression_model", None),
        "xgboost": getattr(cfg.paths.output, "xgboost_extended", None),
        "lightgbm": getattr(cfg.paths.output, "lightgbm_model", None),
        "isolation_forest": getattr(cfg.paths.output, "isolation_forest_model", None),
    }

    # Map script model names to config names
    name_mapping = {
        "logreg_cal": "logistic_regression",
        "rf_cal": "random_forest",
        "xgb": "xgboost",
        "lgbm": "lightgbm",
        "iso_forest": "isolation_forest",
    }

    for name, model in models.items():
        config_name = name_mapping.get(name, name)
        target = artifact_targets.get(config_name)
        if target is None:
            target = Path(paths["models_dir"]) / f"{name}.pkl"
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, target)
        kb = Path(target).stat().st_size / 1024
        logger.info(f"Saved {name} → {target} ({kb:.1f} KB)")


def _train_pattern_specific_models(
    events_df: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    event_train_mask: np.ndarray,
    event_test_mask: np.ndarray,
    trainer: ModelTrainer,
    cfg,
    paths: dict,
) -> tuple[dict, dict, pd.DataFrame]:
    """Train pattern-specific models."""
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING PATTERN-SPECIFIC MODELS")
    logger.info("="*60)
    
    pattern_trainer = PatternSpecificTrainer(cfg.model)
    
    pattern_models, pattern_thresholds, pattern_results = pattern_trainer.train_pattern_models(
        events_df=events_df,
        X=X,
        y=y,
        train_mask=event_train_mask,
        test_mask=event_test_mask,
        preprocess=trainer.preprocessor,
    )
    
    # Save pattern-specific results
    if not pattern_results.empty:
        pattern_results.to_csv(paths["reports_dir"] / "pattern_specific_results.csv", index=False)
        logger.info(f"Saved pattern results → {paths['reports_dir'] / 'pattern_specific_results.csv'}")
        
        # Save pattern models
        pattern_trainer.save_results(paths["models_dir"])
    
    return pattern_models, pattern_thresholds, pattern_results


# ---------------------------
# Main
# ---------------------------
def main() -> int:
    # Initialize logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    setup_logging(level="INFO", log_file=log_dir / "03_train_models.log", log_to_console=True)

    log_section("FUEL THEFT DETECTION - MODEL TRAINING PIPELINE")
    logger.info(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")

    try:
        # Load and validate configuration
        logger.info("Loading configuration from YAML…")
        cfg = load_config(
            detection_config_path=Path("config/detection_config.yaml"),
            model_config_path=Path("config/model_config.yaml"),
            path_config_path=Path("config/paths_config.yaml"),
        )
        validate_config(cfg)

        # Resolve paths and prepare directories
        paths = _resolve_paths_from_config(cfg)
        _setup_output_dirs(paths)

        # Load data
        log_section("LOADING DATA")
        raw_df, events_df = _load_raw_and_events(paths)

        # Split (temporal, per vehicle)
        log_section("CREATING TRAIN/TEST SPLIT")
        train_ratio = getattr(cfg.model.splitting, "train_ratio", 0.80)
        raw_train_mask = create_temporal_split_on_raw_data(raw_df, train_ratio=train_ratio)
        logger.info(f"Raw split: train={raw_train_mask.sum():,} rows ({raw_train_mask.mean()*100:.1f}%), "
                    f"test={(~raw_train_mask).sum():,} rows ({(~raw_train_mask).mean()*100:.1f}%)")

        events_df = map_events_to_split(events_df, raw_df, raw_train_mask)
        event_train_mask = events_df["is_train"].values
        event_test_mask = ~event_train_mask
        logger.info(f"Events split: train={event_train_mask.sum():,} ({event_train_mask.mean()*100:.1f}%), "
                    f"test={event_test_mask.sum():,} ({event_test_mask.mean()*100:.1f}%)")

        # Leakage guard
        validate_no_leakage(events_df, event_train_mask, event_test_mask)
        logger.info("No temporal overlap detected between train/test events.")

        # Features
        log_section("FEATURE ENGINEERING")
        events_enriched = _engineer_features(events_df, raw_df, event_train_mask, cfg)

        # Feature matrix
        log_section("PREPARING FEATURE MATRIX")
        X, y, feature_names = _prepare_feature_matrix(events_enriched)

        # Train baseline models
        log_section("TRAINING BASELINE MODELS")
        models, trainer = _train_models(X, y, event_train_mask, cfg)

        # Evaluate baseline models
        log_section("EVALUATING BASELINE MODELS")
        comparison_df, fi_dict = _evaluate_models(models, X, y, event_test_mask, feature_names, cfg, paths)

        # Save baseline models
        log_section("SAVING BASELINE MODELS")
        _save_models(models, paths, cfg)

        # Train pattern-specific models (optional)
        pattern_models = {}
        pattern_thresholds = {}
        pattern_results = pd.DataFrame()
        
        if hasattr(cfg.model, 'pattern_models') and cfg.model.pattern_models:
            pattern_models, pattern_thresholds, pattern_results = _train_pattern_specific_models(
                events_enriched,
                X,
                y,
                event_train_mask,
                event_test_mask,
                trainer,
                cfg,
                paths,
            )

        # Save enriched events with all features
        log_section("SAVING ENRICHED EVENTS")

        enriched_path = Path('data/events/events_with_features.csv')
        enriched_path.parent.mkdir(parents=True, exist_ok=True)
        events_enriched.to_csv(enriched_path, index=False)

        logger.info(f"✓ Saved events with {len(events_enriched.columns)} features")

        # Summary
        log_section("TRAINING SUMMARY")
        logger.info(f"Events processed: {len(events_enriched):,}")
        logger.info(f"Features used: {len(feature_names)}")
        logger.info(f"Baseline models trained: {len(models)}")
        if pattern_models:
            logger.info(f"Pattern-specific models trained: {len(pattern_models)}")
        if comparison_df is not None and not comparison_df.empty:
            best = comparison_df.sort_values("pr_auc", ascending=False).iloc[0]
            logger.info(f"Best baseline model: {best.model}  PR-AUC={best.pr_auc:.4f}")
        if not pattern_results.empty:
            logger.info("\nPattern-specific results:")
            print(pattern_results[['pattern', 'model_type', 'pr_auc', 'precision', 'recall']].to_string(index=False))
        logger.info(f"Outputs → models: {paths['models_dir']}  reports: {paths['reports_dir']}")

        logger.info("✓ Training pipeline completed successfully")
        logger.info("\n✓ Next step: Run scripts/04_evaluate_models.py")
        return 0

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())