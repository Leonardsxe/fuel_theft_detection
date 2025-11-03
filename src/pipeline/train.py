"""
Training Pipeline
Purpose: End-to-end orchestration of the fuel theft detection training pipeline.
Encapsulates: data loading → splitting → detection → feature engineering → training → evaluation
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from src.config.settings import Config
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.validator import DataValidator
from src.detection.stationary import segment_stationary_periods
from src.detection.thresholds import compute_noise_thresholds, NoiseThresholdCalculator
from src.detection.events import detect_events
from src.detection.nms import nms_events_dataframe, remove_exact_duplicates
from src.clustering.hotspots import cluster_stationary_points
from src.clustering.assignment import assign_events_to_clusters
from src.features.engineering import FeatureEngineer
from src.models.training import ModelTrainer
from src.models.evaluation import ModelEvaluator
from src.models.pattern_models import PatternSpecificTrainer
from src.utils.splitting import (
    create_temporal_split_on_raw_data,
    map_events_to_split,
    validate_no_leakage,
)
from src.utils.io import save_json, save_pickle
from src.utils.timezone import ensure_series_utc

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    End-to-end training pipeline for fuel theft detection.
    
    Orchestrates the complete workflow:
    1. Data loading and validation
    2. Preprocessing and outlier removal
    3. Train/test splitting
    4. Event detection (with train-only threshold/cluster fitting)
    5. Feature engineering
    6. Model training (baseline + pattern-specific)
    7. Evaluation and artifact saving
    
    Example:
        >>> from src.config.loader import load_config
        >>> config = load_config()
        >>> pipeline = TrainingPipeline(config)
        >>> results = pipeline.run("data/processed/combined_dataset.csv")
        >>> print(f"Best model: {results['best_model']}")
    """
    
    def __init__(self, config: Config):
        """
        Initialize training pipeline.
        
        Args:
            config: Complete configuration object
        """
        self.config = config
        self.raw_df: Optional[pd.DataFrame] = None
        self.events_df: Optional[pd.DataFrame] = None
        self.models: Dict = {}
        self.pattern_models: Dict = {}
        self.artifacts: Dict = {}
        
        # Initialize components
        self.data_loader = DataLoader(config.paths)
        self.preprocessor = DataPreprocessor(config.detection)
        self.validator = DataValidator()
        self.feature_engineer = FeatureEngineer(config.model.features)
        self.model_trainer = ModelTrainer(config.model)
        self.evaluator = ModelEvaluator.from_model_config(
            config.model,
            output_dir=config.paths.data.reports
        )
        
        logger.info("Training pipeline initialized")
    
    def run(self, raw_data_path: Path) -> Dict:
        """
        Execute complete training pipeline.
        
        Args:
            raw_data_path: Path to combined/preprocessed raw data CSV
        
        Returns:
            Dictionary with pipeline results:
            - events_df: Detected events DataFrame
            - models: Trained models
            - metrics: Evaluation metrics
            - best_model: Name of best performing model
            - artifacts_paths: Paths to saved artifacts
        """
        
        logger.info("="*70)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("="*70)
        start_time = datetime.now()
        
        try:
            # 1. Load and validate data
            logger.info("\n[1/8] Loading and validating data...")
            self.raw_df = self._load_and_validate_data(raw_data_path)
            
            # 2. Preprocess data
            logger.info("\n[2/8] Preprocessing data...")
            self.raw_df = self._preprocess_data(self.raw_df)
            
            # 3. Create train/test split
            logger.info("\n[3/8] Creating train/test split...")
            train_mask = self._create_split(self.raw_df)
            
            # 4. Detect events (fit on train only)
            logger.info("\n[4/8] Detecting events...")
            self.events_df = self._detect_events(self.raw_df, train_mask)
            
            if self.events_df.empty:
                logger.error("No events detected - cannot train models")
                return {"error": "No events detected"}
            
            # 5. Map events to train/test split
            logger.info("\n[5/8] Mapping events to train/test split...")
            self.events_df = self._map_events_to_split(self.events_df, self.raw_df, train_mask)
            event_train_mask = self.events_df["is_train"].values
            event_test_mask = ~event_train_mask
            
            # 6. Engineer features
            logger.info("\n[6/8] Engineering features...")
            self.events_df = self._engineer_features(self.events_df, self.raw_df, event_train_mask)
            
            # 7. Train models
            logger.info("\n[7/8] Training models...")
            X, y, feature_names = self._prepare_features(self.events_df)
            self.models = self._train_models(X, y, event_train_mask)
            
            # Train pattern-specific models (optional)
            if self.config.model.pattern_models:
                logger.info("\n[7b/8] Training pattern-specific models...")
                self.pattern_models = self._train_pattern_models(
                    self.events_df, X, y, event_train_mask, event_test_mask
                )
            
            # 8. Evaluate models
            logger.info("\n[8/8] Evaluating models...")
            metrics = self._evaluate_models(X, y, event_test_mask, feature_names)
            
            # Save artifacts
            logger.info("\nSaving artifacts...")
            artifact_paths = self._save_artifacts()
            
            # Compute elapsed time
            elapsed = (datetime.now() - start_time).total_seconds()
            
            # Summary
            logger.info("\n" + "="*70)
            logger.info("TRAINING PIPELINE COMPLETE")
            logger.info("="*70)
            logger.info(f"Elapsed time: {elapsed:.1f}s")
            logger.info(f"Events detected: {len(self.events_df):,}")
            logger.info(f"Features engineered: {len(feature_names)}")
            logger.info(f"Models trained: {len(self.models)}")
            if self.pattern_models:
                logger.info(f"Pattern-specific models: {len(self.pattern_models)}")
            
            best_model = metrics['comparison_df'].iloc[0]['model'] if not metrics['comparison_df'].empty else None
            if best_model:
                best_metrics = metrics['comparison_df'].iloc[0]
                logger.info(f"\nBest model: {best_model}")
                logger.info(f"  PR-AUC: {best_metrics['pr_auc']:.4f}")
                logger.info(f"  Precision: {best_metrics['precision']:.4f}")
                logger.info(f"  Recall: {best_metrics['recall']:.4f}")
            
            return {
                "events_df": self.events_df,
                "models": self.models,
                "pattern_models": self.pattern_models,
                "metrics": metrics,
                "best_model": best_model,
                "artifacts_paths": artifact_paths,
                "elapsed_seconds": elapsed,
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
    
    def _load_and_validate_data(self, path: Path) -> pd.DataFrame:
        """Load and validate raw data."""
        df = pd.read_csv(path)
        df["timestamp"] = ensure_series_utc(df["timestamp"])
        
        # Validate
        quality_report = self.validator.generate_quality_report(df)
        
        # Save quality report
        quality_path = self.config.paths.output.data_quality_report
        quality_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(quality_report, quality_path)
        
        logger.info(f"Loaded {len(df):,} rows, {df['vehicle_id'].nunique()} vehicles")
        logger.info(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
        
        return df
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess and clean data."""
        df = self.preprocessor.fit_transform(df)
        logger.info(f"After preprocessing: {len(df):,} rows")
        return df
    
    def _create_split(self, df: pd.DataFrame) -> np.ndarray:
        """Create temporal train/test split."""
        train_mask = create_temporal_split_on_raw_data(
            df,
            train_ratio=self.config.model.splitting.train_ratio
        )
        
        logger.info(f"Train: {train_mask.sum():,} rows ({train_mask.mean()*100:.1f}%)")
        logger.info(f"Test: {(~train_mask).sum():,} rows ({(~train_mask).mean()*100:.1f}%)")
        
        return train_mask
    
    def _detect_events(self, df: pd.DataFrame, train_mask: np.ndarray) -> pd.DataFrame:
        """Detect fuel theft events (fit thresholds/clusters on train only)."""
        
        # Segment stationary periods
        segments = segment_stationary_periods(df, self.config.detection)
        logger.info(f"Segmented {len(segments)} stationary periods")
        
        # Compute noise thresholds (TRAIN ONLY)
        thresholds = compute_noise_thresholds(df, self.config.detection, train_mask)
        logger.info(f"Computed thresholds for {len(thresholds) // 2} vehicles x 2 states")
        
        # Store thresholds artifact
        self.artifacts['noise_thresholds'] = thresholds
        
        # Cluster stationary locations (TRAIN ONLY)
        stationary_mask = df['stationary']
        stationary_with_coords = df[
            stationary_mask & df['latitude'].notna() & df['longitude'].notna()
        ].copy()
        
        stationary_train_mask = train_mask[stationary_with_coords.index]
        stationary_pts = cluster_stationary_points(
            stationary_with_coords,
            self.config.detection.clustering,
            stationary_train_mask
        )
        
        if not stationary_pts.empty:
            n_clusters = stationary_pts[stationary_pts["cluster_id"] >= 0]["cluster_id"].nunique()
            logger.info(f"Clustered {n_clusters} hotspot locations")
        
        # Store clustering artifact
        self.artifacts['stationary_pts'] = stationary_pts
        
        # Detect events
        events_df = detect_events(segments, df, self.config.detection, thresholds)
        
        if events_df.empty:
            logger.warning("No events detected")
            return events_df
        
        logger.info(f"Detected {len(events_df)} raw events")
        
        # Remove duplicates
        events_df = remove_exact_duplicates(events_df)
        
        # Apply NMS
        events_df = nms_events_dataframe(events_df, self.config.detection.nms.iou_threshold)
        logger.info(f"After NMS: {len(events_df)} unique events")
        
        # Assign to hotspot clusters
        events_df = assign_events_to_clusters(events_df, stationary_pts)
        
        n_hotspot = events_df["is_hotspot"].sum() if "is_hotspot" in events_df.columns else 0
        logger.info(f"Hotspot events: {n_hotspot} ({n_hotspot/len(events_df)*100:.1f}%)")
        
        return events_df
    
    def _map_events_to_split(
        self,
        events_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        train_mask: np.ndarray
    ) -> pd.DataFrame:
        """Map detected events to train/test split."""
        events_df = map_events_to_split(events_df, raw_df, train_mask)
        
        event_train_mask = events_df["is_train"].values
        event_test_mask = ~event_train_mask
        
        logger.info(f"Train events: {event_train_mask.sum():,} ({event_train_mask.mean()*100:.1f}%)")
        logger.info(f"Test events: {event_test_mask.sum():,} ({event_test_mask.mean()*100:.1f}%)")
        
        # Validate no leakage
        validate_no_leakage(events_df, event_train_mask, event_test_mask)
        logger.info("✓ No temporal leakage detected")
        
        return events_df
    
    def _engineer_features(
        self,
        events_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        train_mask: np.ndarray
    ) -> pd.DataFrame:
        """Engineer features (fit normalizers on train only)."""
        
        events_df = self.feature_engineer.fit_transform(events_df, raw_df, train_mask)
        
        new_features = set(events_df.columns) - set([
            'vehicle_id', 'start_time', 'end_time', 'pattern', 'y', 'is_train'
        ])
        logger.info(f"Engineered {len(new_features)} features")
        
        return events_df
    
    def _prepare_features(self, events_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, list]:
        """Prepare feature matrix and labels."""
        
        from src.models.training import _dedupe_columns
        
        # Feature columns
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
        X = _dedupe_columns(X)
        feature_names = list(X.columns)
        
        y = events_df["y"].fillna(0).astype(int).values if "y" in events_df.columns else np.zeros(len(events_df), dtype=int)
        
        logger.info(f"Feature matrix: {X.shape}, positives: {int(y.sum())} ({y.mean()*100:.1f}%)")
        
        return X, y, feature_names
    
    def _train_models(self, X: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray) -> Dict:
        """Train baseline models."""
        
        X_train = X.loc[train_mask]
        y_train = y[train_mask]
        
        models = self.model_trainer.train_all(X_train, y_train)
        
        logger.info(f"Trained {len(models)} baseline models: {', '.join(models.keys())}")
        
        return models
    
    def _train_pattern_models(
        self,
        events_df: pd.DataFrame,
        X: pd.DataFrame,
        y: np.ndarray,
        train_mask: np.ndarray,
        test_mask: np.ndarray,
    ) -> Dict:
        """Train pattern-specific models."""
        
        pattern_trainer = PatternSpecificTrainer(self.config.model)
        
        pattern_models, pattern_thresholds, pattern_results = pattern_trainer.train_pattern_models(
            events_df=events_df,
            X=X,
            y=y,
            train_mask=train_mask,
            test_mask=test_mask,
            preprocess=self.model_trainer.preprocessor,
        )
        
        # Store artifacts
        self.artifacts['pattern_thresholds'] = pattern_thresholds
        self.artifacts['pattern_results'] = pattern_results
        
        logger.info(f"Trained {len(pattern_models)} pattern-specific models")
        
        return pattern_models
    
    def _evaluate_models(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_mask: np.ndarray,
        feature_names: list
    ) -> Dict:
        """Evaluate all models."""
        
        X_test = X.loc[test_mask]
        y_test = y[test_mask]
        
        comparison_df, fi_dict = self.evaluator.evaluate_multiple(
            self.models,
            X_test,
            y_test,
            feature_names
        )
        
        return {
            'comparison_df': comparison_df,
            'feature_importance': fi_dict,
        }
    
    def _save_artifacts(self) -> Dict[str, Path]:
        """Save all pipeline artifacts."""
        
        paths = {}
        
        # Save events
        events_path = self.config.paths.output.events_csv
        events_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_df.to_csv(events_path, index=False)
        paths['events'] = events_path
        logger.info(f"Saved events → {events_path}")
        
        # Save models
        models_dir = self.config.paths.data.models
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = models_dir / f"{name}.pkl"
            save_pickle(model, model_path)
            paths[f'model_{name}'] = model_path
        
        logger.info(f"Saved {len(self.models)} models → {models_dir}")
        
        # Save pattern models
        if self.pattern_models:
            for name, model in self.pattern_models.items():
                model_path = models_dir / f"pattern_model_{name}.pkl"
                save_pickle(model, model_path)
                paths[f'pattern_model_{name}'] = model_path
            
            logger.info(f"Saved {len(self.pattern_models)} pattern models → {models_dir}")
        
        # Save noise thresholds
        if 'noise_thresholds' in self.artifacts:
            thresholds_path = self.config.paths.output.noise_thresholds
            thresholds_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert tuple keys to strings for JSON serialization
            thresholds_serializable = {
                f"{vid}_{state}": {"step": step, "cumulative": cum}
                for (vid, state), (step, cum) in self.artifacts['noise_thresholds'].items()
            }
            save_json(thresholds_serializable, thresholds_path)
            paths['noise_thresholds'] = thresholds_path
            logger.info(f"Saved noise thresholds → {thresholds_path}")
        
        # Save pattern thresholds
        if 'pattern_thresholds' in self.artifacts:
            pattern_thresholds_path = self.config.paths.output.pattern_thresholds
            pattern_thresholds_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(self.artifacts['pattern_thresholds'], pattern_thresholds_path)
            paths['pattern_thresholds'] = pattern_thresholds_path
            logger.info(f"Saved pattern thresholds → {pattern_thresholds_path}")
        
        return paths