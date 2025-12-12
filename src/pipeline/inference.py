"""
Inference Pipeline
Purpose: Production inference pipeline for real-time fuel theft detection.
Loads trained artifacts and processes new telemetry data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd

from src.config.loader import load_config
from src.detection.stationary import segment_stationary_periods
from src.detection.events import detect_events
from src.detection.nms import nms_events_dataframe, remove_exact_duplicates
from src.features.temporal import add_all_temporal_features
from src.data.preprocessor import DataPreprocessor
from src.features.engineering import FeatureEngineer
from src.config.settings import FeatureConfig
from src.data.loader import standardize_columns, validate_required_columns
from src.utils.io import load_json, load_model

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Production inference pipeline for fuel theft detection.
    
    Loads pre-trained models and artifacts, then processes new telemetry
    data to detect potential fuel theft events.
    
    Example:
        >>> pipeline = InferencePipeline(
        ...     model_path="data/models/random_forest_calibrated.pkl",
        ...     config_path="config",
        ... )
        >>> results = pipeline.predict(new_telemetry_df)
        >>> print(f"Detected {len(results)} suspicious events")
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        config_path: Union[str, Path],
        pattern_thresholds_path: Optional[Path] = None,
        noise_thresholds_path: Optional[Path] = None,
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to trained model (.pkl file)
            config_path: Path to config directory or detection_config.yaml
            pattern_thresholds_path: Optional path to pattern thresholds JSON
            noise_thresholds_path: Optional path to noise thresholds JSON
        """
        
        # Load configuration
        if Path(config_path).is_dir():
            full_config = load_config(
                detection_config_path=Path(config_path) / "detection_config.yaml",
                model_config_path=Path(config_path) / "model_config.yaml",
                path_config_path=Path(config_path) / "paths_config.yaml",
            )
            self.config = full_config.detection
            self.paths = full_config.paths
            self.model_config = getattr(full_config, "model", None)
        else:
            # Assume single config file
            from src.config.loader import load_detection_config
            self.config = load_detection_config(Path(config_path))
            self.paths = None
            self.model_config = None
        
        # Preprocessor aligns with training pipeline
        self.preprocessor = DataPreprocessor(self.config)
        
        # Load model (prefer joblib for sklearn pipelines)
        self.model = load_model(Path(model_path))
        logger.info(f"Loaded model from {model_path}")
        
        # Load thresholds (optional)
        self.pattern_thresholds = {}
        if pattern_thresholds_path and Path(pattern_thresholds_path).exists():
            self.pattern_thresholds = load_json(Path(pattern_thresholds_path))
            logger.info(f"Loaded pattern thresholds: {len(self.pattern_thresholds)} patterns")
        
        self.noise_thresholds = {}
        if noise_thresholds_path and Path(noise_thresholds_path).exists():
            loaded = load_json(Path(noise_thresholds_path))
            # Convert string keys back to tuples
            self.noise_thresholds = {
                tuple(k.split('_', 1)): (v['step'], v['cumulative'])
                for k, v in loaded.items()
            }
            logger.info(f"Loaded noise thresholds: {len(self.noise_thresholds)} vehicle-state pairs")
        
        logger.info("Inference pipeline initialized")
    
    def predict(
        self,
        telemetry_df: pd.DataFrame,
        return_raw_events: bool = False,
        confidence_threshold: float = 0.5,
        source_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Process telemetry data and predict fuel theft events.
        
        Args:
            telemetry_df: Raw telemetry DataFrame with columns:
                - vehicle_id, timestamp, total_fuel_gal, speed_kmh, ignition, etc.
            return_raw_events: If True, return all detected events with scores
            confidence_threshold: Minimum probability to classify as theft
        
        Returns:
            DataFrame with detected theft events and predictions:
            - All event columns (start_time, end_time, drop_gal, etc.)
            - theft_probability: Model confidence [0, 1]
            - is_theft: Binary prediction (>= confidence_threshold)
            - risk_level: Categorical risk (low/medium/high/critical)
        """
        
        logger.info(f"Processing {len(telemetry_df):,} telemetry rows")
        
        # 1. Preprocess telemetry
        telemetry_df = self._preprocess_telemetry(telemetry_df, source_name=source_name)
        
        # 2. Detect candidate events
        events_df = self._detect_events(telemetry_df)
        
        if events_df.empty:
            logger.info("No candidate events detected")
            return pd.DataFrame()
        
        logger.info(f"Detected {len(events_df)} candidate events")
        
        # 3. Engineer features
        events_df = self._engineer_features(events_df, telemetry_df)
        
        # 4. Predict with model
        predictions = self._predict_probabilities(events_df)
        
        events_df['theft_probability'] = predictions
        events_df['is_theft'] = (predictions >= confidence_threshold).astype(int)
        
        # 5. Assign risk levels
        events_df['risk_level'] = self._assign_risk_levels(predictions)
        
        # 6. Filter to predicted thefts (unless raw requested)
        if not return_raw_events:
            events_df = events_df[events_df['is_theft'] == 1].copy()
        
        logger.info(f"Predicted {(events_df['is_theft'] == 1).sum()} theft events")
        
        return events_df
    
    def predict_single_event(self, event_features: Dict) -> Dict:
        """
        Classify a single event given its features.
        
        Args:
            event_features: Dictionary with feature values
        
        Returns:
            Dictionary with:
            - is_theft: Binary prediction
            - confidence: Probability [0, 1]
            - risk_level: Categorical risk
            - explanation: List of contributing factors
        """
        
        # Convert to DataFrame
        event_df = pd.DataFrame([event_features])
        
        # Ensure required features exist
        required_features = self._get_required_features()
        missing = [f for f in required_features if f not in event_df.columns]
        
        if missing:
            # Fill missing with defaults
            for feat in missing:
                event_df[feat] = 0.0
        
        # Predict
        proba = self.model.predict_proba(event_df[required_features])[:, 1][0]
        
        # Get threshold (use pattern-specific if available)
        pattern = event_features.get('pattern', 'unknown')
        threshold = self.pattern_thresholds.get(pattern, 0.5)
        
        is_theft = int(proba >= threshold)
        risk_level = self._assign_risk_levels(np.array([proba]))[0]
        
        # Generate explanation
        explanation = self._explain_prediction(event_features, proba, is_theft)
        
        return {
            'is_theft': bool(is_theft),
            'confidence': float(proba),
            'risk_level': risk_level,
            'threshold_used': float(threshold),
            'explanation': explanation,
        }
    
    def process_stream(
        self,
        telemetry_batch: pd.DataFrame,
        batch_id: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> Dict:
        """
        Process a batch of streaming telemetry data.
        
        Designed for real-time systems where telemetry arrives in batches.
        
        Args:
            telemetry_batch: Batch of telemetry records
            batch_id: Optional identifier for this batch
        
        Returns:
            Dictionary with:
            - events: Detected events DataFrame
            - n_events: Number of events detected
            - n_thefts: Number predicted as theft
            - batch_id: Echo of batch identifier
            - timestamp: Processing timestamp
        """
        
        batch_id = batch_id or f"batch_{datetime.now():%Y%m%d_%H%M%S}"
        
        logger.info(f"Processing stream batch: {batch_id}")
        
        events_df = self.predict(
            telemetry_batch,
            return_raw_events=False,
            source_name=source_name or batch_id
        )
        
        result = {
            'events': events_df,
            'n_events': len(events_df),
            'n_thefts': int((events_df['is_theft'] == 1).sum()) if not events_df.empty else 0,
            'batch_id': batch_id,
            'timestamp': datetime.now().isoformat(),
        }
        
        logger.info(f"Batch {batch_id}: {result['n_thefts']} thefts in {result['n_events']} events")
        
        return result
    
    # ========== Internal Methods ==========
    
    def _standardize_telemetry(
        self,
        df: pd.DataFrame,
        source_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Standardize raw telemetry columns to canonical names expected by preprocessing.
        
        Uses the same flexible column mapping as the combination pipeline
        (handles Spanish/English headers, speed units, ignition booleans, etc.).
        """
        required = {"vehicle_id", "timestamp", "total_fuel_gal", "speed_kmh", "ignition"}
        
        # If already standardized, return as-is
        if required.issubset(set(df.columns)):
            return df
        
        mapping = getattr(self, "paths", None).column_mapping if getattr(self, "paths", None) else None
        logger.info("Standardizing telemetry columns for inference...")
        standardized = standardize_columns(
            df.copy(),
            column_mapping=mapping,
            source_filename=source_name
        )
        validate_required_columns(standardized, required=list(required))
        return standardized
    
    def _preprocess_telemetry(
        self,
        df: pd.DataFrame,
        source_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Preprocess telemetry data using same pipeline as training."""
        
        standardized = self._standardize_telemetry(df, source_name=source_name)
        
        # Reuse training preprocessor for consistency (normalize timestamps,
        # derive movement/fuel deltas, remove outliers, interpolate gaps, etc.)
        processed = self.preprocessor.fit_transform(standardized)
        return processed
    
    def _detect_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect candidate fuel theft events."""
        
        # Segment stationary periods
        segments = segment_stationary_periods(df, self.config)
        
        if segments.empty:
            return pd.DataFrame()
        
        # Use loaded noise thresholds or compute defaults
        if self.noise_thresholds:
            thresholds = self.noise_thresholds
        else:
            # Compute on-the-fly (not ideal for production)
            from src.detection.thresholds import compute_noise_thresholds
            train_mask = np.ones(len(df), dtype=bool)  # Use all data
            thresholds = compute_noise_thresholds(df, self.config, train_mask)
        
        # Detect events
        events_df = detect_events(segments, df, self.config, thresholds)
        
        if events_df.empty:
            return events_df
        
        # Remove duplicates and apply NMS
        events_df = remove_exact_duplicates(events_df)
        events_df = nms_events_dataframe(events_df, self.config.nms.iou_threshold)
        
        return events_df
    
    def _engineer_features(
        self,
        events_df: pd.DataFrame,
        telemetry_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Engineer features for prediction."""
        
        # Use same feature engineering stack as training
        fe_cfg = FeatureConfig(
            behavioral_lookback_hours=getattr(getattr(self.model_config, "features", None), "behavioral_lookback_hours", 2),
            enable_vehicle_normalization=getattr(getattr(self.model_config, "features", None), "enable_vehicle_normalization", True),
            enable_behavioral_features=getattr(getattr(self.model_config, "features", None), "enable_behavioral_features", True),
            enable_temporal_features=getattr(getattr(self.model_config, "features", None), "enable_temporal_features", True),
            enable_spatial_features=getattr(getattr(self.model_config, "features", None), "enable_spatial_features", True),
        )
        
        fe = FeatureEngineer(fe_cfg)
        engineered = fe.fit_transform(events_df, raw_df=telemetry_df)
        
        # Guarantee rate_gpm exists (defensive)
        if 'rate_gpm' not in engineered.columns and 'drop_gal' in engineered.columns and 'duration_min' in engineered.columns:
            engineered['rate_gpm'] = engineered['drop_gal'] / (engineered['duration_min'] + 1e-6)
        
        return engineered
    
    def _predict_probabilities(self, events_df: pd.DataFrame) -> np.ndarray:
        """Predict theft probabilities."""
        
        required_features = self._get_required_features()
        
        # Ensure all required features exist; fill missing with zeros
        missing = [f for f in required_features if f not in events_df.columns]
        if missing:
            logger.warning(f"Missing features in events_df; filling with zeros: {missing}")
            for feat in missing:
                events_df[feat] = 0.0
        
        # Prepare feature matrix
        X = events_df[required_features].copy()
        
        # Impute: numeric -> 0, categorical -> "unknown"
        num_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = [c for c in X.columns if c not in num_cols]
        if len(num_cols):
            X[num_cols] = X[num_cols].fillna(0.0)
        if len(cat_cols):
            X[cat_cols] = X[cat_cols].fillna("unknown")
        
        # Predict
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return probabilities
    
    def _get_required_features(self) -> List[str]:
        """Get list of required features for model."""
        
        # Standard feature set (matches training)
        features = [
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
            "is_hotspot", "is_weekend", "is_night",
            "pattern",
        ]
        
        # Prefer model-declared feature names if available
        try:
            if hasattr(self.model, 'feature_names_in_'):
                return list(self.model.feature_names_in_)
        except Exception:
            pass
        
        return features
    
    def _assign_risk_levels(self, probabilities: np.ndarray) -> np.ndarray:
        """Assign categorical risk levels based on probabilities."""
        
        risk_levels = np.full(len(probabilities), "low", dtype=object)
        
        risk_levels[probabilities >= 0.3] = "medium"
        risk_levels[probabilities >= 0.6] = "high"
        risk_levels[probabilities >= 0.85] = "critical"
        
        return risk_levels
    
    def _explain_prediction(
        self,
        event_features: Dict,
        probability: float,
        is_theft: int
    ) -> List[str]:
        """Generate human-readable explanation for prediction."""
        
        explanations = []
        
        # Confidence level
        if is_theft:
            confidence = "high" if probability > 0.8 else "moderate" if probability > 0.6 else "low"
            explanations.append(f"{confidence} confidence theft ({probability:.1%})")
        else:
            explanations.append(f"Low theft probability ({probability:.1%})")
        
        # Key factors
        drop_gal = event_features.get('drop_gal', 0)
        duration_min = event_features.get('duration_min', 0)
        rate_gpm = event_features.get('rate_gpm', 0)
        
        if drop_gal > 10:
            explanations.append(f"Large fuel drop: {drop_gal:.1f} gallons")
        elif drop_gal > 5:
            explanations.append(f"Moderate fuel drop: {drop_gal:.1f} gallons")
        
        if duration_min < 10:
            explanations.append(f"Short duration: {duration_min:.1f} minutes")
        elif duration_min > 30:
            explanations.append(f"Extended duration: {duration_min:.1f} minutes")
        
        if rate_gpm > 0.5:
            explanations.append(f"Fast drain rate: {rate_gpm:.2f} gal/min")
        
        # Hotspot
        if event_features.get('is_hotspot'):
            explanations.append("Occurred at known hotspot location")
        
        # Pattern
        pattern = event_features.get('pattern', 'unknown')
        if pattern != 'unknown':
            explanations.append(f"Pattern: {pattern}")
        
        return explanations


class BatchInferencePipeline:
    """
    Batch inference pipeline for processing multiple files.
    
    Optimized for processing large historical datasets or
    multiple vehicle fleets in parallel.
    """
    
    def __init__(self, inference_pipeline: InferencePipeline):
        """
        Initialize batch pipeline.
        
        Args:
            inference_pipeline: Configured InferencePipeline instance
        """
        self.pipeline = inference_pipeline
        logger.info("Batch inference pipeline initialized")
    
    def process_files(
        self,
        file_paths: List[Path],
        output_dir: Path,
        save_results: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Process multiple telemetry files.
        
        Args:
            file_paths: List of CSV file paths
            output_dir: Directory to save results
            save_results: If True, save predictions to files
        
        Returns:
            Dictionary mapping filename to predictions DataFrame
        """
        
        results = {}
        
        for file_path in file_paths:
            logger.info(f"Processing {file_path.name}...")
            
            try:
                # Load telemetry
                telemetry_df = pd.read_csv(file_path)
                
                # Predict
                predictions = self.pipeline.predict(telemetry_df)
                
                results[file_path.name] = predictions
                
                # Save if requested
                if save_results and not predictions.empty:
                    output_path = output_dir / f"predictions_{file_path.stem}.csv"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    predictions.to_csv(output_path, index=False)
                    logger.info(f"Saved {len(predictions)} predictions → {output_path}")
                
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                results[file_path.name] = pd.DataFrame()
        
        logger.info(f"Batch processing complete: {len(results)} files")
        
        return results
