"""
Integration tests for pipeline components.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.pipeline.train import TrainingPipeline
from src.pipeline.inference import InferencePipeline
from src.pipeline.validation import (
    create_temporal_split_on_raw_data,
    map_events_to_split,
    validate_no_leakage,
    validate_split_balance,
)


@pytest.mark.integration
class TestTrainingPipeline:
    """Integration tests for TrainingPipeline."""
    
    def test_pipeline_initialization(self, complete_config):
        """Test pipeline initialization."""
        pipeline = TrainingPipeline(complete_config)
        
        assert pipeline.config is not None
        assert pipeline.data_loader is not None
        assert pipeline.preprocessor is not None
        assert pipeline.feature_engineer is not None
        assert pipeline.model_trainer is not None
    
    @pytest.mark.slow
    def test_pipeline_run_minimal(self, complete_config, sample_telemetry_df, temp_output_dir):
        """Test running pipeline with minimal data."""
        # Save sample data
        data_path = temp_output_dir / "test_data.csv"
        sample_telemetry_df.to_csv(data_path, index=False)
        
        # Update config path
        complete_config.paths.input.combined_csv = data_path
        
        pipeline = TrainingPipeline(complete_config)
        
        # Run pipeline
        results = pipeline.run(data_path)
        
        # Check results structure
        assert 'events_df' in results
        assert 'models' in results
        assert 'metrics' in results
        
        # Models should be trained
        assert len(results['models']) > 0
    
    def test_pipeline_handles_no_events(self, complete_config, temp_output_dir):
        """Test pipeline when no events are detected."""
        # Create data with no theft events
        df = pd.DataFrame({
            'vehicle_id': ['VEH1'] * 100,
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min', tz='UTC'),
            'total_fuel_gal': [50.0] * 100,  # Constant fuel
            'speed_kmh': [60] * 100,  # Always moving
            'ignition': [True] * 100,
            'latitude': [40.7] * 100,
            'longitude': [-74.0] * 100,
        })
        
        # Add derived columns
        df['dt_s'] = 60.0
        df['dfuel'] = 0.0
        df['moving'] = True
        df['stationary_on'] = False
        df['ign_off'] = False
        df['stationary'] = False
        
        data_path = temp_output_dir / "no_events_data.csv"
        df.to_csv(data_path, index=False)
        
        complete_config.paths.input.combined_csv = data_path
        
        pipeline = TrainingPipeline(complete_config)
        
        # Should handle gracefully
        results = pipeline.run(data_path)
        
        assert 'error' in results or len(results.get('events_df', pd.DataFrame())) == 0


@pytest.mark.integration
class TestInferencePipeline:
    """Integration tests for InferencePipeline."""
    
    @pytest.fixture
    def trained_simple_model(self, temp_output_dir):
        """Create a simple trained model for testing."""
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        
        X_train = np.random.randn(100, 5)
        y_train = np.random.choice([0, 1], 100, p=[0.7, 0.3])
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        model_path = temp_output_dir / "test_model.pkl"
        joblib.dump(model, model_path)
        
        return model_path
    
    def test_inference_pipeline_initialization(self, trained_simple_model, temp_output_dir, complete_config):
        """Test InferencePipeline initialization."""
        # Save minimal config
        config_dir = temp_output_dir / "config"
        config_dir.mkdir(exist_ok=True)
        
        from src.config.loader import save_detection_config
        save_detection_config(complete_config.detection, config_dir / "detection_config.yaml")
        
        pipeline = InferencePipeline(
            model_path=trained_simple_model,
            config_path=config_dir / "detection_config.yaml",
        )
        
        assert pipeline.model is not None
        assert pipeline.config is not None
    
    def test_inference_predict_single_event(self, trained_simple_model, temp_output_dir, complete_config):
        """Test predicting single event."""
        config_dir = temp_output_dir / "config"
        config_dir.mkdir(exist_ok=True)
        
        from src.config.loader import save_detection_config
        save_detection_config(complete_config.detection, config_dir / "detection_config.yaml")
        
        pipeline = InferencePipeline(
            model_path=trained_simple_model,
            config_path=config_dir / "detection_config.yaml",
        )
        
        event_features = {
            'drop_gal': 6.5,
            'duration_min': 8.0,
            'rate_gpm': 0.8,
            'pattern': 'short_4_10m_3gal',
        }
        
        result = pipeline.predict_single_event(event_features)
        
        # Check result structure
        assert 'is_theft' in result
        assert 'confidence' in result
        assert 'risk_level' in result
        assert 'explanation' in result
        
        assert isinstance(result['is_theft'], bool)
        assert 0 <= result['confidence'] <= 1
        assert result['risk_level'] in ['low', 'medium', 'high', 'critical']


@pytest.mark.integration
class TestValidationUtilities:
    """Integration tests for validation utilities."""
    
    def test_temporal_split_integration(self, sample_telemetry_df):
        """Test temporal split on real-ish data."""
        train_mask = create_temporal_split_on_raw_data(
            sample_telemetry_df,
            train_ratio=0.80
        )
        
        assert len(train_mask) == len(sample_telemetry_df)
        assert train_mask.dtype == bool
        
        # Should have roughly 80% train
        train_pct = train_mask.mean()
        assert 0.75 < train_pct < 0.85
    
    def test_map_events_integration(self, sample_telemetry_df, sample_events_df):
        """Test mapping events to split."""
        train_mask = create_temporal_split_on_raw_data(
            sample_telemetry_df,
            train_ratio=0.80
        )
        
        events_with_split = map_events_to_split(
            sample_events_df,
            sample_telemetry_df,
            train_mask
        )
        
        assert 'is_train' in events_with_split.columns
        assert len(events_with_split) == len(sample_events_df)
        assert events_with_split['is_train'].dtype == bool
    
    def test_validate_no_leakage_integration(self, sample_telemetry_df, sample_events_df):
        """Test leakage validation on real-ish data."""
        train_mask = create_temporal_split_on_raw_data(
            sample_telemetry_df,
            train_ratio=0.80
        )
        
        events_with_split = map_events_to_split(
            sample_events_df,
            sample_telemetry_df,
            train_mask
        )
        
        event_train_mask = events_with_split['is_train'].values
        event_test_mask = ~event_train_mask
        
        is_valid, report = validate_no_leakage(
            events_with_split,
            event_train_mask,
            event_test_mask,
        )
        
        # Should be valid (no leakage)
        assert is_valid
        assert report is not None
        assert not report.empty
    
    def test_validate_split_balance_integration(self, sample_events_df):
        """Test split balance validation."""
        # Create artificial split
        train_mask = np.array([True] * 2 + [False] * 1)
        test_mask = ~train_mask
        
        is_balanced = validate_split_balance(
            sample_events_df,
            train_mask,
            test_mask,
            min_train_positives=1,
            min_test_positives=1,
        )
        
        # May or may not be balanced depending on data
        assert isinstance(is_balanced, bool)


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Integration tests for end-to-end workflows."""
    
    @pytest.mark.slow
    def test_detection_to_features_workflow(
        self,
        sample_telemetry_df,
        detection_config,
        feature_config
    ):
        """Test workflow from detection to feature engineering."""
        from src.detection.stationary import segment_stationary_periods
        from src.detection.thresholds import compute_noise_thresholds
        from src.detection.events import detect_events
        from src.features.engineering import FeatureEngineer
        
        # 1. Segment stationary periods
        segments = segment_stationary_periods(sample_telemetry_df, detection_config)
        
        assert not segments.empty, "Should detect some stationary segments"
        
        # 2. Compute thresholds
        train_mask = np.ones(len(sample_telemetry_df), dtype=bool)
        thresholds = compute_noise_thresholds(
            sample_telemetry_df,
            detection_config,
            train_mask
        )
        
        assert len(thresholds) > 0, "Should compute thresholds"
        
        # 3. Detect events
        events_df = detect_events(
            segments,
            sample_telemetry_df,
            detection_config,
            thresholds
        )
        
        if events_df.empty:
            pytest.skip("No events detected in sample data")
        
        # 4. Engineer features
        fe = FeatureEngineer(feature_config)
        event_train_mask = np.ones(len(events_df), dtype=bool)
        
        enriched_events = fe.fit_transform(
            events_df,
            sample_telemetry_df,
            event_train_mask
        )
        
        # Should have more features than input
        assert len(enriched_events.columns) > len(events_df.columns)
        assert len(enriched_events) == len(events_df)
    
    @pytest.mark.slow
    def test_features_to_training_workflow(
        self,
        sample_events_df,
        model_config
    ):
        """Test workflow from features to model training."""
        from src.models.training import ModelTrainer
        
        # Prepare features
        X = sample_events_df[['drop_gal', 'duration_min', 'min_step_gal']].copy()
        y = sample_events_df['y'].values
        
        # Train models
        trainer = ModelTrainer(model_config)
        models = trainer.train_all(X, y)
        
        # Should train at least one model
        assert len(models) > 0
        
        # Models should be able to predict
        for name, model in models.items():
            proba = model.predict_proba(X)
            assert proba.shape == (len(X), 2)
            assert (proba >= 0).all() and (proba <= 1).all()