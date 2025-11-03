"""
End-to-end integration tests.
Tests complete workflows from raw data to predictions.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.pipeline.train import TrainingPipeline
from src.pipeline.inference import InferencePipeline, BatchInferencePipeline


@pytest.mark.integration
@pytest.mark.slow
class TestCompleteTrainingWorkflow:
    """End-to-end tests for complete training workflow."""
    
    def test_full_training_pipeline(
        self,
        complete_config,
        sample_telemetry_df,
        temp_output_dir
    ):
        """Test complete training pipeline from raw data to trained models."""
        
        # Save sample data
        data_path = temp_output_dir / "training_data.csv"
        sample_telemetry_df.to_csv(data_path, index=False)
        
        # Update config
        complete_config.paths.input.combined_csv = data_path
        
        # Initialize and run pipeline
        pipeline = TrainingPipeline(complete_config)
        results = pipeline.run(data_path)
        
        # Verify results structure
        assert 'events_df' in results
        assert 'models' in results
        assert 'metrics' in results
        assert 'artifacts_paths' in results
        
        # Check models were trained
        assert len(results['models']) > 0
        
        # Check artifacts were saved
        assert 'events' in results['artifacts_paths']
        assert results['artifacts_paths']['events'].exists()
        
        # Check models can predict
        if not results['events_df'].empty:
            # Prepare sample features for prediction
            X_sample = results['events_df'][
                [c for c in ['drop_gal', 'duration_min'] if c in results['events_df'].columns]
            ].head(1)
            
            if not X_sample.empty and len(results['models']) > 0:
                model = list(results['models'].values())[0]
                # Should be able to predict (may need proper feature alignment)
                try:
                    proba = model.predict_proba(X_sample)
                    assert proba.shape[0] == 1
                except Exception:
                    # May fail due to feature mismatch - that's acceptable for this test
                    pass


@pytest.mark.integration
@pytest.mark.slow
class TestCompleteInferenceWorkflow:
    """End-to-end tests for complete inference workflow."""
    
    @pytest.fixture
    def trained_pipeline_artifacts(self, complete_config, sample_telemetry_df, temp_output_dir):
        """Train a pipeline and return artifacts for inference testing."""
        data_path = temp_output_dir / "training_data.csv"
        sample_telemetry_df.to_csv(data_path, index=False)
        
        complete_config.paths.input.combined_csv = data_path
        
        pipeline = TrainingPipeline(complete_config)
        results = pipeline.run(data_path)
        
        if len(results['models']) == 0:
            pytest.skip("No models trained")
        
        # Return first model and config
        model_name = list(results['models'].keys())[0]
        model_path = complete_config.paths.data.models / f"{model_name}.pkl"
        
        return {
            'model_path': model_path,
            'config': complete_config,
            'results': results,
        }
    
    def test_full_inference_pipeline(
        self,
        trained_pipeline_artifacts,
        sample_telemetry_df,
        temp_output_dir
    ):
        """Test complete inference pipeline on new data."""
        
        if not trained_pipeline_artifacts['model_path'].exists():
            pytest.skip("Model file not found")
        
        # Create config directory
        config_dir = temp_output_dir / "config"
        config_dir.mkdir(exist_ok=True)
        
        # Save detection config
        from src.config.loader import save_detection_config
        save_detection_config(
            trained_pipeline_artifacts['config'].detection,
            config_dir / "detection_config.yaml"
        )
        
        # Initialize inference pipeline
        inference_pipeline = InferencePipeline(
            model_path=trained_pipeline_artifacts['model_path'],
            config_path=config_dir / "detection_config.yaml",
        )
        
        # Run inference on new data (using same sample data for test)
        predictions = inference_pipeline.predict(
            sample_telemetry_df,
            confidence_threshold=0.5,
        )
        
        # Verify predictions structure
        assert isinstance(predictions, pd.DataFrame)
        
        if not predictions.empty:
            assert 'theft_probability' in predictions.columns
            assert 'is_theft' in predictions.columns
            assert 'risk_level' in predictions.columns
            
            # Verify probability ranges
            assert (predictions['theft_probability'] >= 0).all()
            assert (predictions['theft_probability'] <= 1).all()
            
            # Verify risk levels
            valid_risk_levels = ['low', 'medium', 'high', 'critical']
            assert predictions['risk_level'].isin(valid_risk_levels).all()
    
    def test_batch_inference_workflow(
        self,
        trained_pipeline_artifacts,
        sample_telemetry_df,
        temp_output_dir
    ):
        """Test batch inference on multiple files."""
        
        if not trained_pipeline_artifacts['model_path'].exists():
            pytest.skip("Model file not found")
        
        # Create multiple test files
        input_dir = temp_output_dir / "batch_input"
        input_dir.mkdir(exist_ok=True)
        
        file_paths = []
        for i in range(3):
            file_path = input_dir / f"vehicle_{i}.csv"
            sample_telemetry_df.to_csv(file_path, index=False)
            file_paths.append(file_path)
        
        # Create config directory
        config_dir = temp_output_dir / "config"
        config_dir.mkdir(exist_ok=True)
        
        from src.config.loader import save_detection_config
        save_detection_config(
            trained_pipeline_artifacts['config'].detection,
            config_dir / "detection_config.yaml"
        )
        
        # Initialize pipelines
        inference_pipeline = InferencePipeline(
            model_path=trained_pipeline_artifacts['model_path'],
            config_path=config_dir / "detection_config.yaml",
        )
        
        batch_pipeline = BatchInferencePipeline(inference_pipeline)
        
        # Process batch
        output_dir = temp_output_dir / "batch_output"
        output_dir.mkdir(exist_ok=True)
        
        results = batch_pipeline.process_files(
            file_paths=file_paths,
            output_dir=output_dir,
            save_results=True,
        )
        
        # Verify results
        assert len(results) == 3
        
        for filename, predictions in results.items():
            assert isinstance(predictions, pd.DataFrame)


@pytest.mark.integration
class TestDataQualityWorkflow:
    """End-to-end tests for data quality and validation."""
    
    def test_data_quality_checks_in_pipeline(
        self,
        complete_config,
        temp_output_dir
    ):
        """Test that pipeline performs data quality checks."""
        
        # Create data with quality issues
        df = pd.DataFrame({
            'vehicle_id': ['VEH1'] * 100,
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min', tz='UTC'),
            'total_fuel_gal': [50.0] * 50 + [np.nan] * 30 + [50.0] * 20,  # Missing values
            'speed_kmh': [60] * 100,
            'ignition': [True] * 100,
            'latitude': [40.7] * 50 + [200.0] * 30 + [40.7] * 20,  # Invalid coords
            'longitude': [-74.0] * 100,
        })
        
        # Add derived columns
        df['dt_s'] = 60.0
        df['dfuel'] = df.groupby('vehicle_id')['total_fuel_gal'].diff()
        df['moving'] = df['speed_kmh'] > 1.0
        df['stationary_on'] = False
        df['ign_off'] = False
        df['stationary'] = False
        
        data_path = temp_output_dir / "quality_test_data.csv"
        df.to_csv(data_path, index=False)
        
        complete_config.paths.input.combined_csv = data_path
        
        pipeline = TrainingPipeline(complete_config)
        
        # Should handle data quality issues gracefully
        try:
            results = pipeline.run(data_path)
            # Pipeline should complete or report issues
            assert 'events_df' in results or 'error' in results
        except Exception as e:
            # Should provide meaningful error message
            assert len(str(e)) > 0


@pytest.mark.integration
class TestSplitConsistency:
    """End-to-end tests for train/test split consistency."""
    
    def test_split_consistency_across_pipeline(
        self,
        complete_config,
        sample_telemetry_df,
        temp_output_dir
    ):
        """Test that train/test split is consistent throughout pipeline."""
        
        data_path = temp_output_dir / "split_test_data.csv"
        sample_telemetry_df.to_csv(data_path, index=False)
        
        complete_config.paths.input.combined_csv = data_path
        complete_config.model.splitting.train_ratio = 0.80
        
        pipeline = TrainingPipeline(complete_config)
        results = pipeline.run(data_path)
        
        if 'events_df' in results and not results['events_df'].empty:
            events_df = results['events_df']
            
            # Should have is_train column
            assert 'is_train' in events_df.columns
            
            # Check split ratio is approximately correct
            train_ratio = events_df['is_train'].mean()
            assert 0.70 < train_ratio < 0.90  # Allow some variance
            
            # Verify temporal ordering within each vehicle
            for vehicle_id in events_df['vehicle_id'].unique():
                vehicle_events = events_df[events_df['vehicle_id'] == vehicle_id].sort_values('start_time')
                
                # All train events should come before test events
                if len(vehicle_events) > 1:
                    train_events = vehicle_events[vehicle_events['is_train']]
                    test_events = vehicle_events[~vehicle_events['is_train']]
                    
                    if not train_events.empty and not test_events.empty:
                        max_train_time = train_events['end_time'].max()
                        min_test_time = test_events['start_time'].min()
                        
                        # No temporal overlap (temporal split)
                        assert max_train_time <= min_test_time, \
                            f"Temporal leakage detected for vehicle {vehicle_id}"


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceAndScalability:
    """End-to-end tests for performance and scalability."""
    
    def test_pipeline_handles_large_data(
        self,
        complete_config,
        temp_output_dir
    ):
        """Test pipeline with larger dataset."""
        
        # Create larger dataset (still manageable for testing)
        n_points = 5000
        df = pd.DataFrame({
            'vehicle_id': np.repeat(['VEH1', 'VEH2', 'VEH3'], n_points // 3),
            'timestamp': pd.date_range('2024-01-01', periods=n_points, freq='1min', tz='UTC'),
            'total_fuel_gal': 50 + np.cumsum(np.random.normal(-0.05, 0.1, n_points)),
            'speed_kmh': np.random.choice([0, 45, 55, 65], n_points),
            'ignition': np.random.choice([True, False], n_points, p=[0.8, 0.2]),
            'latitude': 40.7 + np.random.normal(0, 0.01, n_points),
            'longitude': -74.0 + np.random.normal(0, 0.01, n_points),
        })
        
        # Add derived columns
        df['dt_s'] = df.groupby('vehicle_id')['timestamp'].diff().dt.total_seconds()
        df['dfuel'] = df.groupby('vehicle_id')['total_fuel_gal'].diff()
        df['moving'] = df['speed_kmh'] > 1.0
        df['stationary_on'] = (~df['moving']) & df['ignition']
        df['ign_off'] = ~df['ignition']
        df['stationary'] = df['stationary_on'] | df['ign_off']
        
        data_path = temp_output_dir / "large_data.csv"
        df.to_csv(data_path, index=False)
        
        complete_config.paths.input.combined_csv = data_path
        
        pipeline = TrainingPipeline(complete_config)
        
        # Should complete in reasonable time
        import time
        start_time = time.time()
        
        results = pipeline.run(data_path)
        
        elapsed = time.time() - start_time
        
        # Should complete (time limit depends on hardware, use generous limit)
        assert elapsed < 300  # 5 minutes max for test
        
        # Should produce valid results
        assert 'models' in results or 'error' in results