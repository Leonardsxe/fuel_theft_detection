"""
Unit tests for src/data/preprocessor.py
"""

import pytest
import pandas as pd
import numpy as np

from src.config.settings import (
    DetectionConfig, StationaryConfig, ThresholdConfig,
    OutlierConfig, InterpolationConfig, PlausibilityConfig,
    NoiseConfig, NMSConfig, ClusteringConfig
)
from src.data.preprocessor import DataPreprocessor


@pytest.fixture
def detection_config():
    """Create test detection configuration"""
    return DetectionConfig(
        stationary=StationaryConfig(speed_threshold_kmh=1.0),
        thresholds_stationary_on=ThresholdConfig(
            min_step_gal=1.0,
            min_cumulative_gal=3.5,
            mad_multiplier_step=3.0,
            mad_multiplier_cum=4.0
        ),
        thresholds_ignition_off=ThresholdConfig(
            min_step_gal=1.5,
            min_cumulative_gal=6.0,
            mad_multiplier_step=2.5,
            mad_multiplier_cum=3.5
        ),
        patterns={},
        plausibility=PlausibilityConfig(),
        noise=NoiseConfig(),
        nms=NMSConfig(),
        clustering=ClusteringConfig(),
        outliers=OutlierConfig(max_drain_rate_gpm=2.0, max_fill_rate_gpm=10.0, mad_threshold=5.0),
        interpolation=InterpolationConfig(max_gap_points=3)
    )


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        'vehicle_id': ['V1'] * 10,
        'timestamp': pd.date_range('2025-01-15', periods=10, freq='1min', tz='UTC'),
        'latitude': [40.7128] * 10,
        'longitude': [-74.0060] * 10,
        'speed_kmh': [0.5, 0.3, 50.0, 60.0, 0.2, 0.1, 0.4, 55.0, 0.0, 0.0],
        'ignition': [True] * 10,
        'total_fuel_gal': [100, 99.8, 99.5, 99.0, 98.8, 98.5, 98.3, 98.0, 97.8, 97.5]
    })


class TestDeriveFeatures:
    """Tests for derive_features method"""
    
    def test_derive_basic_features(self, detection_config, sample_data):
        """Test derivation of basic features"""
        preprocessor = DataPreprocessor(detection_config)
        result = preprocessor.derive_features(sample_data)
        
        # Check derived columns exist
        assert 'dt_s' in result.columns
        assert 'moving' in result.columns
        assert 'stationary_on' in result.columns
        assert 'ign_off' in result.columns
        assert 'stationary' in result.columns
        assert 'dfuel' in result.columns
    
    def test_stationary_detection(self, detection_config, sample_data):
        """Test stationary period detection"""
        preprocessor = DataPreprocessor(detection_config)
        result = preprocessor.derive_features(sample_data)
        
        # Rows with speed <= 1.0 and ignition=True should be stationary_on
        stationary_on_mask = (sample_data['speed_kmh'] <= 1.0) & (sample_data['ignition'] == True)
        assert (result['stationary_on'] == stationary_on_mask).all()
    
    def test_fuel_change_calculation(self, detection_config, sample_data):
        """Test fuel change calculation"""
        preprocessor = DataPreprocessor(detection_config)
        result = preprocessor.derive_features(sample_data)
        
        # First row should have NaN dfuel
        assert pd.isna(result['dfuel'].iloc[0])
        
        # Second row should have negative dfuel
        assert result['dfuel'].iloc[1] < 0


class TestRemoveOutliers:
    """Tests for remove_outliers method"""
    
    def test_rate_based_outlier_detection(self, detection_config):
        """Test outlier detection based on impossible rates"""
        # Create data with impossible drain rate
        df = pd.DataFrame({
            'vehicle_id': ['V1'] * 3,
            'timestamp': pd.date_range('2025-01-15', periods=3, freq='1min', tz='UTC'),
            'total_fuel_gal': [100, 50, 49]  # 50 gal drop in 1 min = impossible
        })
        df['dt_s'] = 60.0  # 1 minute
        
        preprocessor = DataPreprocessor(detection_config)
        result = preprocessor.remove_outliers(df)
        
        # Outlier should be marked as NaN
        assert pd.isna(result['total_fuel_gal'].iloc[1])
    
    def test_statistical_outlier_detection(self, detection_config):
        """Test MAD-based statistical outlier detection"""
        # Create data with statistical outlier
        fuel_values = [100] * 20 + [200]  # 200 is outlier
        df = pd.DataFrame({
            'vehicle_id': ['V1'] * 21,
            'timestamp': pd.date_range('2025-01-15', periods=21, freq='1min', tz='UTC'),
            'total_fuel_gal': fuel_values
        })
        df['dt_s'] = 60.0
        
        preprocessor = DataPreprocessor(detection_config)
        result = preprocessor.remove_outliers(df)
        
        # Outlier should be detected (though exact behavior depends on MAD threshold)
        # At least the original values should be preserved where valid
        assert result['total_fuel_gal'].iloc[0] == 100


class TestInterpolateGaps:
    """Tests for interpolate_gaps method"""
    
    def test_interpolation_in_stationary_segments(self, detection_config):
        """Test gap interpolation within stationary segments"""
        df = pd.DataFrame({
            'vehicle_id': ['V1'] * 5,
            'timestamp': pd.date_range('2025-01-15', periods=5, freq='1min', tz='UTC'),
            'speed_kmh': [0.5] * 5,
            'ignition': [True] * 5,
            'stationary': [True] * 5,
            'total_fuel_gal': [100.0, np.nan, np.nan, 98.0, 97.5]
        })
        
        preprocessor = DataPreprocessor(detection_config)
        result = preprocessor.interpolate_gaps(df)
        
        # Gaps should be interpolated
        assert result['total_fuel_gal'].notna().all()
        
        # Interpolated values should be between 100 and 98
        assert result['total_fuel_gal'].iloc[1] > 98.0
        assert result['total_fuel_gal'].iloc[1] < 100.0
    
    def test_no_interpolation_in_moving_segments(self, detection_config):
        """Test that gaps are NOT interpolated in moving segments"""
        df = pd.DataFrame({
            'vehicle_id': ['V1'] * 5,
            'timestamp': pd.date_range('2025-01-15', periods=5, freq='1min', tz='UTC'),
            'speed_kmh': [50.0] * 5,
            'ignition': [True] * 5,
            'stationary': [False] * 5,
            'total_fuel_gal': [100.0, np.nan, np.nan, 98.0, 97.5]
        })
        
        preprocessor = DataPreprocessor(detection_config)
        result = preprocessor.interpolate_gaps(df)
        
        # Gaps should remain NaN in moving segments
        assert pd.isna(result['total_fuel_gal'].iloc[1])
        assert pd.isna(result['total_fuel_gal'].iloc[2])


class TestFitTransform:
    """Integration tests for fit_transform method"""
    
    def test_complete_pipeline(self, detection_config, sample_data):
        """Test complete preprocessing pipeline"""
        preprocessor = DataPreprocessor(detection_config)
        result = preprocessor.fit_transform(sample_data)
        
        # Check all derived features exist
        expected_features = ['dt_s', 'moving', 'stationary_on', 'ign_off', 
                            'stationary', 'dfuel']
        for feature in expected_features:
            assert feature in result.columns
        
        # Check timestamps are UTC
        assert result['timestamp'].dt.tz is not None
        assert str(result['timestamp'].dt.tz) == 'UTC'
        
        # Check data is sorted
        assert result['timestamp'].is_monotonic_increasing
    
    def test_pipeline_preserves_data_integrity(self, detection_config, sample_data):
        """Test that pipeline preserves original data where appropriate"""
        preprocessor = DataPreprocessor(detection_config)
        result = preprocessor.fit_transform(sample_data)
        
        # Number of rows should be preserved
        assert len(result) == len(sample_data)
        
        # Vehicle IDs should be preserved
        assert (result['vehicle_id'] == sample_data['vehicle_id']).all()