"""
Unit tests for feature engineering modules.
"""

import pytest
import numpy as np
import pandas as pd

from src.features.temporal import (
    add_time_cyclical_features,
    add_all_temporal_features,
)
from src.features.engineering import FeatureEngineer


class TestTemporalFeatures:
    """Tests for temporal feature engineering."""
    
    def test_add_time_cyclical_features(self, sample_events_df):
        """Test cyclical time features."""
        result = add_time_cyclical_features(sample_events_df)
        
        assert 'hod_sin' in result.columns
        assert 'hod_cos' in result.columns
        assert 'weekday' in result.columns
        assert 'is_weekend' in result.columns
        assert 'is_night' in result.columns
        
        # Check value ranges
        assert (result['hod_sin'] >= -1).all() and (result['hod_sin'] <= 1).all()
        assert (result['hod_cos'] >= -1).all() and (result['hod_cos'] <= 1).all()
        assert (result['weekday'] >= 0).all() and (result['weekday'] <= 6).all()
        assert result['is_weekend'].isin([0, 1]).all()
        assert result['is_night'].isin([0, 1]).all()
    
    def test_cyclical_features_continuity(self):
        """Test that cyclical features are continuous at midnight."""
        # Create events at 23:00 and 01:00
        df = pd.DataFrame({
            'start_time': pd.to_datetime(['2024-01-01 23:00:00', '2024-01-02 01:00:00'], utc=True),
            'end_time': pd.to_datetime(['2024-01-01 23:30:00', '2024-01-02 01:30:00'], utc=True),
        })
        
        result = add_time_cyclical_features(df)
        
        # Values should be similar (not jump discontinuously)
        hod_diff = abs(result['hod_sin'].iloc[0] - result['hod_sin'].iloc[1])
        assert hod_diff < 0.5  # Should be close
    
    def test_is_night_detection(self):
        """Test night-time detection."""
        df = pd.DataFrame({
            'start_time': pd.to_datetime([
                '2024-01-01 02:00:00',  # Night
                '2024-01-01 12:00:00',  # Day
                '2024-01-01 23:00:00',  # Night
            ], utc=True),
            'end_time': pd.to_datetime([
                '2024-01-01 02:30:00',
                '2024-01-01 12:30:00',
                '2024-01-01 23:30:00',
            ], utc=True),
        })
        
        result = add_time_cyclical_features(df)
        
        assert result['is_night'].iloc[0] == 1
        assert result['is_night'].iloc[1] == 0
        assert result['is_night'].iloc[2] == 1
    
    def test_weekend_detection(self):
        """Test weekend detection."""
        df = pd.DataFrame({
            'start_time': pd.to_datetime([
                '2024-01-01 12:00:00',  # Monday
                '2024-01-06 12:00:00',  # Saturday
                '2024-01-07 12:00:00',  # Sunday
            ], utc=True),
            'end_time': pd.to_datetime([
                '2024-01-01 12:30:00',
                '2024-01-06 12:30:00',
                '2024-01-07 12:30:00',
            ], utc=True),
        })
        
        result = add_time_cyclical_features(df)
        
        assert result['is_weekend'].iloc[0] == 0
        assert result['is_weekend'].iloc[1] == 1
        assert result['is_weekend'].iloc[2] == 1
    
    def test_add_all_temporal_features(self, sample_events_df):
        """Test adding all temporal features."""
        result = add_all_temporal_features(sample_events_df)
        
        # Check all expected features are present
        expected_features = [
            'hod_sin', 'hod_cos', 'weekday', 'is_weekend', 'is_night'
        ]
        for feat in expected_features:
            assert feat in result.columns


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""
    
    def test_feature_engineer_initialization(self, feature_config):
        """Test FeatureEngineer initialization."""
        fe = FeatureEngineer(feature_config)
        
        assert fe.config is not None
        assert fe.config.behavioral_lookback_hours == 2
    
    def test_feature_engineer_fit_transform(
        self,
        sample_events_df,
        sample_telemetry_df,
        feature_config,
        sample_train_mask
    ):
        """Test fit_transform method."""
        fe = FeatureEngineer(feature_config)
        
        # Ensure train_mask aligns with events
        event_train_mask = sample_train_mask[:len(sample_events_df)]
        
        result = fe.fit_transform(
            sample_events_df,
            sample_telemetry_df,
            event_train_mask
        )
        
        # Should have more columns than input
        assert len(result.columns) > len(sample_events_df.columns)
        
        # Should preserve original rows
        assert len(result) == len(sample_events_df)
    
    def test_vehicle_normalization(
        self,
        sample_events_df,
        sample_telemetry_df,
        feature_config,
        sample_train_mask
    ):
        """Test vehicle-specific normalization."""
        feature_config.enable_vehicle_normalization = True
        fe = FeatureEngineer(feature_config)
        
        event_train_mask = sample_train_mask[:len(sample_events_df)]
        
        result = fe.fit_transform(
            sample_events_df,
            sample_telemetry_df,
            event_train_mask
        )
        
        # Check for normalized features
        normalized_features = [c for c in result.columns if 'zscore' in c or 'pct_of' in c]
        assert len(normalized_features) > 0
    
    def test_feature_engineer_respects_config_flags(
        self,
        sample_events_df,
        sample_telemetry_df,
        sample_train_mask
    ):
        """Test that config flags control feature generation."""
        from src.config.settings import FeatureConfig
        
        # Disable all optional features
        config = FeatureConfig(
            behavioral_lookback_hours=2,
            enable_vehicle_normalization=False,
            enable_behavioral_features=False,
            enable_temporal_features=False,
            enable_spatial_features=False,
        )
        
        fe = FeatureEngineer(config)
        event_train_mask = sample_train_mask[:len(sample_events_df)]
        
        result = fe.fit_transform(
            sample_events_df,
            sample_telemetry_df,
            event_train_mask
        )
        
        # Should have minimal new features
        new_features = set(result.columns) - set(sample_events_df.columns)
        # Some basic features may still be added, but should be minimal
        assert len(new_features) < 10


class TestFeatureEngineeringEdgeCases:
    """Tests for edge cases in feature engineering."""
    
    def test_empty_events(self, sample_telemetry_df, feature_config, sample_train_mask):
        """Test feature engineering with empty events."""
        empty_events = pd.DataFrame(columns=['vehicle_id', 'start_time', 'end_time'])
        
        fe = FeatureEngineer(feature_config)
        
        result = fe.fit_transform(empty_events, sample_telemetry_df, np.array([]))
        
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)
    
    def test_missing_columns_handled_gracefully(
        self,
        feature_config,
        sample_train_mask
    ):
        """Test that missing columns are handled gracefully."""
        # Events without some expected columns
        minimal_events = pd.DataFrame({
            'vehicle_id': ['VEH1'],
            'start_time': pd.to_datetime(['2024-01-01 10:00:00'], utc=True),
            'end_time': pd.to_datetime(['2024-01-01 10:10:00'], utc=True),
        })
        
        minimal_telemetry = pd.DataFrame({
            'vehicle_id': ['VEH1'] * 20,
            'timestamp': pd.date_range('2024-01-01 09:00', periods=20, freq='1min', tz='UTC'),
            'total_fuel_gal': [50.0] * 20,
        })
        
        fe = FeatureEngineer(feature_config)
        event_train_mask = np.array([True])
        
        # Should not crash
        result = fe.fit_transform(minimal_events, minimal_telemetry, event_train_mask)
        
        assert len(result) == 1
    
    def test_single_vehicle(self, feature_config):
        """Test feature engineering with single vehicle."""
        events = pd.DataFrame({
            'vehicle_id': ['VEH1'] * 3,
            'start_time': pd.to_datetime([
                '2024-01-01 10:00:00',
                '2024-01-01 14:00:00',
                '2024-01-01 18:00:00',
            ], utc=True),
            'end_time': pd.to_datetime([
                '2024-01-01 10:10:00',
                '2024-01-01 14:10:00',
                '2024-01-01 18:10:00',
            ], utc=True),
            'drop_gal': [5.0, 6.0, 4.5],
        })
        
        telemetry = pd.DataFrame({
            'vehicle_id': ['VEH1'] * 100,
            'timestamp': pd.date_range('2024-01-01 09:00', periods=100, freq='5min', tz='UTC'),
            'total_fuel_gal': 50 + np.random.normal(0, 0.5, 100),
        })
        
        fe = FeatureEngineer(feature_config)
        train_mask = np.array([True, True, False])
        
        result = fe.fit_transform(events, telemetry, train_mask)
        
        assert len(result) == 3
        assert 'vehicle_id' in result.columns
    
    def test_normalization_with_constant_features(self, feature_config):
        """Test normalization when features have zero variance."""
        # All events have same drop_gal
        events = pd.DataFrame({
            'vehicle_id': ['VEH1'] * 3,
            'start_time': pd.to_datetime([
                '2024-01-01 10:00:00',
                '2024-01-01 14:00:00',
                '2024-01-01 18:00:00',
            ], utc=True),
            'end_time': pd.to_datetime([
                '2024-01-01 10:10:00',
                '2024-01-01 14:10:00',
                '2024-01-01 18:10:00',
            ], utc=True),
            'drop_gal': [5.0, 5.0, 5.0],  # Constant
            'duration_min': [10.0, 10.0, 10.0],  # Constant
        })
        
        telemetry = pd.DataFrame({
            'vehicle_id': ['VEH1'] * 50,
            'timestamp': pd.date_range('2024-01-01 09:00', periods=50, freq='5min', tz='UTC'),
            'total_fuel_gal': [50.0] * 50,
        })
        
        feature_config.enable_vehicle_normalization = True
        fe = FeatureEngineer(feature_config)
        train_mask = np.array([True, True, False])
        
        # Should handle zero variance gracefully (normalize to 0)
        result = fe.fit_transform(events, telemetry, train_mask)
        
        assert len(result) == 3
        # Check normalized features exist and are finite
        zscore_cols = [c for c in result.columns if 'zscore' in c]
        for col in zscore_cols:
            assert result[col].notna().all()