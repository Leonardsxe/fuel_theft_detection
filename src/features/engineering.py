"""
Feature engineering orchestrator.
Coordinates all feature modules and provides unified interface.
"""

import pandas as pd
import numpy as np
import logging

from src.config.settings import FeatureConfig
from src.features.temporal import add_all_temporal_features
from src.features.spatial import add_location_features
from src.features.behavioral import add_pre_event_context, add_movement_variability
from src.features.normalization import VehicleNormalizer
from src.features.event_features import add_event_statistics

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Complete feature engineering pipeline.
    Coordinates all feature modules with train/test awareness.
    """
    
    def __init__(self, config: FeatureConfig):
        """
        Initialize feature engineer.
        
        Args:
            config: Feature configuration
        """
        self.config = config
        self.normalizer = VehicleNormalizer() if config.enable_vehicle_normalization else None
        self.is_fitted = False
    
    def fit(self, raw_df: pd.DataFrame, train_mask: np.ndarray) -> 'FeatureEngineer':
        """
        Fit on training data (for normalization).
        
        Args:
            raw_df: Raw telemetry DataFrame
            train_mask: Boolean mask for training data
        
        Returns:
            Self for chaining
        """
        logger.info("Fitting feature engineer on training data...")
        
        # Fit normalizer if enabled
        if self.normalizer:
            # Need events for normalization - this is called after event detection
            pass
        
        self.is_fitted = True
        logger.info("✓ Feature engineer fitted")
        
        return self
    
    def transform(
        self,
        events_df: pd.DataFrame,
        raw_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Args:
            events_df: Events DataFrame
            raw_df: Raw telemetry DataFrame
        
        Returns:
            Events DataFrame with all features added
        """
        logger.info("Engineering features...")
        
        if events_df.empty:
            logger.warning("No events to engineer features for")
            return events_df
        
        # 1. Temporal features
        if self.config.enable_temporal_features:
            events_df = add_all_temporal_features(events_df)
        
        # 2. Spatial features
        if self.config.enable_spatial_features:
            events_df = add_location_features(events_df, raw_df)
        
        # 3. Event statistics
        events_df = add_event_statistics(events_df, raw_df)
        
        # 4. Behavioral features
        if self.config.enable_behavioral_features:
            events_df = add_pre_event_context(
                events_df, raw_df,
                lookback_hours=self.config.behavioral_lookback_hours
            )
            events_df = add_movement_variability(events_df, raw_df)
        
        # 5. Vehicle normalization
        if self.config.enable_vehicle_normalization and self.normalizer:
            events_df = self.normalizer.transform(events_df)
            events_df = self.normalizer.add_relative_features(events_df, raw_df)
        
        logger.info(f"✓ Feature engineering complete: {len(events_df.columns)} total columns")
        
        return events_df
    
    def fit_transform(
        self,
        events_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        train_mask: np.ndarray
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            events_df: Events DataFrame
            raw_df: Raw telemetry DataFrame
            train_mask: Boolean mask for training events
        
        Returns:
            Events DataFrame with features
        """
        # Fit normalizer on training events
        if self.normalizer:
            self.normalizer.fit(events_df, train_mask)
        
        self.is_fitted = True
        
        return self.transform(events_df, raw_df)


def engineer_features(
    events_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    config: FeatureConfig,
    train_mask: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Convenience function for feature engineering.
    
    Args:
        events_df: Events DataFrame
        raw_df: Raw telemetry DataFrame
        config: Feature configuration
        train_mask: Optional training mask
    
    Returns:
        Events DataFrame with features
    """
    engineer = FeatureEngineer(config)
    
    if train_mask is not None:
        return engineer.fit_transform(events_df, raw_df, train_mask)
    else:
        return engineer.transform(events_df, raw_df)