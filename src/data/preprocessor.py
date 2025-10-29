"""
Data preprocessing: cleaning, outlier removal, feature derivation.
Single Responsibility: Transform raw data into analysis-ready format.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging

from src.config.settings import DetectionConfig
from src.utils.timezone import ensure_series_utc

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Data preprocessing pipeline for fuel theft detection.
    Handles outlier removal, gap interpolation, and feature derivation.
    """
    
    def __init__(self, config: DetectionConfig):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Detection configuration
        """
        self.config = config
    
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and remove fuel sensor outliers using rate-based and statistical methods.
        
        Args:
            df: DataFrame with fuel data
        
        Returns:
            DataFrame with outliers removed (set to NaN for interpolation)
        """
        logger.info("Detecting and removing fuel sensor outliers...")
        
        df = df.copy()
        total_outliers = 0
        
        for vid, vehicle_data in df.groupby("vehicle_id"):
            vehicle_data = vehicle_data.sort_values("timestamp")
            
            # Calculate rate of change
            fuel_diff = vehicle_data["total_fuel_gal"].diff()
            time_diff_min = vehicle_data["dt_s"] / 60.0
            
            # Rate (gal/min)
            rate = fuel_diff / (time_diff_min + 1e-6)
            
            # Flag impossible rates
            impossible_drain = rate < -self.config.outliers.max_drain_rate_gpm
            impossible_fill = rate > self.config.outliers.max_fill_rate_gpm
            
            # Statistical outlier detection using MAD
            fuel_vals = vehicle_data["total_fuel_gal"].dropna()
            if len(fuel_vals) > 10:
                median = fuel_vals.median()
                mad = np.median(np.abs(fuel_vals - median))
                
                # Conservative 5-MAD threshold
                threshold = self.config.outliers.mad_threshold * 1.4826 * mad
                outlier_values = np.abs(vehicle_data["total_fuel_gal"] - median) > threshold
            else:
                outlier_values = pd.Series(False, index=vehicle_data.index)
            
            # Combine outlier flags
            outlier_mask = impossible_drain | impossible_fill | outlier_values
            
            # Mark outliers as NaN
            df.loc[vehicle_data.index[outlier_mask], "total_fuel_gal"] = np.nan
            
            n_outliers = outlier_mask.sum()
            total_outliers += n_outliers
            
            if n_outliers > 0:
                logger.info(f"  Vehicle {vid}: Removed {n_outliers} outlier readings")
        
        logger.info(f"✓ Removed {total_outliers:,} total outlier fuel readings")
        
        return df
    
    def interpolate_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate small gaps in fuel readings within stationary segments.
        
        Args:
            df: DataFrame with potential gaps
        
        Returns:
            DataFrame with gaps interpolated
        """
        logger.info("Interpolating fuel gaps in stationary segments...")
        
        df = df.copy()
        df = df.sort_values(["vehicle_id", "timestamp"])
        
        # Identify stationary segments
        vehicle_change = df["vehicle_id"] != df["vehicle_id"].shift()
        stationary_change = df["stationary"] != df["stationary"].shift()
        segment_id = (vehicle_change | stationary_change).cumsum()
        df["_segment"] = segment_id
        
        # Mark which segments are stationary
        df["_is_stat_seg"] = df.groupby("_segment")["stationary"].transform("first")
        
        # Convert fuel to numeric
        df["total_fuel_gal"] = pd.to_numeric(df["total_fuel_gal"], errors="coerce")
        
        # Create a mask for stationary rows
        stationary_mask = df["_is_stat_seg"] == True
        
        # Interpolate within stationary segments only
        if stationary_mask.any():
            df.loc[stationary_mask, "total_fuel_gal"] = (
                df.loc[stationary_mask]
                .groupby("_segment")["total_fuel_gal"]
                .transform(lambda x: x.interpolate(
                    method="linear",
                    limit=self.config.interpolation.max_gap_points,
                    limit_area="inside"
                ))
            )
        
        # Clean up temporary columns
        df = df.drop(columns=["_segment", "_is_stat_seg"])
        
        logger.info("✓ Gap interpolation complete")
        
        return df
    
    def derive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive basic features needed for detection.
        
        Args:
            df: DataFrame with raw data
        
        Returns:
            DataFrame with derived features
        """
        logger.info("Deriving basic features...")
        
        df = df.copy()
        df = df.sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True)
        
        # Time difference in seconds
        df["dt_s"] = df.groupby("vehicle_id")["timestamp"].diff().dt.total_seconds()
        
        # Movement detection
        df["moving"] = df["speed_kmh"].fillna(np.inf) > self.config.stationary.speed_threshold_kmh
        
        # Stationary states
        df["stationary_on"] = (~df["moving"]) & (df["ignition"] == True)
        df["ign_off"] = df["ignition"] == False
        df["stationary"] = df["stationary_on"] | df["ign_off"]
        
        # Fuel change
        df["dfuel"] = df.groupby("vehicle_id")["total_fuel_gal"].diff()
        
        logger.info("✓ Derived features: dt_s, moving, stationary, stationary_on, ign_off, dfuel")
        
        return df
    
    def normalize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all timestamps are UTC-normalized.
        
        Args:
            df: DataFrame with timestamp column
        
        Returns:
            DataFrame with UTC timestamps
        """
        logger.info("Normalizing timestamps to UTC...")
        
        df = df.copy()
        df["timestamp"] = ensure_series_utc(df["timestamp"])
        
        logger.info("✓ Timestamps normalized to UTC")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply complete preprocessing pipeline.
        
        Args:
            df: Raw DataFrame
        
        Returns:
            Preprocessed DataFrame ready for event detection
        """
        logger.info("Starting preprocessing pipeline...")
        
        # 1. Normalize timestamps
        df = self.normalize_timestamps(df)
        
        # 2. Sort data
        df = df.sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True)
        
        # 3. Derive basic features (needed for stationary detection)
        df = self.derive_features(df)
        
        # 4. Remove outliers
        df = self.remove_outliers(df)
        
        # 5. Interpolate gaps in stationary segments
        df = self.interpolate_gaps(df)
        
        # 6. Recalculate dfuel after interpolation
        df["dfuel"] = df.groupby("vehicle_id")["total_fuel_gal"].diff()
        
        logger.info(f"✓ Preprocessing complete: {len(df):,} rows, "
                   f"{df['vehicle_id'].nunique()} vehicles")
        
        return df


def preprocess_data(
    df: pd.DataFrame,
    config: DetectionConfig
) -> pd.DataFrame:
    """
    Convenience function for preprocessing.
    
    Args:
        df: Raw DataFrame
        config: Detection configuration
    
    Returns:
        Preprocessed DataFrame
    """
    preprocessor = DataPreprocessor(config)
    return preprocessor.fit_transform(df)