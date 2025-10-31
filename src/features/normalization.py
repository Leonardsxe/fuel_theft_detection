"""
Vehicle-specific feature normalization.
Accounts for differences in vehicle sensors and fuel capacity.
"""

import pandas as pd
import numpy as np
from typing import Dict

class VehicleNormalizer:
    """
    Normalize features by vehicle-specific statistics.
    Fits on training data only to prevent leakage.
    """
    
    def __init__(self):
        self.vehicle_stats = {}
    
    def fit(self, events_df: pd.DataFrame, train_mask: np.ndarray) -> 'VehicleNormalizer':
        """
        Calculate vehicle-specific statistics on training data.
        
        Args:
            events_df: Events DataFrame
            train_mask: Boolean mask for training data
        
        Returns:
            Self for chaining
        """
        train_events = events_df.loc[train_mask]
        
        features_to_normalize = [
            "drop_gal", "rate_gpm", "min_step_gal", "duration_min"
        ]
        
        for vid in train_events["vehicle_id"].unique():
            vehicle_data = train_events[train_events["vehicle_id"] == vid]
            
            stats = {}
            for feat in features_to_normalize:
                if feat in vehicle_data.columns:
                    stats[f"{feat}_mean"] = float(vehicle_data[feat].mean())
                    stats[f"{feat}_std"] = float(vehicle_data[feat].std())
            
            self.vehicle_stats[vid] = stats
        
        return self
    
    def transform(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply vehicle-specific normalization.
        
        Creates z-score normalized features: {feature}_vehicle_zscore
        """
        events_df = events_df.copy()
        
        features_to_normalize = [
            "drop_gal", "rate_gpm", "min_step_gal", "duration_min"
        ]
        
        for feat in features_to_normalize:
            events_df[f"{feat}_vehicle_zscore"] = 0.0
        
        for vid in events_df["vehicle_id"].unique():
            mask = events_df["vehicle_id"] == vid
            
            if vid not in self.vehicle_stats:
                continue
            
            stats = self.vehicle_stats[vid]
            
            for feat in features_to_normalize:
                if feat not in events_df.columns:
                    continue
                
                mean_val = stats.get(f"{feat}_mean", 0)
                std_val = stats.get(f"{feat}_std", 1)
                
                if std_val > 0:
                    events_df.loc[mask, f"{feat}_vehicle_zscore"] = (
                        (events_df.loc[mask, feat] - mean_val) / std_val
                    )
        
        return events_df
    
    def add_relative_features(
        self,
        events_df: pd.DataFrame,
        raw_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add features relative to typical fuel levels.
        
        Features:
        - drop_pct_of_avg: Drop as % of average fuel level
        - drop_pct_of_max: Drop as % of maximum fuel level
        """
        vehicle_fuel_stats = raw_df.groupby("vehicle_id")["total_fuel_gal"].agg([
            ("avg_fuel_level", "mean"),
            ("max_fuel_level", "max")
        ]).reset_index()
        
        events_df = events_df.merge(vehicle_fuel_stats, on="vehicle_id", how="left")
        
        events_df["drop_pct_of_avg"] = (
            events_df["drop_gal"] / (events_df["avg_fuel_level"] + 1e-6)
        ).fillna(0)
        
        events_df["drop_pct_of_max"] = (
            events_df["drop_gal"] / (events_df["max_fuel_level"] + 1e-6)
        ).fillna(0)
        
        return events_df