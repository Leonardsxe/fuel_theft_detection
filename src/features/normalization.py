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
        self.is_fitted = False
    
    def fit(self, events_df: pd.DataFrame, train_mask: np.ndarray) -> 'VehicleNormalizer':
        """
        Calculate vehicle-specific statistics on training data.
        
        Args:
            events_df: Events DataFrame
            train_mask: Boolean mask for training data
        
        Returns:
            Self for chaining
        """
        train_events = events_df.loc[train_mask].copy()
        
        features_to_normalize = [
            "drop_gal", "rate_gpm", "min_step_gal", "duration_min"
        ]
        
        for vid in train_events["vehicle_id"].unique():
            vehicle_data = train_events[train_events["vehicle_id"] == vid]
            
            stats = {}
            for feat in features_to_normalize:
                if feat in vehicle_data.columns:
                    s = pd.to_numeric(vehicle_data[feat], errors="coerce")
                    m = float(s.mean(skipna=True)) if len(s) else 0.0
                    sd = float(s.std(skipna=True)) if len(s) else 0.0
                    # avoid degenerate std (divide by zero)
                    if not np.isfinite(sd) or sd == 0.0:
                        sd = 1e-6
                    stats[f"{feat}_mean"] = m
                    stats[f"{feat}_std"] = sd
            
            self.vehicle_stats[vid] = stats
        
        self.is_fitted = True
        
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
                if feat in events_df.columns:
                    events_df[f"{feat}_vehicle_zscore"] = 0.0
        
        for vid in events_df["vehicle_id"].unique():
            mask = events_df["vehicle_id"] == vid
            
            if vid not in self.vehicle_stats:
                continue
            
            stats = self.vehicle_stats[vid]
            
            for feat in features_to_normalize:
                if feat not in events_df.columns:
                    continue
                
                mean_val = float(stats.get(f"{feat}_mean", 0.0))
                std_val = float(stats.get(f"{feat}_std", 1.0))
                if not np.isfinite(std_val) or std_val <= 0.0:
                    std_val = 1e-6
                x = pd.to_numeric(events_df.loc[mask, feat], errors="coerce").astype(float)
                events_df.loc[mask, f"{feat}_vehicle_zscore"] = (x - mean_val) / std_val

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
        s = pd.to_numeric(raw_df["total_fuel_gal"], errors="coerce")
        tmp = raw_df.assign(total_fuel_gal=s)
        vehicle_fuel_stats = (
            tmp.groupby("vehicle_id", as_index=False)["total_fuel_gal"]
            .agg(avg_fuel_level="mean", max_fuel_level="max")
        )
        
        events_df = events_df.merge(vehicle_fuel_stats, on="vehicle_id", how="left")
        
        if "drop_gal" in events_df.columns:
            events_df["drop_pct_of_avg"] = (
                pd.to_numeric(events_df["drop_gal"], errors="coerce") / (events_df["avg_fuel_level"].astype(float) + 1e-6)
            ).fillna(0.0)
            events_df["drop_pct_of_max"] = (
                pd.to_numeric(events_df["drop_gal"], errors="coerce") / (events_df["max_fuel_level"].astype(float) + 1e-6)
            ).fillna(0.0)
        else:
            events_df["drop_pct_of_avg"] = 0.0
            events_df["drop_pct_of_max"] = 0.0
        
        return events_df