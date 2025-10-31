"""
Behavioral feature engineering.
Captures vehicle behavior before events (2-hour lookback).
"""

import pandas as pd
import numpy as np
from src.utils.distance import calculate_trip_distance

def add_pre_event_context(
    events_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    lookback_hours: int = 2
) -> pd.DataFrame:
    """
    Add features based on behavior before the event.
    
    Features:
    - pre_event_distance_km: Distance traveled in lookback period
    - pre_event_avg_speed: Average speed
    - pre_event_moving_pct: Percentage of time moving
    - pre_event_fuel_change: Net fuel change (detects refueling)
    """
    events_df = events_df.copy()
    
    def get_pre_context(row):
        lookback_start = row["start_time"] - pd.Timedelta(hours=lookback_hours)
        lookback_end = row["start_time"]
        
        mask = (
            (raw_df["vehicle_id"] == row["vehicle_id"]) &
            (raw_df["timestamp"] >= lookback_start) &
            (raw_df["timestamp"] < lookback_end)
        )
        
        pre_data = raw_df.loc[mask]
        
        if len(pre_data) < 5:
            return pd.Series({
                "pre_event_distance_km": 0.0,
                "pre_event_avg_speed": 0.0,
                "pre_event_moving_pct": 0.0,
                "pre_event_fuel_change": 0.0
            })
        
        # Distance traveled
        coords = pre_data[["latitude", "longitude"]].dropna()
        distance_km = calculate_trip_distance(coords)
        
        # Speed statistics
        speed_vals = pre_data["speed_kmh"].dropna()
        avg_speed = float(speed_vals.mean()) if len(speed_vals) > 0 else 0.0
        moving_pct = float((speed_vals > 5).mean()) if len(speed_vals) > 0 else 0.0
        
        # Fuel change
        fuel_vals = pre_data["total_fuel_gal"].dropna()
        fuel_change = fuel_vals.iloc[-1] - fuel_vals.iloc[0] if len(fuel_vals) >= 2 else 0.0
        
        return pd.Series({
            "pre_event_distance_km": float(distance_km),
            "pre_event_avg_speed": avg_speed,
            "pre_event_moving_pct": moving_pct,
            "pre_event_fuel_change": float(fuel_change)
        })
    
    pre_context = events_df.apply(get_pre_context, axis=1)
    events_df = pd.concat([events_df, pre_context], axis=1)
    
    return events_df


def add_movement_variability(events_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add movement variability features during event.
    
    Features:
    - speed_std: Speed standard deviation
    - speed_max: Maximum speed
    - movement_variability: Speed variance
    """
    # Similar pattern to pre_event_context
    pass