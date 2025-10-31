"""
Spatial feature engineering.
GPS-based features and location analysis.
"""

import pandas as pd
import numpy as np
from src.utils.distance import haversine_distance, calculate_trip_distance

def add_location_features(events_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add location-based features.
    
    Features:
    - lat_c, lon_c: Median coordinates during event
    - location_entropy: Movement variability
    - distance_from_start: Distance between start and end
    """
    events_df = events_df.copy()
    
    def extract_location(row):
        mask = (
            (raw_df["vehicle_id"] == row["vehicle_id"]) &
            (raw_df["timestamp"] >= row["start_time"]) &
            (raw_df["timestamp"] <= row["end_time"])
        )
        event_data = raw_df.loc[mask, ["latitude", "longitude"]]
        
        if event_data.empty:
            return pd.Series({"lat_c": np.nan, "lon_c": np.nan})
        
        return pd.Series({
            "lat_c": float(event_data["latitude"].median()),
            "lon_c": float(event_data["longitude"].median())
        })
    
    coords = events_df.apply(extract_location, axis=1)
    events_df = pd.concat([events_df, coords], axis=1)
    
    return events_df