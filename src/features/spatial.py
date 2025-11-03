"""
Spatial features for siphoning events:
- Event centroid (median lat/lon within window)
- Location variability (lat/lon std) and max coord radius (km)
- Optional in-window trip distance (km) for added context
"""
from __future__ import annotations

import pandas as pd
from typing import Tuple

from src.utils.distance import haversine_distance, calculate_trip_distance

__all__ = ["add_location_features"]


def _window_mask(raw: pd.DataFrame, row: pd.Series) -> pd.Series:
    return (
        (raw["vehicle_id"] == row["vehicle_id"])
        & (raw["timestamp"] >= row["start_time"])
        & (raw["timestamp"] <= row["end_time"])
    )


def _event_centroid_and_var(raw: pd.DataFrame, row: pd.Series) -> Tuple[float, float, float, float, float]:
    m = _window_mask(raw, row)
    coords = raw.loc[m, ["latitude", "longitude"]].dropna()

    if coords.empty:
        return float("nan"), float("nan"), 0.0, 0.0, 0.0

    lat_c = float(coords["latitude"].median())
    lon_c = float(coords["longitude"].median())
    lat_std = float(coords["latitude"].std()) if len(coords) > 1 else 0.0
    lon_std = float(coords["longitude"].std()) if len(coords) > 1 else 0.0

    # radius: farthest point (km) from median using your haversine_distance(lat1, lon1, lat2, lon2)
    max_r_km = 0.0
    for lat, lon in zip(coords["latitude"].values, coords["longitude"].values):
        d = haversine_distance(lat_c, lon_c, float(lat), float(lon))
        if d > max_r_km:
            max_r_km = d

    return lat_c, lon_c, lat_std, lon_std, max_r_km


def add_location_features(events_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds spatial features to events_df using raw_df telemetry.
    Required:
      events_df: ['vehicle_id','start_time','end_time']
      raw_df:    ['vehicle_id','timestamp','latitude','longitude'] (+ optional speed/fuel for trip distance)
    """
    df = events_df.copy()

    vals = df.apply(lambda r: _event_centroid_and_var(raw_df, r), axis=1, result_type="expand")
    df[["lat_c", "lon_c", "lat_std", "lon_std", "coord_range_km"]] = vals

    # Optional: in-window trip distance for added motion context (km)
    def _window_trip(row):
        m = _window_mask(raw_df, row)
        return float(calculate_trip_distance(raw_df.loc[m, ["latitude", "longitude"]]))

    df["window_trip_km"] = df.apply(_window_trip, axis=1)
    return df
