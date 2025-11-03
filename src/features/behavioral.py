"""
Behavioral (context) features:
- Pre-event context window: distance, avg speed, % moving, fuel delta
- In-window movement variability: speed std & max
- Pre-event parking time (hours) up to 12h lookback
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.distance import calculate_trip_distance

__all__ = ["add_pre_event_context", "add_movement_variability"]


def add_pre_event_context(events_df: pd.DataFrame, raw_df: pd.DataFrame, lookback_hours: int = 2) -> pd.DataFrame:
    """
    Adds 2h (configurable) pre-event context derived from raw telemetry.
    Requires:
      events_df: ['vehicle_id','start_time']
      raw_df: ['vehicle_id','timestamp','latitude','longitude','speed_kmh','total_fuel_gal']
    """
    df = events_df.copy().sort_values(["vehicle_id", "start_time"])

    def _ctx(row: pd.Series) -> pd.Series:
        lb, rb = row["start_time"] - pd.Timedelta(hours=int(lookback_hours)), row["start_time"]
        m = (raw_df["vehicle_id"] == row["vehicle_id"]) & (raw_df["timestamp"] >= lb) & (raw_df["timestamp"] < rb)
        pre = raw_df.loc[m]

        if len(pre) < 5:
            return pd.Series({
                "pre_event_distance_km": 0.0,
                "pre_event_avg_speed": 0.0,
                "pre_event_moving_pct": 0.0,
                "pre_event_fuel_change": 0.0,
            })

        dist_km = float(calculate_trip_distance(pre[["latitude", "longitude"]]))
        sp = pre["speed_kmh"].dropna()
        avg_speed = float(sp.mean()) if len(sp) else 0.0
        moving_pct = float((sp > 5).mean()) if len(sp) else 0.0

        fuel = pre["total_fuel_gal"].dropna()
        fuel_change = float(fuel.iloc[-1] - fuel.iloc[0]) if len(fuel) >= 2 else 0.0

        return pd.Series({
            "pre_event_distance_km": dist_km,
            "pre_event_avg_speed": avg_speed,
            "pre_event_moving_pct": moving_pct,
            "pre_event_fuel_change": fuel_change,
        })

    ctx = df.apply(_ctx, axis=1)
    return pd.concat([df, ctx], axis=1)


def add_movement_variability(events_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds speed variability inside each event window.
    Requires:
      events_df: ['vehicle_id','start_time','end_time']
      raw_df: ['vehicle_id','timestamp','speed_kmh']
    """
    df = events_df.copy()

    def _mv(row: pd.Series) -> pd.Series:
        m = (
            (raw_df["vehicle_id"] == row["vehicle_id"])
            & (raw_df["timestamp"] >= row["start_time"])
            & (raw_df["timestamp"] <= row["end_time"])
        )
        e = raw_df.loc[m]
        if e.empty:
            return pd.Series({"speed_std": 0.0, "speed_max": 0.0})
        sp = e["speed_kmh"].dropna()
        return pd.Series(
            {
                "speed_std": float(sp.std()) if len(sp) > 1 else 0.0,
                "speed_max": float(sp.max()) if len(sp) else 0.0,
            }
        )

    mv = df.apply(_mv, axis=1)
    return pd.concat([df, mv], axis=1)
