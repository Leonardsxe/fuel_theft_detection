"""
Temporal features for siphoning events:
- Cyclical hour-of-day
- Weekday/weekend & night flags
- Event duration (min/hours/log)
- Spacing to previous event (per vehicle) and first-of-day flag
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["add_all_temporal_features"]


def _ensure_duration(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    if "duration_min" not in out.columns:
        dur = (out["end_time"] - out["start_time"]).dt.total_seconds() / 60.0
        out["duration_min"] = dur
    out["duration_hours"] = out["duration_min"] / 60.0
    out["duration_log"] = np.log1p(out["duration_min"])
    return out


def _add_time_signals(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    mid_ts = out["start_time"] + (out["end_time"] - out["start_time"]) / 2
    hour = mid_ts.dt.hour + mid_ts.dt.minute / 60.0

    out["hod_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hod_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["weekday"] = mid_ts.dt.weekday
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    out["is_night"] = ((mid_ts.dt.hour >= 22) | (mid_ts.dt.hour <= 5)).astype(int)
    return out


def _add_spacing(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy().sort_values(["vehicle_id", "start_time"])
    out["time_since_last_event_hours"] = (
        out.groupby("vehicle_id")["start_time"].diff().dt.total_seconds() / 3600.0
    ).fillna(-1)
    out["is_first_event_of_day"] = (
        out.groupby(["vehicle_id", out["start_time"].dt.date]).cumcount() == 0
    ).astype(int)
    return out


def add_all_temporal_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Stateless transform that returns a copy of events_df with temporal features.
    Required columns: ['vehicle_id','start_time','end_time'] (datetime)
    """
    df = events_df.copy()
    df = _ensure_duration(df)
    df = _add_time_signals(df)
    df = _add_spacing(df)
    return df
