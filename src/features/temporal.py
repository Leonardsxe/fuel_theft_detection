"""
Temporal feature engineering.
Time-based features including cyclical encoding and time-of-day indicators.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def add_time_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical time features using sine/cosine encoding.
    
    Cyclical encoding preserves the circular nature of time
    (e.g., 23:59 is close to 00:00).
    
    Args:
        df: DataFrame with timestamp or start_time column
    
    Returns:
        DataFrame with cyclical time features added
    """
    df = df.copy()
    
    # Determine which timestamp column to use
    if "start_time" in df.columns:
        ts_col = "start_time"
    elif "timestamp" in df.columns:
        ts_col = "timestamp"
    else:
        logger.warning("No timestamp column found")
        return df
    
    # Calculate midpoint for events (if start_time and end_time exist)
    if "start_time" in df.columns and "end_time" in df.columns:
        mid_ts = df["start_time"] + (df["end_time"] - df["start_time"]) / 2
    else:
        mid_ts = df[ts_col]
    
    # Extract hour (as float with minutes)
    df["hour"] = mid_ts.dt.hour + mid_ts.dt.minute / 60.0
    
    # Cyclical encoding for hour of day (0-24 hours)
    df["hod_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hod_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    
    # Day of week (0=Monday, 6=Sunday)
    df["weekday"] = mid_ts.dt.weekday
    
    # Cyclical encoding for day of week
    df["dow_sin"] = np.sin(2 * np.pi * df["weekday"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["weekday"] / 7.0)
    
    logger.debug("Added cyclical time features: hod_sin, hod_cos, dow_sin, dow_cos")
    
    return df


def add_time_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add categorical time features.
    
    Args:
        df: DataFrame with timestamp
    
    Returns:
        DataFrame with categorical time features
    """
    df = df.copy()
    
    # Determine timestamp column
    if "start_time" in df.columns and "end_time" in df.columns:
        mid_ts = df["start_time"] + (df["end_time"] - df["start_time"]) / 2
    elif "start_time" in df.columns:
        mid_ts = df["start_time"]
    elif "timestamp" in df.columns:
        mid_ts = df["timestamp"]
    else:
        logger.warning("No timestamp column found")
        return df
    
    # Weekend indicator
    df["is_weekend"] = (mid_ts.dt.weekday >= 5).astype(int)
    
    # Night indicator (22:00 - 05:59)
    hour = mid_ts.dt.hour
    df["is_night"] = ((hour >= 22) | (hour <= 5)).astype(int)
    
    # Business hours (08:00 - 17:59)
    df["is_business_hours"] = ((hour >= 8) & (hour < 18)).astype(int)
    
    logger.debug("Added categorical time features: is_weekend, is_night, is_business_hours")
    
    return df


def add_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add duration-based features.
    
    Args:
        df: DataFrame with start_time and end_time
    
    Returns:
        DataFrame with duration features
    """
    df = df.copy()
    
    if "start_time" not in df.columns or "end_time" not in df.columns:
        logger.warning("Missing start_time or end_time columns")
        return df
    
    # Duration in minutes (if not already present)
    if "duration_min" not in df.columns:
        df["duration_min"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60.0
    
    # Log-scaled duration (helps with skewed distributions)
    df["duration_log"] = np.log1p(df["duration_min"])
    
    # Duration bins
    df["duration_bin"] = pd.cut(
        df["duration_min"],
        bins=[0, 5, 10, 15, 30, 60, np.inf],
        labels=["very_short", "short", "medium", "medium_long", "long", "very_long"]
    ).astype(str)
    
    logger.debug("Added duration features: duration_log, duration_bin")
    
    return df


def add_all_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all temporal features.
    
    Args:
        df: DataFrame with timestamps
    
    Returns:
        DataFrame with all temporal features added
    """
    logger.info("Adding temporal features...")
    
    df = add_time_cyclical_features(df)
    df = add_time_categorical_features(df)
    df = add_duration_features(df)
    
    logger.info("✓ Temporal features added")
    
    return df