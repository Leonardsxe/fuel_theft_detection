"""
Event-specific feature engineering.
Statistical features extracted from event windows.
"""

import pandas as pd
import numpy as np

def add_event_statistics(events_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add statistical features for each event.
    
    Features:
    - n_points: Number of telemetry points in event
    - median_dt_s: Median time between points
    - n_negative_steps: Number of negative fuel steps
    - mean_step_neg: Mean magnitude of negative steps
    - p95_abs_dfuel: 95th percentile of absolute fuel change
    - median_speed: Median speed during event
    - pct_ign_on: Percentage of time with ignition on
    """
    events_df = events_df.copy()
    
    def extract_stats(row):
        mask = (
            (raw_df["vehicle_id"] == row["vehicle_id"]) &
            (raw_df["timestamp"] >= row["start_time"]) &
            (raw_df["timestamp"] <= row["end_time"])
        )
        
        event_data = raw_df.loc[mask]
        
        if event_data.empty:
            return pd.Series({
                "n_points": 0,
                "median_dt_s": np.nan,
                "n_negative_steps": 0,
                "mean_step_neg": np.nan,
                "p95_abs_dfuel": np.nan,
                "median_speed": np.nan,
                "pct_ign_on": np.nan
            })
        
        # Negative fuel steps
        neg_steps = event_data["dfuel"].dropna()
        neg_steps = neg_steps[neg_steps < -0.10]  # Below noise threshold
        
        return pd.Series({
            "n_points": int(len(event_data)),
            "median_dt_s": float(np.nanmedian(event_data["dt_s"])),
            "n_negative_steps": int(len(neg_steps)),
            "mean_step_neg": float(-neg_steps.mean()) if len(neg_steps) > 0 else np.nan,
            "p95_abs_dfuel": float(np.nanpercentile(np.abs(event_data["dfuel"].dropna()), 95)) if event_data["dfuel"].notna().any() else np.nan,
            "median_speed": float(np.nanmedian(event_data["speed_kmh"])),
            "pct_ign_on": float((event_data["ignition"] == True).mean())
        })
    
    stats = events_df.apply(extract_stats, axis=1)

    # If the detector already provided n_negative_steps, keep it and drop the recomputed one.
    if "n_negative_steps" in events_df.columns and "n_negative_steps" in stats.columns:
        stats = stats.drop(columns=["n_negative_steps"])

    # Assign (overwrites shared names you *want* to overwrite; avoids duplicate columns)
    events_df[stats.columns] = stats.values
    return events_df