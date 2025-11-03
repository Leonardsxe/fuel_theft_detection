from __future__ import annotations
import logging
from typing import Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def create_temporal_split_on_raw_data(df: pd.DataFrame, train_ratio: float = 0.80) -> np.ndarray:
    """
    Time-aware split per vehicle on RAW telemetry rows.

    Returns
    -------
    np.ndarray of bool, True for train rows.
    """
    if "vehicle_id" not in df or "timestamp" not in df:
        raise ValueError("Expected columns: vehicle_id, timestamp")
    # defensive: ensure sorted per vehicle
    train_mask = np.zeros(len(df), dtype=bool)
    logger.info(f"Creating temporal split ({train_ratio:.0%} train / {1-train_ratio:.0%} test)")
    for vid, g in df.groupby("vehicle_id"):
        g = g.sort_values("timestamp")
        cutoff_idx = int(len(g) * train_ratio)
        cutoff_idx = max(1, min(cutoff_idx, len(g) - 1))
        cutoff_ts = g["timestamp"].iloc[cutoff_idx - 1]
        train_mask[g.index] = g["timestamp"] <= cutoff_ts
        logger.debug(f"  {vid}: cutoff={cutoff_ts}  train_rows={train_mask[g.index].sum()}")
    return train_mask

def map_events_to_split(events_df: pd.DataFrame, raw_df: pd.DataFrame, raw_train_mask: np.ndarray) -> pd.DataFrame:
    """
    Tag each event with is_train by comparing its midpoint to the raw-data cutoff used per vehicle.
    """
    if events_df.empty:
        return events_df.assign(is_train=False)
    ev = events_df.copy()
    ev["mid_time"] = ev["start_time"] + (ev["end_time"] - ev["start_time"]) / 2
    ev["is_train"] = False

    train_mask_series = pd.Series(raw_train_mask, index=raw_df.index)
    for vid in ev["vehicle_id"].unique():
        m = raw_df["vehicle_id"] == vid
        tr = train_mask_series[m]
        if not tr.any():
            continue
        cutoff = raw_df.loc[m & tr, "timestamp"].max()
        ev.loc[ev["vehicle_id"] == vid, "is_train"] = ev.loc[ev["vehicle_id"] == vid, "mid_time"] <= cutoff
    logger.info(f"Events mapped to split: train={int(ev['is_train'].sum())}, test={int((~ev['is_train']).sum())}")
    return ev

def validate_no_leakage(events_df: pd.DataFrame, train_mask: np.ndarray, test_mask: np.ndarray) -> None:
    """
    Ensure no temporal overlap between train and test event windows per vehicle.
    Raises ValueError on overlap.
    """
    ev = events_df.copy()
    ev["__set"] = np.where(train_mask, "train", np.where(test_mask, "test", "none"))
    for vid, g in ev.groupby("vehicle_id"):
        g = g.sort_values("start_time")
        train_windows = g[g["__set"] == "train"][["start_time", "end_time"]].to_numpy()
        test_windows  = g[g["__set"] == "test" ][["start_time", "end_time"]].to_numpy()
        for s1, e1 in train_windows:
            for s2, e2 in test_windows:
                if (s1 <= e2) and (s2 <= e1):
                    raise ValueError(f"Temporal overlap found between train/test for vehicle {vid}")
