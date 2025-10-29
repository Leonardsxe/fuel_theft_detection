"""
Stationary period segmentation.
Identifies continuous stationary periods with state-aware gap handling.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

from src.config.settings import DetectionConfig

logger = logging.getLogger(__name__)


class StationarySegmenter:
    """
    Segment continuous stationary periods for event detection.
    Handles different gap thresholds for stationary_on vs ignition_off states.
    """
    
    def __init__(self, config: DetectionConfig):
        """
        Initialize segmenter with configuration.
        
        Args:
            config: Detection configuration
        """
        self.config = config
    
    def segment_stationary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify continuous stationary periods with state-aware gap handling.
        
        Key features:
        - Different gap thresholds for stationary_on (5 min) vs ign_off (120 min)
        - Split on mode changes (stationary_on ↔ ign_off)
        - Minimum segment duration filtering
        
        Args:
            df: DataFrame with stationary flags
        
        Returns:
            DataFrame with columns: vehicle_id, start_time, end_time, is_ign_off, duration_min
        """
        logger.info("Segmenting stationary periods...")
        
        df = df.sort_values(["vehicle_id", "timestamp"]).copy()
        
        # Add previous state flags using shift with fill_value
        df["prev_stationary"] = df.groupby("vehicle_id")["stationary"].shift(1, fill_value=False).astype(bool)
        df["prev_stationary_on"] = df.groupby("vehicle_id")["stationary_on"].shift(1, fill_value=False).astype(bool)
        df["prev_ign_off"] = df.groupby("vehicle_id")["ign_off"].shift(1, fill_value=False).astype(bool)
        
        # Identify gaps that should break segments
        gap_on = (df["prev_stationary_on"]) & (df["dt_s"] > self.config.stationary.gap_limit_on_min * 60)
        gap_off = (df["prev_ign_off"]) & (df["dt_s"] > self.config.stationary.gap_limit_off_min * 60)
        
        # Mode flip: stationary but mode changed (stationary_on ↔ ign_off)
        mode_flip = (
            df["stationary"] & 
            df["prev_stationary"] & 
            (df["ign_off"].astype(int) != df["prev_ign_off"].astype(int))
        )
        
        # Segment start flag
        seg_start_flag = (
            df["stationary"] & 
            (~df["prev_stationary"] | gap_on | gap_off | mode_flip)
        )
        
        # Per-vehicle cumulative segment id on stationary rows
        df["seg_id"] = np.nan
        for vid, g in df.groupby("vehicle_id"):
            idx = g.index
            seg_ids = seg_start_flag.loc[idx].cumsum()
            seg_ids = seg_ids.where(g["stationary"], np.nan)
            df.loc[idx, "seg_id"] = seg_ids
        
        # Aggregate segments
        segments = (
            df.loc[df["stationary"] & df["seg_id"].notna()]
            .groupby(["vehicle_id", "seg_id"])
            .agg(
                start_time=("timestamp", "min"),
                end_time=("timestamp", "max"),
                is_ign_off=("ign_off", lambda x: x.mean() > 0.5),
                duration_min=("timestamp", lambda s: (s.max() - s.min()).total_seconds() / 60.0)
            )
            .reset_index(drop=False)
        )
        
        # Filter by minimum duration
        segments = segments[
            segments["duration_min"] >= self.config.stationary.min_segment_duration_min
        ].reset_index(drop=True)
        
        logger.info(f"✓ Found {len(segments)} stationary segments")
        
        # Log segment statistics
        if not segments.empty:
            state_counts = segments["is_ign_off"].value_counts()
            logger.info(f"  Stationary ON segments: {state_counts.get(False, 0)}")
            logger.info(f"  Ignition OFF segments: {state_counts.get(True, 0)}")
            logger.info(f"  Duration range: {segments['duration_min'].min():.1f} - {segments['duration_min'].max():.1f} min")
        
        return segments


def segment_stationary_periods(
    df: pd.DataFrame,
    config: DetectionConfig
) -> pd.DataFrame:
    """
    Convenience function for stationary segmentation.
    
    Args:
        df: DataFrame with stationary flags
        config: Detection configuration
    
    Returns:
        DataFrame with stationary segments
    """
    segmenter = StationarySegmenter(config)
    return segmenter.segment_stationary(df)