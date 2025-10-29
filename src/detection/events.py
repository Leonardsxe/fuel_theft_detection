"""
Event detection engine.
Core logic for detecting fuel theft events using sliding windows and patterns.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

from src.config.settings import DetectionConfig
from src.detection.patterns import DetectionPattern, PatternManager
from src.detection.thresholds import NoiseThresholdCalculator

logger = logging.getLogger(__name__)


class EventDetector:
    """
    Detect fuel theft events using sliding window and pattern matching.
    """
    
    def __init__(
        self,
        config: DetectionConfig,
        thresholds: Dict[Tuple[str, str], Tuple[float, float]],
        raw_df: pd.DataFrame
    ):
        """
        Initialize event detector.
        
        Args:
            config: Detection configuration
            thresholds: Pre-computed noise thresholds
            raw_df: Raw telemetry data for event context
        """
        self.config = config
        self.thresholds = thresholds
        self.raw_df = raw_df
        self.pattern_manager = PatternManager(config.patterns)
    
    def detect_events_in_segment(self, segment: pd.Series) -> List[Dict]:
        """
        Detect potential theft events within a stationary segment.
        
        Args:
            segment: Segment row with vehicle_id, start_time, end_time, is_ign_off
        
        Returns:
            List of detected events
        """
        vid = segment["vehicle_id"]
        state = "ign_off" if segment["is_ign_off"] else "stationary_on"
        
        # Get segment data
        mask = (
            (self.raw_df["vehicle_id"] == vid) &
            (self.raw_df["timestamp"] >= segment["start_time"]) &
            (self.raw_df["timestamp"] <= segment["end_time"])
        )
        
        g = self.raw_df.loc[mask].sort_values("timestamp")
        
        if len(g) < 2:
            return []
        
        # Get thresholds
        threshold_step, threshold_cum = self.thresholds.get(
            (vid, state),
            self._get_default_thresholds(state)
        )
        
        # Get applicable patterns for this state
        patterns = self.pattern_manager.get_patterns_for_state(state)
        
        if not patterns:
            return []
        
        # Detect events using sliding window
        events = []
        
        for pattern_key, pattern in patterns.items():
            pattern_events = self._scan_with_pattern(
                g, pattern, threshold_step, threshold_cum, state
            )
            events.extend(pattern_events)
        
        return events
    
    def _scan_with_pattern(
        self,
        data: pd.DataFrame,
        pattern: DetectionPattern,
        threshold_step: float,
        threshold_cum: float,
        state: str
    ) -> List[Dict]:
        """
        Scan data for events matching a specific pattern.
        
        Args:
            data: Segment data
            pattern: Detection pattern
            threshold_step: Step threshold
            threshold_cum: Cumulative threshold
            state: State name
        
        Returns:
            List of detected events
        """
        events = []
        
        times = data["timestamp"].values
        dfuel = data["dfuel"].values.astype(float)
        n = len(data)
        
        # Sliding window
        for start_idx in range(n):
            for end_idx in range(start_idx + 1, n):
                # Calculate duration
                duration_min = float(
                    (times[end_idx] - times[start_idx]).astype("timedelta64[s]").astype(float) / 60.0
                )
                
                # Check if duration is in pattern range
                if not pattern.validate_duration(duration_min):
                    # If already past max duration, break inner loop
                    if pattern.max_duration_min is not None and duration_min > pattern.max_duration_min:
                        break
                    continue
                
                # Extract fuel changes in window
                window_fuel = dfuel[start_idx:end_idx + 1]
                neg = window_fuel[np.isfinite(window_fuel) & (window_fuel < self.config.noise.min_noise_dfuel)]
                
                if neg.size == 0:
                    continue
                
                cum_drop = float(-neg.sum())
                max_step = float(-neg.min())
                n_neg_steps = int(neg.size)
                
                # Check thresholds based on state
                if state == "ign_off":
                    # For ignition off, use strict cumulative threshold
                    threshold_met = cum_drop >= threshold_cum
                    
                    # Additional OFF-specific checks
                    if threshold_met and pattern.min_points is not None:
                        n_points = end_idx - start_idx + 1
                        if n_points < pattern.min_points:
                            threshold_met = False
                    
                    if threshold_met and pattern.min_median_dt_s is not None:
                        seg_dt = pd.to_numeric(data["dt_s"].iloc[start_idx + 1:end_idx + 1], errors="coerce").dropna()
                        if seg_dt.empty or np.nanmedian(seg_dt) < pattern.min_median_dt_s:
                            threshold_met = False
                else:
                    # For stationary_on, allow step OR cumulative
                    threshold_met = (cum_drop >= threshold_cum) or (max_step >= threshold_step)
                    
                    # Short pattern: require at least 2 negative steps
                    if pattern.name.startswith("short") and n_neg_steps < 2:
                        threshold_met = False
                
                if not threshold_met:
                    continue
                
                # Create event
                event = {
                    "vehicle_id": vid,
                    "start_time": pd.Timestamp(times[start_idx]),
                    "end_time": pd.Timestamp(times[end_idx]),
                    "duration_min": duration_min,
                    "drop_gal": cum_drop,
                    "min_step_gal": max_step,
                    "pattern": pattern.name,
                    "n_negative_steps": n_neg_steps
                }
                
                # Plausibility check
                if self._is_plausible(event):
                    events.append(event)
        
        return events
    
    def _is_plausible(self, event: Dict) -> bool:
        """
        Check if event is physically plausible.
        
        Args:
            event: Event dictionary
        
        Returns:
            True if plausible
        """
        duration = max(event["duration_min"], 0.001)
        rate = event["drop_gal"] / duration
        
        # Check maximum rate
        if rate > self.config.plausibility.max_rate_gpm:
            return False
        
        # Check maximum single step
        if event["min_step_gal"] > self.config.plausibility.max_step_gal:
            return False
        
        return True
    
    def _get_default_thresholds(self, state: str) -> Tuple[float, float]:
        """Get default thresholds if vehicle-specific not available"""
        if state == "stationary_on":
            return (
                self.config.thresholds_stationary_on.min_step_gal,
                self.config.thresholds_stationary_on.min_cumulative_gal
            )
        else:
            return (
                self.config.thresholds_ignition_off.min_step_gal,
                self.config.thresholds_ignition_off.min_cumulative_gal
            )


def detect_events(
    segments: pd.DataFrame,
    raw_df: pd.DataFrame,
    config: DetectionConfig,
    thresholds: Dict[Tuple[str, str], Tuple[float, float]]
) -> pd.DataFrame:
    """
    Detect events across all segments.
    
    Args:
        segments: Stationary segments DataFrame
        raw_df: Raw telemetry data
        config: Detection configuration
        thresholds: Noise thresholds
    
    Returns:
        DataFrame with detected events
    """
    logger.info(f"Detecting events in {len(segments)} segments...")
    
    detector = EventDetector(config, thresholds, raw_df)
    
    all_events = []
    for _, segment in segments.iterrows():
        segment_events = detector.detect_events_in_segment(segment)
        all_events.extend(segment_events)
    
    if not all_events:
        logger.warning("No events detected")
        return pd.DataFrame()
    
    events_df = pd.DataFrame(all_events)
    
    logger.info(f"✓ Detected {len(events_df)} events before NMS")
    
    # Log pattern distribution
    if not events_df.empty:
        pattern_counts = events_df["pattern"].value_counts()
        logger.info("Event distribution by pattern:")
        for pattern, count in pattern_counts.items():
            logger.info(f"  {pattern}: {count}")
    
    return events_df