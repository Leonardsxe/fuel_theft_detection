"""
Non-Maximum Suppression (NMS) for overlapping events.
Removes duplicate detections, keeping the strongest (largest drop).
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def calculate_temporal_iou(
    event1: Dict,
    event2: Dict
) -> float:
    """
    Calculate Intersection over Union (IoU) for two time intervals.
    
    Args:
        event1: First event with start_time and end_time
        event2: Second event with start_time and end_time
    
    Returns:
        IoU value between 0 and 1
    """
    s1, e1 = event1["start_time"], event1["end_time"]
    s2, e2 = event2["start_time"], event2["end_time"]
    
    # Calculate intersection
    intersection = max(pd.Timedelta(0), min(e1, e2) - max(s1, s2))
    
    # Calculate union
    union = (e1 - s1) + (e2 - s2) - intersection
    
    # Return IoU
    if union > pd.Timedelta(0):
        return intersection.total_seconds() / union.total_seconds()
    else:
        return 0.0


def apply_nms(
    events: List[Dict],
    iou_threshold: float = 0.5
) -> List[Dict]:
    """
    Apply Non-Maximum Suppression to remove overlapping events.
    
    Keeps events with largest fuel drop when overlapping.
    
    Args:
        events: List of event dictionaries
        iou_threshold: IoU threshold for considering events as overlapping
    
    Returns:
        List of events after NMS
    """
    if not events:
        return []
    
    # Sort by fuel drop (descending) - strongest first
    sorted_events = sorted(events, key=lambda d: d["drop_gal"], reverse=True)
    
    keep = []
    used = np.zeros(len(sorted_events), dtype=bool)
    
    for i in range(len(sorted_events)):
        if used[i]:
            continue
        
        keep.append(sorted_events[i])
        
        # Mark overlapping events as used
        for j in range(i + 1, len(sorted_events)):
            if used[j]:
                continue
            
            # Only compare events from same vehicle
            if sorted_events[i]["vehicle_id"] != sorted_events[j]["vehicle_id"]:
                continue
            
            # Calculate IoU
            iou = calculate_temporal_iou(sorted_events[i], sorted_events[j])
            
            if iou >= iou_threshold:
                used[j] = True
    
    return keep


def nms_events_dataframe(
    events_df: pd.DataFrame,
    iou_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Apply NMS to events DataFrame.
    
    Args:
        events_df: DataFrame with events
        iou_threshold: IoU threshold
    
    Returns:
        DataFrame with NMS applied
    """
    if events_df.empty:
        return events_df
    
    logger.info(f"Applying NMS with IoU threshold={iou_threshold}")
    
    initial_count = len(events_df)
    
    # Convert to list of dicts
    events_list = events_df.to_dict("records")
    
    # Apply NMS
    kept_events = apply_nms(events_list, iou_threshold)
    
    # Convert back to DataFrame
    result_df = pd.DataFrame(kept_events)
    
    removed = initial_count - len(result_df)
    logger.info(f"✓ NMS removed {removed} overlapping events")
    logger.info(f"✓ Kept {len(result_df)} unique events")
    
    return result_df


def remove_exact_duplicates(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate events.
    
    Args:
        events_df: DataFrame with events
    
    Returns:
        DataFrame with duplicates removed
    """
    if events_df.empty:
        return events_df
    
    initial_count = len(events_df)
    
    # Remove exact duplicates
    events_df = events_df.drop_duplicates(
        subset=["vehicle_id", "start_time", "end_time", "drop_gal"]
    ).reset_index(drop=True)
    
    removed = initial_count - len(events_df)
    if removed > 0:
        logger.info(f"Removed {removed} exact duplicate events")
    
    return events_df