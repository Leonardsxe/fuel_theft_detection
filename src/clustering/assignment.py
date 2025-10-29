"""
Assign events to hotspot clusters.
Adds cluster-based features to detected events.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def assign_events_to_clusters(
    events_df: pd.DataFrame,
    stationary_pts: pd.DataFrame
) -> pd.DataFrame:
    """
    Assign detected events to hotspot clusters.
    
    Args:
        events_df: DataFrame with detected events
        stationary_pts: DataFrame with clustered stationary points
    
    Returns:
        DataFrame with cluster features added
    """
    if events_df.empty:
        logger.warning("No events to assign to clusters")
        events_df["cluster_id"] = []
        events_df["cluster_count"] = []
        events_df["is_hotspot"] = []
        return events_df
    
    if stationary_pts.empty or "cluster_id" not in stationary_pts.columns:
        logger.warning("No clustered stationary points available")
        events_df["cluster_id"] = -1
        events_df["cluster_count"] = 0
        events_df["is_hotspot"] = 0
        return events_df
    
    logger.info("Assigning events to clusters...")
    
    def get_event_cluster(event):
        """Get cluster assignment for a single event"""
        mask = (
            (stationary_pts["vehicle_id"] == event["vehicle_id"]) &
            (stationary_pts["timestamp"] >= event["start_time"]) &
            (stationary_pts["timestamp"] <= event["end_time"])
        )
        
        cluster_ids = stationary_pts.loc[mask, "cluster_id"]
        
        if cluster_ids.empty:
            return pd.Series({
                "cluster_id": -1,
                "cluster_count": 0,
                "is_hotspot": 0
            })
        
        # Use mode (most common cluster) for the event
        mode_cluster = int(cluster_ids.mode().iloc[0]) if not cluster_ids.mode().empty else -1
        count = int((cluster_ids == mode_cluster).sum())
        
        return pd.Series({
            "cluster_id": mode_cluster,
            "cluster_count": count,
            "is_hotspot": int(mode_cluster >= 0)
        })
    
    # Apply to all events
    cluster_features = events_df.apply(get_event_cluster, axis=1)
    events_df = pd.concat([events_df, cluster_features], axis=1)
    
    # Log statistics
    n_hotspot = events_df["is_hotspot"].sum()
    pct_hotspot = 100 * n_hotspot / len(events_df) if len(events_df) > 0 else 0
    
    logger.info(f"✓ Events at hotspots: {n_hotspot} ({pct_hotspot:.1f}%)")
    
    return events_df


def calculate_hotspot_statistics(
    events_df: pd.DataFrame,
    stationary_pts: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate hotspot statistics for analysis.
    
    Args:
        events_df: DataFrame with events
        stationary_pts: DataFrame with clustered stationary points
    
    Returns:
        DataFrame with hotspot statistics
    """
    if events_df.empty or stationary_pts.empty:
        return pd.DataFrame()
    
    # Events at each hotspot
    hotspot_events = events_df[events_df["is_hotspot"] == 1]
    
    if hotspot_events.empty:
        return pd.DataFrame()
    
    stats = hotspot_events.groupby(["vehicle_id", "cluster_id"]).agg(
        n_events=("cluster_id", "size"),
        total_fuel_drop=("drop_gal", "sum"),
        mean_fuel_drop=("drop_gal", "mean"),
        max_fuel_drop=("drop_gal", "max"),
        mean_duration=("duration_min", "mean")
    ).reset_index()
    
    # Add cluster centroids
    cluster_centroids = stationary_pts.groupby(["vehicle_id", "cluster_id"]).agg({
        "latitude": "mean",
        "longitude": "mean"
    }).reset_index()
    
    stats = stats.merge(cluster_centroids, on=["vehicle_id", "cluster_id"], how="left")
    
    # Sort by number of events
    stats = stats.sort_values("n_events", ascending=False)
    
    logger.info(f"Identified {len(stats)} unique hotspots")
    
    return stats