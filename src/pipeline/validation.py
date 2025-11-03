"""
Pipeline Validation Utilities
Purpose: Consolidated validation functions for data splitting and leakage prevention.
Previously scattered in utils/splitting.py - now centralized in pipeline package.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_temporal_split_on_raw_data(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    time_column: str = "timestamp",
    group_column: str = "vehicle_id",
) -> np.ndarray:
    """
    Create time-based train/test split on raw telemetry data.
    
    Splits each vehicle's data temporally (chronologically) to prevent
    future information leaking into training set.
    
    Args:
        df: Raw telemetry DataFrame
        train_ratio: Proportion of data for training (e.g., 0.80 = 80%)
        time_column: Name of timestamp column
        group_column: Name of grouping column (e.g., vehicle_id)
    
    Returns:
        Boolean mask array where True = train, False = test
    
    Example:
        >>> train_mask = create_temporal_split_on_raw_data(df, train_ratio=0.80)
        >>> train_df = df[train_mask]
        >>> test_df = df[~train_mask]
    """
    
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    
    if time_column not in df.columns:
        raise ValueError(f"Time column '{time_column}' not found in DataFrame")
    
    if group_column not in df.columns:
        raise ValueError(f"Group column '{group_column}' not found in DataFrame")
    
    train_mask = np.zeros(len(df), dtype=bool)
    
    # Split per vehicle
    for group_id, group_df in df.groupby(group_column):
        # Find temporal cutoff for this vehicle
        cutoff = group_df[time_column].quantile(train_ratio)
        
        # Mark rows before cutoff as training
        vehicle_train_mask = group_df[time_column] <= cutoff
        train_mask[group_df.index] = vehicle_train_mask.values
    
    n_train = train_mask.sum()
    n_test = (~train_mask).sum()
    
    logger.info(
        f"Temporal split created: "
        f"train={n_train:,} ({n_train/len(df)*100:.1f}%), "
        f"test={n_test:,} ({n_test/len(df)*100:.1f}%)"
    )
    
    return train_mask


def map_events_to_split(
    events_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    raw_train_mask: np.ndarray,
    time_column: str = "timestamp",
    group_column: str = "vehicle_id",
) -> pd.DataFrame:
    """
    Map detected events to train/test split based on their midpoint timestamp.
    
    Events are assigned to train/test based on when they occurred relative
    to the raw data split cutoff for each vehicle.
    
    Args:
        events_df: Detected events DataFrame with start_time, end_time
        raw_df: Raw telemetry DataFrame used for splitting
        raw_train_mask: Boolean mask from create_temporal_split_on_raw_data
        time_column: Name of timestamp column in raw_df
        group_column: Name of grouping column
    
    Returns:
        Events DataFrame with added 'is_train' boolean column
    
    Example:
        >>> events_df = map_events_to_split(events_df, raw_df, train_mask)
        >>> train_events = events_df[events_df['is_train']]
        >>> test_events = events_df[~events_df['is_train']]
    """
    
    if len(raw_train_mask) != len(raw_df):
        raise ValueError(
            f"Mask length {len(raw_train_mask)} != DataFrame length {len(raw_df)}"
        )
    
    events_df = events_df.copy()
    
    # Calculate event midpoint
    events_df['mid_time'] = (
        events_df['start_time'] + (events_df['end_time'] - events_df['start_time']) / 2
    )
    
    events_df['is_train'] = False
    
    # For each vehicle, find train/test cutoff
    for vehicle_id in events_df[group_column].unique():
        # Get vehicle's raw data rows
        vehicle_mask = raw_df[group_column] == vehicle_id
        vehicle_raw = raw_df[vehicle_mask]
        
        # Get train rows for this vehicle (using boolean indexing on aligned mask)
        train_mask_series = pd.Series(raw_train_mask, index=raw_df.index)
        vehicle_train_rows = vehicle_raw[train_mask_series[vehicle_mask]]
        
        if vehicle_train_rows.empty:
            logger.warning(f"No training data for vehicle {vehicle_id}")
            continue
        
        # Find temporal cutoff (last timestamp in training data)
        train_cutoff = vehicle_train_rows[time_column].max()
        
        # Mark events as train if midpoint is before cutoff
        vehicle_events_mask = events_df[group_column] == vehicle_id
        events_df.loc[vehicle_events_mask, 'is_train'] = (
            events_df.loc[vehicle_events_mask, 'mid_time'] <= train_cutoff
        )
    
    # Clean up temporary column
    events_df = events_df.drop(columns=['mid_time'])
    
    n_train = events_df['is_train'].sum()
    n_test = (~events_df['is_train']).sum()
    
    logger.info(
        f"Events mapped to split: "
        f"train={n_train:,} ({n_train/len(events_df)*100:.1f}%), "
        f"test={n_test:,} ({n_test/len(events_df)*100:.1f}%)"
    )
    
    return events_df


def validate_no_leakage(
    events_df: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    group_column: str = "vehicle_id",
    verbose: bool = True,
) -> Tuple[bool, Optional[pd.DataFrame]]:
    """
    Verify no temporal information leakage from test to train.
    
    Checks that for each vehicle:
    - All training events occur before all test events
    - No temporal overlap between train and test events
    
    Args:
        events_df: Events DataFrame with start_time, end_time, is_train
        train_mask: Boolean mask for training events
        test_mask: Boolean mask for test events
        group_column: Grouping column (e.g., vehicle_id)
        verbose: If True, log detailed validation results
    
    Returns:
        Tuple of (is_valid, leakage_report_df)
        - is_valid: True if no leakage detected
        - leakage_report_df: DataFrame with per-vehicle validation results
    
    Raises:
        ValueError: If critical leakage detected (can be disabled)
    """
    
    train_events = events_df[train_mask]
    test_events = events_df[test_mask]
    
    if verbose:
        logger.info(f"Validating split: {len(train_events)} train, {len(test_events)} test events")
    
    if train_events.empty:
        logger.warning("Empty training set")
        return False, None
    
    if test_events.empty:
        logger.warning("Empty test set")
        return False, None
    
    # Check per vehicle
    leakage_records = []
    has_leakage = False
    
    for vehicle_id in events_df[group_column].unique():
        train_vid = train_events[train_events[group_column] == vehicle_id]
        test_vid = test_events[test_events[group_column] == vehicle_id]
        
        if train_vid.empty or test_vid.empty:
            continue
        
        max_train_time = train_vid['end_time'].max()
        min_test_time = test_vid['start_time'].min()
        
        # Check for overlap
        overlap = max_train_time >= min_test_time
        
        if overlap:
            overlap_duration = (max_train_time - min_test_time).total_seconds() / 3600  # hours
            logger.warning(
                f"⚠️  Temporal overlap detected for vehicle {vehicle_id}: "
                f"{overlap_duration:.1f} hours"
            )
            has_leakage = True
        elif verbose:
            gap_duration = (min_test_time - max_train_time).total_seconds() / 3600  # hours
            logger.info(
                f"✓ Vehicle {vehicle_id}: No overlap "
                f"(gap: {gap_duration:.1f} hours)"
            )
        
        leakage_records.append({
            'vehicle_id': vehicle_id,
            'n_train_events': len(train_vid),
            'n_test_events': len(test_vid),
            'max_train_time': max_train_time,
            'min_test_time': min_test_time,
            'has_overlap': overlap,
            'gap_hours': (min_test_time - max_train_time).total_seconds() / 3600 if not overlap else -overlap_duration,
        })
    
    leakage_report = pd.DataFrame(leakage_records)
    
    if has_leakage:
        logger.error("❌ Data leakage detected!")
        n_leaking = leakage_report['has_overlap'].sum()
        logger.error(f"   {n_leaking}/{len(leakage_report)} vehicles have temporal overlap")
        return False, leakage_report
    else:
        logger.info("✓ Leakage validation passed - no temporal overlap detected")
        return True, leakage_report


def validate_split_balance(
    events_df: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    label_column: str = "y",
    min_train_positives: int = 10,
    min_test_positives: int = 5,
    warn_only: bool = True,
) -> bool:
    """
    Check if train/test split has sufficient positive examples.
    
    Args:
        events_df: Events DataFrame
        train_mask: Boolean mask for training events
        test_mask: Boolean mask for test events
        label_column: Name of label column
        min_train_positives: Minimum positive examples in train
        min_test_positives: Minimum positive examples in test
        warn_only: If True, only warn; if False, raise ValueError
    
    Returns:
        True if split is balanced, False otherwise
    
    Raises:
        ValueError: If split is severely imbalanced and warn_only=False
    """
    
    if label_column not in events_df.columns:
        logger.warning(f"Label column '{label_column}' not found - skipping balance check")
        return True
    
    train_events = events_df[train_mask]
    test_events = events_df[test_mask]
    
    train_positives = train_events[label_column].sum()
    test_positives = test_events[label_column].sum()
    
    train_positive_rate = train_positives / len(train_events) if len(train_events) > 0 else 0
    test_positive_rate = test_positives / len(test_events) if len(test_events) > 0 else 0
    
    logger.info(
        f"Split balance: "
        f"train={train_positives}/{len(train_events)} ({train_positive_rate*100:.1f}%), "
        f"test={test_positives}/{len(test_events)} ({test_positive_rate*100:.1f}%)"
    )
    
    issues = []
    
    if train_positives < min_train_positives:
        issues.append(f"Insufficient training positives: {train_positives} < {min_train_positives}")
    
    if test_positives < min_test_positives:
        issues.append(f"Insufficient test positives: {test_positives} < {min_test_positives}")
    
    # Warn about extreme imbalance
    if train_positive_rate < 0.01:
        issues.append(f"Extreme train imbalance: {train_positive_rate*100:.2f}% positive")
    
    if test_positive_rate < 0.01:
        issues.append(f"Extreme test imbalance: {test_positive_rate*100:.2f}% positive")
    
    if issues:
        message = "Split balance issues detected:\n  " + "\n  ".join(issues)
        if warn_only:
            logger.warning(message)
            return False
        else:
            raise ValueError(message)
    
    logger.info("✓ Split balance validated")
    return True


def stratified_temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    time_column: str = "timestamp",
    group_column: str = "vehicle_id",
    stratify_column: Optional[str] = None,
) -> np.ndarray:
    """
    Create stratified temporal split (advanced).
    
    Similar to create_temporal_split_on_raw_data, but attempts to maintain
    class balance across groups when stratify_column is provided.
    
    Args:
        df: DataFrame
        train_ratio: Proportion for training
        time_column: Timestamp column
        group_column: Grouping column
        stratify_column: Optional column to stratify by (e.g., 'stationary_drain')
    
    Returns:
        Boolean train mask
    
    Note:
        This is more complex and may not be feasible for time-series data
        where temporal ordering must be preserved. Use with caution.
    """
    
    if stratify_column is None:
        # Fall back to standard temporal split
        return create_temporal_split_on_raw_data(df, train_ratio, time_column, group_column)
    
    logger.info(f"Creating stratified temporal split on '{stratify_column}'")
    
    # Not implemented - requires careful handling of temporal dependencies
    # For now, fall back to standard temporal split
    logger.warning("Stratified temporal split not yet implemented - using standard temporal split")
    
    return create_temporal_split_on_raw_data(df, train_ratio, time_column, group_column)