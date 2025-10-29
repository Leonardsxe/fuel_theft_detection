"""
Data quality validation and checks.
"Errors should never pass silently" - explicit validation before processing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import json
from pathlib import Path

from src.utils.timezone import ensure_series_utc

logger = logging.getLogger(__name__)


def check_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Check for and handle duplicate records.
    
    Args:
        df: DataFrame to check
        subset: Columns to check for duplicates (default: vehicle_id + timestamp)
    
    Returns:
        DataFrame with duplicates removed (keeping first occurrence)
    """
    if subset is None:
        subset = ["vehicle_id", "timestamp"]
    
    # Check for duplicates
    dup_mask = df.duplicated(subset=subset, keep=False)
    n_duplicates = dup_mask.sum()
    
    if n_duplicates > 0:
        logger.warning(f"Found {n_duplicates:,} duplicate records on {subset}")
        logger.info("Keeping first occurrence of duplicates")
        df = df.drop_duplicates(subset=subset, keep="first")
    else:
        logger.info("✓ No duplicate records found")
    
    return df


def validate_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate GPS coordinates and flag invalid values.
    
    Args:
        df: DataFrame with latitude/longitude columns
    
    Returns:
        DataFrame with invalid coordinates set to NaN
    """
    if "latitude" not in df.columns or "longitude" not in df.columns:
        logger.warning("GPS columns not found - skipping coordinate validation")
        return df
    
    df = df.copy()
    
    # Check valid ranges
    valid_coords = (
        (df["latitude"].between(-90, 90, inclusive='both')) &
        (df["longitude"].between(-180, 180, inclusive='both')) &
        (df["latitude"].notna()) & 
        (df["longitude"].notna())
    )
    
    invalid_count = (~valid_coords).sum()
    
    if invalid_count > 0:
        logger.warning(f"Found {invalid_count:,} invalid coordinates - setting to NaN")
        df.loc[~valid_coords, ["latitude", "longitude"]] = np.nan
    else:
        logger.info("✓ All coordinates valid")
    
    return df


def validate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate timestamp column and remove invalid dates.
    
    Args:
        df: DataFrame with timestamp column
    
    Returns:
        DataFrame with invalid timestamps removed
    """
    if "timestamp" not in df.columns:
        raise ValueError("Timestamp column not found")
    
    # Normalize timezone to UTC to avoid naive vs aware comparisons
    ts = ensure_series_utc(df["timestamp"]) if not getattr(df["timestamp"].dtype, 'tz', None) else df["timestamp"]

    # Check for NaT values
    invalid_mask = ts.isna()
    n_invalid = invalid_mask.sum()

    if n_invalid > 0:
        logger.warning(f"Removing {n_invalid:,} rows with invalid timestamps")
        df = df.loc[~invalid_mask].copy()
        ts = ts.loc[df.index]
    else:
        logger.info("✓ All timestamps valid")

    # Check for future dates (likely data errors)
    now = pd.Timestamp.now(tz='UTC')
    future_mask = ts > now
    n_future = future_mask.sum()

    if n_future > 0:
        logger.warning(f"Found {n_future:,} future timestamps - removing")
        df = df.loc[~future_mask].copy()
        ts = ts.loc[df.index]

    # Persist normalized timestamp column
    df["timestamp"] = ts

    return df


def check_data_gaps(
    df: pd.DataFrame, 
    max_gap_hours: float = 1.0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Identify large time gaps in data that might affect analysis.
    
    Args:
        df: DataFrame sorted by vehicle_id and timestamp
        max_gap_hours: Threshold for flagging large gaps
        verbose: Whether to log detailed gap information
    
    Returns:
        DataFrame with gap analysis (same as input, for chaining)
    """
    df = df.sort_values(["vehicle_id", "timestamp"])
    
    # Calculate time differences
    df["_dt_check"] = df.groupby("vehicle_id")["timestamp"].diff().dt.total_seconds() / 3600.0
    
    # Find large gaps
    large_gaps = df[df["_dt_check"] > max_gap_hours]
    
    if not large_gaps.empty:
        logger.warning(f"Found {len(large_gaps):,} time gaps > {max_gap_hours} hours")
        
        if verbose:
            # Summarize gaps by vehicle
            gap_summary = large_gaps.groupby("vehicle_id")["_dt_check"].agg([
                ("count", "count"),
                ("max_gap_hours", "max"),
                ("mean_gap_hours", "mean")
            ]).round(2)
            
            logger.info("Gap summary by vehicle:")
            for vid, row in gap_summary.iterrows():
                logger.info(f"  {vid}: {int(row['count'])} gaps, "
                          f"max={row['max_gap_hours']:.1f}h, "
                          f"mean={row['mean_gap_hours']:.1f}h")
    else:
        logger.info(f"✓ No time gaps > {max_gap_hours} hours")
    
    # Clean up temporary column
    df = df.drop(columns=["_dt_check"])
    
    return df


def check_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    """
    Check for missing values in critical columns.
    
    Args:
        df: DataFrame to check
    
    Returns:
        Dictionary with missing value counts per column
    """
    missing = df.isna().sum()
    missing_dict = {col: int(count) for col, count in missing.items() if count > 0}
    
    if missing_dict:
        logger.warning("Missing values detected:")
        for col, count in missing_dict.items():
            pct = 100 * count / len(df)
            logger.warning(f"  {col}: {count:,} ({pct:.2f}%)")
    else:
        logger.info("✓ No missing values in dataset")
    
    return missing_dict


def check_value_ranges(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Check value ranges for numeric columns.
    
    Args:
        df: DataFrame to check
    
    Returns:
        Dictionary with range statistics per column
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    ranges = {}
    for col in numeric_cols:
        ranges[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std())
        }
    
    return ranges


def generate_quality_report(df: pd.DataFrame, output_path: Optional[Path] = None) -> Dict:
    """
    Generate comprehensive data quality report.
    
    Args:
        df: DataFrame to analyze
        output_path: Optional path to save JSON report
    
    Returns:
        Dictionary with quality metrics
    """
    logger.info("Generating data quality report...")
    
    report = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "vehicles": int(df["vehicle_id"].nunique()) if "vehicle_id" in df.columns else 0,
        "date_range": {
            "start": str(df["timestamp"].min()) if "timestamp" in df.columns else None,
            "end": str(df["timestamp"].max()) if "timestamp" in df.columns else None
        },
        "missing_values": check_missing_values(df),
        "duplicates": int(df.duplicated(subset=["vehicle_id", "timestamp"]).sum()),
        "value_ranges": check_value_ranges(df)
    }
    
    # Coordinate validity
    if "latitude" in df.columns and "longitude" in df.columns:
        valid_coords = (
            df["latitude"].between(-90, 90) & 
            df["longitude"].between(-180, 180)
        ).sum()
        report["valid_coordinates"] = int(valid_coords)
        report["valid_coordinates_pct"] = float(100 * valid_coords / len(df))
    
    # Save report if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Quality report saved to {output_path}")
    
    logger.info("✓ Quality report generated")
    
    return report


def validate_data_pipeline(df: pd.DataFrame, max_gap_hours: float = 1.0) -> pd.DataFrame:
    """
    Run complete validation pipeline.
    
    Args:
        df: DataFrame to validate
        max_gap_hours: Threshold for time gap warnings
    
    Returns:
        Validated DataFrame (cleaned)
    """
    logger.info("Starting data validation pipeline...")
    
    # 1. Check and remove duplicates
    df = check_duplicates(df)
    
    # 2. Validate timestamps
    df = validate_timestamps(df)
    
    # 3. Validate coordinates
    df = validate_coordinates(df)
    
    # 4. Check for data gaps
    df = check_data_gaps(df, max_gap_hours)
    
    # 5. Check missing values
    check_missing_values(df)
    
    logger.info("✓ Data validation complete")
    
    return df
