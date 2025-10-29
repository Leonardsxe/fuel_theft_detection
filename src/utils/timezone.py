"""
Robust timezone handling utilities.
Ensures all timestamps are UTC-normalized for consistency.
"""

import pandas as pd
from typing import Union
import logging

logger = logging.getLogger(__name__)


def ensure_utc_timestamp(value: Union[pd.Timestamp, str, None]) -> pd.Timestamp:
    """
    Return a UTC timestamp regardless of the input's timezone awareness.
    
    Args:
        value: Timestamp, string, or None
    
    Returns:
        UTC-normalized pd.Timestamp (or pd.NaT if input is None/invalid)
    
    Examples:
        >>> ensure_utc_timestamp("2025-01-15 10:30:00")
        Timestamp('2025-01-15 10:30:00+0000', tz='UTC')
        
        >>> ensure_utc_timestamp(pd.Timestamp("2025-01-15 10:30:00", tz="US/Eastern"))
        Timestamp('2025-01-15 15:30:00+0000', tz='UTC')
    """
    if pd.isna(value):
        return pd.NaT
    
    try:
        ts = pd.Timestamp(value)
        
        # If no timezone info, assume UTC
        if ts.tzinfo is None:
            return ts.tz_localize('UTC')
        
        # If has timezone, convert to UTC
        return ts.tz_convert('UTC')
    
    except Exception as e:
        logger.warning(f"Failed to convert timestamp '{value}': {e}")
        return pd.NaT


def ensure_series_utc(series: pd.Series) -> pd.Series:
    """
    Normalize an entire series to UTC without losing datetime dtype.
    
    Args:
        series: Series with datetime-like values
    
    Returns:
        Series with UTC-normalized timestamps
    
    Examples:
        >>> df['timestamp'] = ensure_series_utc(df['timestamp'])
    """
    # Convert to datetime first
    ts_series = pd.to_datetime(series, errors='coerce')
    
    # Check if already timezone-aware
    if getattr(ts_series.dtype, 'tz', None) is None:
        # No timezone - assume UTC
        return ts_series.dt.tz_localize('UTC')
    
    # Has timezone - convert to UTC
    return ts_series.dt.tz_convert('UTC')


def get_timezone_info(series: pd.Series) -> dict:
    """
    Get timezone information about a datetime series.
    
    Args:
        series: Series with datetime values
    
    Returns:
        Dictionary with timezone statistics
    """
    ts_series = pd.to_datetime(series, errors='coerce')
    
    info = {
        'dtype': str(ts_series.dtype),
        'has_timezone': getattr(ts_series.dtype, 'tz', None) is not None,
        'timezone': str(getattr(ts_series.dtype, 'tz', None)),
        'null_count': ts_series.isna().sum(),
        'valid_count': ts_series.notna().sum()
    }
    
    return info


def validate_utc_series(series: pd.Series, name: str = "timestamp") -> None:
    """
    Validate that a series is properly UTC-normalized.
    
    Args:
        series: Series to validate
        name: Column name for error messages
    
    Raises:
        ValueError: If series is not UTC-normalized
    """
    if not pd.api.types.is_datetime64_any_dtype(series):
        raise ValueError(f"Column '{name}' is not datetime type")
    
    tz = getattr(series.dtype, 'tz', None)
    
    if tz is None:
        raise ValueError(f"Column '{name}' is not timezone-aware")
    
    if str(tz) != 'UTC':
        raise ValueError(f"Column '{name}' has timezone '{tz}', expected 'UTC'")
    
    logger.debug(f"✓ Column '{name}' is properly UTC-normalized")