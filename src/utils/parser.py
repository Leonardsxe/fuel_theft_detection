"""
Advanced parsing utilities for various data formats.
Handles messy real-world CSV data with mixed formats.
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def parse_coordinates(coord_str) -> Tuple[float, float]:
    """
    Parse coordinate string into (latitude, longitude) tuple.
    
    Handles formats like:
    - "7.078715, -73.857758"
    - "7.078715,-73.857758"
    
    Args:
        coord_str: Coordinate string
    
    Returns:
        Tuple of (latitude, longitude) or (nan, nan) if invalid
    
    Examples:
        >>> parse_coordinates("40.7128, -74.0060")
        (40.7128, -74.0060)
    """
    try:
        if pd.isna(coord_str) or coord_str == '':
            return np.nan, np.nan
        
        # Convert to string and strip
        coord_str = str(coord_str).strip()
        
        if ',' in coord_str:
            parts = coord_str.split(',')
            lat = float(parts[0].strip())
            lon = float(parts[1].strip())
            return lat, lon
        
        return np.nan, np.nan
    
    except (ValueError, AttributeError, IndexError):
        return np.nan, np.nan


def parse_speed(speed_val) -> float:
    """
    Parse speed value that might include units.
    
    Handles formats like:
    - "50"
    - "50.5"
    - "50 km/h"
    - "50.5 km/h"
    
    Args:
        speed_val: Speed value (numeric or string)
    
    Returns:
        Speed as float or nan if invalid
    
    Examples:
        >>> parse_speed("50 km/h")
        50.0
        >>> parse_speed("60.5")
        60.5
    """
    try:
        if pd.isna(speed_val):
            return np.nan
        
        # Convert to string and extract numeric part
        speed_str = str(speed_val)
        
        # Extract first numeric value (handles "XX km/h" format)
        match = re.search(r'([-+]?\d*\.?\d+)', speed_str)
        if match:
            return float(match.group(1))
        
        return np.nan
    
    except (ValueError, AttributeError):
        return np.nan


def parse_ignition(ign_val) -> bool:
    """
    Parse ignition status from various formats.
    
    Handles:
    - Boolean: True/False
    - Numeric: 0/1, 0.0/1.0
    - String (English): "true"/"false", "on"/"off", "yes"/"no"
    - String (Spanish): "encendido"/"apagado", "si"/"no", "sí"/"no"
    
    Args:
        ign_val: Ignition value
    
    Returns:
        Boolean ignition status
    
    Examples:
        >>> parse_ignition("encendido")
        True
        >>> parse_ignition(0)
        False
        >>> parse_ignition("on")
        True
    """
    try:
        if pd.isna(ign_val):
            return False
        
        ign_str = str(ign_val).strip().lower()
        
        # Map various representations to boolean
        true_vals = ["1", "true", "on", "encendido", "si", "sí", "yes"]
        false_vals = ["0", "false", "off", "apagado", "no"]
        
        if ign_str in true_vals:
            return True
        elif ign_str in false_vals:
            return False
        
        # Try numeric conversion
        try:
            return float(ign_str) > 0
        except:
            return False
    
    except:
        return False


def parse_timestamp_column(series: pd.Series) -> pd.Series:
    """
    Parse timestamp column handling multiple datetime formats.
    
    Handles:
    - ISO format: "2025-01-15 10:30:00"
    - Dotted format: "15.01.2025 10:30:00" (day-first)
    - Other formats with dayfirst hint
    
    Args:
        series: Series with timestamp strings
    
    Returns:
        Series with parsed datetime values
    
    Examples:
        >>> parse_timestamp_column(pd.Series(["2025-01-15 10:30:00", "15.01.2025 10:30:00"]))
    """
    # Normalize to string and strip whitespace
    ser = series.astype(str).str.strip()
    ser = ser.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    
    # Primary ISO-like format
    parsed = pd.to_datetime(ser, errors="coerce", format="%Y-%m-%d %H:%M:%S")
    
    # Try day-first dotted format (e.g., 01.12.2024 00:41:05)
    mask = parsed.isna() & ser.notna()
    if mask.any():
        alt = pd.to_datetime(ser[mask], errors="coerce", format="%d.%m.%Y %H:%M:%S")
        parsed.loc[mask] = alt
    
    # Fallback to generic parser with dayfirst hint
    mask = parsed.isna() & ser.notna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(ser[mask], errors="coerce", dayfirst=True)
    
    return parsed


def parse_label_column(series: pd.Series) -> pd.Series:
    """
    Parse label column from various formats to binary integer.
    
    Handles:
    - Boolean: True/False → 1/0
    - Numeric: 1.0/0.0
    - String: "true"/"yes"/"si"/"sí" → 1, others → 0
    
    Args:
        series: Series with label values
    
    Returns:
        Series with binary integer labels (0/1)
    
    Examples:
        >>> parse_label_column(pd.Series([True, "yes", "si", 0]))
        0    1
        1    1
        2    1
        3    0
    """
    if series.dtype == bool:
        return series.astype(int)
    
    elif series.dtype in [np.int64, np.float64]:
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    
    else:
        # String type
        return series.astype(str).str.lower().isin(["1", "true", "yes", "si", "sí"]).astype(int)


def pick_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    """
    Find matching column from candidates list.
    
    Tries:
    1. Exact match (case insensitive)
    2. Partial match (case insensitive)
    
    Args:
        columns: List of actual column names
        candidates: List of candidate names to match
    
    Returns:
        Matched column name or None
    
    Examples:
        >>> pick_column(["Velocidad (km/h)", "Tiempo"], ["speed", "velocidad"])
        "Velocidad (km/h)"
    """
    # Keep both original and trimmed names
    cleaned = [(col, col.strip()) for col in columns]
    
    # Exact match (case insensitive)
    for candidate in candidates:
        for original, stripped in cleaned:
            if stripped.lower() == candidate.lower():
                return original
    
    # Partial match (case insensitive)
    for candidate in candidates:
        for original, stripped in cleaned:
            if candidate.lower() in stripped.lower():
                return original
    
    return None


def extract_vehicle_id_from_filename(filename: str) -> str:
    """
    Extract vehicle ID from filename.
    
    Tries to extract first alphanumeric token from filename.
    Falls back to cleaned filename if no clear token found.
    
    Args:
        filename: CSV filename
    
    Returns:
        Vehicle ID string
    
    Examples:
        >>> extract_vehicle_id_from_filename("GQU478_data.csv")
        "GQU478"
        >>> extract_vehicle_id_from_filename("vehicle_123.csv")
        "vehicle"
    """
    # Remove .csv extension
    base = filename.split(".csv")[0]
    
    # Try to extract first alphanumeric token
    tokens = re.split(r"[_\s\-\.]", base)
    if tokens:
        token = tokens[0]
        if re.match(r"^[A-Za-z0-9]+$", token):
            return token
    
    # Fallback to full base name (cleaned)
    return base.replace(" ", "_")


def combine_fuel_tanks(
    left_tank: pd.Series,
    right_tank: pd.Series
) -> pd.Series:
    """
    Combine left and right fuel tank readings into total fuel.
    
    Args:
        left_tank: Left tank fuel level
        right_tank: Right tank fuel level
    
    Returns:
        Combined total fuel
    """
    left = pd.to_numeric(left_tank, errors="coerce").fillna(0)
    right = pd.to_numeric(right_tank, errors="coerce").fillna(0)
    
    return left + right


def try_multiple_encodings(filepath, encodings=['utf-8', 'latin1', 'utf-8']):
    """
    Try reading CSV with multiple encodings.
    
    Args:
        filepath: Path to CSV file
        encodings: List of encodings to try
    
    Returns:
        DataFrame or raises exception if all fail
    """
    for i, encoding in enumerate(encodings):
        try:
            if i == len(encodings) - 1:
                # Last attempt: use errors='ignore'
                df = pd.read_csv(filepath, low_memory=False, encoding=encoding, errors='ignore')
            else:
                df = pd.read_csv(filepath, low_memory=False, encoding=encoding)
            
            logger.info(f"Successfully read file with encoding: {encoding}")
            return df
        
        except (UnicodeDecodeError, Exception) as e:
            if i == len(encodings) - 1:
                logger.error(f"Failed to read file with any encoding")
                raise
            continue