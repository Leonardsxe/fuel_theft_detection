"""
Raw data loading with column standardization.
Handles various CSV formats and naming conventions including Spanish names.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from src.utils.parser import (
    parse_coordinates,
    parse_speed,
    parse_ignition,
    parse_timestamp_column,
    parse_label_column,
    pick_column,
    extract_vehicle_id_from_filename,
    combine_fuel_tanks,
    try_multiple_encodings
)

logger = logging.getLogger(__name__)


def load_raw_csv(path: Path, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Load raw CSV file with error handling and multiple encoding support.
    
    Args:
        path: Path to CSV file
        encoding: Initial encoding to try (default: utf-8)
    
    Returns:
        DataFrame with raw data
    
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    logger.info(f"Loading data from {path}")
    
    # Try multiple encodings
    try:
        df = try_multiple_encodings(path, encodings=['utf-8', 'latin1', 'utf-8'])
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        return df
    
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        raise


def standardize_columns(
    df: pd.DataFrame,
    column_mapping: Optional[Dict[str, List[str]]] = None,
    vehicle_id: Optional[str] = None,
    source_filename: Optional[str] = None
) -> pd.DataFrame:
    """
    Standardize column names and parse data with enhanced format support.
    
    Handles:
    - Multiple column name formats (English/Spanish)
    - Coordinate strings ("lat, lon")
    - Speed with units ("50 km/h")
    - Boolean formats (Spanish/English)
    - Multiple timestamp formats
    - Separate fuel tanks
    
    Args:
        df: DataFrame with potentially varying column names
        column_mapping: Dictionary mapping canonical names to possible variants
        vehicle_id: Optional vehicle ID (extracted from filename if None)
        source_filename: Source filename (for vehicle ID extraction)
    
    Returns:
        DataFrame with standardized columns and parsed data
    """
    df = df.copy()
    
    # Lowercase and strip whitespace from all columns
    df.columns = [c.strip().lower() for c in df.columns]
    raw_columns = list(df.columns)
    
    # Extended column mapping with Spanish names
    if column_mapping is None:
        column_mapping = {
            "vehicle_id": ["vehicle_id", "vehiculo", "veh_id", "vehicleid", "equipo"],
            "timestamp": ["tiempo", "timestamp", "time", "datetime", "fecha", "fechahora", "fecha_hora"],
            "coordinates": ["coordenadas", "coordinates", "coords", "coordinate"],
            "latitude": ["latitude", "lat"],
            "longitude": ["longitude", "lon", "lng", "longitud"],
            "speed_kmh": ["velocidad", "speed", "speed_kmh", "velocidad km/h"],
            "ignition": ["ignición", "ignicion", "ignition", "ignition_status"],
            "total_fuel_gal": ["tanque total", "total_fuel", "fuel_level", "fuel", "totalfuel"],
            "left_tank": ["tanque izquierdo", "left_tank", "left_fuel"],
            "right_tank": ["tanque derecho", "right_tank", "right_fuel"],
            "label": ["stationary_drain", "drain", "fuel_drain", "label"]
        }
    
    # Find columns using flexible matching
    vehicle_col = pick_column(raw_columns, column_mapping.get("vehicle_id", []))
    timestamp_col = pick_column(raw_columns, column_mapping["timestamp"])
    coord_col = pick_column(raw_columns, column_mapping.get("coordinates", []))
    lat_col = pick_column(raw_columns, column_mapping["latitude"])
    lon_col = pick_column(raw_columns, column_mapping["longitude"])
    speed_col = pick_column(raw_columns, column_mapping["speed_kmh"])
    ignition_col = pick_column(raw_columns, column_mapping["ignition"])
    total_fuel_col = pick_column(raw_columns, column_mapping["total_fuel_gal"])
    left_tank_col = pick_column(raw_columns, column_mapping.get("left_tank", []))
    right_tank_col = pick_column(raw_columns, column_mapping.get("right_tank", []))
    label_col = pick_column(raw_columns, column_mapping.get("label", []))
    
    # Create standardized dataframe
    sdf = pd.DataFrame()
    
    # 1. Timestamp (required)
    if timestamp_col is None:
        raise ValueError("No timestamp column found")
    
    sdf["timestamp"] = parse_timestamp_column(df[timestamp_col])
    
    # 2. Vehicle ID
    if vehicle_id is not None:
        sdf["vehicle_id"] = vehicle_id
    elif vehicle_col:
        sdf["vehicle_id"] = df[vehicle_col].astype(str).str.strip()
    elif source_filename:
        sdf["vehicle_id"] = extract_vehicle_id_from_filename(source_filename)
    else:
        sdf["vehicle_id"] = "UNKNOWN"
    
    # 3. Coordinates (try combined column first, then separate)
    if coord_col:
        coords = df[coord_col].apply(parse_coordinates)
        sdf['latitude'] = coords.apply(lambda x: x[0])
        sdf['longitude'] = coords.apply(lambda x: x[1])
    elif lat_col and lon_col:
        sdf['latitude'] = pd.to_numeric(df[lat_col], errors="coerce")
        sdf['longitude'] = pd.to_numeric(df[lon_col], errors="coerce")
    else:
        sdf['latitude'] = np.nan
        sdf['longitude'] = np.nan

    # Maintain shorthand columns expected by downstream code/tests
    if 'latitude' in sdf.columns:
        sdf['lat'] = sdf['latitude']
    if 'longitude' in sdf.columns:
        sdf['lon'] = sdf['longitude']
    
    # 4. Speed (with unit parsing)
    if speed_col:
        sdf["speed_kmh"] = df[speed_col].apply(parse_speed)
    else:
        sdf["speed_kmh"] = np.nan
    
    # 5. Ignition (flexible boolean parsing)
    if ignition_col:
        sdf["ignition"] = df[ignition_col].apply(parse_ignition)
    else:
        sdf["ignition"] = False
    
    # 6. Fuel (total or combined tanks)
    if total_fuel_col:
        sdf["total_fuel_gal"] = pd.to_numeric(df[total_fuel_col], errors="coerce")
    elif left_tank_col or right_tank_col:
        left = df[left_tank_col] if left_tank_col else pd.Series([0] * len(df))
        right = df[right_tank_col] if right_tank_col else pd.Series([0] * len(df))
        sdf["total_fuel_gal"] = combine_fuel_tanks(left, right)
        logger.info("Combined left and right fuel tanks")
    else:
        sdf["total_fuel_gal"] = np.nan
    
    # 7. Label (if exists)
    if label_col:
        sdf["stationary_drain"] = parse_label_column(df[label_col])
        logger.info("Found and parsed label column")
    
    logger.info(f"Standardized to columns: {list(sdf.columns)}")
    
    return sdf


def validate_required_columns(df: pd.DataFrame, required: Optional[List[str]] = None) -> None:
    """
    Validate that required columns are present.
    
    Args:
        df: DataFrame to validate
        required: List of required column names
    
    Raises:
        ValueError: If any required columns are missing
    """
    if required is None:
        required = ["vehicle_id", "timestamp", "total_fuel_gal"]
    
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    logger.info("✓ All required columns present")


def clean_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and deduplicate loaded data.
    
    Args:
        df: DataFrame to clean
    
    Returns:
        Cleaned DataFrame
    """
    initial_len = len(df)
    
    # Remove rows with invalid timestamps
    df = df.dropna(subset=["timestamp"])
    
    # Sort by timestamp
    df = df.sort_values("timestamp")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=["vehicle_id", "timestamp"])
    
    removed = initial_len - len(df)
    if removed > 0:
        logger.info(f"Removed {removed:,} invalid/duplicate rows")
    
    return df.reset_index(drop=True)


def load_and_standardize(
    path: Path,
    column_mapping: Optional[Dict[str, List[str]]] = None,
    vehicle_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Load CSV and apply standardization pipeline.
    
    Args:
        path: Path to CSV file
        column_mapping: Column name mapping dictionary
        vehicle_id: Optional vehicle ID override
    
    Returns:
        Standardized DataFrame
    """
    # Load raw data
    df = load_raw_csv(path)
    
    # Standardize columns and parse data
    df = standardize_columns(
        df,
        column_mapping=column_mapping,
        vehicle_id=vehicle_id,
        source_filename=path.name
    )
    
    # Validate required columns
    validate_required_columns(df)
    
    # Clean and deduplicate
    df = clean_and_deduplicate(df)
    
    logger.info(f"✓ Successfully loaded and standardized {len(df):,} rows")
    
    return df


def load_multiple_sources(
    paths: List[Path],
    column_mapping: Optional[Dict[str, List[str]]] = None
) -> List[pd.DataFrame]:
    """
    Load multiple CSV files with standardization.
    
    Args:
        paths: List of paths to CSV files
        column_mapping: Column name mapping dictionary
    
    Returns:
        List of standardized DataFrames
    """
    dataframes = []
    
    for path in paths:
        try:
            df = load_and_standardize(path, column_mapping)
            dataframes.append(df)
            logger.info(f"✓ Loaded {path.name}: {len(df):,} rows")
        
        except Exception as e:
            logger.warning(f"⚠ Failed to load {path.name}: {e}")
            continue
    
    if not dataframes:
        raise ValueError("No data files loaded successfully")
    
    logger.info(f"✓ Loaded {len(dataframes)} data sources")
    
    return dataframes