"""
Raw data loading with column standardization.
Handles various CSV formats and naming conventions.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def load_raw_csv(path: Path, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Load raw CSV file with error handling.
    
    Args:
        path: Path to CSV file
        encoding: File encoding (default: utf-8)
    
    Returns:
        DataFrame with raw data
    
    Raises:
        FileNotFoundError: If file doesn't exist
        pd.errors.ParserError: If CSV is malformed
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    logger.info(f"Loading data from {path}")
    
    try:
        df = pd.read_csv(path, encoding=encoding)
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        return df
    
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV: {e}")
        raise
    
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        raise


def standardize_columns(df: pd.DataFrame, column_mapping: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
    """
    Standardize column names to canonical format.
    
    Args:
        df: DataFrame with potentially varying column names
        column_mapping: Dictionary mapping canonical names to possible variants
    
    Returns:
        DataFrame with standardized column names
    
    Raises:
        ValueError: If required columns are missing after mapping
    """
    df = df.copy()
    
    # Lowercase and strip whitespace from all columns
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Default column mapping if not provided
    if column_mapping is None:
        column_mapping = {
            "vehicle_id": ["vehicle_id", "vehiculo", "id_vehiculo", "vehicle"],
            "timestamp": ["timestamp", "time", "fecha_hora", "datetime", "date_time"],
            "latitude": ["latitude", "lat"],
            "longitude": ["longitude", "lon", "lng", "longitud"],
            "speed_kmh": ["speed_kmh", "speed", "velocidad", "speed(km/h)"],
            "ignition": ["ignition", "ign", "ignition_status", "switch_ignition"],
            "total_fuel_gal": ["total_fuel_gal", "totalfuel", "fuel_total_gal", "total_fuel"]
        }
    
    # Map columns to standardized names
    resolved_mapping = {}
    for canonical_name, variants in column_mapping.items():
        for variant in variants:
            if variant in df.columns:
                resolved_mapping[variant] = canonical_name
                logger.debug(f"Mapped '{variant}' → '{canonical_name}'")
                break
    
    # Rename columns
    df = df.rename(columns=resolved_mapping)
    
    logger.info(f"Standardized column names: {list(df.columns)}")
    
    return df


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
        required = ["vehicle_id", "timestamp", "latitude", "longitude", 
                   "speed_kmh", "ignition", "total_fuel_gal"]
    
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    logger.info("✓ All required columns present")


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to appropriate data types.
    
    Args:
        df: DataFrame with standardized columns
    
    Returns:
        DataFrame with converted types
    """
    df = df.copy()
    
    # Timestamp conversion (will be converted to UTC in preprocessor)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    
    # Ignition as boolean
    df["ignition"] = (pd.to_numeric(df["ignition"], errors="coerce") > 0).astype(bool)
    
    # Numeric conversions
    df["speed_kmh"] = pd.to_numeric(df["speed_kmh"], errors="coerce")
    df["total_fuel_gal"] = pd.to_numeric(df["total_fuel_gal"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    
    logger.info("✓ Data types converted")
    
    return df


def load_and_standardize(
    path: Path,
    column_mapping: Optional[Dict[str, List[str]]] = None,
    required_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load CSV and apply standardization pipeline.
    
    Args:
        path: Path to CSV file
        column_mapping: Column name mapping dictionary
        required_columns: List of required columns
    
    Returns:
        Standardized DataFrame
    """
    # Load raw data
    df = load_raw_csv(path)
    
    # Standardize columns
    df = standardize_columns(df, column_mapping)
    
    # Validate required columns
    validate_required_columns(df, required_columns)
    
    # Convert data types
    df = convert_data_types(df)
    
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