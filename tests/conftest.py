"""
Pytest configuration and shared fixtures.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_vehicle_data():
    """
    Create sample vehicle telemetry data for testing.
    
    Returns:
        DataFrame with realistic vehicle data
    """
    n_records = 100
    
    return pd.DataFrame({
        'vehicle_id': ['V1'] * 50 + ['V2'] * 50,
        'timestamp': pd.date_range('2025-01-15', periods=n_records, freq='5min', tz='UTC'),
        'latitude': np.random.uniform(40.7, 40.8, n_records),
        'longitude': np.random.uniform(-74.1, -74.0, n_records),
        'speed_kmh': np.random.uniform(0, 80, n_records),
        'ignition': np.random.choice([True, False], n_records),
        'total_fuel_gal': 100 - np.cumsum(np.random.uniform(0, 0.5, n_records))
    })


@pytest.fixture
def sample_theft_event():
    """
    Create sample data with a clear theft event.
    
    Returns:
        DataFrame with theft event pattern
    """
    # Normal operation
    normal_data = pd.DataFrame({
        'vehicle_id': ['V1'] * 20,
        'timestamp': pd.date_range('2025-01-15 08:00', periods=20, freq='5min', tz='UTC'),
        'latitude': [40.7128] * 20,
        'longitude': [-74.0060] * 20,
        'speed_kmh': [0.5] * 20,  # Stationary
        'ignition': [True] * 20,
        'total_fuel_gal': [100 - i * 0.1 for i in range(20)]  # Slow normal drain
    })
    
    # Theft event (rapid drain)
    theft_data = pd.DataFrame({
        'vehicle_id': ['V1'] * 5,
        'timestamp': pd.date_range('2025-01-15 09:40', periods=5, freq='2min', tz='UTC'),
        'latitude': [40.7128] * 5,
        'longitude': [-74.0060] * 5,
        'speed_kmh': [0.3] * 5,
        'ignition': [True] * 5,
        'total_fuel_gal': [98.0, 95.0, 92.0, 89.0, 86.0]  # Rapid drain
    })
    
    # After theft
    after_data = pd.DataFrame({
        'vehicle_id': ['V1'] * 10,
        'timestamp': pd.date_range('2025-01-15 09:50', periods=10, freq='5min', tz='UTC'),
        'latitude': [40.7128] * 10,
        'longitude': [-74.0060] * 10,
        'speed_kmh': [0.4] * 10,
        'ignition': [True] * 10,
        'total_fuel_gal': [86.0 - i * 0.1 for i in range(10)]
    })
    
    return pd.concat([normal_data, theft_data, after_data], ignore_index=True)


@pytest.fixture
def temp_csv_file(tmp_path):
    """
    Create a temporary CSV file for testing.
    
    Args:
        tmp_path: Pytest tmp_path fixture
    
    Returns:
        Path to temporary CSV file
    """
    csv_path = tmp_path / "test_data.csv"
    
    test_data = pd.DataFrame({
        'Vehicle_ID': ['V1', 'V2', 'V1'],
        'Timestamp': ['2025-01-15 10:00:00', '2025-01-15 10:00:00', '2025-01-15 10:05:00'],
        'Latitude': [40.7128, 34.0522, 40.7130],
        'Longitude': [-74.0060, -118.2437, -74.0062],
        'Speed': [50.0, 60.0, 55.0],
        'Ignition': [1, 1, 1],
        'TotalFuel': [100.0, 100.0, 99.5]
    })
    
    test_data.to_csv(csv_path, index=False)
    
    return csv_path


@pytest.fixture
def temp_spanish_csv_file(tmp_path):
    """
    Create a temporary CSV file with Spanish column names.
    
    Args:
        tmp_path: Pytest tmp_path fixture
    
    Returns:
        Path to temporary CSV file
    """
    csv_path = tmp_path / "datos_vehiculo.csv"
    
    test_data = pd.DataFrame({
        'Vehiculo': ['V1', 'V2', 'V1'],
        'Tiempo': ['2025-01-15 10:00:00', '2025-01-15 10:00:00', '2025-01-15 10:05:00'],
        'Coordenadas': ['40.7128, -74.0060', '34.0522, -118.2437', '40.7130, -74.0062'],
        'Velocidad': ['50 km/h', '60 km/h', '55.5 km/h'],
        'Ignición': ['encendido', 'encendido', 'apagado'],
        'Tanque Total': [100.0, 100.0, 99.5]
    })
    
    test_data.to_csv(csv_path, index=False)
    
    return csv_path


@pytest.fixture
def temp_separate_tanks_csv(tmp_path):
    """
    Create CSV with separate left/right fuel tanks.
    
    Args:
        tmp_path: Pytest tmp_path fixture
    
    Returns:
        Path to temporary CSV file
    """
    csv_path = tmp_path / "vehicle_tanks.csv"
    
    test_data = pd.DataFrame({
        'Vehicle_ID': ['V1', 'V1', 'V1'],
        'Timestamp': ['2025-01-15 10:00:00', '2025-01-15 10:05:00', '2025-01-15 10:10:00'],
        'Left_Tank': [50.0, 49.5, 49.0],
        'Right_Tank': [48.0, 47.5, 47.0],
        'Speed': [0, 0, 0],
        'Ignition': [1, 1, 1]
    })
    
    test_data.to_csv(csv_path, index=False)
    
    return csv_path


@pytest.fixture
def sample_coordinates():
    """
    Sample GPS coordinates for testing distance calculations.
    
    Returns:
        DataFrame with lat/lon coordinates
    """
    return pd.DataFrame({
        'latitude': [40.7128, 34.0522, 51.5074],  # NYC, LA, London
        'longitude': [-74.0060, -118.2437, -0.1278]
    })


@pytest.fixture
def sample_coordinate_strings():
    """
    Sample coordinate strings for testing parsing.
    
    Returns:
        List of coordinate strings
    """
    return [
        "40.7128, -74.0060",    # NYC
        "34.0522,-118.2437",    # LA (no spaces)
        "invalid",               # Invalid
        "",                      # Empty
        np.nan                   # NaN
    ]


@pytest.fixture
def sample_speed_values():
    """
    Sample speed values in various formats for testing parsing.
    
    Returns:
        List of speed values
    """
    return [
        50.0,           # Numeric float
        60,             # Numeric int
        "45.5",         # String numeric
        "50 km/h",      # With units
        "60.5 km/h",    # With units and decimal
        "invalid",      # Invalid
        np.nan          # NaN
    ]


@pytest.fixture
def sample_ignition_values():
    """
    Sample ignition values in various formats for testing parsing.
    
    Returns:
        Dictionary of ignition values by language/format
    """
    return {
        'boolean': [True, False],
        'numeric': [1, 0, 1.0, 0.0],
        'english': ["true", "false", "on", "off", "yes", "no"],
        'spanish': ["encendido", "apagado", "si", "sí", "no"]
    }


# Configure pytest
def pytest_configure(config):
    """Pytest configuration hook"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "parser: marks tests for parser utilities"
    )


# Ignore specific warnings
@pytest.fixture(autouse=True)
def ignore_warnings():
    """Ignore specific warnings during tests"""
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)