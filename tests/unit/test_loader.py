"""
Unit tests for src/data/loader.py
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile

from src.data.loader import (
    load_raw_csv,
    standardize_columns,
    validate_required_columns,
    convert_data_types,
    load_and_standardize
)


class TestLoadRawCSV:
    """Tests for load_raw_csv function"""
    
    def test_load_valid_csv(self, tmp_path):
        """Test loading a valid CSV file"""
        # Create test CSV
        test_file = tmp_path / "test.csv"
        test_data = pd.DataFrame({
            'Vehicle_ID': ['V1', 'V2'],
            'Timestamp': ['2025-01-15 10:00:00', '2025-01-15 10:05:00'],
            'Fuel': [50.0, 49.5]
        })
        test_data.to_csv(test_file, index=False)
        
        # Load
        df = load_raw_csv(test_file)
        
        assert len(df) == 2
        assert list(df.columns) == ['Vehicle_ID', 'Timestamp', 'Fuel']
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error"""
        with pytest.raises(FileNotFoundError):
            load_raw_csv(Path("nonexistent.csv"))


class TestStandardizeColumns:
    """Tests for standardize_columns function"""
    
    def test_standardize_column_names(self):
        """Test column name standardization"""
        df = pd.DataFrame({
            'Vehicle_ID': ['V1'],
            'TIMESTAMP': ['2025-01-15 10:00:00'],
            'Lat': [40.7128],
            'Lon': [-74.0060],
            'Speed': [50.0],
            'IGN': [1],
            'TotalFuel': [50.0]
        })
        
        result = standardize_columns(df)
        
        # Check lowercase and stripped
        assert 'vehicle_id' in result.columns
        assert 'timestamp' in result.columns
        assert 'lat' in result.columns
    
    def test_resolve_column_mapping(self):
        """Test mapping of various column name conventions"""
        df = pd.DataFrame({
            'vehiculo': ['V1'],
            'fecha_hora': ['2025-01-15 10:00:00'],
            'latitude': [40.7128],
            'longitud': [-74.0060],
            'velocidad': [50.0],
            'switch_ignition': [1],
            'fuel_total_gal': [50.0]
        })
        
        column_mapping = {
            "vehicle_id": ["vehiculo", "vehicle_id"],
            "timestamp": ["fecha_hora", "timestamp"],
            "latitude": ["latitude", "lat"],
            "longitude": ["longitud", "longitude"],
            "speed_kmh": ["velocidad", "speed_kmh"],
            "ignition": ["switch_ignition", "ignition"],
            "total_fuel_gal": ["fuel_total_gal", "total_fuel_gal"]
        }
        
        result = standardize_columns(df, column_mapping)
        
        assert 'vehicle_id' in result.columns
        assert 'timestamp' in result.columns
        assert 'longitude' in result.columns


class TestValidateRequiredColumns:
    """Tests for validate_required_columns function"""
    
    def test_all_columns_present(self):
        """Test validation passes with all required columns"""
        df = pd.DataFrame({
            'vehicle_id': ['V1'],
            'timestamp': ['2025-01-15 10:00:00'],
            'latitude': [40.7128],
            'longitude': [-74.0060],
            'speed_kmh': [50.0],
            'ignition': [1],
            'total_fuel_gal': [50.0]
        })
        
        # Should not raise
        validate_required_columns(df)
    
    def test_missing_columns_raises_error(self):
        """Test validation fails with missing columns"""
        df = pd.DataFrame({
            'vehicle_id': ['V1'],
            'timestamp': ['2025-01-15 10:00:00']
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_required_columns(df)


class TestConvertDataTypes:
    """Tests for convert_data_types function"""
    
    def test_convert_timestamp(self):
        """Test timestamp conversion"""
        df = pd.DataFrame({
            'timestamp': ['2025-01-15 10:00:00', '2025-01-15 11:00:00']
        })
        
        result = convert_data_types(df)
        
        assert pd.api.types.is_datetime64_any_dtype(result['timestamp'])
    
    def test_convert_ignition_to_boolean(self):
        """Test ignition conversion to boolean"""
        df = pd.DataFrame({
            'ignition': [0, 1, 1, 0]
        })
        
        result = convert_data_types(df)
        
        assert result['ignition'].dtype == bool
        assert result['ignition'].tolist() == [False, True, True, False]
    
    def test_convert_numeric_columns(self):
        """Test numeric column conversions"""
        df = pd.DataFrame({
            'speed_kmh': ['50.5', '60.0', 'invalid'],
            'total_fuel_gal': ['100', '99.5', '98.0'],
            'latitude': ['40.7128', '34.0522'],
            'longitude': ['-74.0060', '-118.2437']
        })
        
        result = convert_data_types(df)
        
        assert pd.api.types.is_numeric_dtype(result['speed_kmh'])
        assert pd.api.types.is_numeric_dtype(result['total_fuel_gal'])
        assert pd.api.types.is_numeric_dtype(result['latitude'])
        assert pd.api.types.is_numeric_dtype(result['longitude'])
        
        # Check invalid value becomes NaN
        assert pd.isna(result['speed_kmh'].iloc[2])


class TestLoadAndStandardize:
    """Integration tests for load_and_standardize"""
    
    def test_complete_pipeline(self, tmp_path):
        """Test complete loading and standardization pipeline"""
        # Create test CSV with various column names
        test_file = tmp_path / "test.csv"
        test_data = pd.DataFrame({
            'Vehiculo': ['V1', 'V2'],
            'Fecha_Hora': ['2025-01-15 10:00:00', '2025-01-15 10:05:00'],
            'Lat': [40.7128, 34.0522],
            'Lon': [-74.0060, -118.2437],
            'Velocidad': [50.0, 60.0],
            'IGN': [1, 0],
            'TotalFuel': [100.0, 99.5]
        })
        test_data.to_csv(test_file, index=False)
        
        # Load and standardize
        df = load_and_standardize(test_file)
        
        # Check standardized columns
        expected_cols = ['vehicle_id', 'timestamp', 'latitude', 'longitude', 
                        'speed_kmh', 'ignition', 'total_fuel_gal']
        for col in expected_cols:
            assert col in df.columns
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])
        assert df['ignition'].dtype == bool
        assert pd.api.types.is_numeric_dtype(df['speed_kmh'])