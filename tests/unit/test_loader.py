"""
Unit tests for src/data/loader.py
Tests enhanced loading with Spanish names, coordinates, speed with units, etc.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.data.loader import (
    load_raw_csv,
    standardize_columns,
    validate_required_columns,
    clean_and_deduplicate,
    load_and_standardize,
    load_multiple_sources
)


class TestLoadRawCSV:
    """Tests for load_raw_csv function"""
    
    def test_load_valid_csv(self, tmp_path):
        """Test loading a valid CSV file"""
        test_file = tmp_path / "test.csv"
        test_data = pd.DataFrame({
            'Vehicle_ID': ['V1', 'V2'],
            'Timestamp': ['2025-01-15 10:00:00', '2025-01-15 10:05:00'],
            'Fuel': [50.0, 49.5]
        })
        test_data.to_csv(test_file, index=False)
        
        df = load_raw_csv(test_file)
        
        assert len(df) == 2
        assert list(df.columns) == ['Vehicle_ID', 'Timestamp', 'Fuel']
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error"""
        with pytest.raises(FileNotFoundError):
            load_raw_csv(Path("nonexistent.csv"))
    
    def test_load_utf8_encoding(self, tmp_path):
        """Test loading UTF-8 encoded file"""
        test_file = tmp_path / "utf8.csv"
        test_data = pd.DataFrame({
            'Vehículo': ['V1', 'V2'],
            'Tiempo': ['2025-01-15 10:00:00', '2025-01-15 10:05:00']
        })
        test_data.to_csv(test_file, index=False, encoding='utf-8')
        
        df = load_raw_csv(test_file)
        
        assert len(df) == 2
        assert 'Vehículo' in df.columns


class TestStandardizeColumns:
    """Tests for standardize_columns function with enhanced parsing"""
    
    def test_standardize_english_columns(self):
        """Test standardization of English column names"""
        df = pd.DataFrame({
            'Vehicle_ID': ['V1'],
            'TIMESTAMP': ['2025-01-15 10:00:00'],
            'Latitude': [40.7128],
            'Longitude': [-74.0060],
            'Speed': [50.0],
            'Ignition': [1],
            'TotalFuel': [50.0]
        })
        
        result = standardize_columns(df, vehicle_id='V1')
        
        # Check standardized columns exist
        assert 'vehicle_id' in result.columns
        assert 'timestamp' in result.columns
        assert 'latitude' in result.columns
        assert 'longitude' in result.columns
        assert 'speed_kmh' in result.columns
        assert 'ignition' in result.columns
        assert 'total_fuel_gal' in result.columns
    
    def test_standardize_spanish_columns(self):
        """Test standardization of Spanish column names"""
        df = pd.DataFrame({
            'Vehiculo': ['V1'],
            'Tiempo': ['2025-01-15 10:00:00'],
            'Velocidad': ['50 km/h'],
            'Ignición': ['encendido'],
            'Tanque Total': [100.0]
        })
        
        result = standardize_columns(df, vehicle_id='V1')
        
        assert result['vehicle_id'].iloc[0] == 'V1'
        assert 'timestamp' in result.columns
        assert result['speed_kmh'].iloc[0] == 50.0  # Parsed from "50 km/h"
        assert result['ignition'].iloc[0] == True    # Parsed from "encendido"
        assert result['total_fuel_gal'].iloc[0] == 100.0
    
    def test_parse_coordinate_string(self):
        """Test parsing coordinate string column"""
        df = pd.DataFrame({
            'Vehiculo': ['V1', 'V2'],
            'Tiempo': ['2025-01-15 10:00:00', '2025-01-15 10:05:00'],
            'Coordenadas': ['40.7128, -74.0060', '34.0522, -118.2437'],
            'Speed': [50, 60],
            'Ignition': [1, 1],
            'Fuel': [100, 99]
        })
        
        result = standardize_columns(df, vehicle_id='V1')
        
        # Check coordinates were parsed correctly
        assert result['latitude'].iloc[0] == 40.7128
        assert result['longitude'].iloc[0] == -74.0060
        assert result['latitude'].iloc[1] == 34.0522
        assert result['longitude'].iloc[1] == -118.2437
    
    def test_parse_speed_with_units(self):
        """Test parsing speed values with units"""
        df = pd.DataFrame({
            'Vehicle_ID': ['V1', 'V2', 'V3'],
            'Timestamp': ['2025-01-15 10:00:00'] * 3,
            'Velocidad': ['50 km/h', '60.5 km/h', '45'],
            'Ignition': [1, 1, 1],
            'Fuel': [100, 99, 98]
        })
        
        result = standardize_columns(df, vehicle_id='V1')
        
        # Check speed values were parsed correctly
        assert result['speed_kmh'].iloc[0] == 50.0
        assert result['speed_kmh'].iloc[1] == 60.5
        assert result['speed_kmh'].iloc[2] == 45.0
    
    def test_parse_ignition_spanish(self):
        """Test parsing Spanish ignition values"""
        df = pd.DataFrame({
            'Vehicle_ID': ['V1', 'V2', 'V3', 'V4'],
            'Timestamp': ['2025-01-15 10:00:00'] * 4,
            'Ignición': ['encendido', 'apagado', 'si', 'no'],
            'Speed': [50, 0, 50, 0],
            'Fuel': [100, 99, 98, 97]
        })
        
        result = standardize_columns(df, vehicle_id='V1')
        
        # Check ignition values were parsed correctly
        assert result['ignition'].iloc[0] == True   # "encendido"
        assert result['ignition'].iloc[1] == False  # "apagado"
        assert result['ignition'].iloc[2] == True   # "si"
        assert result['ignition'].iloc[3] == False  # "no"
    
    def test_combine_separate_fuel_tanks(self):
        """Test combining left and right fuel tanks"""
        df = pd.DataFrame({
            'Vehicle_ID': ['V1', 'V2'],
            'Timestamp': ['2025-01-15 10:00:00', '2025-01-15 10:05:00'],
            'Tanque Izquierdo': [50.0, 49.5],
            'Tanque Derecho': [48.0, 47.5],
            'Speed': [0, 0],
            'Ignition': [1, 1]
        })
        
        result = standardize_columns(df, vehicle_id='V1')
        
        # Check fuel tanks were combined
        assert result['total_fuel_gal'].iloc[0] == 98.0  # 50 + 48
        assert result['total_fuel_gal'].iloc[1] == 97.0  # 49.5 + 47.5
    
    def test_parse_multiple_timestamp_formats(self):
        """Test parsing various timestamp formats"""
        df = pd.DataFrame({
            'Vehicle_ID': ['V1', 'V2', 'V3'],
            'Tiempo': [
                '2025-01-15 10:00:00',      # ISO format
                '15.01.2025 10:00:00',      # Day-first dotted
                '15/01/2025 10:00:00'       # Day-first slashed
            ],
            'Speed': [50, 60, 55],
            'Ignition': [1, 1, 1],
            'Fuel': [100, 99, 98]
        })
        
        result = standardize_columns(df, vehicle_id='V1')
        
        # All timestamps should be parsed (not NaT)
        assert result['timestamp'].notna().all()
        assert pd.api.types.is_datetime64_any_dtype(result['timestamp'])
    
    def test_extract_vehicle_id_from_filename(self):
        """Test vehicle ID extraction from filename"""
        df = pd.DataFrame({
            'Timestamp': ['2025-01-15 10:00:00'],
            'Speed': [50],
            'Ignition': [1],
            'Fuel': [100]
        })
        
        result = standardize_columns(df, source_filename='GQU478_telemetry.csv')
        
        assert result['vehicle_id'].iloc[0] == 'GQU478'
    
    def test_parse_label_column(self):
        """Test parsing various label formats"""
        df = pd.DataFrame({
            'Vehicle_ID': ['V1', 'V2', 'V3', 'V4'],
            'Timestamp': ['2025-01-15 10:00:00'] * 4,
            'Speed': [0] * 4,
            'Ignition': [1] * 4,
            'Fuel': [100, 99, 98, 97],
            'Stationary_drain': [True, 'yes', 1, 'si']
        })
        
        result = standardize_columns(df, vehicle_id='V1')
        
        # Check label was parsed to binary
        assert 'stationary_drain' in result.columns
        assert result['stationary_drain'].dtype == int
        assert result['stationary_drain'].tolist() == [1, 1, 1, 1]
    
    def test_missing_timestamp_raises_error(self):
        """Test that missing timestamp column raises error"""
        df = pd.DataFrame({
            'Vehicle_ID': ['V1'],
            'Speed': [50],
            'Fuel': [100]
        })
        
        with pytest.raises(ValueError, match="No timestamp column found"):
            standardize_columns(df)


class TestValidateRequiredColumns:
    """Tests for validate_required_columns function"""
    
    def test_all_required_columns_present(self):
        """Test validation passes with all required columns"""
        df = pd.DataFrame({
            'vehicle_id': ['V1'],
            'timestamp': ['2025-01-15 10:00:00'],
            'total_fuel_gal': [100.0]
        })
        
        # Should not raise
        validate_required_columns(df)
    
    def test_missing_required_columns_raises_error(self):
        """Test validation fails with missing columns"""
        df = pd.DataFrame({
            'vehicle_id': ['V1'],
            'timestamp': ['2025-01-15 10:00:00']
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_required_columns(df)
    
    def test_custom_required_columns(self):
        """Test validation with custom required columns"""
        df = pd.DataFrame({
            'vehicle_id': ['V1'],
            'timestamp': ['2025-01-15 10:00:00']
        })
        
        # Should not raise with custom requirements
        validate_required_columns(df, required=['vehicle_id', 'timestamp'])


class TestCleanAndDeduplicate:
    """Tests for clean_and_deduplicate function"""
    
    def test_remove_invalid_timestamps(self):
        """Test removal of rows with invalid timestamps"""
        df = pd.DataFrame({
            'vehicle_id': ['V1', 'V2', 'V3'],
            'timestamp': pd.to_datetime([
                '2025-01-15 10:00:00',
                pd.NaT,
                '2025-01-15 11:00:00'
            ], utc=True),
            'fuel': [100, 99, 98]
        })
        
        result = clean_and_deduplicate(df)
        
        assert len(result) == 2
        assert result['timestamp'].notna().all()
    
    def test_remove_duplicates(self):
        """Test removal of duplicate rows"""
        df = pd.DataFrame({
            'vehicle_id': ['V1', 'V1', 'V2'],
            'timestamp': pd.to_datetime([
                '2025-01-15 10:00:00',
                '2025-01-15 10:00:00',  # Duplicate
                '2025-01-15 10:00:00'
            ], utc=True),
            'fuel': [100, 100, 99]
        })
        
        result = clean_and_deduplicate(df)
        
        assert len(result) == 2
    
    def test_sort_by_timestamp(self):
        """Test that data is sorted by timestamp"""
        df = pd.DataFrame({
            'vehicle_id': ['V1'] * 3,
            'timestamp': pd.to_datetime([
                '2025-01-15 11:00:00',
                '2025-01-15 09:00:00',
                '2025-01-15 10:00:00'
            ], utc=True),
            'fuel': [98, 100, 99]
        })
        
        result = clean_and_deduplicate(df)
        
        assert result['timestamp'].is_monotonic_increasing


class TestLoadAndStandardize:
    """Integration tests for load_and_standardize"""
    
    def test_load_english_csv(self, tmp_path):
        """Test loading English format CSV"""
        test_file = tmp_path / "english.csv"
        test_data = pd.DataFrame({
            'Vehicle_ID': ['V1', 'V2'],
            'Timestamp': ['2025-01-15 10:00:00', '2025-01-15 10:05:00'],
            'Latitude': [40.7128, 34.0522],
            'Longitude': [-74.0060, -118.2437],
            'Speed': [50.0, 60.0],
            'Ignition': [1, 0],
            'TotalFuel': [100.0, 99.5]
        })
        test_data.to_csv(test_file, index=False)
        
        df = load_and_standardize(test_file)
        
        # Check all standard columns
        expected_cols = ['vehicle_id', 'timestamp', 'latitude', 'longitude', 
                        'speed_kmh', 'ignition', 'total_fuel_gal']
        for col in expected_cols:
            assert col in df.columns
        
        assert len(df) == 2
    
    def test_load_spanish_csv(self, temp_spanish_csv_file):
        """Test loading Spanish format CSV"""
        df = load_and_standardize(temp_spanish_csv_file)
        
        # Check standardized columns
        assert 'vehicle_id' in df.columns
        assert 'timestamp' in df.columns
        assert 'latitude' in df.columns
        assert 'longitude' in df.columns
        assert 'speed_kmh' in df.columns
        assert 'ignition' in df.columns
        assert 'total_fuel_gal' in df.columns
        
        # Check parsed values
        assert df['speed_kmh'].iloc[0] == 50.0  # From "50 km/h"
        assert df['ignition'].iloc[0] == True    # From "encendido"
    
    def test_load_separate_tanks_csv(self, temp_separate_tanks_csv):
        """Test loading CSV with separate fuel tanks"""
        df = load_and_standardize(temp_separate_tanks_csv)
        
        # Check tanks were combined
        assert 'total_fuel_gal' in df.columns
        assert df['total_fuel_gal'].iloc[0] == 98.0  # 50 + 48
    
    def test_vehicle_id_from_filename(self, tmp_path):
        """Test vehicle ID extraction from filename"""
        test_file = tmp_path / "GQU478_data.csv"
        test_data = pd.DataFrame({
            'Timestamp': ['2025-01-15 10:00:00'],
            'Speed': [50],
            'Ignition': [1],
            'TotalFuel': [100]
        })
        test_data.to_csv(test_file, index=False)
        
        df = load_and_standardize(test_file)
        
        assert df['vehicle_id'].iloc[0] == 'GQU478'


class TestLoadMultipleSources:
    """Tests for load_multiple_sources function"""
    
    def test_load_multiple_files(self, tmp_path):
        """Test loading multiple CSV files"""
        # Create multiple test files
        files = []
        for i in range(3):
            test_file = tmp_path / f"vehicle_{i}.csv"
            test_data = pd.DataFrame({
                'Vehicle_ID': [f'V{i}'],
                'Timestamp': ['2025-01-15 10:00:00'],
                'Speed': [50],
                'Ignition': [1],
                'TotalFuel': [100]
            })
            test_data.to_csv(test_file, index=False)
            files.append(test_file)
        
        dataframes = load_multiple_sources(files)
        
        assert len(dataframes) == 3
        for df in dataframes:
            assert 'vehicle_id' in df.columns
            assert 'timestamp' in df.columns
    
    def test_skip_failed_files(self, tmp_path):
        """Test that failed files are skipped"""
        # Create valid file
        valid_file = tmp_path / "valid.csv"
        pd.DataFrame({
            'Timestamp': ['2025-01-15 10:00:00'],
            'Speed': [50],
            'Fuel': [100]
        }).to_csv(valid_file, index=False)
        
        # Create invalid file (no timestamp)
        invalid_file = tmp_path / "invalid.csv"
        pd.DataFrame({
            'Speed': [50],
            'Fuel': [100]
        }).to_csv(invalid_file, index=False)
        
        # Should load valid file and skip invalid
        dataframes = load_multiple_sources([valid_file, invalid_file])
        
        assert len(dataframes) == 1
    
    def test_no_valid_files_raises_error(self, tmp_path):
        """Test that error is raised if no files load successfully"""
        # Create file with no timestamp
        test_file = tmp_path / "invalid.csv"
        pd.DataFrame({
            'Speed': [50],
            'Fuel': [100]
        }).to_csv(test_file, index=False)
        
        with pytest.raises(ValueError, match="No data files loaded successfully"):
            load_multiple_sources([test_file])


class TestRealWorldScenarios:
    """Tests for real-world data scenarios"""
    
    def test_mixed_format_csv(self, tmp_path):
        """Test CSV with mixed column formats"""
        test_file = tmp_path / "mixed.csv"
        test_data = pd.DataFrame({
            'Vehiculo': ['V1', 'V2', 'V3'],
            'Tiempo': ['2025-01-15 10:00:00', '15.01.2025 11:00:00', '2025-01-15 12:00:00'],
            'Coordenadas': ['40.7128, -74.0060', '34.0522,-118.2437', 'invalid'],
            'Velocidad': ['50 km/h', '60.5', 'invalid'],
            'Ignición': ['encendido', 'on', '1'],
            'Tanque Total': [100, 99, 98]
        })
        test_data.to_csv(test_file, index=False)
        
        df = load_and_standardize(test_file)
        
        # Should handle all formats gracefully
        assert len(df) == 3
        assert df['speed_kmh'].notna().sum() >= 2  # At least 2 valid speeds
        assert df['ignition'].all()  # All should be True
    
    def test_large_dataset(self, tmp_path):
        """Test loading larger dataset"""
        test_file = tmp_path / "large.csv"
        n_records = 10000
        test_data = pd.DataFrame({
            'Vehicle_ID': ['V1'] * n_records,
            'Timestamp': pd.date_range('2025-01-15', periods=n_records, freq='1min'),
            'Speed': np.random.uniform(0, 100, n_records),
            'Ignition': np.random.choice([0, 1], n_records),
            'TotalFuel': 100 - np.cumsum(np.random.uniform(0, 0.05, n_records))
        })
        test_data.to_csv(test_file, index=False)
        
        df = load_and_standardize(test_file)
        
        assert len(df) == n_records
        assert df['vehicle_id'].nunique() == 1
    
    def test_csv_with_missing_values(self, tmp_path):
        """Test CSV with missing values in various columns"""
        test_file = tmp_path / "missing.csv"
        test_data = pd.DataFrame({
            'Vehicle_ID': ['V1', 'V2', 'V3'],
            'Timestamp': ['2025-01-15 10:00:00', '2025-01-15 11:00:00', '2025-01-15 12:00:00'],
            'Speed': [50.0, np.nan, 60.0],
            'Ignition': [1, 0, np.nan],
            'TotalFuel': [100, 99, 98],
            'Coordenadas': ['40.7128, -74.0060', '', '34.0522, -118.2437']
        })
        test_data.to_csv(test_file, index=False)
        
        df = load_and_standardize(test_file)
        
        # Should handle missing values gracefully
        assert len(df) == 3
        assert pd.isna(df['speed_kmh'].iloc[1])
        assert pd.isna(df['latitude'].iloc[1])