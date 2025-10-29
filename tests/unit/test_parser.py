"""
Unit tests for src/utils/parser.py
Tests the enhanced parsing functions for various data formats.
"""

import pytest
import pandas as pd
import numpy as np

from src.utils.parser import (
    parse_coordinates,
    parse_speed,
    parse_ignition,
    parse_timestamp_column,
    parse_label_column,
    pick_column,
    extract_vehicle_id_from_filename,
    combine_fuel_tanks
)


class TestParseCoordinates:
    """Tests for parse_coordinates function"""
    
    def test_valid_coordinates(self):
        """Test parsing valid coordinate strings"""
        lat, lon = parse_coordinates("40.7128, -74.0060")
        assert lat == 40.7128
        assert lon == -74.0060
    
    def test_coordinates_no_spaces(self):
        """Test parsing coordinates without spaces"""
        lat, lon = parse_coordinates("40.7128,-74.0060")
        assert lat == 40.7128
        assert lon == -74.0060
    
    def test_invalid_coordinates(self):
        """Test parsing invalid coordinate strings"""
        lat, lon = parse_coordinates("invalid")
        assert np.isnan(lat)
        assert np.isnan(lon)
    
    def test_empty_coordinates(self):
        """Test parsing empty coordinate strings"""
        lat, lon = parse_coordinates("")
        assert np.isnan(lat)
        assert np.isnan(lon)
    
    def test_nan_coordinates(self):
        """Test parsing NaN coordinates"""
        lat, lon = parse_coordinates(np.nan)
        assert np.isnan(lat)
        assert np.isnan(lon)


class TestParseSpeed:
    """Tests for parse_speed function"""
    
    def test_numeric_speed(self):
        """Test parsing numeric speed values"""
        assert parse_speed(50.0) == 50.0
        assert parse_speed(60) == 60.0
    
    def test_speed_with_units(self):
        """Test parsing speed with km/h units"""
        assert parse_speed("50 km/h") == 50.0
        assert parse_speed("60.5 km/h") == 60.5
    
    def test_speed_string_numeric(self):
        """Test parsing speed as string"""
        assert parse_speed("45.5") == 45.5
    
    def test_invalid_speed(self):
        """Test parsing invalid speed values"""
        assert np.isnan(parse_speed("invalid"))
        assert np.isnan(parse_speed(""))
    
    def test_nan_speed(self):
        """Test parsing NaN speed"""
        assert np.isnan(parse_speed(np.nan))


class TestParseIgnition:
    """Tests for parse_ignition function"""
    
    def test_boolean_ignition(self):
        """Test parsing boolean ignition values"""
        assert parse_ignition(True) == True
        assert parse_ignition(False) == False
    
    def test_numeric_ignition(self):
        """Test parsing numeric ignition values"""
        assert parse_ignition(1) == True
        assert parse_ignition(0) == False
        assert parse_ignition(1.0) == True
        assert parse_ignition(0.0) == False
    
    def test_string_ignition_english(self):
        """Test parsing English string ignition values"""
        assert parse_ignition("true") == True
        assert parse_ignition("false") == False
        assert parse_ignition("on") == True
        assert parse_ignition("off") == False
        assert parse_ignition("yes") == True
        assert parse_ignition("no") == False
    
    def test_string_ignition_spanish(self):
        """Test parsing Spanish string ignition values"""
        assert parse_ignition("encendido") == True
        assert parse_ignition("apagado") == False
        assert parse_ignition("si") == True
        assert parse_ignition("sí") == True
        assert parse_ignition("no") == False
    
    def test_invalid_ignition(self):
        """Test parsing invalid ignition values"""
        assert parse_ignition("invalid") == False
        assert parse_ignition("") == False
    
    def test_nan_ignition(self):
        """Test parsing NaN ignition"""
        assert parse_ignition(np.nan) == False


class TestParseTimestampColumn:
    """Tests for parse_timestamp_column function"""
    
    def test_iso_format(self):
        """Test parsing ISO datetime format"""
        series = pd.Series(["2025-01-15 10:30:00", "2025-01-15 11:00:00"])
        result = parse_timestamp_column(series)
        
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert not result.isna().any()
    
    def test_dotted_format(self):
        """Test parsing day-first dotted format"""
        series = pd.Series(["15.01.2025 10:30:00", "15.01.2025 11:00:00"])
        result = parse_timestamp_column(series)
        
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert not result.isna().any()
    
    def test_mixed_formats(self):
        """Test parsing mixed datetime formats"""
        series = pd.Series(["2025-01-15 10:30:00", "15.01.2025 11:00:00"])
        result = parse_timestamp_column(series)
        
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert not result.isna().any()
    
    def test_invalid_timestamps(self):
        """Test parsing invalid timestamps"""
        series = pd.Series(["invalid", "2025-01-15 10:30:00"])
        result = parse_timestamp_column(series)
        
        assert result.isna().iloc[0]
        assert not result.isna().iloc[1]


class TestParseLabelColumn:
    """Tests for parse_label_column function"""
    
    def test_boolean_labels(self):
        """Test parsing boolean labels"""
        series = pd.Series([True, False, True])
        result = parse_label_column(series)
        
        assert result.tolist() == [1, 0, 1]
        assert result.dtype == int
    
    def test_numeric_labels(self):
        """Test parsing numeric labels"""
        series = pd.Series([1, 0, 1, 0])
        result = parse_label_column(series)
        
        assert result.tolist() == [1, 0, 1, 0]
    
    def test_string_labels(self):
        """Test parsing string labels"""
        series = pd.Series(["yes", "no", "true", "false", "si", "sí"])
        result = parse_label_column(series)
        
        assert result.tolist() == [1, 0, 1, 0, 1, 1]


class TestPickColumn:
    """Tests for pick_column function"""
    
    def test_exact_match(self):
        """Test exact column name matching"""
        columns = ["Vehicle_ID", "Timestamp", "Speed"]
        result = pick_column(columns, ["timestamp", "time"])
        
        assert result == "Timestamp"
    
    def test_partial_match(self):
        """Test partial column name matching"""
        columns = ["Velocidad (km/h)", "Tiempo", "Fuel"]
        result = pick_column(columns, ["velocidad", "speed"])
        
        assert result == "Velocidad (km/h)"
    
    def test_no_match(self):
        """Test when no column matches"""
        columns = ["Vehicle_ID", "Timestamp"]
        result = pick_column(columns, ["fuel", "tank"])
        
        assert result is None
    
    def test_case_insensitive(self):
        """Test case-insensitive matching"""
        columns = ["VEHICLE_ID", "timestamp", "Speed"]
        result = pick_column(columns, ["vehicle_id", "vehiculo"])
        
        assert result == "VEHICLE_ID"


class TestExtractVehicleIdFromFilename:
    """Tests for extract_vehicle_id_from_filename function"""
    
    def test_simple_filename(self):
        """Test extracting ID from simple filename"""
        assert extract_vehicle_id_from_filename("GQU478_data.csv") == "GQU478"
        assert extract_vehicle_id_from_filename("V1.csv") == "V1"
    
    def test_complex_filename(self):
        """Test extracting ID from complex filename"""
        assert extract_vehicle_id_from_filename("vehicle_123_telemetry.csv") == "vehicle"
    
    def test_filename_with_spaces(self):
        """Test extracting ID from filename with spaces"""
        result = extract_vehicle_id_from_filename("My Vehicle.csv")
        assert "_" in result or result == "My"


class TestCombineFuelTanks:
    """Tests for combine_fuel_tanks function"""
    
    def test_valid_tanks(self):
        """Test combining valid fuel tank readings"""
        left = pd.Series([50.0, 45.0, 40.0])
        right = pd.Series([48.0, 44.0, 39.0])
        result = combine_fuel_tanks(left, right)
        
        assert result.tolist() == [98.0, 89.0, 79.0]
    
    def test_missing_values(self):
        """Test combining tanks with missing values"""
        left = pd.Series([50.0, np.nan, 40.0])
        right = pd.Series([48.0, 44.0, np.nan])
        result = combine_fuel_tanks(left, right)
        
        # Missing values should be treated as 0
        assert result.iloc[0] == 98.0
        assert result.iloc[1] == 44.0
        assert result.iloc[2] == 40.0
    
    def test_non_numeric_tanks(self):
        """Test combining non-numeric tank values"""
        left = pd.Series(["50", "invalid", "40"])
        right = pd.Series(["48", "44", "39"])
        result = combine_fuel_tanks(left, right)
        
        assert result.iloc[0] == 98.0
        assert result.iloc[1] == 44.0  # Invalid becomes 0
        assert result.iloc[2] == 79.0