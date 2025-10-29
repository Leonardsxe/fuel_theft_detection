"""
Unit tests for src/data/validator.py
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.validator import (
    check_duplicates,
    validate_coordinates,
    validate_timestamps,
    check_data_gaps,
    check_missing_values,
    generate_quality_report
)


class TestCheckDuplicates:
    """Tests for check_duplicates function"""
    
    def test_no_duplicates(self):
        """Test with no duplicate records"""
        df = pd.DataFrame({
            'vehicle_id': ['V1', 'V1', 'V2'],
            'timestamp': pd.date_range('2025-01-15', periods=3, freq='5min', tz='UTC'),
            'fuel': [100, 99, 98]
        })
        
        result = check_duplicates(df)
        
        assert len(result) == 3
    
    def test_with_duplicates(self):
        """Test duplicate detection and removal"""
        df = pd.DataFrame({
            'vehicle_id': ['V1', 'V1', 'V1', 'V2'],
            'timestamp': pd.to_datetime([
                '2025-01-15 10:00:00',
                '2025-01-15 10:00:00',  # Duplicate
                '2025-01-15 10:05:00',
                '2025-01-15 10:00:00'
            ], utc=True),
            'fuel': [100, 100, 99, 98]
        })
        
        result = check_duplicates(df)
        
        # Should keep first occurrence
        assert len(result) == 3
        assert result['fuel'].tolist() == [100, 99, 98]


class TestValidateCoordinates:
    """Tests for validate_coordinates function"""
    
    def test_valid_coordinates(self):
        """Test with valid GPS coordinates"""
        df = pd.DataFrame({
            'latitude': [40.7128, 34.0522, 51.5074],
            'longitude': [-74.0060, -118.2437, -0.1278]
        })
        
        result = validate_coordinates(df)
        
        # All coordinates should remain valid
        assert result['latitude'].notna().all()
        assert result['longitude'].notna().all()
    
    def test_invalid_latitude(self):
        """Test with invalid latitude values"""
        df = pd.DataFrame({
            'latitude': [40.7128, 91.0, -95.0],  # 91 and -95 are invalid
            'longitude': [-74.0060, -118.2437, -0.1278]
        })
        
        result = validate_coordinates(df)
        
        # Invalid coordinates should be NaN
        assert result['latitude'].notna().sum() == 1
        assert pd.isna(result['latitude'].iloc[1])
        assert pd.isna(result['latitude'].iloc[2])
    
    def test_invalid_longitude(self):
        """Test with invalid longitude values"""
        df = pd.DataFrame({
            'latitude': [40.7128, 34.0522],
            'longitude': [-74.0060, 185.0]  # 185 is invalid
        })
        
        result = validate_coordinates(df)
        
        assert pd.isna(result['longitude'].iloc[1])
    
    def test_missing_coordinate_columns(self):
        """Test with missing coordinate columns"""
        df = pd.DataFrame({
            'vehicle_id': ['V1', 'V2']
        })
        
        # Should return unchanged (with warning logged)
        result = validate_coordinates(df)
        
        assert len(result) == 2


class TestValidateTimestamps:
    """Tests for validate_timestamps function"""
    
    def test_valid_timestamps(self):
        """Test with valid timestamps"""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-15', periods=3, freq='1h', tz='UTC')
        })
        
        result = validate_timestamps(df)
        
        assert len(result) == 3
    
    def test_invalid_timestamps(self):
        """Test with invalid timestamps"""
        df = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2025-01-15 10:00:00',
                'invalid',
                '2025-01-15 11:00:00'
            ], errors='coerce', utc=True)
        })
        
        result = validate_timestamps(df)
        
        # Invalid timestamp should be removed
        assert len(result) == 2
    
    def test_future_timestamps(self):
        """Test removal of future timestamps"""
        df = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2025-01-15 10:00:00',
                '2030-01-15 10:00:00'  # Future date
            ], utc=True)
        })
        
        result = validate_timestamps(df)
        
        # Future timestamp should be removed
        assert len(result) == 1


class TestCheckDataGaps:
    """Tests for check_data_gaps function"""
    
    def test_no_gaps(self):
        """Test with regular time intervals"""
        df = pd.DataFrame({
            'vehicle_id': ['V1'] * 5,
            'timestamp': pd.date_range('2025-01-15', periods=5, freq='5min', tz='UTC')
        })
        
        result = check_data_gaps(df, max_gap_hours=1.0, verbose=False)
        
        assert len(result) == 5
    
    def test_with_large_gap(self):
        """Test detection of large time gaps"""
        timestamps = [
            '2025-01-15 10:00:00',
            '2025-01-15 10:05:00',
            '2025-01-15 13:00:00'  # 3-hour gap
        ]
        df = pd.DataFrame({
            'vehicle_id': ['V1'] * 3,
            'timestamp': pd.to_datetime(timestamps, utc=True)
        })
        
        # Should detect gap (returns same DataFrame)
        result = check_data_gaps(df, max_gap_hours=1.0, verbose=True)
        
        assert len(result) == 3  # No rows removed, just flagged


class TestCheckMissingValues:
    """Tests for check_missing_values function"""
    
    def test_no_missing_values(self):
        """Test with complete data"""
        df = pd.DataFrame({
            'vehicle_id': ['V1', 'V2'],
            'fuel': [100, 99],
            'speed': [50, 60]
        })
        
        result = check_missing_values(df)
        
        assert len(result) == 0  # No missing values
    
    def test_with_missing_values(self):
        """Test detection of missing values"""
        df = pd.DataFrame({
            'vehicle_id': ['V1', 'V2', 'V3'],
            'fuel': [100, np.nan, 98],
            'speed': [50, 60, np.nan]
        })
        
        result = check_missing_values(df)
        
        assert 'fuel' in result
        assert 'speed' in result
        assert result['fuel'] == 1
        assert result['speed'] == 1


class TestGenerateQualityReport:
    """Tests for generate_quality_report function"""
    
    def test_basic_report(self):
        """Test basic quality report generation"""
        df = pd.DataFrame({
            'vehicle_id': ['V1', 'V1', 'V2'],
            'timestamp': pd.date_range('2025-01-15', periods=3, freq='1h', tz='UTC'),
            'latitude': [40.7128, 34.0522, 51.5074],
            'longitude': [-74.0060, -118.2437, -0.1278],
            'fuel': [100, 99, 98]
        })
        
        report = generate_quality_report(df)
        
        assert report['total_rows'] == 3
        assert report['vehicles'] == 2
        assert 'date_range' in report
        assert 'valid_coordinates' in report
        assert report['valid_coordinates'] == 3
    
    def test_report_with_missing_values(self):
        """Test report includes missing value information"""
        df = pd.DataFrame({
            'vehicle_id': ['V1', 'V2'],
            'timestamp': pd.date_range('2025-01-15', periods=2, freq='1h', tz='UTC'),
            'fuel': [100, np.nan]
        })
        
        report = generate_quality_report(df)
        
        assert 'missing_values' in report
        assert 'fuel' in report['missing_values']
    
    def test_report_saved_to_file(self, tmp_path):
        """Test report is saved to JSON file"""
        df = pd.DataFrame({
            'vehicle_id': ['V1'],
            'timestamp': pd.date_range('2025-01-15', periods=1, tz='UTC'),
            'fuel': [100]
        })
        
        output_path = tmp_path / "quality_report.json"
        report = generate_quality_report(df, output_path=output_path)
        
        assert output_path.exists()