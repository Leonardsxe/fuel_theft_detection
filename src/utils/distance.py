"""
Geospatial distance calculations.
Uses haversine formula for accurate GPS distance computation.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple
from math import radians, sin, cos, sqrt, atan2


EARTH_RADIUS_KM = 6371.0
EARTH_RADIUS_M = 6371000.0


def haversine_distance(
    lat1: Union[float, np.ndarray],
    lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
    unit: str = 'km'
) -> Union[float, np.ndarray]:
    """
    Calculate distance between two GPS coordinates using haversine formula.
    
    Args:
        lat1: Latitude of first point (degrees)
        lon1: Longitude of first point (degrees)
        lat2: Latitude of second point (degrees)
        lon2: Longitude of second point (degrees)
        unit: Return unit ('km' or 'm')
    
    Returns:
        Distance in specified unit
    
    Examples:
        >>> haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)  # NYC to LA
        3935.746
    """
    # Convert to numpy arrays for vectorized operations
    lat1, lon1 = np.atleast_1d(lat1), np.atleast_1d(lon1)
    lat2, lon2 = np.atleast_1d(lat2), np.atleast_1d(lon2)
    
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Calculate distance
    if unit == 'km':
        distance = EARTH_RADIUS_KM * c
    elif unit == 'm':
        distance = EARTH_RADIUS_M * c
    else:
        raise ValueError(f"Invalid unit '{unit}'. Use 'km' or 'm'")
    
    # Return scalar if input was scalar
    return float(distance[0]) if distance.size == 1 else distance


def calculate_trip_distance(coords: pd.DataFrame) -> float:
    """
    Calculate total distance traveled from a sequence of GPS coordinates.
    
    Args:
        coords: DataFrame with 'latitude' and 'longitude' columns
    
    Returns:
        Total distance in kilometers
    
    Examples:
        >>> coords = pd.DataFrame({
        ...     'latitude': [40.7128, 40.7580, 40.7589],
        ...     'longitude': [-74.0060, -73.9855, -73.9851]
        ... })
        >>> calculate_trip_distance(coords)
        5.123
    """
    if len(coords) < 2:
        return 0.0
    
    coords = coords.dropna(subset=['latitude', 'longitude'])
    
    if len(coords) < 2:
        return 0.0
    
    # Calculate distances between consecutive points
    lat1 = coords['latitude'].values[:-1]
    lon1 = coords['longitude'].values[:-1]
    lat2 = coords['latitude'].values[1:]
    lon2 = coords['longitude'].values[1:]
    
    distances = haversine_distance(lat1, lon1, lat2, lon2, unit='km')
    
    return float(np.sum(distances))


def haversine_pairwise(coords1: np.ndarray, coords2: np.ndarray, unit: str = 'km') -> np.ndarray:
    """
    Calculate pairwise distances between two sets of coordinates.
    Optimized for clustering algorithms (DBSCAN).
    
    Args:
        coords1: Array of shape (n, 2) with [lat, lon] in degrees
        coords2: Array of shape (m, 2) with [lat, lon] in degrees
        unit: Return unit ('km' or 'm')
    
    Returns:
        Distance matrix of shape (n, m)
    
    Examples:
        >>> points1 = np.array([[40.7128, -74.0060], [34.0522, -118.2437]])
        >>> points2 = np.array([[51.5074, -0.1278]])
        >>> haversine_pairwise(points1, points2)  # NYC, LA to London
        array([[5576.23], [8756.45]])
    """
    # Convert to radians
    coords1_rad = np.radians(coords1)
    coords2_rad = np.radians(coords2)
    
    # Reshape for broadcasting
    lat1 = coords1_rad[:, 0][:, np.newaxis]
    lon1 = coords1_rad[:, 1][:, np.newaxis]
    lat2 = coords2_rad[:, 0][np.newaxis, :]
    lon2 = coords2_rad[:, 1][np.newaxis, :]
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Calculate distance
    if unit == 'km':
        distances = EARTH_RADIUS_KM * c
    elif unit == 'm':
        distances = EARTH_RADIUS_M * c
    else:
        raise ValueError(f"Invalid unit '{unit}'. Use 'km' or 'm'")
    
    return distances


def point_to_line_distance(
    point: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float]
) -> float:
    """
    Calculate perpendicular distance from a point to a line segment.
    Useful for trajectory simplification (Douglas-Peucker algorithm).
    
    Args:
        point: (lat, lon) of the point
        line_start: (lat, lon) of line start
        line_end: (lat, lon) of line end
    
    Returns:
        Perpendicular distance in kilometers
    """
    # If line is actually a point
    if line_start == line_end:
        return haversine_distance(point[0], point[1], line_start[0], line_start[1])
    
    # Calculate distances
    d_start_end = haversine_distance(line_start[0], line_start[1], line_end[0], line_end[1])
    d_start_point = haversine_distance(line_start[0], line_start[1], point[0], point[1])
    d_point_end = haversine_distance(point[0], point[1], line_end[0], line_end[1])
    
    # Check if point projects onto line segment
    if d_start_end == 0:
        return d_start_point
    
    # Use Heron's formula for triangle area, then calculate height
    s = (d_start_end + d_start_point + d_point_end) / 2  # semi-perimeter
    
    # Avoid negative values under sqrt due to floating point errors
    area_squared = s * (s - d_start_end) * (s - d_start_point) * (s - d_point_end)
    
    if area_squared <= 0:
        return min(d_start_point, d_point_end)
    
    area = sqrt(area_squared)
    perpendicular_distance = (2 * area) / d_start_end
    
    return perpendicular_distance


def is_within_radius(
    lat: float,
    lon: float,
    center_lat: float,
    center_lon: float,
    radius_km: float
) -> bool:
    """
    Check if a point is within a given radius of a center point.
    
    Args:
        lat: Latitude of point to check
        lon: Longitude of point to check
        center_lat: Latitude of center
        center_lon: Longitude of center
        radius_km: Radius in kilometers
    
    Returns:
        True if point is within radius
    
    Examples:
        >>> is_within_radius(40.7580, -73.9855, 40.7128, -74.0060, 10)  # Within 10km?
        True
    """
    distance = haversine_distance(lat, lon, center_lat, center_lon, unit='km')
    return distance <= radius_km