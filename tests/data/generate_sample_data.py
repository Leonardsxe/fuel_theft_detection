"""
Generate sample data for testing.

Run this script to create tests/data/sample_data.csv

Usage:
    python tests/data/generate_sample_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


def generate_sample_data(n_vehicles=3, days=7, output_path=None):
    """
    Generate synthetic telemetry data with known theft patterns.
    
    Args:
        n_vehicles: Number of vehicles
        days: Number of days of data
        output_path: Path to save CSV (default: tests/data/sample_data.csv)
    """
    
    np.random.seed(42)
    
    all_data = []
    
    for veh_num in range(n_vehicles):
        vehicle_id = f"TEST_VEH_{veh_num+1:03d}"
        
        # Generate timestamps (1 minute intervals)
        start_time = datetime(2024, 1, 1, tzinfo=None)
        n_points = days * 24 * 60  # Points per vehicle
        timestamps = [start_time + timedelta(minutes=i) for i in range(n_points)]
        
        # Simulate fuel level with gradual consumption
        fuel = np.zeros(n_points)
        fuel[0] = np.random.uniform(60, 90)  # Starting fuel
        
        for i in range(1, n_points):
            # Normal consumption
            if i % 60 == 0:  # Hourly check
                # Occasional refueling
                if np.random.random() < 0.05 and fuel[i-1] < 30:
                    fuel[i] = fuel[i-1] + np.random.uniform(30, 50)
                else:
                    # Gradual decrease with noise
                    fuel[i] = fuel[i-1] - np.random.uniform(0.05, 0.15) + np.random.normal(0, 0.05)
            else:
                fuel[i] = fuel[i-1] + np.random.normal(0, 0.05)  # Small noise
        
        fuel = np.clip(fuel, 5, 100)
        
        # Inject synthetic theft events
        theft_times = []
        for day in range(days):
            if np.random.random() < 0.4:  # 40% chance of theft per day
                # Random time during the day
                theft_start = day * 24 * 60 + np.random.randint(0, 24 * 60)
                theft_duration = np.random.randint(5, 15)  # 5-15 minutes
                theft_amount = np.random.uniform(4, 10)  # 4-10 gallons
                
                if theft_start + theft_duration < n_points:
                    # Create fuel drop
                    fuel[theft_start:theft_start+theft_duration] -= theft_amount / theft_duration
                    theft_times.append((theft_start, theft_start+theft_duration))
        
        # Generate speed (simulating driving patterns)
        speed = np.zeros(n_points)
        for i in range(n_points):
            hour = (i // 60) % 24
            # More driving during work hours
            if 6 <= hour <= 22:
                if np.random.random() < 0.3:  # 30% moving during day
                    speed[i] = np.random.choice([45, 55, 65, 75])
                else:
                    speed[i] = 0
            else:
                speed[i] = 0  # Stationary at night
        
        # Ignition follows speed (mostly)
        ignition = (speed > 0) | (np.random.random(n_points) < 0.1)
        
        # GPS coordinates (centered around NYC with small variations)
        base_lat = 40.7128
        base_lon = -74.0060
        latitude = base_lat + np.cumsum(np.random.normal(0, 0.0001, n_points))
        longitude = base_lon + np.cumsum(np.random.normal(0, 0.0001, n_points))
        
        # Clip to reasonable range
        latitude = np.clip(latitude, base_lat - 0.1, base_lat + 0.1)
        longitude = np.clip(longitude, base_lon - 0.1, base_lon + 0.1)
        
        # Create labels (1 if point is during theft)
        labels = np.zeros(n_points, dtype=int)
        for start, end in theft_times:
            labels[start:end] = 1
        
        # Create DataFrame for this vehicle
        vehicle_data = pd.DataFrame({
            'vehicle_id': vehicle_id,
            'timestamp': pd.to_datetime(timestamps),
            'total_fuel_gal': fuel,
            'speed_kmh': speed,
            'ignition': ignition,
            'latitude': latitude,
            'longitude': longitude,
            'stationary_drain': labels,  # Ground truth labels
        })
        
        all_data.append(vehicle_data)
    
    # Combine all vehicles
    df = pd.concat(all_data, ignore_index=True)
    
    # Add derived columns
    df['dt_s'] = df.groupby('vehicle_id')['timestamp'].diff().dt.total_seconds()
    df['dfuel'] = df.groupby('vehicle_id')['total_fuel_gal'].diff()
    df['moving'] = df['speed_kmh'] > 1.0
    df['stationary_on'] = (~df['moving']) & df['ignition']
    df['ign_off'] = ~df['ignition']
    df['stationary'] = df['stationary_on'] | df['ign_off']
    
    # Sort by vehicle and time
    df = df.sort_values(['vehicle_id', 'timestamp']).reset_index(drop=True)
    
    # Save to CSV
    if output_path is None:
        output_path = Path(__file__).parent / "sample_data.csv"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    print(f"Generated sample data:")
    print(f"  Vehicles: {n_vehicles}")
    print(f"  Days: {days}")
    print(f"  Total rows: {len(df):,}")
    print(f"  Theft events (labeled points): {df['stationary_drain'].sum():,}")
    print(f"  Saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    # Generate sample data
    generate_sample_data(
        n_vehicles=3,
        days=7,
        output_path=Path(__file__).parent / "sample_data.csv"
    )