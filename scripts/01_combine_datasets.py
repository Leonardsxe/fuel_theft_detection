"""
Script to combine multiple raw data sources into unified dataset.
Demonstrates usage of the data loading and preprocessing pipeline with enhanced parsing.

Usage:
    python scripts/01_combine_datasets.py

This script will:
1. Load multiple CSV files from data/raw/ (with glob pattern support)
2. Parse various formats (Spanish/English, coordinates, speed with units)
3. Standardize column names
4. Validate data quality
5. Remove outliers and interpolate gaps
6. Calculate fuel rate and apply signal conditioning
7. Save combined dataset to data/processed/combined_dataset.csv
"""

import sys
from pathlib import Path
import glob

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.loader import load_config, validate_config
from src.data.combiner import DataCombiner
from src.utils.logging_config import setup_logging, log_section
import logging


def resolve_glob_patterns(raw_sources: list) -> list:
    """
    Resolve glob patterns in source paths.
    
    Args:
        raw_sources: List of paths (may include glob patterns)
    
    Returns:
        List of resolved Path objects
    """
    resolved = []
    for source in raw_sources:
        source_path = Path(source)
        
        # Check if it's a glob pattern
        if '*' in str(source_path):
            # Expand glob
            matches = list(source_path.parent.glob(source_path.name))
            resolved.extend(matches)
        elif source_path.exists():
            resolved.append(source_path)
    
    return sorted(set(resolved))  # Remove duplicates and sort


def main():
    """Main execution function"""
    
    # Setup logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    setup_logging(
        level="INFO",
        log_file=log_dir / "01_combine_datasets.log",
        log_to_console=True
    )
    
    logger = logging.getLogger(__name__)
    
    log_section("FUEL THEFT DETECTION - DATA COMBINATION")
    
    try:
        # 1. Load configuration
        logger.info("Loading configuration...")
        config = load_config(
            detection_config_path=Path("config/detection_config.yaml"),
            model_config_path=Path("config/model_config.yaml"),
            path_config_path=Path("config/paths_config.yaml")
        )
        
        # Validate configuration
        validate_config(config)
        
        # 2. Initialize data combiner
        logger.info("Initializing data combiner...")
        combiner = DataCombiner(
            path_config=config.paths,
            detection_config=config.detection
        )
        
        # 3. Resolve glob patterns in raw sources
        raw_sources = config.paths.input.raw_sources
        existing_sources = resolve_glob_patterns(raw_sources)
        
        if not existing_sources:
            logger.error("No raw data sources found!")
            logger.info(f"Expected sources in: {config.paths.data.raw}")
            logger.info("Please add CSV files to data/raw/ directory")
            logger.info("Or update paths in config/paths_config.yaml")
            logger.info("\nSupported formats:")
            logger.info("  - English column names: timestamp, speed_kmh, ignition, etc.")
            logger.info("  - Spanish column names: Tiempo, Velocidad, Ignición, etc.")
            logger.info("  - Combined coordinates: 'lat, lon' format")
            logger.info("  - Speed with units: '50 km/h', '60.5 km/h'")
            logger.info("  - Separate fuel tanks: left_tank + right_tank")
            return 1
        
        logger.info(f"Found {len(existing_sources)} data sources:")
        for source in existing_sources:
            size_mb = source.stat().st_size / (1024 * 1024)
            logger.info(f"  - {source.name} ({size_mb:.2f} MB)")
        
        # 4. Process data sources
        log_section("PROCESSING DATA SOURCES")
        
        combined_df = combiner.process_multiple_sources(
            source_paths=existing_sources,
            save_output=True
        )
        
        # 5. Summary statistics
        log_section("SUMMARY STATISTICS")
        
        logger.info(f"Total rows: {len(combined_df):,}")
        logger.info(f"Total vehicles: {combined_df['vehicle_id'].nunique()}")
        
        if 'timestamp' in combined_df.columns:
            logger.info(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
            duration = (combined_df['timestamp'].max() - combined_df['timestamp'].min()).days
            logger.info(f"Duration: {duration} days")
        
        # Check for new features
        new_features = ['rate_gpm', 'fuel_med5', 'is_moving', 'is_stationary_on', 'is_ign_off']
        available_features = [f for f in new_features if f in combined_df.columns]
        if available_features:
            logger.info(f"\nEnhanced features available: {', '.join(available_features)}")
        
        # Per-vehicle statistics
        vehicle_stats = combined_df.groupby('vehicle_id').agg({
            'timestamp': 'count',
            'total_fuel_gal': ['min', 'max', 'mean']
        }).round(2)
        
        logger.info("\nPer-vehicle statistics:")
        logger.info(f"\n{vehicle_stats.to_string()}")
        
        # State distribution
        if all(col in combined_df.columns for col in ['is_moving', 'is_stationary_on', 'is_ign_off']):
            logger.info("\nVehicle state distribution:")
            state_counts = {
                'Moving': combined_df['is_moving'].sum(),
                'Stationary (ON)': combined_df['is_stationary_on'].sum(),
                'Ignition OFF': combined_df['is_ign_off'].sum()
            }
            for state, count in state_counts.items():
                pct = 100 * count / len(combined_df)
                logger.info(f"  {state}: {count:,} ({pct:.1f}%)")
        
        # Label statistics (if available)
        if 'stationary_drain' in combined_df.columns:
            n_labels = combined_df['stationary_drain'].sum()
            pct_labels = 100 * n_labels / len(combined_df)
            logger.info(f"\nLabeled theft events: {n_labels:,} ({pct_labels:.2f}%)")
        
        # 6. Output locations
        log_section("OUTPUT FILES")
        
        logger.info(f"Combined dataset: {config.paths.input.combined_csv}")
        logger.info(f"Quality report: {config.paths.output.data_quality_report}")
        
        log_section("DATA COMBINATION COMPLETE")
        logger.info("✓ Dataset is ready for event detection")
        logger.info("\nNew features added:")
        logger.info("  - rate_gpm: Fuel consumption rate (gal/min)")
        logger.info("  - fuel_med5: Rolling median filtered fuel level")
        logger.info("  - Enhanced state detection (moving, stationary_on, ign_off)")
        logger.info("\n✓ Next step: Run scripts/02_detect_events.py")
        
        return 0
    
    except Exception as e:
        logger.error(f"Data combination failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)