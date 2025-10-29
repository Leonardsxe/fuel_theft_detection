"""
Script to combine multiple raw data sources into unified dataset.
Demonstrates usage of the data loading and preprocessing pipeline.

Usage:
    python scripts/01_combine_datasets.py

This script will:
1. Load multiple CSV files from data/raw/
2. Standardize column names
3. Validate data quality
4. Remove outliers and interpolate gaps
5. Save combined dataset to data/processed/combined_dataset.csv
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.loader import load_config, validate_config
from src.data.combiner import DataCombiner
from src.utils.logging_config import setup_logging, log_section
import logging


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
        
        # 3. Check if raw sources exist
        raw_sources = config.paths.input.raw_sources
        existing_sources = [p for p in raw_sources if p.exists()]
        
        if not existing_sources:
            logger.error("No raw data sources found!")
            logger.info(f"Expected sources in: {config.paths.data.raw}")
            logger.info("Please add CSV files to data/raw/ directory")
            logger.info("Or update paths in config/paths_config.yaml")
            return 1
        
        logger.info(f"Found {len(existing_sources)} data sources:")
        for source in existing_sources:
            logger.info(f"  - {source.name}")
        
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
        
        # Per-vehicle statistics
        vehicle_stats = combined_df.groupby('vehicle_id').agg({
            'timestamp': 'count',
            'total_fuel_gal': ['min', 'max', 'mean']
        }).round(2)
        
        logger.info("\nPer-vehicle statistics:")
        logger.info(f"\n{vehicle_stats.to_string()}")
        
        # 6. Output locations
        log_section("OUTPUT FILES")
        
        logger.info(f"Combined dataset: {config.paths.input.combined_csv}")
        logger.info(f"Quality report: {config.paths.output.data_quality_report}")
        
        log_section("DATA COMBINATION COMPLETE")
        logger.info("✓ Dataset is ready for event detection")
        logger.info("✓ Next step: Run scripts/02_detect_events.py")
        
        return 0
    
    except Exception as e:
        logger.error(f"Data combination failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)