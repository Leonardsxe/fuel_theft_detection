"""
Combine multiple data sources into unified dataset.
Addresses the need to unify preprocessing currently split across files.
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
import logging

from src.config.settings import PathConfig, DetectionConfig
from src.data.loader import load_multiple_sources
from src.data.validator import validate_data_pipeline, generate_quality_report
from src.data.preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)


class DataCombiner:
    """
    Combine multiple CSV sources into a single preprocessed dataset.
    Handles deduplication, alignment, and quality checks.
    """
    
    def __init__(
        self,
        path_config: PathConfig,
        detection_config: DetectionConfig
    ):
        """
        Initialize combiner with configuration.
        
        Args:
            path_config: Path configuration
            detection_config: Detection configuration
        """
        self.path_config = path_config
        self.detection_config = detection_config
        self.preprocessor = DataPreprocessor(detection_config)
    
    def deduplicate_across_sources(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate records that may appear across multiple sources.
        
        Args:
            df: Combined DataFrame
        
        Returns:
            Deduplicated DataFrame
        """
        logger.info("Deduplicating across sources...")
        
        initial_count = len(df)
        
        # Sort to keep most recent data (last source wins)
        df = df.sort_values(["vehicle_id", "timestamp", "total_fuel_gal"])
        
        # Remove duplicates, keeping last occurrence
        df = df.drop_duplicates(
            subset=["vehicle_id", "timestamp"],
            keep="last"
        ).reset_index(drop=True)
        
        removed = initial_count - len(df)
        logger.info(f"✓ Removed {removed:,} duplicate records")
        
        return df
    
    def align_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure consistent timestamp alignment across sources.
        
        Args:
            df: Combined DataFrame
        
        Returns:
            DataFrame with aligned timestamps
        """
        logger.info("Aligning timestamps...")
        
        # Already handled in preprocessor's normalize_timestamps
        # This is a placeholder for additional alignment logic if needed
        
        df = df.sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True)
        
        logger.info("✓ Timestamps aligned")
        
        return df
    
    def combine_dataframes(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Concatenate multiple DataFrames with consistency checks.
        
        Args:
            dataframes: List of DataFrames to combine
        
        Returns:
            Combined DataFrame
        """
        logger.info(f"Combining {len(dataframes)} data sources...")
        
        if not dataframes:
            raise ValueError("No dataframes to combine")
        
        # Check that all have same columns
        first_cols = set(dataframes[0].columns)
        for i, df in enumerate(dataframes[1:], start=1):
            if set(df.columns) != first_cols:
                missing = first_cols - set(df.columns)
                extra = set(df.columns) - first_cols
                logger.warning(f"Source {i+1} has different columns:")
                if missing:
                    logger.warning(f"  Missing: {missing}")
                if extra:
                    logger.warning(f"  Extra: {extra}")
        
        # Concatenate
        combined = pd.concat(dataframes, ignore_index=True)
        
        logger.info(f"✓ Combined to {len(combined):,} total rows")
        
        return combined
    
    def process_multiple_sources(
        self,
        source_paths: Optional[List[Path]] = None,
        save_output: bool = True
    ) -> pd.DataFrame:
        """
        Complete pipeline: load → combine → validate → preprocess → save.
        
        Args:
            source_paths: List of paths to source CSVs (uses config if None)
            save_output: Whether to save combined dataset
        
        Returns:
            Preprocessed combined DataFrame
        """
        logger.info("="*60)
        logger.info("STARTING DATA COMBINATION PIPELINE")
        logger.info("="*60)
        
        # Use paths from config if not provided
        if source_paths is None:
            source_paths = self.path_config.input.raw_sources
        
        # 1. Load all sources
        logger.info(f"Step 1/6: Loading {len(source_paths)} source files...")
        dataframes = load_multiple_sources(
            source_paths,
            column_mapping=self.path_config.column_mapping
        )
        
        # 2. Combine
        logger.info("Step 2/6: Combining sources...")
        combined = self.combine_dataframes(dataframes)
        
        # 3. Deduplicate
        logger.info("Step 3/6: Deduplicating...")
        combined = self.deduplicate_across_sources(combined)
        
        # 4. Align timestamps
        logger.info("Step 4/6: Aligning timestamps...")
        combined = self.align_timestamps(combined)
        
        # 5. Validate
        logger.info("Step 5/6: Validating data quality...")
        combined = validate_data_pipeline(combined)
        
        # Generate quality report
        report = generate_quality_report(
            combined,
            output_path=self.path_config.output.data_quality_report
        )
        
        # 6. Preprocess
        logger.info("Step 6/6: Preprocessing...")
        combined = self.preprocessor.fit_transform(combined)
        
        # Save if requested
        if save_output:
            output_path = self.path_config.input.combined_csv
            output_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(output_path, index=False)
            logger.info(f"✓ Saved combined dataset to {output_path}")
        
        logger.info("="*60)
        logger.info("DATA COMBINATION PIPELINE COMPLETE")
        logger.info(f"Final dataset: {len(combined):,} rows, "
                   f"{combined['vehicle_id'].nunique()} vehicles")
        logger.info("="*60)
        
        return combined
    
    def load_combined(self) -> pd.DataFrame:
        """
        Load previously combined dataset.
        
        Returns:
            Combined DataFrame
        """
        path = self.path_config.input.combined_csv
        
        if not path.exists():
            raise FileNotFoundError(
                f"Combined dataset not found: {path}\n"
                f"Run process_multiple_sources() first to create it."
            )
        
        logger.info(f"Loading combined dataset from {path}")
        df = pd.read_csv(path)
        
        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        
        logger.info(f"✓ Loaded {len(df):,} rows, {df['vehicle_id'].nunique()} vehicles")
        
        return df


def combine_data_sources(
    source_paths: List[Path],
    path_config: PathConfig,
    detection_config: DetectionConfig,
    save_output: bool = True
) -> pd.DataFrame:
    """
    Convenience function for combining data sources.
    
    Args:
        source_paths: List of paths to source CSVs
        path_config: Path configuration
        detection_config: Detection configuration
        save_output: Whether to save output
    
    Returns:
        Combined and preprocessed DataFrame
    """
    combiner = DataCombiner(path_config, detection_config)
    return combiner.process_multiple_sources(source_paths, save_output)