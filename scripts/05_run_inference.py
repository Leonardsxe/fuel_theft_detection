#!/usr/bin/env python3
"""
Script 05: Run Inference

Production inference script for processing new telemetry data.
Loads trained models and detects fuel theft events in real-time or batch mode.

Usage:
    # Single file inference
    python scripts/05_run_inference.py --input data/new_telemetry.csv
    
    # Batch inference on multiple files
    python scripts/05_run_inference.py --input data/raw/*.csv --batch
    
    # Streaming mode (process and exit)
    python scripts/05_run_inference.py --input data/stream_batch.csv --stream
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import glob

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.pipeline.inference import InferencePipeline, BatchInferencePipeline
from src.utils.logging_config import setup_logging, log_section

import logging

logger = logging.getLogger(__name__)


def read_telemetry_csv(file_path: Path) -> pd.DataFrame:
    """
    Load telemetry CSV while tolerating malformed rows.
    
    Uses the python engine so that on_bad_lines works consistently.
    """
    try:
        return pd.read_csv(
            file_path,
            engine="python",
            on_bad_lines="warn",  # skip and log malformed lines instead of failing
        )
    except Exception:
        logger.exception(f"Failed to read telemetry file: {file_path}")
        raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference on new telemetry data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python scripts/05_run_inference.py --input data/new_raw/WOL991_2025_Jun.csv
  
  # Multiple files
  python scripts/05_run_inference.py --input data/raw/*.csv --batch
  
  # With custom threshold
  python scripts/05_run_inference.py --input data/new_data.csv --threshold 0.7
        """,
    )
    
    parser.add_argument(
        '--input',
        '-i',
        type=str,
        required=True,
        help='Input telemetry file(s) - supports glob patterns (e.g., data/*.csv)',
    )
    
    parser.add_argument(
        '--model',
        type=Path,
        default=Path('data/models/random_forest_calibrated.pkl'),
        help='Path to trained model (default: data/models/random_forest_calibrated.pkl)',
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config'),
        help='Path to config directory (default: config/)',
    )
    
    parser.add_argument(
        '--output',
        '-o',
        type=Path,
        help='Output directory for predictions (default: data/predictions/)',
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for theft classification (default: 0.5)',
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch mode: process multiple files',
    )
    
    parser.add_argument(
        '--stream',
        action='store_true',
        help='Streaming mode: process as batch and return summary',
    )
    
    parser.add_argument(
        '--return-raw',
        action='store_true',
        help='Return all detected events (not just predicted thefts)',
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)',
    )
    
    return parser.parse_args()


def main():
    """Main execution."""
    
    args = parse_args()
    
    # Setup logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    setup_logging(
        level=args.log_level,
        log_file=log_dir / "05_run_inference.log",
        log_to_console=True
    )
    
    log_section("FUEL THEFT DETECTION - INFERENCE")
    logger.info(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    
    try:
        # Check model exists
        if not args.model.exists():
            logger.error(f"Model not found: {args.model}")
            logger.info("Please train models first using scripts/03_train_models.py")
            return 1
        
        # Resolve input files
        input_files = []
        if '*' in str(args.input):
            # Glob pattern
            input_files = [Path(p) for p in glob.glob(str(args.input))]
        else:
            input_files = [Path(args.input)]
        
        if not input_files:
            logger.error(f"No input files found matching: {args.input}")
            return 1
        
        logger.info(f"Found {len(input_files)} input file(s)")
        
        # Set output directory
        output_dir = args.output or Path("data/predictions")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize inference pipeline
        logger.info("Initializing inference pipeline...")
        logger.info(f"  Model: {args.model}")
        logger.info(f"  Config: {args.config}")
        logger.info(f"  Threshold: {args.threshold}")
        
        # Try to load optional artifacts
        pattern_thresholds_path = args.config.parent / "data/models/pattern_thresholds.json"
        noise_thresholds_path = args.config.parent / "data/models/noise_thresholds.json"
        
        pipeline = InferencePipeline(
            model_path=args.model,
            config_path=args.config,
            pattern_thresholds_path=pattern_thresholds_path if pattern_thresholds_path.exists() else None,
            noise_thresholds_path=noise_thresholds_path if noise_thresholds_path.exists() else None,
        )
        
        # Process based on mode
        if args.batch and len(input_files) > 1:
            # Batch mode
            log_section("BATCH INFERENCE")
            
            batch_pipeline = BatchInferencePipeline(pipeline)
            results = batch_pipeline.process_files(
                file_paths=input_files,
                output_dir=output_dir,
                save_results=True,
            )
            
            # Summary
            log_section("BATCH SUMMARY")
            
            total_events = sum(len(df) for df in results.values())
            total_thefts = sum((df['is_theft'] == 1).sum() for df in results.values() if not df.empty)
            
            logger.info(f"Files processed: {len(results)}")
            logger.info(f"Total events detected: {total_events}")
            logger.info(f"Total thefts predicted: {total_thefts}")
            
            # Per-file summary
            logger.info("\nPer-file results:")
            for filename, df in results.items():
                if not df.empty:
                    n_thefts = (df['is_theft'] == 1).sum()
                    logger.info(f"  {filename}: {len(df)} events, {n_thefts} thefts")
                else:
                    logger.info(f"  {filename}: No events detected")
        
        elif args.stream:
            # Streaming mode
            log_section("STREAMING INFERENCE")
            
            for file_path in input_files:
                logger.info(f"\nProcessing stream batch: {file_path.name}")
                
                telemetry_df = read_telemetry_csv(file_path)
                
                result = pipeline.process_stream(
                    telemetry_batch=telemetry_df,
                    batch_id=file_path.stem,
                    source_name=file_path.name,
                )
                
                logger.info(f"Batch {result['batch_id']}: {result['n_thefts']} thefts in {result['n_events']} events")
                
                # Save results
                if not result['events'].empty:
                    output_path = output_dir / f"predictions_{result['batch_id']}.csv"
                    result['events'].to_csv(output_path, index=False)
                    logger.info(f"Saved predictions → {output_path}")
        
        else:
            # Single file mode
            log_section("SINGLE FILE INFERENCE")
            
            file_path = input_files[0]
            logger.info(f"Processing: {file_path}")
            
            # Load telemetry
            telemetry_df = read_telemetry_csv(file_path)
            logger.info(f"Loaded {len(telemetry_df):,} telemetry rows")
            
            # Run inference
            predictions = pipeline.predict(
                telemetry_df=telemetry_df,
                return_raw_events=args.return_raw,
                confidence_threshold=args.threshold,
                source_name=file_path.name,
            )
            
            if predictions.empty:
                logger.info("No theft events detected")
                return 0
            
            # Display results
            log_section("INFERENCE RESULTS")
            
            n_events = len(predictions)
            n_thefts = (predictions['is_theft'] == 1).sum()
            
            logger.info(f"Events detected: {n_events}")
            logger.info(f"Predicted thefts: {n_thefts} ({n_thefts/n_events*100:.1f}%)")
            
            # Risk level breakdown
            if 'risk_level' in predictions.columns:
                logger.info("\nRisk level distribution:")
                risk_counts = predictions['risk_level'].value_counts()
                for level, count in risk_counts.items():
                    logger.info(f"  {level}: {count} ({count/n_events*100:.1f}%)")
            
            # Top predictions
            logger.info("\nTop 10 highest confidence predictions:")
            top_preds = predictions.nlargest(10, 'theft_probability')[
                ['vehicle_id', 'start_time', 'duration_min', 'drop_gal', 'theft_probability', 'risk_level']
            ]
            print(top_preds.to_string(index=False))
            
            # Save results
            output_path = output_dir / f"predictions_{file_path.stem}.csv"
            predictions.to_csv(output_path, index=False)
            logger.info(f"\n✓ Predictions saved → {output_path}")
            
            # Summary CSV
            summary_path = output_dir / f"summary_{file_path.stem}.csv"
            summary = predictions.groupby('vehicle_id').agg(
                n_events=('theft_probability', 'size'),
                n_thefts=('is_theft', 'sum'),
                max_confidence=('theft_probability', 'max'),
                total_fuel_drop=('drop_gal', 'sum'),
            ).reset_index()
            summary.to_csv(summary_path, index=False)
            logger.info(f"✓ Summary saved → {summary_path}")
        
        logger.info("\n✓ Inference complete")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\nInference interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
