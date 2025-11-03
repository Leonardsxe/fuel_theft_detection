"""
Event Detection Pipeline Script
Demonstrates the complete fuel theft detection pipeline.

This script:
1. Loads preprocessed data from combined_dataset.csv
2. Creates time-aware train/test split
3. Segments stationary periods
4. Calculates adaptive noise thresholds (train only)
5. Clusters stationary locations (train only)
6. Detects potential theft events
7. Applies non-maximum suppression
8. Assigns events to hotspot clusters
9. Adds basic temporal features
10. Saves detected events with features

Usage:
    python scripts/02_detect_events.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.loader import load_config, validate_config
from src.utils.logging_config import setup_logging, log_section
from src.detection.stationary import segment_stationary_periods
from src.detection.thresholds import compute_noise_thresholds, NoiseThresholdCalculator
from src.clustering.hotspots import cluster_stationary_points
from src.detection.events import detect_events
from src.detection.nms import nms_events_dataframe, remove_exact_duplicates
from src.clustering.assignment import assign_events_to_clusters, calculate_hotspot_statistics
from src.features.temporal import add_all_temporal_features
from src.utils.timezone import ensure_series_utc
from src.utils.splitting import create_temporal_split_on_raw_data
from src.utils.splitting import map_events_to_split

import logging


def attach_labels(events_df: pd.DataFrame, raw_df: pd.DataFrame, label_col: str = "stationary_drain") -> pd.DataFrame:
    """
    Attach ground truth labels to events if available.
    
    Args:
        events_df: Events DataFrame
        raw_df: Raw DataFrame with labels
        label_col: Name of label column
    
    Returns:
        Events DataFrame with y column
    """
    logger = logging.getLogger(__name__)
    
    if label_col not in raw_df.columns:
        logger.info(f"Label column '{label_col}' not found, skipping label attachment")
        events_df["y"] = np.nan
        return events_df
    
    logger.info(f"Attaching labels from '{label_col}' column...")
    
    # Convert label column to numeric
    raw_df = raw_df.copy()
    raw_df[label_col] = pd.to_numeric(raw_df[label_col], errors="coerce").fillna(0).astype(int)
    
    def get_event_label(event):
        """Check if any point in event window has label=1"""
        mask = (
            (raw_df["vehicle_id"] == event["vehicle_id"]) &
            (raw_df["timestamp"] >= event["start_time"]) &
            (raw_df["timestamp"] <= event["end_time"])
        )
        label_values = raw_df.loc[mask, label_col]
        return int((label_values == 1).any())
    
    events_df["y"] = events_df.apply(get_event_label, axis=1)
    
    n_positive = events_df["y"].sum()
    pct_positive = 100 * n_positive / len(events_df) if len(events_df) > 0 else 0
    
    logger.info(f"✓ Labels attached: {n_positive} positive events ({pct_positive:.1f}%)")
    
    return events_df


def generate_summary_statistics(events_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """
    Generate summary statistics of detected events.
    
    Args:
        events_df: Events DataFrame
        output_path: Path to save summary
    
    Returns:
        Summary DataFrame
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating event summary statistics...")
    
    if events_df.empty:
        logger.warning("No events to summarize")
        return pd.DataFrame()
    
    # Overall statistics
    summary = events_df.groupby(["vehicle_id", "pattern"]).agg(
        event_count=("pattern", "size"),
        mean_drop_gal=("drop_gal", "mean"),
        max_drop_gal=("drop_gal", "max"),
        median_duration_min=("duration_min", "median"),
        hotspot_events=("is_hotspot", "sum")
    ).reset_index()
    
    # Add positive labels if available
    if "y" in events_df.columns and events_df["y"].notna().any():
        label_summary = events_df.groupby(["vehicle_id", "pattern"])["y"].sum().reset_index()
        label_summary.columns = ["vehicle_id", "pattern", "positive_labels"]
        summary = summary.merge(label_summary, on=["vehicle_id", "pattern"], how="left")
    
    summary = summary.sort_values(["vehicle_id", "event_count"], ascending=[True, False])
    
    # Save summary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    logger.info(f"✓ Summary saved to {output_path}")
    
    return summary


def main():
    """Main execution function"""
    
    # Setup logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    setup_logging(
        level="INFO",
        log_file=log_dir / "02_detect_events.log",
        log_to_console=True
    )
    
    logger = logging.getLogger(__name__)
    
    log_section("FUEL THEFT DETECTION - EVENT DETECTION PIPELINE")
    
    try:
        # 1. Load configuration
        logger.info("Loading configuration...")
        config = load_config(
            detection_config_path=Path("config/detection_config.yaml"),
            model_config_path=Path("config/model_config.yaml"),
            path_config_path=Path("config/paths_config.yaml")
        )
        validate_config(config)
        
        # 2. Load preprocessed data
        log_section("LOADING PREPROCESSED DATA")
        
        combined_path = config.paths.input.combined_csv
        if not combined_path.exists():
            logger.error(f"Combined dataset not found: {combined_path}")
            logger.info("Please run scripts/01_combine_datasets.py first")
            return 1
        
        logger.info(f"Loading data from {combined_path}")
        df = pd.read_csv(combined_path)
        
        # Ensure UTC timestamps
        df["timestamp"] = ensure_series_utc(df["timestamp"])
        
        logger.info(f"Loaded {len(df):,} rows, {df['vehicle_id'].nunique()} vehicles")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Check required columns
        required_cols = ["vehicle_id", "timestamp", "total_fuel_gal", "speed_kmh", 
                        "ignition", "stationary", "stationary_on", "ign_off", "dfuel"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return 1
        
        # 3. Create time-aware train/test split
        log_section("CREATING TRAIN/TEST SPLIT")
        
        train_mask = create_temporal_split_on_raw_data(df, train_ratio=config.model.splitting.train_ratio)
        
        # 4. Segment stationary periods
        log_section("SEGMENTING STATIONARY PERIODS")
        
        segments = segment_stationary_periods(df, config.detection)
        
        if segments.empty:
            logger.error("No stationary segments found")
            return 1
        
        logger.info(f"Found {len(segments)} stationary segments")
        
        # 5. Calculate adaptive noise thresholds (TRAIN ONLY)
        log_section("CALCULATING ADAPTIVE NOISE THRESHOLDS")
        
        thresholds = compute_noise_thresholds(df, config.detection, train_mask)
        
        logger.info(f"Calculated thresholds for {len(thresholds) // 2} vehicles x 2 states")
        
        # Save thresholds for inspection
        threshold_calc = NoiseThresholdCalculator(config.detection)
        threshold_calc.thresholds = thresholds
        noise_envelope = threshold_calc.compute_noise_envelope(df.loc[train_mask])
        
        if not noise_envelope.empty:
            noise_path = config.paths.output.noise_envelopes
            noise_path.parent.mkdir(parents=True, exist_ok=True)
            noise_envelope.to_csv(noise_path, index=False)
            logger.info(f"Saved noise envelopes to {noise_path}")
        
        # 6. Cluster stationary locations (TRAIN ONLY)
        stationary_mask = df['stationary']

        stationary_with_coords = df[stationary_mask & 
                                df['latitude'].notna() & 
                                df['longitude'].notna()].copy()


        stationary_train_mask = train_mask[stationary_with_coords.index]

        print(f"Final stationary points: {len(stationary_with_coords)}")
        print(f"Final train mask: {len(stationary_train_mask)}")
        print(f"Sizes match: {len(stationary_with_coords) == len(stationary_train_mask)}")

        stationary_pts = cluster_stationary_points(stationary_with_coords, config.detection.clustering, stationary_train_mask)

        if not stationary_pts.empty:
            n_clusters = stationary_pts[stationary_pts["cluster_id"] >= 0]["cluster_id"].nunique()
            n_noise = (stationary_pts["cluster_id"] == -1).sum()
            logger.info(f"Found {n_clusters} hotspot clusters, {n_noise} noise points")        
        
        # 7. Detect events
        log_section("DETECTING POTENTIAL THEFT EVENTS")
        
        events_df = detect_events(segments, df, config.detection, thresholds)
        
        if events_df.empty:
            logger.warning("No events detected")
            # Save empty file
            config.paths.output.events_csv.parent.mkdir(parents=True, exist_ok=True)
            events_df.to_csv(config.paths.output.events_csv, index=False)
            return 0
        
        logger.info(f"Detected {len(events_df)} raw events")
        
        # 8. Remove exact duplicates
        events_df = remove_exact_duplicates(events_df)
        
        # 9. Apply non-maximum suppression
        log_section("APPLYING NON-MAXIMUM SUPPRESSION")
        
        events_df = nms_events_dataframe(events_df, config.detection.nms.iou_threshold)
        
        logger.info(f"After NMS: {len(events_df)} unique events")
        
        # 10. Assign events to hotspot clusters
        log_section("ASSIGNING EVENTS TO HOTSPOT CLUSTERS")
        
        events_df = assign_events_to_clusters(events_df, stationary_pts)
        
        # Calculate hotspot statistics
        hotspot_stats = calculate_hotspot_statistics(events_df, stationary_pts)
        if not hotspot_stats.empty:
            hotspot_path = Path("data/reports/hotspot_statistics.csv")
            hotspot_path.parent.mkdir(parents=True, exist_ok=True)
            hotspot_stats.to_csv(hotspot_path, index=False)
            logger.info(f"Saved hotspot statistics to {hotspot_path}")
        
        # 11. Map events to train/test split
        log_section("MAPPING EVENTS TO TRAIN/TEST SPLIT")
        
        events_df = map_events_to_split(events_df, df, train_mask)
        
        # 12. Attach labels (if available)
        log_section("ATTACHING LABELS")
        
        events_df = attach_labels(events_df, df, label_col="stationary_drain")
        
        # 13. Add temporal features
        log_section("ADDING TEMPORAL FEATURES")
        
        logger.info("Adding temporal features...")
        events_df = add_all_temporal_features(events_df)
        logger.info("✓ Temporal features added")
        
        # 14. Save detected events
        log_section("SAVING RESULTS")
        
        events_path = config.paths.output.events_csv
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events_df.to_csv(events_path, index=False)
        logger.info(f"✓ Saved events to {events_path}")
        
        # 15. Generate and save summary
        summary_path = config.paths.output.event_summary
        summary = generate_summary_statistics(events_df, summary_path)
        
        # 16. Display final statistics
        log_section("DETECTION SUMMARY")
        
        logger.info(f"Total events detected: {len(events_df)}")
        logger.info(f"Vehicles: {events_df['vehicle_id'].nunique()}")
        
        # Pattern distribution
        logger.info("\nEvents by pattern:")
        pattern_counts = events_df["pattern"].value_counts()
        for pattern, count in pattern_counts.items():
            pct = 100 * count / len(events_df)
            logger.info(f"  {pattern}: {count} ({pct:.1f}%)")
        
        # Train/test distribution
        if "is_train" in events_df.columns:
            n_train = events_df["is_train"].sum()
            n_test = (~events_df["is_train"]).sum()
            logger.info(f"\nTrain/test split:")
            logger.info(f"  Train events: {n_train} ({100*n_train/len(events_df):.1f}%)")
            logger.info(f"  Test events: {n_test} ({100*n_test/len(events_df):.1f}%)")
        
        # Label statistics
        if "y" in events_df.columns and events_df["y"].notna().any():
            n_positive = events_df["y"].sum()
            pct_positive = 100 * n_positive / len(events_df)
            logger.info(f"\nLabeled events:")
            logger.info(f"  Positive (theft): {n_positive} ({pct_positive:.1f}%)")
            logger.info(f"  Negative (normal): {len(events_df) - n_positive} ({100 - pct_positive:.1f}%)")
        
        # Hotspot statistics
        n_hotspot = events_df["is_hotspot"].sum()
        pct_hotspot = 100 * n_hotspot / len(events_df)
        logger.info(f"\nHotspot events: {n_hotspot} ({pct_hotspot:.1f}%)")
        
        # Top events by fuel drop
        logger.info("\nTop 10 events by fuel drop:")
        top_events = events_df.nlargest(10, "drop_gal")[
            ["vehicle_id", "start_time", "duration_min", "drop_gal", "pattern", "is_hotspot"]
        ]
        logger.info(f"\n{top_events.to_string(index=False)}")
        
        # Summary by vehicle
        if not summary.empty:
            logger.info("\nSummary by vehicle and pattern:")
            logger.info(f"\n{summary.to_string(index=False)}")
        
        # Output files
        log_section("OUTPUT FILES")
        
        logger.info(f"Events: {events_path}")
        logger.info(f"Summary: {summary_path}")
        logger.info(f"Noise envelopes: {config.paths.output.noise_envelopes}")
        logger.info(f"Hotspot statistics: data/reports/hotspot_statistics.csv")
        
        log_section("EVENT DETECTION COMPLETE")
        
        logger.info("✓ Event detection pipeline executed successfully")
        logger.info("✓ Next step: Run scripts/03_train_models.py (when implemented)")
        
        return 0
    
    except Exception as e:
        logger.error(f"Event detection failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)