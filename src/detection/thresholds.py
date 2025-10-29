"""
Adaptive noise threshold calculation.
Uses Median Absolute Deviation (MAD) for robust threshold estimation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

from src.config.settings import DetectionConfig

logger = logging.getLogger(__name__)


def robust_sigma_mad(series: pd.Series) -> float:
    """
    Calculate robust standard deviation using Median Absolute Deviation.
    
    MAD is more robust to outliers than standard deviation.
    
    Args:
        series: Series of values
    
    Returns:
        Robust sigma estimate
    
    Formula:
        sigma ≈ 1.4826 * MAD
        where MAD = median(|x - median(x)|)
    """
    x = pd.to_numeric(series, errors="coerce").dropna().values
    
    if x.size == 0:
        return 0.0
    
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    
    return 1.4826 * mad


class NoiseThresholdCalculator:
    """
    Calculate adaptive noise thresholds for fuel theft detection.
    Uses vehicle-specific and state-specific thresholds based on MAD.
    """
    
    def __init__(self, config: DetectionConfig):
        """
        Initialize calculator with configuration.
        
        Args:
            config: Detection configuration
        """
        self.config = config
        self.thresholds = {}
    
    def compute_thresholds(
        self,
        df: pd.DataFrame,
        train_mask: Optional[np.ndarray] = None
    ) -> Dict[Tuple[str, str], Tuple[float, float]]:
        """
        Compute adaptive noise thresholds for each vehicle and state.
        
        Thresholds are based on:
        1. Robust sigma (MAD-based) of fuel changes
        2. Minimum absolute thresholds from configuration
        
        Args:
            df: DataFrame with fuel change data
            train_mask: Optional mask for training data only
        
        Returns:
            Dictionary mapping (vehicle_id, state) to (step_threshold, cumulative_threshold)
        """
        logger.info("Computing adaptive noise thresholds...")
        
        # Use only training data if mask provided
        if train_mask is not None:
            df_calc = df.loc[train_mask]
            logger.info("Using training data only for threshold calculation")
        else:
            df_calc = df
        
        thresholds = {}
        
        for vid, vehicle_data in df_calc.groupby("vehicle_id"):
            # Calculate for stationary_on state
            stationary_on_fuel = vehicle_data.loc[vehicle_data["stationary_on"], "dfuel"]
            sigma_on = robust_sigma_mad(stationary_on_fuel)
            
            step_thr_on = max(
                self.config.thresholds_stationary_on.mad_multiplier_step * sigma_on,
                self.config.thresholds_stationary_on.min_step_gal
            )
            cum_thr_on = max(
                self.config.thresholds_stationary_on.mad_multiplier_cum * sigma_on,
                self.config.thresholds_stationary_on.min_cumulative_gal
            )
            
            thresholds[(vid, "stationary_on")] = (step_thr_on, cum_thr_on)
            
            # Calculate for ignition_off state
            ign_off_fuel = vehicle_data.loc[vehicle_data["ign_off"], "dfuel"]
            sigma_off = robust_sigma_mad(ign_off_fuel)
            
            step_thr_off = max(
                self.config.thresholds_ignition_off.mad_multiplier_step * sigma_off,
                self.config.thresholds_ignition_off.min_step_gal
            )
            cum_thr_off = max(
                self.config.thresholds_ignition_off.mad_multiplier_cum * sigma_off,
                self.config.thresholds_ignition_off.min_cumulative_gal
            )
            
            thresholds[(vid, "ign_off")] = (step_thr_off, cum_thr_off)
            
            logger.debug(f"Vehicle {vid}:")
            logger.debug(f"  Stationary ON: step={step_thr_on:.3f}, cum={cum_thr_on:.3f} (sigma={sigma_on:.3f})")
            logger.debug(f"  Ignition OFF: step={step_thr_off:.3f}, cum={cum_thr_off:.3f} (sigma={sigma_off:.3f})")
        
        logger.info(f"✓ Computed thresholds for {len(thresholds) // 2} vehicles")
        
        self.thresholds = thresholds
        return thresholds
    
    def get_threshold(
        self,
        vehicle_id: str,
        state: str
    ) -> Tuple[float, float]:
        """
        Get threshold for specific vehicle and state.
        
        Args:
            vehicle_id: Vehicle identifier
            state: State name ("stationary_on" or "ign_off")
        
        Returns:
            Tuple of (step_threshold, cumulative_threshold)
        """
        key = (vehicle_id, state)
        
        if key in self.thresholds:
            return self.thresholds[key]
        
        # Fallback to minimum thresholds
        logger.warning(f"No threshold found for {key}, using minimum values")
        
        if state == "stationary_on":
            return (
                self.config.thresholds_stationary_on.min_step_gal,
                self.config.thresholds_stationary_on.min_cumulative_gal
            )
        else:
            return (
                self.config.thresholds_ignition_off.min_step_gal,
                self.config.thresholds_ignition_off.min_cumulative_gal
            )
    
    def compute_noise_envelope(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute noise envelope statistics for analysis.
        
        Args:
            df: DataFrame with fuel data
        
        Returns:
            DataFrame with noise statistics per vehicle and state
        """
        results = []
        
        for vid, vehicle_data in df.groupby("vehicle_id"):
            for state_name, state_col in [("stationary_on", "stationary_on"), ("ign_off", "ign_off")]:
                fuel_series = vehicle_data.loc[vehicle_data[state_col], "dfuel"]
                
                if fuel_series.empty:
                    continue
                
                sigma = robust_sigma_mad(fuel_series)
                quantiles = fuel_series.quantile([0.01, 0.05, 0.95, 0.99])
                
                results.append({
                    "vehicle_id": vid,
                    "state": state_name,
                    "n_samples": int(fuel_series.notna().sum()),
                    "sigma_mad": float(sigma),
                    "q01": float(quantiles[0.01]),
                    "q05": float(quantiles[0.05]),
                    "q95": float(quantiles[0.95]),
                    "q99": float(quantiles[0.99])
                })
        
        return pd.DataFrame(results)


def compute_noise_thresholds(
    df: pd.DataFrame,
    config: DetectionConfig,
    train_mask: Optional[np.ndarray] = None
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """
    Convenience function for threshold calculation.
    
    Args:
        df: DataFrame with fuel data
        config: Detection configuration
        train_mask: Optional training data mask
    
    Returns:
        Dictionary of thresholds
    """
    calculator = NoiseThresholdCalculator(config)
    return calculator.compute_thresholds(df, train_mask)