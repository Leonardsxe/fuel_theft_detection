"""
Feature Engineering Orchestration — function-first pipeline using your config & utils.

Pipeline:
  1) add_event_statistics(events_df, raw_df?) — core event metrics (drop_gal, rate_gpm, min_step_gal, duration_min)
  2) add_all_temporal_features(events_df)
  3) add_location_features(events_df, raw_df)              [if config.include_spatial]
  4) add_pre_event_context(..., lookback_hours)            [if config.include_behavioral]
     add_movement_variability(events_df, raw_df)           [if config.include_behavioral]
  5) VehicleNormalizer — fit on TRAIN ONLY, then transform (adds vehicle z-scores + drop% of avg/max if raw available)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any, List

import numpy as np
import pandas as pd

from src.config.settings import FeatureConfig
from src.features.temporal import add_all_temporal_features
from src.features.spatial import add_location_features
from src.features.behavioral import add_pre_event_context, add_movement_variability
from src.features.normalization import VehicleNormalizer
from src.features.event_features import add_event_statistics


@dataclass
class _Defaults:
    lookback_hours: int = 2


class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.normalizer: Optional[VehicleNormalizer] = None
        self.raw_df: Optional[pd.DataFrame] = None
        self._is_fitted: bool = False
        # Back-compat: legacy consumers might look for fitted norms dict
        self._legacy_norms: Optional[dict] = None

    def is_fitted(self) -> bool:
        """
        True when the engineer itself is fitted AND (if normalization is enabled)
        the normalizer is also fitted. This keeps behavior consistent whether or not
        vehicle normalization is used.
        """
        norm_ok = (self.normalizer is None) or getattr(self.normalizer, "is_fitted", False)
        return bool(self._is_fitted and norm_ok)

    @property
    def fitted_norms(self) -> dict:
        """
        Back-compat alias for code that used to read `self._fitted_norms`.
        Returns per-vehicle stats from the VehicleNormalizer, or {} if unavailable.
        """
        if self.normalizer is not None and hasattr(self.normalizer, "vehicle_stats"):
            return self.normalizer.vehicle_stats
        return {}

    def fit(
            self,
            events_df: pd.DataFrame,
            raw_df: Optional[pd.DataFrame] = None,
            train_mask: Optional[np.ndarray] = None,
            ) -> "FeatureEngineer":
        """Fit any per-vehicle normalization statistics on TRAIN events only."""
        self.raw_df = raw_df
        df = events_df if train_mask is None else events_df.loc[train_mask].copy()

        if getattr(self.config, "enable_vehicle_normalization", True):
            self.normalizer = VehicleNormalizer()
            # Fit on TRAIN events (and optionally raw_df if your normalizer uses raw telemetry)
            self.normalizer.fit(events_df, train_mask=train_mask if train_mask is not None else np.ones(len(events_df), dtype=bool))
            self._legacy_norms = self.normalizer.vehicle_stats
        
        self._is_fitted = True

        return self

    def transform(self, events_df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer.transform() called before fit(). Call fit() or fit_transform() first.")
        if self.raw_df is None:
            raise ValueError("FeatureEngineer has no raw_df. Pass raw_df to fit()/fit_transform().")

        df = events_df.copy()
        # 1) Core event stats first (ensures drop_gal/rate_gpm/min_step_gal/duration_min exist)
        df = add_event_statistics(df, self.raw_df)

        # 2) Temporal
        df = add_all_temporal_features(df)

        # 3) Spatial
        if getattr(self.config, "include_spatial", True) and self.raw_df is not None:
            df = add_location_features(df, self.raw_df)

        # 4) Behavioral
        if getattr(self.config, "include_behavioral", True) and self.raw_df is not None:
            lookback = getattr(self.config, "lookback_hours", _Defaults.lookback_hours)
            df = add_pre_event_context(df, self.raw_df, lookback_hours=int(lookback))
            df = add_movement_variability(df, self.raw_df)

        # 5) Vehicle normalization (fit-on-train-only; then transform)
        if getattr(self.config, "enable_vehicle_normalization", True) and self.normalizer is not None:
            # z-scores by vehicle
            df = self.normalizer.transform(df)
            # percentages vs avg/max tank scale from raw telemetry
            df = self.normalizer.add_relative_features(df, self.raw_df)

        return df

    def fit_transform(
        self,
        events_df: pd.DataFrame,
        raw_df: Optional[pd.DataFrame] = None,
        train_mask: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        self.fit(events_df, raw_df=raw_df, train_mask=train_mask)
        return self.transform(events_df)
