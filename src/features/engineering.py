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
from typing import Optional

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
    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        self.config = config or FeatureConfig()
        self.normalizer = VehicleNormalizer()
        self._train_mask: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        return self.normalizer.is_fitted

    def fit(self, events_df: pd.DataFrame, train_mask: np.ndarray) -> "FeatureEngineer":
        # We defer fitting until after we’ve computed event stats + temporal/spatial/behavioral in transform().
        # Here we only store the mask.
        self._train_mask = train_mask
        return self

    def transform(self, events_df: pd.DataFrame, raw_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if self._train_mask is None:
            # If user forgot to pass fit(), we still allow inference-only transforms (no fitting).
            self._train_mask = np.zeros(len(events_df), dtype=bool)

        df = events_df.copy()

        # 1) Core event stats first (ensures drop_gal/rate_gpm/min_step_gal/duration_min exist)
        df = add_event_statistics(df)

        # 2) Temporal
        df = add_all_temporal_features(df)

        # 3) Spatial
        if getattr(self.config, "include_spatial", True) and raw_df is not None:
            df = add_location_features(df, raw_df)

        # 4) Behavioral
        if getattr(self.config, "include_behavioral", True) and raw_df is not None:
            lookback = getattr(self.config, "lookback_hours", _Defaults.lookback_hours)
            df = add_pre_event_context(df, raw_df, lookback_hours=int(lookback))
            df = add_movement_variability(df, raw_df)

        # 5) Vehicle normalization (fit-on-train-only; then transform)
        if getattr(self.config, "include_normalization", True):
            if not self.normalizer.is_fitted and self._train_mask is not None:
                self.normalizer.fit(df, self._train_mask)
            df = self.normalizer.transform(df, raw_df)

        return df

    def fit_transform(self, events_df: pd.DataFrame, raw_df: Optional[pd.DataFrame], train_mask: np.ndarray) -> pd.DataFrame:
        self.fit(events_df, train_mask)
        return self.transform(events_df, raw_df)
