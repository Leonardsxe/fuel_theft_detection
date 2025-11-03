"""
Base Model Interface
Purpose: Abstract base classes for fuel theft detection models and ensembles,
with sklearn-compatible APIs and robust probability semantics.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)

# -----------------------------
# helpers
# -----------------------------

def _to_2col_proba(p: np.ndarray) -> np.ndarray:
    """
    Ensure probability-like output is shape (n, 2).
    Accepts:
      - (n,) as positive-class probability -> expands to (1-p, p)
      - (n, 1) similarly
      - (n, 2) returned as-is
    """
    p = np.asarray(p)
    if p.ndim == 2 and p.shape[1] == 2:
        return p
    if p.ndim == 2 and p.shape[1] == 1:
        pos = p.ravel()
    elif p.ndim == 1:
        pos = p
    else:
        raise ValueError(f"Unsupported probability array shape: {p.shape}")
    pos = np.clip(pos, 0.0, 1.0)
    return np.vstack([1.0 - pos, pos]).T


def _scores_to_pos_proba(scores: np.ndarray) -> np.ndarray:
    """
    Map arbitrary decision scores to [0, 1] with min-max normalization.
    If scores are constant, returns zeros.
    """
    s = np.asarray(scores, dtype=float)
    mn, mx = np.min(s), np.max(s)
    if mx - mn < 1e-12:
        return np.zeros_like(s)
    return (s - mn) / (mx - mn)


# -----------------------------
# Base interface
# -----------------------------

class BaseTheftDetector(BaseEstimator, ClassifierMixin, ABC):
    """
    Abstract base class for fuel theft detection models with sklearn semantics.
    Implementations should set self.is_fitted = True and self.classes_ after fit().
    """

    def __init__(self) -> None:
        self.model: Any = None
        self.is_fitted: bool = False
        self.feature_names: Optional[List[str]] = None
        # Binary classification by design:
        self.classes_: np.ndarray = np.array([0, 1], dtype=int)

    # ---- required API ----
    @abstractmethod
    def fit(self, X: Any, y: Optional[np.ndarray] = None) -> "BaseTheftDetector":
        """Train the model and set self.is_fitted = True."""
        raise NotImplementedError

    # ---- unified probability interface ----
    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Return probabilities shape (n, 2). Tries, in order:
          - underlying model.predict_proba
          - underlying model.decision_function -> min-max normalized to [0,1]
        """
        self._ensure_fitted()
        m = self.model if self.model is not None else self
        # 1) native predict_proba
        if hasattr(m, "predict_proba"):
            try:
                p = m.predict_proba(X)
                return _to_2col_proba(p)
            except Exception as e:
                logger.debug(f"predict_proba failed on {m}: {e}")
        # 2) decision_function -> normalize
        if hasattr(m, "decision_function"):
            scores = m.decision_function(X)  # type: ignore[attr-defined]
            pos = _scores_to_pos_proba(scores)
            return _to_2col_proba(pos)
        # 3) last resort: predict -> cast to {0,1}
        if hasattr(m, "predict"):
            labels = np.asarray(m.predict(X)).astype(int)
            return _to_2col_proba(labels)
        raise AttributeError("Neither predict_proba nor decision_function nor predict available.")

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        """Binary predictions from positive-class probability with a threshold."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= float(threshold)).astype(int)

    # ---- interpretability ----
    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Return a DataFrame with columns ['feature', 'importance'].
        Implementations should handle their own internal extraction (coef_, feature_importances_, etc.).
        """
        raise NotImplementedError

    # ---- sklearn plumbing ----
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Expose parameters for sklearn compatibility.
        Base class has no tunable hyperparameters; subclasses should override if needed.
        """
        return {"model": self.model}

    def set_params(self, **params: Any) -> "BaseTheftDetector":
        """Set arbitrary attributes for sklearn compatibility."""
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # ---- utils ----
    def _ensure_fitted(self) -> None:
        if not getattr(self, "is_fitted", False):
            raise RuntimeError("This detector is not fitted yet. Call fit() first.")


# -----------------------------
# Ensemble base
# -----------------------------

class BaseEnsembleDetector(BaseTheftDetector):
    """
    Generic ensemble that blends probabilities from multiple base models.
    Parameters
    ----------
    base_models : list
        List of fitted or unfitted models (sklearn-compatible). Each must implement predict_proba or decision_function.
    weights : list or None
        Optional per-model weights (same length as base_models). Defaults to uniform.
    combine : {"mean","max","median","rank_average"}
        Combination rule applied to positive-class probabilities.
    refit_base_models : bool
        If True, calls fit(X, y) on each base model during ensemble fit.
    """

    def __init__(
        self,
        base_models: Sequence[Any],
        weights: Optional[Sequence[float]] = None,
        combine: str = "mean",
        refit_base_models: bool = True,
    ) -> None:
        super().__init__()
        self.base_models: List[Any] = list(base_models)
        self.weights: Optional[np.ndarray] = (
            np.asarray(weights, dtype=float) if weights is not None else None
        )
        self.combine: str = str(combine).lower()
        self.refit_base_models: bool = bool(refit_base_models)

        if self.weights is not None:
            if len(self.weights) != len(self.base_models):
                raise ValueError("weights must have same length as base_models")
            if np.sum(self.weights) <= 0:
                raise ValueError("weights must sum to a positive value")

    def fit(self, X: Any, y: Optional[np.ndarray] = None) -> "BaseEnsembleDetector":
        for m in self.base_models:
            if self.refit_base_models and hasattr(m, "fit"):
                m.fit(X, y)
        self.is_fitted = True
        return self

    def _stack_pos_probs(self, X: Any) -> np.ndarray:
        """Return an (n_models, n_samples) matrix of positive-class probabilities."""
        probs: List[np.ndarray] = []
        for m in self.base_models:
            if hasattr(m, "predict_proba"):
                p = m.predict_proba(X)
                pos = _to_2col_proba(p)[:, 1]
            elif hasattr(m, "decision_function"):
                pos = _scores_to_pos_proba(m.decision_function(X))  # type: ignore[attr-defined]
            else:
                pos = np.asarray(m.predict(X)).astype(float)  # type: ignore[attr-defined]
            probs.append(pos)
        return np.vstack(probs)  # shape: (M, N)

    def predict_proba(self, X: Any) -> np.ndarray:
        self._ensure_fitted()
        P = self._stack_pos_probs(X)  # (M, N)
        if self.combine == "max":
            pos = np.max(P, axis=0)
        elif self.combine == "median":
            pos = np.median(P, axis=0)
        elif self.combine == "rank_average":
            # average ranks across models, then min-max to [0,1]
            ranks = np.argsort(np.argsort(P, axis=1), axis=1).astype(float)
            avg_rank = np.mean(ranks, axis=0)
            pos = _scores_to_pos_proba(avg_rank)
        else:
            # mean (weighted if provided)
            if self.weights is None:
                pos = np.mean(P, axis=0)
            else:
                w = self.weights / np.sum(self.weights)
                pos = np.average(P, axis=0, weights=w)
        return _to_2col_proba(pos)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Average feature importances across base models that expose them.
        If models operate on different transformed spaces, names may not align; caller should handle mapping if needed.
        """
        frames: List[pd.DataFrame] = []
        for m in self.base_models:
            gi = None
            # Allow both custom detectors and raw sklearn models inside pipelines
            if hasattr(m, "get_feature_importance"):
                try:
                    gi = m.get_feature_importance()
                except Exception as e:
                    logger.debug(f"Feature importance unavailable for {m}: {e}")
            elif hasattr(m, "feature_importances_"):
                imp = np.asarray(m.feature_importances_)
                gi = pd.DataFrame({"feature": [f"f{i}" for i in range(len(imp))], "importance": imp})
            elif hasattr(m, "coef_"):
                coef = np.asarray(m.coef_)
                coef = coef.ravel() if coef.ndim > 1 else coef
                gi = pd.DataFrame({"feature": [f"f{i}" for i in range(len(coef))], "importance": coef})
            if gi is not None:
                frames.append(gi[["feature", "importance"]].copy())

        if not frames:
            return pd.DataFrame(columns=["feature", "importance"])
        combined = pd.concat(frames, ignore_index=True)
        out = (
            combined.groupby("feature", as_index=False)["importance"]
            .mean()
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        return out