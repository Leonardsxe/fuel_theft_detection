"""
Model Training Pipeline
Purpose: Build and train all theft detection models using typed configs,
preprocessing pipelines, and calibrated probability outputs.

Design Principles:
- Explicit is better than implicit (type-safe configs)
- Single Responsibility (each function does one thing)
- DRY (shared preprocessing pipeline)
- Type safety (type hints throughout)
- Separation of concerns (training logic separate from config)
"""

from __future__ import annotations

import logging
import inspect
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBClassifier = None  # type: ignore

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    LGBMClassifier = None  # type: ignore

from src.config.settings import ModelConfig
from src.utils.io import save_pickle

logger = logging.getLogger(__name__)

# ========================================
# Constants
# ========================================

# OneHotEncoder kwargs (handle sklearn version compatibility)
_ONE_HOT_ENCODER_KWARGS = {"handle_unknown": "ignore"}
if "sparse_output" in inspect.signature(OneHotEncoder.__init__).parameters:
    _ONE_HOT_ENCODER_KWARGS["sparse_output"] = True
else:
    _ONE_HOT_ENCODER_KWARGS["sparse"] = True


# ========================================
# Preprocessing Pipeline
# ========================================

def build_preprocessing_pipeline(
    numeric_cols: List[str],
    binary_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """
    Build sklearn preprocessing pipeline.
    
    Design: Single Responsibility - only builds pipeline, doesn't fit
    
    Args:
        numeric_cols: Numeric feature names
        binary_cols: Binary (0/1) feature names
        categorical_cols: Categorical feature names
    
    Returns:
        Configured ColumnTransformer
    
    Example:
        >>> pipeline = build_preprocessing_pipeline(
        ...     numeric_cols=['drop_gal', 'duration_min'],
        ...     binary_cols=['is_hotspot'],
        ...     categorical_cols=['pattern']
        ... )
        >>> X_transformed = pipeline.fit_transform(X)
    """
    
    transformers = []
    
    # Numeric: impute + scale
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]),
            numeric_cols,
        ))
    else:
        transformers.append(("num", "drop", []))
    
    # Binary: pass through (already 0/1)
    if binary_cols:
        transformers.append(("bin", "passthrough", binary_cols))
    else:
        transformers.append(("bin", "drop", []))
    
    # Categorical: one-hot encode
    if categorical_cols:
        transformers.append((
            "cat",
            OneHotEncoder(**_ONE_HOT_ENCODER_KWARGS),
            categorical_cols,
        ))
    else:
        transformers.append(("cat", "drop", []))
    
    preprocess = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,  # Use sparse matrices for efficiency
    )
    
    return preprocess


def _infer_columns(X: pd.DataFrame) -> tuple[List[str], List[str], List[str]]:
    """
    Infer numeric, binary, and categorical columns from DataFrame.
    
    Args:
        X: Feature DataFrame
    
    Returns:
        Tuple of (numeric_cols, binary_cols, categorical_cols)
    """
    if not isinstance(X, pd.DataFrame):
        # Best effort for arrays
        n_cols = X.shape[1] if hasattr(X, 'shape') else 0
        return list(range(n_cols)), [], []
    
    # Get numeric columns
    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    
    # Binary is subset of numeric: exactly {0, 1} (ignoring NaNs)
    binary_cols = []
    for c in num_cols:
        vals = pd.Series(X[c]).dropna().unique()
        if len(vals) > 0:
            unique_vals = set(pd.Series(vals).astype(float).unique())
            if unique_vals.issubset({0.0, 1.0}):
                binary_cols.append(c)
    
    # Remaining numeric (exclude binary)
    numeric_cols = [c for c in num_cols if c not in binary_cols]
    
    # Categorical columns
    cat_cols = list(X.select_dtypes(include=["object", "category", "string"]).columns)
    
    return numeric_cols, binary_cols, cat_cols


def _dedupe_columns(X: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all column names are unique.
    
    Args:
        X: DataFrame potentially with duplicate columns
    
    Returns:
        DataFrame with unique column names
    
    Note:
        Appends __dupN suffix to duplicates
    """
    if X.columns.is_unique:
        return X
    
    X = X.copy()
    counts: Dict[str, int] = {}
    new_cols = []
    
    for col in X.columns:
        if col in counts:
            counts[col] += 1
            new_cols.append(f"{col}__dup{counts[col]}")
        else:
            counts[col] = 0
            new_cols.append(col)
    
    dup_total = sum(v for v in counts.values())
    X.columns = new_cols
    
    logger.warning(
        f"Detected {dup_total} duplicate feature names. "
        f"Renamed with __dupN suffix: {[col for col, count in counts.items() if count > 0]}"
    )
    
    return X


# ========================================
# Model Training Functions
# ========================================

def _xgb_params_from_config(xgb_cfg, which: str) -> Dict[str, Any]:
    """
    Extract XGBoost parameters from config.
    
    Args:
        xgb_cfg: XGBoost configuration object
        which: 'extended' or 'short' pattern
    
    Returns:
        Dictionary of XGBoost parameters
    
    Raises:
        ValueError: If which is not 'extended' or 'short'
    """
    which = which.lower()
    if which not in ("extended", "short"):
        raise ValueError(f"Unknown pattern selector: {which!r}. Must be 'extended' or 'short'")
    
    patt = getattr(xgb_cfg, f"{which}_pattern", None)
    if patt is None:
        raise ValueError(f"XGBoost {which}_pattern config missing")
    
    params = {
        'n_estimators': int(patt.n_estimators),
        'max_depth': int(patt.max_depth),
        'learning_rate': float(patt.learning_rate),
        'subsample': float(patt.subsample),
        'colsample_bytree': float(patt.colsample_bytree),
        'min_child_weight': int(patt.min_child_weight),
        'gamma': float(patt.gamma),
        'reg_alpha': float(patt.reg_alpha),
        'reg_lambda': float(patt.reg_lambda),
        'eval_metric': getattr(xgb_cfg, "eval_metric", "aucpr"),
        'random_state': int(getattr(xgb_cfg, "random_state", 42)),
        'tree_method': str(getattr(xgb_cfg, "tree_method", "hist")),
    }
    
    return params


def train_random_forest(
    preprocess: ColumnTransformer,
    X: pd.DataFrame,
    y: np.ndarray,
    config: ModelConfig
) -> Pipeline:
    """
    Train calibrated Random Forest classifier.
    
    Args:
        preprocess: Fitted ColumnTransformer
        X: Features
        y: Labels
        config: Model configuration
    
    Returns:
        Trained pipeline with calibration
    """
    rf_cfg = config.random_forest
    
    rf = RandomForestClassifier(
        n_estimators=rf_cfg.n_estimators,
        max_depth=rf_cfg.max_depth,
        min_samples_split=rf_cfg.min_samples_split,
        min_samples_leaf=rf_cfg.min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=rf_cfg.random_state,
        n_jobs=rf_cfg.n_jobs,
    )
    
    # Calibrate probabilities
    rf_calibrated = CalibratedClassifierCV(
        rf,
        method=rf_cfg.calibration_method,
        cv=rf_cfg.calibration_cv,
    )
    
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", rf_calibrated),
    ])
    
    pipe.fit(X, y)
    logger.info("✓ Trained Random Forest (calibrated)")
    
    return pipe


def train_logistic_regression(
    preprocess: ColumnTransformer,
    X: pd.DataFrame,
    y: np.ndarray,
    config: ModelConfig
) -> Pipeline:
    """
    Train calibrated Logistic Regression classifier.
    
    Args:
        preprocess: Fitted ColumnTransformer
        X: Features
        y: Labels
        config: Model configuration
    
    Returns:
        Trained pipeline with calibration
    """
    lr_cfg = config.logistic_regression
    
    lr = LogisticRegression(
        max_iter=lr_cfg.max_iter,
        C=lr_cfg.C,
        class_weight=lr_cfg.class_weight,
        random_state=lr_cfg.random_state,
        solver=lr_cfg.solver,
    )
    
    # Calibrate probabilities
    lr_calibrated = CalibratedClassifierCV(
        lr,
        method=lr_cfg.calibration_method,
        cv=lr_cfg.calibration_cv,
    )
    
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", lr_calibrated),
    ])
    
    pipe.fit(X, y)
    logger.info("✓ Trained Logistic Regression (calibrated)")
    
    return pipe


def train_xgboost(
    preprocess: ColumnTransformer,
    X: pd.DataFrame,
    y: np.ndarray,
    config: ModelConfig,
    which: Optional[str] = None
) -> Pipeline:
    """
    Train XGBoost classifier with pattern-specific parameters.
    
    Args:
        preprocess: Fitted ColumnTransformer
        X: Features
        y: Labels
        config: Model configuration
        which: 'extended' or 'short' for pattern-specific params
    
    Returns:
        Trained pipeline
    
    Raises:
        ImportError: If XGBoost not installed
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    
    xgb_cfg = config.xgboost
    
    # Get pattern-specific params
    which_final = (which or "extended").lower()
    params = _xgb_params_from_config(xgb_cfg, which_final)
    
    # Compute class balance
    y_np = np.asarray(y)
    n_pos = int((y_np == 1).sum())
    n_neg = int((y_np == 0).sum())
    scale_pos_weight = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0
    
    xgb = XGBClassifier(
        **params,
        n_jobs=getattr(xgb_cfg, "n_jobs", -1),
        scale_pos_weight=scale_pos_weight,
    )
    
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", xgb),
    ])
    
    pipe.fit(X, y)
    logger.info(f"✓ Trained XGBoost ({which_final} pattern)")
    
    return pipe


def train_lightgbm(
    preprocess: ColumnTransformer,
    X: pd.DataFrame,
    y: np.ndarray,
    config: ModelConfig
) -> Pipeline:
    """
    Train LightGBM classifier.
    
    Args:
        preprocess: Fitted ColumnTransformer
        X: Features
        y: Labels
        config: Model configuration
    
    Returns:
        Trained pipeline
    
    Raises:
        ImportError: If LightGBM not installed
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
    
    lgb_cfg = config.lightgbm
    
    lgb = LGBMClassifier(
        n_estimators=lgb_cfg.n_estimators,
        max_depth=lgb_cfg.max_depth,
        num_leaves=lgb_cfg.num_leaves,
        min_child_samples=lgb_cfg.min_child_samples,
        learning_rate=lgb_cfg.learning_rate,
        subsample=lgb_cfg.subsample,
        colsample_bytree=lgb_cfg.colsample_bytree,
        reg_alpha=lgb_cfg.reg_alpha,
        reg_lambda=lgb_cfg.reg_lambda,
        random_state=lgb_cfg.random_state,
        n_jobs=lgb_cfg.n_jobs,
        objective="binary",
        verbose=-1,  # Suppress output
    )
    
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", lgb),
    ])
    
    pipe.fit(X, y)
    logger.info("✓ Trained LightGBM")
    
    return pipe


class IsoForestWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper to expose IsolationForest as probability-producing classifier.
    
    Design: Adapter pattern to make sklearn's IsolationForest compatible
            with our probability interface.
    
    Note:
        predict_proba returns normalized anomaly scores where
        higher values = more anomalous = more theft-like
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.05,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model_: Optional[IsolationForest] = None
    
    def fit(self, X: Any, y: Optional[np.ndarray] = None) -> "IsoForestWrapper":
        """Fit IsolationForest (y is ignored - unsupervised)."""
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X)
        return self
    
    def decision_scores_(self, X: Any) -> np.ndarray:
        """
        Compute anomaly scores normalized to [0, 1].
        
        Higher = more anomalous/theft-like
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
        
        # sklearn: higher decision_function = more normal (inlier)
        # We flip: higher = more anomalous
        raw = self.model_.decision_function(X)
        flipped = -raw
        
        # Normalize to [0, 1]
        mn, mx = np.min(flipped), np.max(flipped)
        if mx - mn < 1e-12:
            return np.zeros_like(flipped)
        
        return (flipped - mn) / (mx - mn)
    
    def predict_proba(self, X: Any) -> np.ndarray:
        """Return 2-column probabilities."""
        scores = self.decision_scores_(X)
        return np.vstack([1.0 - scores, scores]).T
    
    def predict(self, X: Any) -> np.ndarray:
        """Binary predictions (threshold = 0.5)."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_isolation_forest(
    preprocess: ColumnTransformer,
    X: pd.DataFrame,
    y: np.ndarray,
    config: ModelConfig
) -> Pipeline:
    """
    Train Isolation Forest (unsupervised baseline).
    
    Args:
        preprocess: Fitted ColumnTransformer
        X: Features
        y: Labels (ignored - unsupervised)
        config: Model configuration
    
    Returns:
        Trained pipeline
    """
    if_cfg = config.isolation_forest
    
    iso = IsoForestWrapper(
        n_estimators=if_cfg.n_estimators,
        contamination=if_cfg.contamination,
        random_state=if_cfg.random_state,
        n_jobs=if_cfg.n_jobs,
    )
    
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", iso),
    ])
    
    pipe.fit(X)  # y not used
    logger.info("✓ Trained Isolation Forest (unsupervised)")
    
    return pipe


# ========================================
# Model Trainer Class
# ========================================

class ModelTrainer:
    """
    Orchestrates training of multiple models.
    
    Design Principles:
    - Single Responsibility: Coordinates model training
    - Explicit configuration: Type-safe ModelConfig
    - DRY: Shared preprocessing pipeline
    
    Attributes:
        config: Model configuration
        preprocessor: Fitted ColumnTransformer
        models: Dictionary of trained models
    
    Example:
        >>> trainer = ModelTrainer(config)
        >>> models = trainer.train_all(X_train, y_train)
        >>> predictions = trainer.predict(X_test)
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Model configuration object
        """
        self.config = config
        self.preprocessor: Optional[ColumnTransformer] = None
        self.models: Dict[str, Pipeline] = {}
    
    def fit_preprocessor(self, X: pd.DataFrame) -> None:
        """
        Build and fit preprocessor from data.
        
        Args:
            X: Feature DataFrame
        """
        num_cols, bin_cols, cat_cols = _infer_columns(X)
        self.preprocessor = build_preprocessing_pipeline(num_cols, bin_cols, cat_cols)
        
        logger.info(
            f"Preprocessor configured: "
            f"numeric={len(num_cols)}, binary={len(bin_cols)}, categorical={len(cat_cols)}"
        )
    
    def train_all(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Dict[str, Pipeline]:
        """
        Train all configured models.
        
        Args:
            X_train: Training features
            y_train: Training labels
        
        Returns:
            Dictionary of trained models {name: pipeline}
        """
        # Ensure unique column names
        X_train = _dedupe_columns(X_train)
        
        # Fit preprocessor if not already fitted
        if self.preprocessor is None:
            self.fit_preprocessor(X_train)
        
        self.models = {}
        
        # Logistic Regression (calibrated)
        try:
            self.models["logreg_cal"] = train_logistic_regression(
                self.preprocessor, X_train, y_train, self.config
            )
        except Exception as e:
            logger.error(f"Failed to train Logistic Regression: {e}")
        
        # Random Forest (calibrated)
        try:
            self.models["rf_cal"] = train_random_forest(
                self.preprocessor, X_train, y_train, self.config
            )
        except Exception as e:
            logger.error(f"Failed to train Random Forest: {e}")
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            try:
                self.models["xgb"] = train_xgboost(
                    self.preprocessor, X_train, y_train, self.config
                )
            except Exception as e:
                logger.error(f"Failed to train XGBoost: {e}")
        else:
            logger.warning("XGBoost not available - skipping")
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            try:
                self.models["lgbm"] = train_lightgbm(
                    self.preprocessor, X_train, y_train, self.config
                )
            except Exception as e:
                logger.error(f"Failed to train LightGBM: {e}")
        else:
            logger.warning("LightGBM not available - skipping")
        
        # Isolation Forest (unsupervised baseline)
        try:
            self.models["iso_forest"] = train_isolation_forest(
                self.preprocessor, X_train, y_train, self.config
            )
        except Exception as e:
            logger.error(f"Failed to train Isolation Forest: {e}")
        
        logger.info(f"✓ Trained {len(self.models)} models: {', '.join(self.models.keys())}")
        
        return self.models
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate probability predictions from all models.
        
        Args:
            X: Feature matrix
        
        Returns:
            Dictionary of predictions {name: probabilities}
        """
        predictions: Dict[str, np.ndarray] = {}
        
        for name, model in self.models.items():
            try:
                predictions[name] = model.predict_proba(X)[:, 1]
            except Exception as e:
                logger.warning(f"Could not get predictions from {name}: {e}")
        
        return predictions
    
    def save_all(self, output_dir: str) -> None:
        """
        Save all trained models.
        
        Args:
            output_dir: Directory to save models
        """
        from pathlib import Path
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            path = output_dir / f"{name}.pkl"
            save_pickle(model, path)
            logger.info(f"Saved {name} → {path}")