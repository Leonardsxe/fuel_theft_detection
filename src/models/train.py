"""
Model Training Pipeline
Purpose: Build and train all theft detection models using typed configs,
preprocessing pipelines, and a calibrated / probability-producing interface.
"""
 
import numpy as np
import pandas as pd

from typing import Dict, Tuple, List, Optional, Any
import logging
import inspect
 
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

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
 
from src.config.settings import ModelConfig
from src.utils.io import save_pickle

logger = logging.getLogger(__name__)

_ONE_HOT_ENCODER_KWARGS = {"handle_unknown": "ignore"}
# scikit-learn 1.2+ renamed `sparse` to `sparse_output`; detect dynamically
if "sparse_output" in inspect.signature(OneHotEncoder.__init__).parameters:
    _ONE_HOT_ENCODER_KWARGS["sparse_output"] = True
else:
    _ONE_HOT_ENCODER_KWARGS["sparse"] = True


def build_preprocessing_pipeline(
    numeric_cols: List[str],
    binary_cols: List[str],
    categorical_cols: List[str]
) -> ColumnTransformer:

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric_cols,
            )
            if numeric_cols
            else ("num", "drop", []),
            ("bin", "passthrough", binary_cols) if binary_cols else ("bin", "drop", []),
            (
                "cat",
                OneHotEncoder(**_ONE_HOT_ENCODER_KWARGS),
                categorical_cols,
            )
            if categorical_cols
            else ("cat", "drop", []),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    return preprocess
 
def _infer_columns(X: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Infer numeric, binary, and categorical columns from a pandas frame."""
    if not isinstance(X, pd.DataFrame):
        # best effort: treat all as numeric
        return list(range(X.shape[1])), [], []

    num_cols = list(X.select_dtypes(include=[np.number]).columns)

    # binary ⊆ numeric: exactly {0,1} (ignoring NaNs)
    binary_cols = []
    for c in num_cols:
        vals = pd.Series(X[c]).dropna().unique()
        if len(vals) > 0 and set(pd.Series(vals).astype(float).unique()).issubset({0.0, 1.0}):
            binary_cols.append(c)

    numeric_cols = [c for c in num_cols if c not in binary_cols]
    cat_cols = list(X.select_dtypes(include=["object", "category", "string"]).columns)

    return numeric_cols, binary_cols, cat_cols

def _dedupe_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Ensure all column names are unique by appending __dupN to duplicates."""
    
    if X.columns.is_unique:
        return X
    
    X = X.copy()
    counts = {}
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
    
    logger.warning(f"Detected duplicate feature names; renamed {dup_total} duplicates with __dupN suffix.")
    logger.warning(f'Duplicate feature names: {[col for col, count in counts.items() if count > 0]}')

    return X

def _subset_numpy(y, mask):
    """Return y[mask] with robust handling for Series/ndarray mask and y."""
    y_np = np.asarray(y)
    m = mask.values if hasattr(mask, "values") else np.asarray(mask)
    return y_np[m]

def _xgb_params_from_pattern_cfg(xgb_cfg, which: str) -> dict:
    """
    Build XGBoost params dict from pattern-specific config ('extended' or 'short')
    plus general xgb settings (eval_metric, random_state, tree_method).
    """
    which = which.lower()
    if which not in ("extended", "short"):
        raise ValueError(f"Unknown pattern selector: {which!r}")
    patt = getattr(xgb_cfg, f"{which}_pattern", None)
    if patt is None:
        raise ValueError(f"XGBoost {which}_pattern config missing in settings.")
    params = dict(
        n_estimators=int(patt.n_estimators),
        max_depth=int(patt.max_depth),
        learning_rate=float(patt.learning_rate),
        subsample=float(patt.subsample),
        colsample_bytree=float(patt.colsample_bytree),
        min_child_weight=int(patt.min_child_weight),
        gamma=float(patt.gamma),
        reg_alpha=float(patt.reg_alpha),
        reg_lambda=float(patt.reg_lambda),
        eval_metric=getattr(xgb_cfg, "eval_metric", "aucpr"),
        random_state=int(getattr(xgb_cfg, "random_state", 42)),
        tree_method=str(getattr(xgb_cfg, "tree_method", "hist")),
    )
    return params


 
def train_random_forest(preprocess: ColumnTransformer, X, y, config: ModelConfig) -> Pipeline:
    """Train a calibrated Random Forest classifier."""
    
    rf_cfg = config.random_forest
    calib = config.logistic_regression.calibration_method if hasattr(config, "logistic_regression") else "sigmoid"
    calib_cv = config.logistic_regression.calibration_cv if hasattr(config, "logistic_regression") else 5
    
    rf = RandomForestClassifier(
        n_estimators=rf_cfg.n_estimators,
        max_depth=rf_cfg.max_depth,
        min_samples_split=rf_cfg.min_samples_split,
        min_samples_leaf=rf_cfg.min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=rf_cfg.random_state,
        n_jobs=rf_cfg.n_jobs,
    )
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("clf", CalibratedClassifierCV(rf, method=calib, cv=calib_cv)),
        ]
    )
    pipe.fit(X, y)

    return pipe
 
 
def train_logistic_regression(preprocess: ColumnTransformer, X, y, config: ModelConfig) -> Pipeline:
    """Train a calibrated Logistic Regression classifier."""
    lr_cfg = config.logistic_regression
    lr = LogisticRegression(
        max_iter=lr_cfg.max_iter,
        C=lr_cfg.C,
        class_weight=lr_cfg.class_weight,
        random_state=lr_cfg.random_state,
        solver=lr_cfg.solver,
    )
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("clf", CalibratedClassifierCV(lr, method=lr_cfg.calibration_method, cv=lr_cfg.calibration_cv)),
        ]
    )
    pipe.fit(X, y)
    
    return pipe
 
 
def train_xgboost(preprocess: ColumnTransformer, X, y, config: ModelConfig, which: Optional[str] = None) -> Pipeline:
    """Train an XGBoost classifier using pattern-specific params if `which` provided."""
    
    xgb_cfg = config.xgboost
    # choose params from pattern config or default to 'extended'
    which_final = (which or "extended").lower()
    params = _xgb_params_from_pattern_cfg(xgb_cfg, which_final)
    # compute scale_pos_weight from this subset's class balance
    y_np = np.asarray(y)
    n_pos = int((y_np == 1).sum())
    n_neg = int((y_np == 0).sum())
    scale_pos_weight = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0
    xgb = XGBClassifier(
        **params,
        n_jobs=getattr(getattr(config, "xgboost", None), "n_jobs", -1),
        scale_pos_weight=scale_pos_weight,
    )
    pipe = Pipeline(steps=[("preprocess", preprocess), ("clf", xgb)])
    pipe.fit(X, y)
    return pipe
 
 
def train_lightgbm(preprocess: ColumnTransformer, X, y, config: ModelConfig) -> Pipeline:
    """Train a LightGBM classifier."""
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
    )
    pipe = Pipeline(steps=[("preprocess", preprocess), ("clf", lgb)])
    pipe.fit(X, y)
    return pipe


class IsoForestWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper to expose IsolationForest as a probability-like classifier.
    predict_proba returns a 2-col array where col1 = 1 - score, col2 = score,
    and score is normalized anomaly score in [0,1] with higher = more theft-like.
    """
    def __init__(self, n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model_: Optional[IsolationForest] = None

    def fit(self, X, y=None):
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X)
        return self

    def decision_scores_(self, X) -> np.ndarray:
        # In sklearn, larger decision_function => more normal (inlier).
        # We flip and normalize so larger => more anomalous/theft-like.
        raw = self.model_.decision_function(X)  # higher = normal
        flipped = -raw
        mn, mx = np.min(flipped), np.max(flipped)
        if mx - mn < 1e-12:
            return np.zeros_like(flipped)
        return (flipped - mn) / (mx - mn)

    def predict_proba(self, X) -> np.ndarray:
        s = self.decision_scores_(X)
        return np.vstack([1.0 - s, s]).T

    def predict(self, X) -> np.ndarray:
        # default 0.5 threshold
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_isolation_forest(preprocess: ColumnTransformer, X, y, config: ModelConfig) -> Pipeline:
    """Unsupervised baseline trained on features only; uses probability-like scores."""
    if_cfg = config.isolation_forest
    iso = IsoForestWrapper(
        n_estimators=if_cfg.n_estimators,
        contamination=if_cfg.contamination,
        random_state=if_cfg.random_state,
        n_jobs=if_cfg.n_jobs,
    )
    pipe = Pipeline(steps=[("preprocess", preprocess), ("clf", iso)])
    pipe.fit(X)  # y not used
    return pipe
 
 
class ModelTrainer:
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.preprocessor: Optional[ColumnTransformer] = None
        self.models: Dict[str, Any] = {}
 

    def fit_preprocessor(self, X: pd.DataFrame) -> None:
        """Fit (or build) preprocessor using inferred column types."""
        num_cols, bin_cols, cat_cols = _infer_columns(X)
        self.preprocessor = build_preprocessing_pipeline(num_cols, bin_cols, cat_cols)
        # Pipelines will own the preprocessor; nothing to fit here explicitly.
        logger.info(
            f"Preprocessor columns — numeric: {len(num_cols)}, binary: {len(bin_cols)}, categorical: {len(cat_cols)}"
        )
 
    def train_all(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Pipeline]:
        """Train all configured models using a shared preprocessing pipeline."""

        X_train = _dedupe_columns(X_train)

        if self.preprocessor is None:
            self.fit_preprocessor(X_train)
        self.models = {}
        # Logistic Regression (calibrated)
        self.models["logreg_cal"] = train_logistic_regression(self.preprocessor, X_train, y_train, self.config)
        # Random Forest (calibrated)
        self.models["rf_cal"] = train_random_forest(self.preprocessor, X_train, y_train, self.config)
        # XGBoost
        if XGBOOST_AVAILABLE:
            self.models["xgb"] = train_xgboost(self.preprocessor, X_train, y_train, self.config)
        else:
            logger.warning("XGBoost not available; skipping.")
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            self.models["lgbm"] = train_lightgbm(self.preprocessor, X_train, y_train, self.config)
        else:
            logger.warning("LightGBM not available; skipping.")
        # Isolation Forest baseline
        self.models["iso_forest"] = train_isolation_forest(self.preprocessor, X_train, y_train, self.config)
        logger.info(f"Trained models: {list(self.models.keys())}")
        return self.models
 
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate probability-like predictions from all models.
        """

        for name, model in self.models.items():
            try:
                if hasattr(model, "predict_proba"):
                    predictions[name] = model.predict_proba(X)[:, 1]
                else:
                    # Last resort: decision_function normalization (shouldn't trigger with pipelines above)
                    scores = model.decision_function(X)  # type: ignore[attr-defined]
                    scores = np.asarray(scores, dtype=float)
                    mn, mx = np.min(scores), np.max(scores)
                    predictions[name] = (scores - mn) / (mx - mn + 1e-12) if mx > mn else np.zeros_like(scores)
            except Exception as e:
                logger.warning(f"Could not get predictions from {name}: {e}")
         
        return predictions

    def save_all(self, output_dir) -> None:
        """Persist all trained models as pickles."""
        out = getattr(self.config, "paths", None)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            path = output_dir / f"{name}.pkl"
            save_pickle(model, path)
            logger.info(f"Saved {name} → {path}")
