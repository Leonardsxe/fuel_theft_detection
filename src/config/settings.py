"""
Configuration dataclasses for fuel theft detection system.
Defines type-safe configuration structures following Python's Zen:
"Explicit is better than implicit"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class StationaryConfig:
    """Configuration for stationary period detection"""
    speed_threshold_kmh: float = 1.0
    gap_limit_on_min: float = 5.0
    gap_limit_off_min: float = 120.0
    min_segment_duration_min: float = 3.0


@dataclass
class ThresholdConfig:
    """Adaptive noise threshold configuration"""
    min_step_gal: float
    min_cumulative_gal: float
    mad_multiplier_step: float
    mad_multiplier_cum: float


@dataclass
class PatternConfig:
    """Detection pattern configuration"""
    name: str
    min_duration_min: float
    max_duration_min: Optional[float] = None
    min_negative_steps: Optional[int] = None
    min_cumulative_gal: Optional[float] = None
    min_points: Optional[int] = None
    min_median_dt_s: Optional[float] = None
    description: str = ""


@dataclass
class PlausibilityConfig:
    """Plausibility filter configuration"""
    max_rate_gpm: float = 2.0
    max_step_gal: float = 50.0
    max_fill_rate_gpm: float = 10.0


@dataclass
class NoiseConfig:
    """Noise filtering configuration"""
    min_noise_dfuel: float = -0.10


@dataclass
class NMSConfig:
    """Non-maximum suppression configuration"""
    iou_threshold: float = 0.5


@dataclass
class ClusteringConfig:
    """Hotspot clustering configuration"""
    eps_meters: float = 60.0
    min_samples: int = 12
    earth_radius_m: float = 6371000.0


@dataclass
class OutlierConfig:
    """Outlier detection configuration"""
    max_drain_rate_gpm: float = 2.0
    max_fill_rate_gpm: float = 10.0
    mad_threshold: float = 5.0


@dataclass
class InterpolationConfig:
    """Gap interpolation configuration"""
    max_gap_points: int = 3


@dataclass
class DetectionConfig:
    """Complete detection configuration"""
    stationary: StationaryConfig
    thresholds_stationary_on: ThresholdConfig
    thresholds_ignition_off: ThresholdConfig
    patterns: Dict[str, PatternConfig]
    plausibility: PlausibilityConfig
    noise: NoiseConfig
    nms: NMSConfig
    clustering: ClusteringConfig
    outliers: OutlierConfig
    interpolation: InterpolationConfig


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    behavioral_lookback_hours: int = 2
    enable_vehicle_normalization: bool = True
    enable_behavioral_features: bool = True
    enable_temporal_features: bool = True
    enable_spatial_features: bool = True


@dataclass
class SplittingConfig:
    """Data splitting configuration"""
    train_ratio: float = 0.80
    strategy: str = "temporal"
    random_seed: int = 42


@dataclass
class PreprocessingConfig:
    """Feature preprocessing configuration"""
    imputation_strategy: str = "constant"
    imputation_fill_value: float = 0.0
    scaler: str = "standard"
    handle_unknown_categories: str = "ignore"


@dataclass
class RandomForestConfig:
    """Random Forest hyperparameters"""
    n_estimators: int = 200
    max_depth: int = 10
    min_samples_split: int = 20
    min_samples_leaf: int = 10
    max_features: str = "sqrt"
    class_weight: str = "balanced_subsample"
    random_state: int = 42
    n_jobs: int = -1
    calibration_method: str = "sigmoid"
    calibration_cv: int = 5


@dataclass
class LogisticRegressionConfig:
    """Logistic Regression hyperparameters"""
    max_iter: int = 1000
    solver: str = "lbfgs"
    C: float = 0.1
    class_weight: str = "balanced"
    random_state: int = 42
    calibration_method: str = "sigmoid"
    calibration_cv: int = 5


@dataclass
class IsolationForestConfig:
    """Isolation Forest hyperparameters"""
    n_estimators: int = 200
    contamination: float = 0.1
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class XGBoostPatternConfig:
    """XGBoost hyperparameters for specific pattern"""
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    min_child_weight: int
    gamma: float
    reg_alpha: float
    reg_lambda: float


@dataclass
class XGBoostConfig:
    """XGBoost configuration"""
    extended_pattern: XGBoostPatternConfig
    short_pattern: XGBoostPatternConfig
    eval_metric: str = "aucpr"
    random_state: int = 42
    tree_method: str = "hist"


@dataclass
class LightGBMConfig:
    """LightGBM hyperparameters"""
    n_estimators: int = 154
    max_depth: int = 8
    num_leaves: int = 83
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 10
    reg_alpha: float = 0.5
    reg_lambda: float = 0.5
    class_weight: str = "balanced"
    random_state: int = 42
    verbose: int = -1


@dataclass
class TuningConfig:
    """Hyperparameter tuning configuration"""
    method: str = "random_search"
    n_iter: int = 20
    cv: int = 3
    scoring: str = "average_precision"
    n_jobs: int = -1
    verbose: int = 1


@dataclass
class PatternModelConfig:
    """Pattern-specific model configuration"""
    model_type: str
    target_fpr: float


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    target_fpr: float = 0.05
    metrics: List[str] = field(default_factory=lambda: [
        "pr_auc", "precision", "recall", "f1_score", "roc_auc"
    ])
    save_pr_curves: bool = True
    save_confusion_matrices: bool = True
    save_feature_importance: bool = True
    save_top_predictions: int = 100


@dataclass
class ModelConfig:
    """Complete model configuration"""
    features: FeatureConfig
    splitting: SplittingConfig
    preprocessing: PreprocessingConfig
    random_forest: RandomForestConfig
    logistic_regression: LogisticRegressionConfig
    isolation_forest: IsolationForestConfig
    xgboost: XGBoostConfig
    lightgbm: LightGBMConfig
    tuning: TuningConfig
    pattern_models: Dict[str, PatternModelConfig]
    evaluation: EvaluationConfig


@dataclass
class DataPaths:
    """Data directory paths"""
    raw: Path
    processed: Path
    events: Path
    models: Path
    reports: Path


@dataclass
class InputPaths:
    """Input file paths"""
    combined_csv: Path
    raw_sources: List[Path]


@dataclass
class OutputPaths:
    """Output file paths"""
    # Events
    events_csv: Path
    events_with_features: Path
    
    # Models
    random_forest_model: Path
    logistic_regression_model: Path
    xgboost_extended: Path
    xgboost_short: Path
    lightgbm_model: Path
    isolation_forest_model: Path
    
    # Model artifacts
    preprocessor: Path
    vehicle_stats: Path
    noise_thresholds: Path
    hotspot_clusters: Path
    pattern_thresholds: Path
    
    # Reports
    overall_metrics: Path
    per_pattern_metrics: Path
    confusion_matrices: Path
    feature_importance: Path
    top_predictions: Path
    pr_curve_overall: Path
    pr_curve_per_pattern: Path
    event_summary: Path
    data_quality_report: Path


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_dir: Path
    log_file: Path
    level: str = "INFO"


@dataclass
class PathConfig:
    """Complete path configuration"""
    data: DataPaths
    input: InputPaths
    output: OutputPaths
    logging: LoggingConfig
    column_mapping: Dict[str, List[str]]


@dataclass
class Config:
    """Master configuration object"""
    detection: DetectionConfig
    model: ModelConfig
    paths: PathConfig