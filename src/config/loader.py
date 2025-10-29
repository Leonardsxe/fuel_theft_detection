"""
Configuration loader from YAML files.
Converts YAML to type-safe dataclass instances.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

from src.config.settings import (
    Config, DetectionConfig, ModelConfig, PathConfig,
    StationaryConfig, ThresholdConfig, PatternConfig,
    PlausibilityConfig, NoiseConfig, NMSConfig, ClusteringConfig,
    OutlierConfig, InterpolationConfig,
    FeatureConfig, SplittingConfig, PreprocessingConfig,
    RandomForestConfig, LogisticRegressionConfig, IsolationForestConfig,
    XGBoostConfig, XGBoostPatternConfig, LightGBMConfig, TuningConfig,
    PatternModelConfig, EvaluationConfig,
    DataPaths, InputPaths, OutputPaths, LoggingConfig
)


def load_yaml(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file"""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_detection_config(config_path: Path) -> DetectionConfig:
    """Load detection configuration from YAML"""
    cfg = load_yaml(config_path)
    
    # Parse stationary config
    stationary = StationaryConfig(**cfg['stationary'])
    
    # Parse threshold configs
    thresholds_on = ThresholdConfig(**cfg['thresholds']['stationary_on'])
    thresholds_off = ThresholdConfig(**cfg['thresholds']['ignition_off'])
    
    # Parse pattern configs
    patterns = {
        name: PatternConfig(name=pattern['name'], **{k: v for k, v in pattern.items() if k != 'name'})
        for name, pattern in cfg['patterns'].items()
    }
    
    # Parse other configs
    plausibility = PlausibilityConfig(**cfg['plausibility'])
    noise = NoiseConfig(**cfg['noise'])
    nms = NMSConfig(**cfg['nms'])
    clustering = ClusteringConfig(**cfg['clustering'])
    outliers = OutlierConfig(**cfg['outliers'])
    interpolation = InterpolationConfig(**cfg['interpolation'])
    
    return DetectionConfig(
        stationary=stationary,
        thresholds_stationary_on=thresholds_on,
        thresholds_ignition_off=thresholds_off,
        patterns=patterns,
        plausibility=plausibility,
        noise=noise,
        nms=nms,
        clustering=clustering,
        outliers=outliers,
        interpolation=interpolation
    )


def load_model_config(config_path: Path) -> ModelConfig:
    """Load model configuration from YAML"""
    cfg = load_yaml(config_path)
    
    # Parse feature config
    features = FeatureConfig(**cfg['features'])
    
    # Parse splitting config
    splitting = SplittingConfig(**cfg['splitting'])
    
    # Parse preprocessing config
    preprocessing = PreprocessingConfig(**cfg['preprocessing'])
    
    # Parse model configs
    random_forest = RandomForestConfig(**cfg['random_forest'])
    logistic_regression = LogisticRegressionConfig(**cfg['logistic_regression'])
    isolation_forest = IsolationForestConfig(**cfg['isolation_forest'])
    
    # Parse XGBoost config
    xgb_extended = XGBoostPatternConfig(**cfg['xgboost']['extended_pattern'])
    xgb_short = XGBoostPatternConfig(**cfg['xgboost']['short_pattern'])
    xgboost = XGBoostConfig(
        extended_pattern=xgb_extended,
        short_pattern=xgb_short,
        eval_metric=cfg['xgboost']['eval_metric'],
        random_state=cfg['xgboost']['random_state'],
        tree_method=cfg['xgboost']['tree_method']
    )
    
    # Parse LightGBM config
    lightgbm = LightGBMConfig(**cfg['lightgbm'])
    
    # Parse tuning config
    tuning = TuningConfig(**cfg['tuning'])
    
    # Parse pattern-specific model configs
    pattern_models = {
        name: PatternModelConfig(**pattern_cfg)
        for name, pattern_cfg in cfg['pattern_models'].items()
    }
    
    # Parse evaluation config
    evaluation = EvaluationConfig(**cfg['evaluation'])
    
    return ModelConfig(
        features=features,
        splitting=splitting,
        preprocessing=preprocessing,
        random_forest=random_forest,
        logistic_regression=logistic_regression,
        isolation_forest=isolation_forest,
        xgboost=xgboost,
        lightgbm=lightgbm,
        tuning=tuning,
        pattern_models=pattern_models,
        evaluation=evaluation
    )


def load_path_config(config_path: Path) -> PathConfig:
    """Load path configuration from YAML"""
    cfg = load_yaml(config_path)
    
    # Parse data paths
    data = DataPaths(
        raw=Path(cfg['data']['raw']),
        processed=Path(cfg['data']['processed']),
        events=Path(cfg['data']['events']),
        models=Path(cfg['data']['models']),
        reports=Path(cfg['data']['reports'])
    )
    
    # Parse input paths
    input_paths = InputPaths(
        combined_csv=Path(cfg['input']['combined_csv']),
        raw_sources=[Path(p) for p in cfg['input']['raw_sources']]
    )
    
    # Parse output paths
    output = OutputPaths(
        events_csv=Path(cfg['output']['events_csv']),
        events_with_features=Path(cfg['output']['events_with_features']),
        noise_envelopes=Path(cfg['output']['noise_envelopes']),
        random_forest_model=Path(cfg['output']['random_forest_model']),
        logistic_regression_model=Path(cfg['output']['logistic_regression_model']),
        xgboost_extended=Path(cfg['output']['xgboost_extended']),
        xgboost_short=Path(cfg['output']['xgboost_short']),
        lightgbm_model=Path(cfg['output']['lightgbm_model']),
        isolation_forest_model=Path(cfg['output']['isolation_forest_model']),
        preprocessor=Path(cfg['output']['preprocessor']),
        vehicle_stats=Path(cfg['output']['vehicle_stats']),
        noise_thresholds=Path(cfg['output']['noise_thresholds']),
        hotspot_clusters=Path(cfg['output']['hotspot_clusters']),
        pattern_thresholds=Path(cfg['output']['pattern_thresholds']),
        overall_metrics=Path(cfg['output']['overall_metrics']),
        per_pattern_metrics=Path(cfg['output']['per_pattern_metrics']),
        confusion_matrices=Path(cfg['output']['confusion_matrices']),
        feature_importance=Path(cfg['output']['feature_importance']),
        top_predictions=Path(cfg['output']['top_predictions']),
        pr_curve_overall=Path(cfg['output']['pr_curve_overall']),
        pr_curve_per_pattern=Path(cfg['output']['pr_curve_per_pattern']),
        event_summary=Path(cfg['output']['event_summary']),
        data_quality_report=Path(cfg['output']['data_quality_report'])
    )
    
    # Parse logging config
    logging = LoggingConfig(
        log_dir=Path(cfg['logging']['log_dir']),
        log_file=Path(cfg['logging']['log_file']),
        level=cfg['logging']['level']
    )
    
    return PathConfig(
        data=data,
        input=input_paths,
        output=output,
        logging=logging,
        column_mapping=cfg['column_mapping']
    )


def load_config(
    detection_config_path: Path = Path("config/detection_config.yaml"),
    model_config_path: Path = Path("config/model_config.yaml"),
    path_config_path: Path = Path("config/paths_config.yaml")
) -> Config:
    """
    Load complete configuration from YAML files.
    
    Args:
        detection_config_path: Path to detection configuration
        model_config_path: Path to model configuration
        path_config_path: Path to path configuration
    
    Returns:
        Complete Config object with all settings
    """
    detection = load_detection_config(detection_config_path)
    model = load_model_config(model_config_path)
    paths = load_path_config(path_config_path)
    
    return Config(
        detection=detection,
        model=model,
        paths=paths
    )


def validate_config(config: Config) -> None:
    """
    Validate configuration for consistency and sanity checks.
    
    Args:
        config: Configuration object to validate
    
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate train ratio
    if not 0 < config.model.splitting.train_ratio < 1:
        raise ValueError("Train ratio must be between 0 and 1")
    
    # Validate FPR targets
    for pattern_name, pattern_cfg in config.model.pattern_models.items():
        if not 0 < pattern_cfg.target_fpr < 1:
            raise ValueError(f"Pattern {pattern_name} has invalid target FPR: {pattern_cfg.target_fpr}")
    
    # Validate clustering parameters
    if config.detection.clustering.eps_meters <= 0:
        raise ValueError("Clustering eps_meters must be positive")
    
    if config.detection.clustering.min_samples < 1:
        raise ValueError("Clustering min_samples must be at least 1")
    
    # Validate paths exist (for input)
    if not config.paths.input.combined_csv.parent.exists():
        config.paths.input.combined_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Create output directories if they don't exist
    for path_attr in ['events', 'models', 'reports']:
        path = getattr(config.paths.data, path_attr)
        path.mkdir(parents=True, exist_ok=True)
    
    # Create log directory
    config.paths.logging.log_dir.mkdir(parents=True, exist_ok=True)
    
    print("✓ Configuration validated successfully")
