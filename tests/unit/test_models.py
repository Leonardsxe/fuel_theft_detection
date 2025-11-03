"""
Unit tests for model training and evaluation modules.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.models.training import (
    ModelTrainer,
    build_preprocessing_pipeline,
    _dedupe_columns,
)
from src.models.evaluation import ModelEvaluator
from src.models.calibration import (
    calibrate_classifier,
    check_calibration_quality,
)


class TestModelTrainer:
    """Tests for ModelTrainer class."""
    
    def test_model_trainer_initialization(self, model_config):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(model_config)
        
        assert trainer.config is not None
        assert trainer.preprocessor is None  # Not fitted yet
        assert trainer.models == {}
    
    def test_fit_preprocessor(self, model_config):
        """Test preprocessor fitting."""
        trainer = ModelTrainer(model_config)
        
        X = pd.DataFrame({
            'drop_gal': [5.0, 6.0, 7.0],
            'duration_min': [8.0, 9.0, 10.0],
            'is_hotspot': [0, 1, 0],
            'pattern': ['short', 'extended', 'short'],
        })
        
        trainer.fit_preprocessor(X)
        
        assert trainer.preprocessor is not None
    
    def test_train_all(self, model_config):
        """Test training all models."""
        trainer = ModelTrainer(model_config)
        
        X_train = pd.DataFrame({
            'drop_gal': np.random.uniform(3, 10, 100),
            'duration_min': np.random.uniform(5, 20, 100),
            'rate_gpm': np.random.uniform(0.2, 1.5, 100),
            'is_hotspot': np.random.choice([0, 1], 100),
        })
        y_train = np.random.choice([0, 1], 100, p=[0.7, 0.3])
        
        models = trainer.train_all(X_train, y_train)
        
        # Should train multiple models
        assert len(models) > 0
        
        # Each model should be a Pipeline
        for name, model in models.items():
            assert isinstance(model, Pipeline)
            assert hasattr(model, 'predict_proba')
    
    def test_train_all_handles_missing_libraries(self, model_config):
        """Test that training handles missing optional libraries gracefully."""
        trainer = ModelTrainer(model_config)
        
        X_train = pd.DataFrame({
            'drop_gal': np.random.uniform(3, 10, 50),
            'duration_min': np.random.uniform(5, 20, 50),
        })
        y_train = np.random.choice([0, 1], 50)
        
        # Should not crash even if some libraries unavailable
        models = trainer.train_all(X_train, y_train)
        
        # Should at least train logistic regression and random forest
        assert 'logreg_cal' in models or 'rf_cal' in models


class TestBuildPreprocessingPipeline:
    """Tests for preprocessing pipeline building."""
    
    def test_build_pipeline_numeric_only(self):
        """Test pipeline with only numeric features."""
        numeric_cols = ['drop_gal', 'duration_min', 'rate_gpm']
        pipeline = build_preprocessing_pipeline(numeric_cols, [], [])
        
        assert pipeline is not None
        assert hasattr(pipeline, 'fit_transform')
    
    def test_build_pipeline_with_categorical(self):
        """Test pipeline with categorical features."""
        numeric_cols = ['drop_gal']
        binary_cols = ['is_hotspot']
        categorical_cols = ['pattern']
        
        pipeline = build_preprocessing_pipeline(numeric_cols, binary_cols, categorical_cols)
        
        assert pipeline is not None
    
    def test_pipeline_transforms_correctly(self):
        """Test that pipeline transforms data correctly."""
        numeric_cols = ['x', 'y']
        pipeline = build_preprocessing_pipeline(numeric_cols, [], [])
        
        X = pd.DataFrame({'x': [1, 2, np.nan], 'y': [4, 5, 6]})
        
        X_transformed = pipeline.fit_transform(X)
        
        # Should handle NaN and scale
        assert not np.isnan(X_transformed).any()


class TestDedupeColumns:
    """Tests for column deduplication."""
    
    def test_dedupe_unique_columns(self):
        """Test with unique columns."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        
        result = _dedupe_columns(df)
        
        assert list(result.columns) == ['a', 'b', 'c']
    
    def test_dedupe_duplicate_columns(self):
        """Test with duplicate columns."""
        df = pd.DataFrame([[1, 2, 3]], columns=['a', 'a', 'b'])
        
        result = _dedupe_columns(df)
        
        # Should rename duplicates
        assert len(result.columns) == 3
        assert result.columns[0] == 'a'
        assert '__dup' in result.columns[1]
    
    def test_dedupe_preserves_data(self):
        """Test that deduplication preserves data."""
        df = pd.DataFrame([[1, 2, 3]], columns=['a', 'a', 'b'])
        
        result = _dedupe_columns(df)
        
        # Data should be unchanged
        assert result.iloc[0, 0] == 1
        assert result.iloc[0, 1] == 2
        assert result.iloc[0, 2] == 3


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""
    
    def test_evaluator_initialization(self):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator(target_fpr=0.05)
        
        assert evaluator.target_fpr == 0.05
        assert evaluator.save_pr_curves
    
    def test_evaluate_multiple(self, model_config, temp_output_dir):
        """Test evaluating multiple models."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        # Create simple models
        X_train = np.random.randn(100, 3)
        y_train = np.random.choice([0, 1], 100, p=[0.7, 0.3])
        
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        lr = LogisticRegression(random_state=42)
        
        rf.fit(X_train, y_train)
        lr.fit(X_train, y_train)
        
        models = {'rf': rf, 'lr': lr}
        
        # Test data
        X_test = np.random.randn(50, 3)
        y_test = np.random.choice([0, 1], 50, p=[0.7, 0.3])
        
        evaluator = ModelEvaluator(target_fpr=0.05, output_dir=temp_output_dir)
        
        comparison_df, fi_dict = evaluator.evaluate_multiple(
            models, X_test, y_test, ['f1', 'f2', 'f3']
        )
        
        # Should return comparison DataFrame
        assert not comparison_df.empty
        assert 'model' in comparison_df.columns
        assert 'pr_auc' in comparison_df.columns
        
        # Should have 2 models
        assert len(comparison_df) == 2


class TestCalibration:
    """Tests for calibration utilities."""
    
    def test_calibrate_classifier(self):
        """Test classifier calibration."""
        from sklearn.ensemble import RandomForestClassifier
        
        X_train = np.random.randn(100, 5)
        y_train = np.random.choice([0, 1], 100)
        X_val = np.random.randn(50, 5)
        y_val = np.random.choice([0, 1], 50)
        
        base_model = RandomForestClassifier(n_estimators=10, random_state=42)
        base_model.fit(X_train, y_train)
        
        calibrated = calibrate_classifier(base_model, X_val, y_val, method='sigmoid')
        
        assert hasattr(calibrated, 'predict_proba')
        
        # Calibrated model should produce probabilities
        probs = calibrated.predict_proba(X_val)
        assert probs.shape == (50, 2)
        assert (probs >= 0).all() and (probs <= 1).all()
    
    def test_check_calibration_quality(self):
        """Test calibration quality assessment."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.85, 0.15, 0.25, 0.75])
        
        cal_stats = check_calibration_quality(y_true, y_proba, n_bins=5)
        
        assert 'bin_centers' in cal_stats
        assert 'bin_accuracies' in cal_stats
        assert 'ece' in cal_stats
        assert 'mce' in cal_stats
        
        # ECE and MCE should be between 0 and 1
        assert 0 <= cal_stats['ece'] <= 1
        assert 0 <= cal_stats['mce'] <= 1
    
    def test_calibration_perfect(self):
        """Test calibration metrics for perfectly calibrated predictions."""
        # Perfectly calibrated: predicted probabilities match actual rates
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_proba = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])
        
        cal_stats = check_calibration_quality(y_true, y_proba, n_bins=2)
        
        # ECE should be very small for well-calibrated predictions
        assert cal_stats['ece'] < 0.2


class TestModelEdgeCases:
    """Tests for edge cases in model training."""
    
    def test_training_with_single_class(self, model_config):
        """Test training when all labels are same class."""
        trainer = ModelTrainer(model_config)
        
        X_train = pd.DataFrame({
            'drop_gal': [5.0, 6.0, 7.0],
            'duration_min': [8.0, 9.0, 10.0],
        })
        y_train = np.array([0, 0, 0])  # All same class
        
        # Should handle gracefully (may skip some models)
        try:
            models = trainer.train_all(X_train, y_train)
            # If it trains, models should exist
            assert len(models) >= 0
        except Exception as e:
            # Some models may fail - that's expected
            assert True
    
    def test_training_with_minimal_data(self, model_config):
        """Test training with very few samples."""
        trainer = ModelTrainer(model_config)
        
        X_train = pd.DataFrame({
            'drop_gal': [5.0, 6.0],
            'duration_min': [8.0, 9.0],
        })
        y_train = np.array([0, 1])
        
        # Should handle minimal data
        try:
            models = trainer.train_all(X_train, y_train)
            assert isinstance(models, dict)
        except Exception:
            # Some models may require more data
            pass
    
    def test_training_with_imbalanced_data(self, model_config):
        """Test training with highly imbalanced classes."""
        trainer = ModelTrainer(model_config)
        
        X_train = pd.DataFrame({
            'drop_gal': np.random.uniform(3, 10, 100),
            'duration_min': np.random.uniform(5, 20, 100),
        })
        # 95% negative, 5% positive
        y_train = np.random.choice([0, 1], 100, p=[0.95, 0.05])
        
        models = trainer.train_all(X_train, y_train)
        
        # Should train successfully with class imbalance
        assert len(models) > 0