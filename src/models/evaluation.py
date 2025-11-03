"""
Model Evaluation
Minimal ModelEvaluator wired to metrics.py and model_config.yaml via settings.ModelConfig.
"""
 
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any, List
from pathlib import Path
 
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
 
from src.config.settings import ModelConfig 
from src.utils.metrics import pr_auc, threshold_at_fpr, classification_report_dict
 

def _get_scores(model, X) -> np.ndarray:
    """Return probability-like scores in [0,1] (or best effort)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # fallback: try decision_function and map to [0,1]
    scores = model.decision_function(X)  # type: ignore[attr-defined]
    scores = np.asarray(scores, dtype=float)
    mn, mx = np.min(scores), np.max(scores)
    return (scores - mn) / (mx - mn + 1e-12) if mx > mn else np.zeros_like(scores)


def _feature_names_from_preprocessor(model) -> Optional[List[str]]:
    """Extract transformed feature names from a pipeline's ColumnTransformer if available."""
    try:
        pre = model.named_steps.get("preprocess")
        if pre is None:
            return None
        fn = pre.get_feature_names_out()
        return list(fn)
    except Exception:
        return None


def _feature_importances(model) -> Optional[np.ndarray]:
    """
    Try multiple paths to recover feature importances / coefficients.
    Works for tree models; returns None if unavailable (e.g., calibrated LR/RF wrapped).
    """
    try:
        clf = model.named_steps.get("clf", model)
    except Exception:
        clf = model
    # Try calibrated wrapper internals
    for candidate in (
        getattr(clf, "base_estimator", None),
        getattr(clf, "estimator", None),
        getattr(clf, "best_estimator_", None),
        clf,
    ):
        if candidate is None:
            continue
        if hasattr(candidate, "feature_importances_"):
            return np.asarray(candidate.feature_importances_)
        if hasattr(candidate, "coef_"):
            coef = np.asarray(candidate.coef_)
            return coef.ravel() if coef.ndim > 1 else coef
    return None


class ModelEvaluator:
    def __init__(
        self,
        target_fpr: float = 0.05,
        save_pr_curves: bool = True,
        save_feature_importance: bool = True,
        output_dir: Optional[Path] = None,
    ):
        self.target_fpr = float(target_fpr)
        self.save_pr_curves = bool(save_pr_curves)
        self.save_feature_importance = bool(save_feature_importance)
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_model_config(cls, model_cfg: ModelConfig, output_dir: Optional[Path] = None):
        ev = getattr(model_cfg, "evaluation", None)
        target = getattr(ev, "target_fpr", 0.05) if ev else 0.05
        save_pr = getattr(ev, "save_pr_curves", True) if ev else True
        save_fi = getattr(ev, "save_feature_importance", True) if ev else True
        return cls(target_fpr=target, save_pr_curves=save_pr, save_feature_importance=save_fi, output_dir=output_dir)

    def evaluate_multiple(
        self,
        models: Dict[str, Any],
        X_test: Any,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Evaluate many models: compute PR-AUC and choose threshold at target FPR.
        Returns (comparison_df, feature_importances_dict).
        """
        rows = []
        fi_dict: Dict[str, pd.DataFrame] = {}

        for name, model in models.items():
            scores = _get_scores(model, X_test)
            pr = pr_auc(scores, y_test)
            th = threshold_at_fpr(scores, y_test, max_fpr=self.target_fpr)

            rows.append(
                dict(
                    model=name,
                    pr_auc=pr,
                    threshold=th.get("threshold", np.nan),
                    recall=th.get("recall", 0.0),
                    precision=th.get("precision", 0.0),
                    fpr=th.get("fpr", 0.0),
                )
            )

            # Optional feature importance
            if self.save_feature_importance:
                fi = _feature_importances(model)
                if fi is not None:
                    # Try to align with transformed names
                    transformed_names = _feature_names_from_preprocessor(model) or feature_names
                    if transformed_names is None or len(transformed_names) != len(fi):
                        # fall back to index
                        transformed_names = [f"f{i}" for i in range(len(fi))]
                    df_fi = pd.DataFrame({"feature": transformed_names, "importance": fi}).sort_values(
                        "importance", ascending=False
                    )
                    fi_dict[name] = df_fi

            # Optional PR curve
            if self.save_pr_curves and self.output_dir is not None:
                prec, rec, _ = precision_recall_curve(y_test, scores)
                fig = plt.figure()
                plt.plot(rec, prec, label=name)
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(f"PR Curve — {name}")
                plt.legend(loc="best")
                fig.savefig(self.output_dir / f"pr_curve_{name}.png", dpi=160, bbox_inches="tight")
                plt.close(fig)

        comparison = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
        if self.output_dir is not None:
            comparison.to_csv(self.output_dir / "model_comparison.csv", index=False)
            # Save feature importances
            if self.save_feature_importance:
                fi_dir = self.output_dir / "feature_importance"
                fi_dir.mkdir(parents=True, exist_ok=True)
                for m, df in fi_dict.items():
                    df.to_csv(fi_dir / f"{m}.csv", index=False)
        return comparison, fi_dict

    def generate_report(
        self,
        comparison: pd.DataFrame,
        top_n: int = 100,
    ) -> pd.DataFrame:
        """
        Placeholder for additional reporting or top-predictions export (caller provides scores).
        For now, we just return the comparison DF (already saved if output_dir provided).
        """
        return comparison
