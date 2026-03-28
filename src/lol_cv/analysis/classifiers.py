"""
Win prediction classifiers using CV-extracted features.

Compares multiple ML methods on the same feature set to determine
which best predicts match outcomes. Supports ablation studies
comparing CV-only, API-only, and combined feature sets.

Models:
    - Random Forest: good baseline, built-in feature importance
    - SVM (RBF): strong on small-medium datasets
    - Gradient Boosting: typically best performer, robust to outliers
    - MLP: neural net baseline for comparison
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

from lol_cv.utils import setup_logger

logger = setup_logger("lol_cv.analysis.classifiers")


def _build_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Create sklearn pipelines for each model (scaler + classifier)."""
    return {
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=random_state
            )),
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=random_state)),
        ]),
        "gradient_boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1, random_state=random_state
            )),
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=500, random_state=random_state
            )),
        ]),
    }


class WinPredictor:
    """Compare ML models for predicting match outcomes from CV features."""

    def __init__(
        self,
        models: list[str] = None,
        cv_folds: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Args:
            models: List of model names to use. Defaults to all four.
            cv_folds: Number of cross-validation folds.
            test_size: Fraction reserved for hold-out test (used in evaluate_holdout).
            random_state: Random seed for reproducibility.
        """
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state

        all_models = _build_models(random_state)
        selected = models or list(all_models.keys())
        self.models = {k: all_models[k] for k in selected if k in all_models}
        self.results = {}
        self.fitted_models = {}

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train all models with stratified cross-validation.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Binary target (1 = blue win, 0 = red win).

        Returns:
            Dict of {model_name: {accuracy, precision, recall, f1, roc_auc}}.
            Each metric is the mean across CV folds.
        """
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

        results = {}
        for name, pipeline in self.models.items():
            logger.info("Cross-validating %s (%d folds)", name, self.cv_folds)
            cv_results = cross_validate(
                pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False
            )
            metrics = {metric: float(np.mean(cv_results[f"test_{metric}"])) for metric in scoring}
            metrics["std_accuracy"] = float(np.std(cv_results["test_accuracy"]))
            results[name] = metrics
            logger.info("%s — acc: %.3f (±%.3f), AUC: %.3f",
                        name, metrics["accuracy"], metrics["std_accuracy"], metrics["roc_auc"])

            # Fit on full data for feature importance / later use
            pipeline.fit(X, y)
            self.fitted_models[name] = pipeline

        self.results = results
        return results

    def feature_importance(self, model_name: str, feature_names: list[str] = None) -> pd.DataFrame:
        """Extract feature importances from a trained model.

        Supports Random Forest and Gradient Boosting (built-in importance).
        For SVM/MLP, returns coefficient magnitudes where applicable.

        Args:
            model_name: Name of the model to inspect.
            feature_names: Optional list of feature names for the columns.

        Returns:
            DataFrame with columns [feature, importance] sorted descending.
        """
        if model_name not in self.fitted_models:
            raise ValueError(f"Model '{model_name}' not fitted. Run train_and_evaluate() first.")

        pipeline = self.fitted_models[model_name]
        clf = pipeline.named_steps["clf"]

        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_).mean(axis=0) if clf.coef_.ndim > 1 else np.abs(clf.coef_[0])
        else:
            raise ValueError(f"Model '{model_name}' does not support feature importance extraction.")

        names = feature_names or [f"feature_{i}" for i in range(len(importances))]
        df = pd.DataFrame({"feature": names, "importance": importances})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def results_to_dataframe(self) -> pd.DataFrame:
        """Convert cross-validation results to a comparison DataFrame.

        Returns:
            DataFrame with models as rows and metrics as columns.
        """
        if not self.results:
            raise ValueError("No results yet. Run train_and_evaluate() first.")
        return pd.DataFrame(self.results).T

    def ablation_study(
        self,
        X_cv: pd.DataFrame,
        X_api: pd.DataFrame,
        X_combined: pd.DataFrame,
        y: pd.Series,
        model_name: str = "gradient_boosting",
    ) -> dict:
        """Compare CV-only, API-only, and combined feature sets.

        This is the core experiment: does CV add value beyond the API?

        Returns:
            Dict with keys 'cv_only', 'api_only', 'combined', each containing metrics.
        """
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scoring = ["accuracy", "f1", "roc_auc"]

        results = {}
        for label, X in [("cv_only", X_cv), ("api_only", X_api), ("combined", X_combined)]:
            pipeline = _build_models(self.random_state)[model_name]
            cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)
            results[label] = {
                metric: float(np.mean(cv_results[f"test_{metric}"])) for metric in scoring
            }
            logger.info("Ablation [%s] — acc: %.3f, AUC: %.3f",
                        label, results[label]["accuracy"], results[label]["roc_auc"])

        return results
