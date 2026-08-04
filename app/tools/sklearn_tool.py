"""Model training and evaluation helpers built on scikit-learn / xgboost."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.tools.pandas_tool import is_numeric


def detect_task(y: pd.Series) -> str:
    """Return 'classification' or 'regression' based on the target column."""
    if is_numeric(y) and y.nunique() > 20:
        return "regression"
    return "classification"


def classification_models(random_state: int = 42) -> dict[str, Any]:
    """Model zoo for classification tasks."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, random_state=random_state),
        "XGBoost": _make_xgb_classifier(random_state),
    }


def regression_models(random_state: int = 42) -> dict[str, Any]:
    """Model zoo for regression tasks."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "KNN": KNeighborsRegressor(),
        "SVM": SVR(),
        "XGBoost": _make_xgb_regressor(random_state),
    }


def _make_xgb_classifier(random_state: int):
    from xgboost import XGBClassifier

    return XGBClassifier(n_estimators=100, random_state=random_state, eval_metric="logloss")


def _make_xgb_regressor(random_state: int):
    from xgboost import XGBRegressor

    return XGBRegressor(n_estimators=100, random_state=random_state)


def train_all(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    model_dir: str | Path,
    tune: bool = False,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train every model in the zoo, cross-validate, and persist the best.

    Returns a serialisable summary with cross-val scores, a held-out test
    evaluation, and the path to the saved estimator.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    label_encoder = None
    if task == "classification":
        label_encoder = LabelEncoder()
        y_enc = label_encoder.fit_transform(y)
        zoo = classification_models(random_state)
    else:
        y_enc = y.astype(float)
        zoo = regression_models(random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=random_state, stratify=(y_enc if task == "classification" and len(np.unique(y_enc)) > 1 else None)
    )

    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for name, estimator in zoo.items():
        try:
            pipe = Pipeline([("scale", StandardScaler(with_mean=False)), ("model", estimator)])
            scores = cross_val_score(pipe, X_train, y_train, cv=cv_folds, scoring=_scoring(task), n_jobs=-1)
            pipe.fit(X_train, y_train)
            metrics = _evaluate(pipe, X_test, y_test, task)

            entry = {
                "model": name,
                "task": task,
                "cv_mean": round(float(np.mean(scores)), 4),
                "cv_std": round(float(np.std(scores)), 4),
                "test_metrics": metrics,
            }
            results.append(entry)
            if best is None or entry["cv_mean"] > best["cv_mean"]:
                best = entry
                artifact = {
                    "pipeline": pipe,
                    "task": task,
                    "label_encoder": label_encoder,
                    "feature_names": list(X.columns),
                }
        except Exception as exc:  # keep the workflow alive if one model fails
            results.append({"model": name, "error": str(exc)})

    if best is None:
        raise RuntimeError("No model could be trained on this dataset.")

    model_path = model_dir / "best_model.joblib"
    joblib.dump(artifact, model_path)

    return {
        "task": task,
        "results": results,
        "best_model": {**best, "path": str(model_path)},
        "label_encoder": label_encoder.classes_.tolist() if label_encoder is not None else None,
        "feature_names": list(X.columns),
    }


def _scoring(task: str) -> str:
    return "accuracy" if task == "classification" else "neg_mean_squared_error"


def _evaluate(pipe, X_test, y_test, task: str) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )

    y_pred = pipe.predict(X_test)
    if task == "classification":
        try:
            y_proba = pipe.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1": round(float(f1_score(y_test, y_pred, average="weighted")), 4),
        }
        if y_proba is not None and len(np.unique(y_test)) == 2:
            metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_proba)), 4)
        return metrics

    y_test = y_test.astype(float)
    return {
        "r2": round(float(r2_score(y_test, y_pred)), 4),
        "mse": round(float(mean_squared_error(y_test, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
    }


def feature_importance(artifact_path: str | Path) -> dict[str, float]:
    """Return feature importances from the persisted best pipeline, if possible."""
    artifact = joblib.load(artifact_path)
    pipe = artifact.get("pipeline")
    model = pipe.named_steps["model"] if pipe else None
    if model is None or not hasattr(model, "feature_importances_"):
        return {}
    names = artifact.get("feature_names", [])
    importances = model.feature_importances_
    ranked = sorted(zip(names, importances), key=lambda item: item[1], reverse=True)
    return {name: round(float(score), 4) for name, score in ranked[:20]}