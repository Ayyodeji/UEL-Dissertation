"""Model definitions, tuning grids, and evaluation."""
from __future__ import annotations

from typing import Any

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import CV_FOLDS, N_ITER_TUNING, RANDOM_STATE
from .preprocessing import needs_smote


def _wrap(estimator, preprocessor, use_smote: bool):
    steps: list[tuple[str, Any]] = [("preprocess", preprocessor), ("model", estimator)]
    if use_smote:
        return ImbPipeline(
            [("preprocess", preprocessor), ("smote", SMOTE(random_state=RANDOM_STATE)), ("model", estimator)]
        )
    return Pipeline(steps)


def get_param_grids() -> dict[str, dict]:
    return {
        "Logistic Regression": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs", "liblinear"],
            "model__max_iter": [2000],
        },
        "Decision Tree": {
            "model__max_depth": [3, 5, 8, 12, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "Random Forest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [5, 10, 15, None],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "SVM": {
            "model__C": [0.1, 1, 10],
            "model__gamma": ["scale", "auto"],
            "model__kernel": ["rbf", "linear"],
        },
        "KNN": {
            "model__n_neighbors": [3, 5, 7, 11, 15],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan"],
        },
        "Gradient Boosting": {
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__n_estimators": [100, 150, 200],
            "model__max_depth": [3, 5, 7],
        },
        "XGBoost": {
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__max_depth": [3, 5, 7],
            "model__n_estimators": [100, 150, 200],
            "model__subsample": [0.8, 1.0],
        },
    }


def get_base_estimators() -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=2000
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
        ),
    }


def tune_model(name: str, pipeline, param_grid: dict, X_train, y_train):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=min(N_ITER_TUNING, max(1, _grid_size(param_grid))),
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def _grid_size(grid: dict) -> int:
    size = 1
    for v in grid.values():
        size *= len(v)
    return size


def build_stacking(preprocessor, use_smote: bool) -> Any:
    bases = get_base_estimators()
    estimators = [
        ("lr", _wrap(clone(bases["Logistic Regression"]), preprocessor, use_smote)),
        ("rf", _wrap(clone(bases["Random Forest"]), preprocessor, use_smote)),
        ("svm", _wrap(clone(bases["SVM"]), preprocessor, use_smote)),
        ("gb", _wrap(clone(bases["Gradient Boosting"]), preprocessor, use_smote)),
    ]
    meta = LogisticRegression(random_state=RANDOM_STATE, max_iter=2000)
    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        cv=CV_FOLDS,
        stack_method="predict_proba",
        n_jobs=-1,
    )


def build_soft_voting(preprocessor, use_smote: bool) -> Any:
    bases = get_base_estimators()
    estimators = [
        ("lr", _wrap(clone(bases["Logistic Regression"]), preprocessor, use_smote)),
        ("rf", _wrap(clone(bases["Random Forest"]), preprocessor, use_smote)),
        ("svm", _wrap(clone(bases["SVM"]), preprocessor, use_smote)),
        ("gb", _wrap(clone(bases["Gradient Boosting"]), preprocessor, use_smote)),
    ]
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


def evaluate_model(model, X_test, y_test) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, zero_division=0, average="binary", pos_label=1)
        ),
        "recall": float(
            recall_score(y_test, y_pred, zero_division=0, average="binary", pos_label=1)
        ),
        "f1": float(f1_score(y_test, y_pred, zero_division=0, average="binary", pos_label=1)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_proba is not None else np.nan,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics
