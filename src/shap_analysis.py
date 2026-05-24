"""SHAP explainability for best tree-based or linear proxy model."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier


def run_shap(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list[str],
    out_path: Path,
    max_samples: int = 200,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample = X_test.sample(
        n=min(max_samples, len(X_test)), random_state=42
    )

    try:
        if hasattr(model, "named_steps"):
            inner = model.named_steps.get("model", model)
            preprocess = model.named_steps.get("preprocess")
            if preprocess is not None:
                X_train_t = preprocess.fit_transform(X_train)
                X_sample_t = preprocess.transform(sample)
            else:
                X_train_t = X_train.values
                X_sample_t = sample.values
            inner = model.named_steps["model"]
        else:
            X_train_t = X_train.values
            X_sample_t = sample.values
            inner = model

        if isinstance(inner, RandomForestClassifier):
            explainer = shap.TreeExplainer(inner)
            shap_values = explainer.shap_values(X_sample_t)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            explainer = shap.Explainer(inner.predict_proba, X_train_t)
            sv = explainer(X_sample_t)
            shap_values = sv.values[:, :, 1] if sv.values.ndim == 3 else sv.values

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X_sample_t,
            feature_names=feature_names[: X_sample_t.shape[1]],
            show=False,
            max_display=15,
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        with open(out_path.with_suffix(".txt"), "w") as f:
            f.write(f"SHAP skipped: {e}\n")


def top_features_from_rf(
    model, feature_names: list[str], top_n: int = 5
) -> list[dict]:
    if hasattr(model, "named_steps"):
        rf = model.named_steps["model"]
        preprocess = model.named_steps["preprocess"]
    else:
        rf = model
        preprocess = None

    if not hasattr(rf, "feature_importances_"):
        return []

    importances = rf.feature_importances_
    if preprocess is not None and hasattr(preprocess, "get_feature_names_out"):
        try:
            feature_names = list(preprocess.get_feature_names_out())
        except Exception:
            pass

    order = np.argsort(importances)[::-1][:top_n]
    rows = []
    for rank, idx in enumerate(order, start=1):
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        score = float(importances[idx])
        clean = str(name).replace("num__", "").replace("cat__", "")
        rows.append(
            {
                "rank": rank,
                "feature": clean,
                "importance_score": round(score, 4),
                "interpretation": f"Rank {rank} predictor for diabetes risk in this model.",
            }
        )
    return rows
