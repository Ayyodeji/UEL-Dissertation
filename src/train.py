"""Train and evaluate models for one dataset."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from .config import FIGURES_DIR, RESULTS_DIR
from .eda import eda_early_stage, eda_pima
from .models import (
    build_soft_voting,
    build_stacking,
    evaluate_model,
    get_base_estimators,
    get_param_grids,
    tune_model,
)
from .preprocessing import (
    build_preprocessor,
    needs_smote,
    preprocess_early_stage,
    preprocess_pima,
    save_json,
    stratified_split,
)
from .shap_analysis import run_shap, top_features_from_rf


def _plot_confusion(cm: list, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_dataset(
    name: str,
    df: pd.DataFrame,
    preprocess_fn,
    eda_fn,
) -> dict:
    result_dir = RESULTS_DIR / name
    fig_dir = FIGURES_DIR / name
    result_dir.mkdir(parents=True, exist_ok=True)

    X, y, prep_meta = preprocess_fn(df)
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    X_test.assign(target=y_test.values).to_csv(result_dir / "test_set.csv", index=False)

    split_meta = {
        **prep_meta,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "train_class_distribution": y_train.value_counts().to_dict(),
        "test_class_distribution": y_test.value_counts().to_dict(),
    }
    save_json(result_dir / "preprocessing_summary.json", split_meta)

    eda_fn(X, y, fig_dir)

    use_smote = needs_smote(y_train)
    split_meta["smote_applied"] = use_smote
    save_json(result_dir / "preprocessing_summary.json", split_meta)

    preprocessor = build_preprocessor(X_train, name)
    param_grids = get_param_grids()
    estimators = get_base_estimators()

    baseline_rows = []
    tuned_models = {}
    best_params_log = {}

    for model_name, estimator in estimators.items():
        from .models import _wrap

        pipe = _wrap(estimator, preprocessor, use_smote)
        tuned, best_params = tune_model(
            model_name, pipe, param_grids[model_name], X_train, y_train
        )
        tuned_models[model_name] = tuned
        best_params_log[model_name] = best_params
        metrics = evaluate_model(tuned, X_test, y_test)
        row = {"model": model_name, **metrics}
        baseline_rows.append(row)
        joblib.dump(tuned, result_dir / f"model_{model_name.replace(' ', '_').lower()}.joblib")

    baseline_df = pd.DataFrame(
        [
            {
                "Model": r["model"],
                "Accuracy": round(r["accuracy"], 4),
                "Precision": round(r["precision"], 4),
                "Recall": round(r["recall"], 4),
                "F1-score": round(r["f1"], 4),
                "ROC-AUC": round(r["roc_auc"], 4),
            }
            for r in baseline_rows
        ]
    )
    baseline_df.to_csv(result_dir / "table_baseline_results.csv", index=False)

    best_idx = baseline_df["Recall"].astype(float).idxmax()
    best_baseline_name = baseline_df.loc[best_idx, "Model"]
    best_baseline = tuned_models[best_baseline_name]
    best_baseline_metrics = evaluate_model(best_baseline, X_test, y_test)

    stacking = build_stacking(preprocessor, use_smote)
    stacking.fit(X_train, y_train)
    stacking_metrics = evaluate_model(stacking, X_test, y_test)
    joblib.dump(stacking, result_dir / "model_stacking.joblib")

    voting = build_soft_voting(preprocessor, use_smote)
    voting.fit(X_train, y_train)
    voting_metrics = evaluate_model(voting, X_test, y_test)
    joblib.dump(voting, result_dir / "model_soft_voting.joblib")

    hybrids = {
        "Stacking": stacking_metrics,
        "Soft Voting": voting_metrics,
    }
    best_hybrid_name = max(hybrids, key=lambda k: hybrids[k]["recall"])
    best_hybrid_metrics = hybrids[best_hybrid_name]
    best_hybrid_model = stacking if best_hybrid_name == "Stacking" else voting

    hybrid_comparison = {
        "dataset": name,
        "best_baseline_model": best_baseline_name,
        "best_baseline_recall": round(best_baseline_metrics["recall"], 4),
        "best_baseline_roc_auc": round(best_baseline_metrics["roc_auc"], 4),
        "hybrid_model": best_hybrid_name,
        "hybrid_recall": round(best_hybrid_metrics["recall"], 4),
        "hybrid_roc_auc": round(best_hybrid_metrics["roc_auc"], 4),
        "hybrid_improves_recall": best_hybrid_metrics["recall"] > best_baseline_metrics["recall"],
        "hybrid_improves_roc_auc": best_hybrid_metrics["roc_auc"] > best_baseline_metrics["roc_auc"],
    }
    pd.DataFrame([hybrid_comparison]).to_csv(
        result_dir / "table_hybrid_comparison.csv", index=False
    )

    pd.DataFrame(
        [
            {"Model": "Stacking", **{k: round(v, 4) if isinstance(v, float) else v for k, v in stacking_metrics.items() if k != "confusion_matrix"}},
            {"Model": "Soft Voting", **{k: round(v, 4) if isinstance(v, float) else v for k, v in voting_metrics.items() if k != "confusion_matrix"}},
        ]
    ).to_csv(result_dir / "table_hybrid_full_metrics.csv", index=False)

    _plot_confusion(
        best_baseline_metrics["confusion_matrix"],
        f"{name}: best baseline ({best_baseline_name})",
        fig_dir / "confusion_matrix_best_baseline.png",
    )
    _plot_confusion(
        best_hybrid_metrics["confusion_matrix"],
        f"{name}: hybrid ({best_hybrid_name})",
        fig_dir / "confusion_matrix_hybrid.png",
    )

    pd.DataFrame(
        best_baseline_metrics["confusion_matrix"],
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"],
    ).to_csv(result_dir / "confusion_matrix_best_baseline.csv")
    pd.DataFrame(
        best_hybrid_metrics["confusion_matrix"],
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"],
    ).to_csv(result_dir / "confusion_matrix_hybrid.csv")

    rf_model = tuned_models["Random Forest"]
    feat_names = list(X.columns)
    run_shap(rf_model, X_train, X_test, feat_names, fig_dir / "shap_summary_rf.png")

    importance_rows = top_features_from_rf(rf_model, feat_names, top_n=5)
    if importance_rows:
        pd.DataFrame(importance_rows).to_csv(
            result_dir / "table_feature_importance.csv", index=False
        )

    save_json(result_dir / "best_hyperparameters.json", best_params_log)

    return {
        "name": name,
        "baseline_df": baseline_df,
        "hybrid_comparison": hybrid_comparison,
        "importance_rows": importance_rows,
        "best_baseline_name": best_baseline_name,
        "best_hybrid_name": best_hybrid_name,
    }


def run_pima(df: pd.DataFrame) -> dict:
    return run_dataset("pima", df, preprocess_pima, eda_pima)


def run_early_stage(df: pd.DataFrame) -> dict:
    return run_dataset("early_stage", df, preprocess_early_stage, eda_early_stage)
