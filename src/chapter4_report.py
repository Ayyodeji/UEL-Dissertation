"""Build Chapter 4 supplementary reports from saved results."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import RESULTS_DIR


def _clean_feature(name: str) -> str:
    return name.replace("num__", "").replace("cat__", "")


def write_reports(pima_hc: dict, early_hc: dict) -> None:
    lines = ["# Chapter 4 — Empirical Findings Summary", ""]

    for name in ("pima", "early_stage"):
        prep_path = RESULTS_DIR / name / "preprocessing_summary.json"
        base_path = RESULTS_DIR / name / "table_baseline_results.csv"
        if not prep_path.exists():
            continue
        prep = json.loads(prep_path.read_text())
        baseline = pd.read_csv(base_path)
        label = "Pima Indians Diabetes" if name == "pima" else "UCI Early Stage Diabetes"
        lines.append(f"## {label}")
        lines.append("")
        lines.append("### Preprocessing")
        lines.append(f"- Train/test: {prep['train_size']} / {prep['test_size']}")
        lines.append(f"- SMOTE on training: {prep.get('smote_applied', False)}")
        if "zeros_replaced" in prep:
            lines.append(f"- Implausible zeros replaced: {prep['zeros_replaced']}")
        lines.append(f"- Class distribution (full): {prep.get('class_distribution', {})}")
        lines.append("")
        lines.append("### Best baseline (by recall)")
        best = baseline.loc[baseline["Recall"].astype(float).idxmax()]
        lines.append(
            f"- **{best['Model']}**: Recall={best['Recall']}, ROC-AUC={best['ROC-AUC']}, "
            f"F1={best['F1-score']}"
        )
        hc = pima_hc if name == "pima" else early_hc
        lines.append("")
        lines.append("### Hybrid comparison")
        lines.append(
            f"- Best hybrid: **{hc['hybrid_model']}** (Recall={hc['hybrid_recall']}, "
            f"ROC-AUC={hc['hybrid_roc_auc']})"
        )
        lines.append(
            f"- Hybrid improves recall vs best baseline: {hc['hybrid_improves_recall']}"
        )
        lines.append(
            f"- Hybrid improves ROC-AUC vs best baseline: {hc['hybrid_improves_roc_auc']}"
        )
        fi_path = RESULTS_DIR / name / "table_feature_importance.csv"
        if fi_path.exists():
            fi = pd.read_csv(fi_path)
            lines.append("")
            lines.append("### Top features (Random Forest)")
            for _, row in fi.iterrows():
                lines.append(
                    f"- {int(row['rank'])}. {_clean_feature(str(row['feature']))} "
                    f"({row['importance_score']})"
                )
        lines.append("")

    out = RESULTS_DIR / "chapter4_empirical_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
