#!/usr/bin/env python3
"""Run full Chapter 4 pipeline for both datasets."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chapter4_report import write_reports
from src.config import RESULTS_DIR
from src.data_loader import load_early_stage, load_pima
from src.train import run_early_stage, run_pima


def update_dissertation_markdown(pima: dict, early: dict) -> None:
    md_path = PROJECT_ROOT / (
        "Hybrid Machine Learning Model for Early Diabetes Prediction "
        "Using Lifestyle and Clinical Data.md"
    )
    if not md_path.exists():
        return

    text = md_path.read_text(encoding="utf-8")

    def fill_baseline_table(df: pd.DataFrame, section_marker: str) -> str:
        pattern = (
            rf"(### \*\*{re.escape(section_marker)}\*\*\n\n"
            r"\| Model \| Accuracy \| Precision \| Recall \| F1-score \| ROC-AUC \|\n"
            r"\| [^\n]+ \|\n)"
            r"((?:\| [^\n]+ \|\n)+)"
        )
        rows = ["| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |",
                "| ----- | ----- | ----- | ----- | ----- | ----- |"]
        for _, r in df.iterrows():
            rows.append(
                f"| {r['Model']} | {r['Accuracy']} | {r['Precision']} | {r['Recall']} | "
                f"{r['F1-score']} | {r['ROC-AUC']} |"
            )
        replacement = r"\1" + "\n".join(rows[2:]) + "\n"
        return re.sub(pattern, replacement, text, count=1)

    text = fill_baseline_table(
        pima["baseline_df"], "Table 3: Baseline Model Results for Pima Dataset"
    )
    text = fill_baseline_table(
        early["baseline_df"], "Table 4: Baseline Model Results for UCI Dataset"
    )

    def fill_hybrid_row(dataset_label: str, hc: dict) -> str:
        row = (
            f"| {dataset_label} | {hc['best_baseline_model']} | {hc['best_baseline_recall']} | "
            f"{hc['hybrid_recall']} | {hc['best_baseline_roc_auc']} | {hc['hybrid_roc_auc']} |"
        )
        return row

    text = text.replace(
        "| Pima | \\[Insert\\] | \\[Insert\\] | \\[Insert\\] | \\[Insert\\] | \\[Insert\\] |",
        fill_hybrid_row("Pima", pima["hybrid_comparison"]),
    )
    text = text.replace(
        "| UCI Early Stage | \\[Insert\\] | \\[Insert\\] | \\[Insert\\] | \\[Insert\\] | \\[Insert\\] |",
        fill_hybrid_row("UCI Early Stage", early["hybrid_comparison"]),
    )

    pima_rows = pima.get("importance_rows") or []
    for r in pima_rows:
        old = f"| {r['rank']} | \\[Insert\\] | \\[Insert\\] | \\[Insert\\] |"
        new = f"| {r['rank']} | {r['feature']} | {r['importance_score']} | {r['interpretation']} |"
        text = text.replace(old, new, 1)

    early_rows = early.get("importance_rows") or []
    if early_rows:
        for r in early_rows:
            old = f"| {r['rank']} | \\[Insert\\] | \\[Insert\\] | \\[Insert\\] |"
            if old in text:
                feat = r["feature"]
                new = (
                    f"| {r['rank']} | {feat} (early-stage) | {r['importance_score']} | "
                    f"{r['interpretation']} |"
                )
                text = text.replace(old, new, 1)

    md_path.write_text(text, encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading Pima Indians Diabetes dataset (OpenML id=37)...")
    pima_df = load_pima()
    print("Loading Early Stage Diabetes Risk dataset (UCI id=529)...")
    early_df = load_early_stage()

    print("Training Pima pipeline...")
    pima_results = run_pima(pima_df)
    print("Training Early Stage pipeline...")
    early_results = run_early_stage(early_df)

    summary = pd.DataFrame(
        [pima_results["hybrid_comparison"], early_results["hybrid_comparison"]]
    )
    summary.to_csv(RESULTS_DIR / "chapter4_hybrid_summary.csv", index=False)

    update_dissertation_markdown(pima_results, early_results)
    write_reports(pima_results["hybrid_comparison"], early_results["hybrid_comparison"])

    print("\nDone. Results saved to:", RESULTS_DIR)
    print("Figures saved to:", PROJECT_ROOT / "figures")
    print("\nPima best baseline:", pima_results["best_baseline_name"])
    print("Pima best hybrid:", pima_results["best_hybrid_name"])
    print("Early-stage best baseline:", early_results["best_baseline_name"])
    print("Early-stage best hybrid:", early_results["best_hybrid_name"])


if __name__ == "__main__":
    main()
