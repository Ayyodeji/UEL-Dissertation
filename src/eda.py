"""Exploratory data analysis figures for Chapter 4."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR, RANDOM_STATE

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def eda_pima(X: pd.DataFrame, y: pd.Series, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    numeric = ["Glucose", "BMI", "BloodPressure", "Insulin", "Age", "Pregnancies"]
    cols = [c for c in X.columns if any(c.lower() == n.lower() for n in numeric)]
    if not cols:
        cols = X.select_dtypes(include=[np.number]).columns.tolist()[:6]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    for ax, col in zip(axes, cols[:6]):
        for label in sorted(y.unique()):
            subset = X.loc[y == label, col].dropna()
            ax.hist(subset, bins=20, alpha=0.6, label=f"class={label}")
        ax.set_title(col)
        ax.legend(fontsize=8)
    fig.suptitle("Pima: feature distributions by class")
    _save(fig, out_dir / "pima_histograms_by_class.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    corr = X[cols].apply(pd.to_numeric, errors="coerce").corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Pima: correlation heatmap")
    _save(fig, out_dir / "pima_correlation_heatmap.png")

    melt_cols = cols[:5]
    plot_df = X[melt_cols].copy()
    plot_df["target"] = y.values
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=plot_df.melt(id_vars="target"), x="variable", y="value", hue="target", ax=ax)
    ax.set_title("Pima: boxplots by class")
    plt.xticks(rotation=45)
    _save(fig, out_dir / "pima_boxplots_by_class.png")


def eda_early_stage(X: pd.DataFrame, y: pd.Series, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    symptom_keywords = [
        "polyuria",
        "polydipsia",
        "sudden",
        "weight",
        "weakness",
        "visual",
        "blurring",
        "delayed",
        "healing",
        "obesity",
    ]
    symptom_cols = [
        c
        for c in X.columns
        if any(k in c.lower() for k in symptom_keywords)
    ]
    if not symptom_cols:
        symptom_cols = [c for c in X.columns if c.lower() != "age"][:10]

    rates = []
    for col in symptom_cols:
        for cls in sorted(y.unique()):
            mask = y == cls
            rate = float(X.loc[mask, col].mean()) if X[col].dtype != object else np.nan
            rates.append({"feature": col, "class": int(cls), "rate": rate})
    rate_df = pd.DataFrame(rates)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=rate_df, x="feature", y="rate", hue="class", ax=ax)
    ax.set_title("Early-stage: symptom prevalence by class")
    plt.xticks(rotation=45, ha="right")
    _save(fig, out_dir / "early_stage_symptom_prevalence.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    y.value_counts().sort_index().plot(kind="bar", ax=ax, color=["steelblue", "coral"])
    ax.set_title("Early-stage: class distribution")
    ax.set_xlabel("class")
    _save(fig, out_dir / "early_stage_class_distribution.png")
