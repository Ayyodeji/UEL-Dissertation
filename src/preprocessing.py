"""Dataset-specific preprocessing utilities."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    IMBALANCE_RATIO_THRESHOLD,
    PIMA_ZERO_COLS,
    RANDOM_STATE,
    TEST_SIZE,
)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "") for c in df.columns]
    return df


def _match_zero_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for name in PIMA_ZERO_COLS:
        for c in df.columns:
            if c.lower() == name.lower():
                cols.append(c)
                break
    return cols


def preprocess_pima(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    df = _normalize_columns(df)
    y = df["target"].astype(int)
    X = df.drop(columns=["target"])

    zero_cols = _match_zero_cols(X)
    zeros_replaced = 0
    for col in zero_cols:
        mask = X[col] == 0
        zeros_replaced += int(mask.sum())
        X.loc[mask, col] = np.nan

    y = y.astype(int)
    meta = {
        "zeros_replaced": zeros_replaced,
        "zero_columns": zero_cols,
        "missing_after_zero_fix": X.isna().sum().to_dict(),
        "class_distribution": y.value_counts().to_dict(),
    }
    return X, y, meta


def preprocess_early_stage(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    df = _normalize_columns(df)
    target_col = None
    for c in df.columns:
        if c.lower() in {"class", "target", "outcome"}:
            target_col = c
            break
    if target_col is None:
        target_col = df.columns[-1]

    y_raw = df[target_col]
    X = df.drop(columns=[target_col])

    encoding_map: dict[str, dict[str, int]] = {}
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == "category":
            uniq = sorted(X[col].dropna().astype(str).unique())
            if set(uniq) <= {"Yes", "No", "yes", "no"}:
                mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0}
                X[col] = X[col].astype(str).map(mapping)
                encoding_map[col] = mapping
            elif col.lower() == "gender":
                mapping = {}
                for i, val in enumerate(sorted(X[col].dropna().astype(str).unique())):
                    mapping[val] = i
                X[col] = X[col].astype(str).map(mapping)
                encoding_map[col] = mapping
            else:
                mapping = {v: i for i, v in enumerate(sorted(X[col].dropna().astype(str).unique()))}
                X[col] = X[col].astype(str).map(mapping)
                encoding_map[col] = mapping

    y = y_raw.astype(str).str.strip().str.lower()
    pos_labels = {"positive", "1", "yes", "diabetic", "p"}
    y = y.apply(lambda v: 1 if v in pos_labels or v == "1" else 0).astype(int)

    meta = {
        "encoding_map": encoding_map,
        "missing": X.isna().sum().to_dict(),
        "class_distribution": y.value_counts().to_dict(),
    }
    return X, y, meta


def stratified_split(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def needs_smote(y_train: pd.Series) -> bool:
    counts = y_train.value_counts()
    if len(counts) < 2:
        return False
    minority = counts.min()
    majority = counts.max()
    return (minority / majority) < IMBALANCE_RATIO_THRESHOLD


def build_preprocessor(X: pd.DataFrame, dataset: str) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        return ColumnTransformer(
            [("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols)]
        )
    return ColumnTransformer(transformers)


def save_json(path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
