"""Load datasets: Pima via OpenML, early-stage via ucimlrepo."""
from __future__ import annotations

import pandas as pd
from sklearn.datasets import fetch_openml
from ucimlrepo import fetch_ucirepo

from .config import EARLY_STAGE_UCI_ID, PIMA_COLUMN_MAP, PIMA_OPENML_ID


def load_pima() -> pd.DataFrame:
    """Pima Indians Diabetes (768 rows) via OpenML id=37."""
    bundle = fetch_openml(data_id=PIMA_OPENML_ID, as_frame=True, parser="auto")
    df = bundle.data.copy()
    if "class" in df.columns:
        y = df["class"]
        df = df.drop(columns=["class"])
    else:
        y = bundle.target
    df = df.rename(columns=PIMA_COLUMN_MAP)
    df["target"] = (y.astype(str).str.contains("positive", case=False)).astype(int)
    return df


def load_early_stage() -> pd.DataFrame:
    """Early Stage Diabetes Risk Prediction (520 rows) via UCI id=529."""
    bundle = fetch_ucirepo(id=EARLY_STAGE_UCI_ID)
    X = bundle.data.features.copy()
    y = bundle.data.targets.iloc[:, 0]
    df = X.copy()
    df["target"] = y.values
    return df
