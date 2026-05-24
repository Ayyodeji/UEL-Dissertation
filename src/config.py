"""Project configuration for diabetes prediction pipeline."""
from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_ITER_TUNING = 20

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Pima: OpenML data_id=37 (UCI id 34 not importable via ucimlrepo)
PIMA_OPENML_ID = 37
EARLY_STAGE_UCI_ID = 529

PIMA_COLUMN_MAP = {
    "preg": "Pregnancies",
    "plas": "Glucose",
    "pres": "BloodPressure",
    "skin": "SkinThickness",
    "insu": "Insulin",
    "mass": "BMI",
    "pedi": "DiabetesPedigreeFunction",
    "age": "Age",
}

PIMA_ZERO_COLS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

# Use SMOTE when minority class is below this fraction of majority
IMBALANCE_RATIO_THRESHOLD = 0.8
