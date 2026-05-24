# Early Diabetes Prediction — ML Pipeline (Chapter 4)

Reproducible pipeline for the UEL dissertation: Pima (clinical) and UCI Early Stage (symptom/lifestyle) datasets, analysed **separately**.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python -m src.run_pipeline
```

## Outputs

| Path | Content |
|------|---------|
| `results/pima/` | Baseline CSV, hybrid comparison, preprocessing JSON, saved models |
| `results/early_stage/` | Same for UCI early-stage dataset |
| `figures/pima/` | EDA, confusion matrices, SHAP |
| `figures/early_stage/` | EDA, confusion matrices, SHAP |

Chapter 4 tables in the dissertation markdown are updated with baseline, hybrid, and feature-importance values after each run.

## Export PDF and Word

```bash
./scripts/export_dissertation.sh
```

Requires `pandoc` and `tectonic` (`brew install pandoc tectonic`). Outputs:

- `Hybrid Machine Learning Model for Early Diabetes Prediction Using Lifestyle and Clinical Data.docx`
- `Hybrid Machine Learning Model for Early Diabetes Prediction Using Lifestyle and Clinical Data.pdf`

## Datasets

- Pima Indians Diabetes — OpenML data id **37** (768 rows; UCI id 34 is not importable via `ucimlrepo`)
- Early Stage Diabetes Risk Prediction — UCI id **529** via `ucimlrepo`
