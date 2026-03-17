# TabularAutoML

> An end-to-end Automated Machine Learning platform for tabular classification — from raw CSV to deployed REST API, with zero manual model selection.

## Overview

Most ML projects stop at model training. **TabularAutoML** goes further — it automates the entire ML lifecycle:

- Ingests any tabular CSV or OpenML dataset
- Auto-preprocesses (missing values, encoding, scaling, outlier removal)
- Selects the best features automatically
- Trains **13 models**  (10 classical ML + 3 deep learning)
- Optimizes hyperparameters via **Bayesian HPO** (Optuna) for top candidates
- Builds a **stacking/voting ensemble** from the top 3 models
- Applies **Platt calibration** for well-calibrated probabilities
- Tracks every experiment in **MLflow**
- Serves predictions via a **FastAPI REST API**

The motivation: AutoML systems like Google AutoML and H2O exist but are black boxes. This project replicates the core architecture from scratch to understand every design decision — preprocessing order, why HPO only on 3 models, why interactions only for linear models, why stacking beats voting.


[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-orange)](https://mlflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-DL_models-red)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [API Endpoints](#api-endpoints)
- [Results](#results)
- [Future Work](#future-work)
- [Author](#author)

---

## System Architecture

```
Raw CSV / OpenML Dataset
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                    INGESTION LAYER                       │
│   DataLoader  ──►  DataValidator (schema + profiling)   │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                  PREPROCESSING LAYER                     │
│  OutlierHandler → MissingValueHandler → AutoEncoder     │
│                        → AutoScaler                     │
│                                                         │
│  ORDER MATTERS:                                         │
│  outlier removal → impute → encode → scale              │
│  (outliers distort scaler mean/std if not removed first)│
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│               FEATURE ENGINEERING LAYER                  │
│  FeatureSelector (auto: RFECV / SHAP / KBest by size)   │
│  InteractionFeatureGenerator (linear models only)       │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                   TRAINING LAYER                         │
│                                                         │
│  10 ML Models:                  3 DL Models:            │
│  LogisticRegression             MLP (PyTorch)           │
│  DecisionTree                   TabNet                  │
│  RandomForest ──► HPO           1D-CNN                  │
│  GradientBoosting                                       │
│  XGBoost      ──► HPO           HPO via Optuna          │
│  LightGBM     ──► HPO           (3 models only —        │
│  SVM                             highest ROI)           │
│  KNN                                                    │
│  ElasticNet                                             │
│  ExtraTrees                                             │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                   ENSEMBLE LAYER                         │
│  Top 3 models → Stacking vs Voting → Platt Calibration  │
│  Champion selected by validation AUC                    │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                  TRACKING & SERVING                      │
│  MLflow (metrics, params, artifacts, model registry)    │
│  FastAPI REST API (/train, /predict, /leaderboard)      │
└─────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Reason |
|---|---|
| HPO only on XGBoost, LightGBM, RandomForest | Highest AUC sensitivity + best ROI on compute (80/20 rule) |
| Interactions only for linear models | Trees learn interactions via splits; applying to all 13 models would bloat feature space unnecessarily |
| Stacking over Voting | Stacking learns optimal model weights; Voting weights equally |
| DL models not HPO'd | One HPO trial = 5–15 min for DL; early stopping acts as implicit regularization |
| Outlier removal before scaling | Outliers distort mean/std used by StandardScaler/RobustScaler |
| Feature selection before train/test split | Prevents data leakage from test set influencing feature selection |

---

## Project Structure

```
TabularAutoML/
├── src/
│   ├── ingestion/
│   │   ├── loader.py           # CSV + OpenML data loading
│   │   └── validator.py        # Schema validation, data profiling
│   ├── preprocessing/
│   │   ├── missing.py          # Missing value imputation
│   │   ├── encoder.py          # Auto categorical encoding
│   │   ├── scaler.py           # StandardScaler / RobustScaler (by skewness)
│   │   └── outlier.py          # IQR + IsolationForest
│   ├── feature_engineering/
│   │   ├── selector.py         # RFECV / SHAP / KBest auto selection
│   │   └── interactions.py     # Polynomial features (linear models only)
│   ├── models/
│   │   ├── ml_models.py        # 10 classical ML models
│   │   ├── dl_mlp.py           # PyTorch MLP
│   │   ├── dl_tabnet.py        # TabNet wrapper
│   │   ├── dl_cnn.py           # 1D CNN
│   │   └── ensemble.py         # Stacking + Voting ensemble builder
│   ├── hpo/
│   │   └── optuna_tuner.py     # Bayesian HPO for XGB/LGBM/RF
│   ├── pipeline/
│   │   └── automl_pipeline.py  # Master orchestrator
│   └── serving/
│       ├── api.py              # FastAPI REST endpoints
│       └── schemas.py          # Pydantic request/response schemas
├── data/
│   └── raw/                    # Place your CSV datasets here
├── mlruns/                     # MLflow artifacts (auto-generated)
├── mlflow.db                   # MLflow SQLite tracking database
├── automl.log                  # Training logs (auto-generated)
├── main.py                     # Entry point for training
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Category | Library | Purpose |
|---|---|---|
| ML Models | scikit-learn | 8 classical models + preprocessing |
| Boosting | XGBoost, LightGBM | Gradient boosting models |
| Deep Learning | PyTorch | MLP, 1D-CNN |
| Tabular DL | pytorch-tabnet | TabNet attention model |
| HPO | Optuna | Bayesian hyperparameter optimization |
| Experiment Tracking | MLflow | Metrics, params, model registry |
| API Serving | FastAPI | REST endpoints |
| Data | pandas, numpy | Data manipulation |
| OpenML | openml | Public dataset access |

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/shivang-maurya/TabularAutoML.git
cd TabularAutoML

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Option A — Train directly via Python (main.py)

The project was benchmarked on the **Electricity dataset** (OpenML ID 151). `main.py` is pre-configured to run it:

```bash
python main.py
```

This loads OpenML dataset 151 (45,312 rows, binary classification — electricity price UP or DOWN), trains all 13 models, builds the champion ensemble, and registers it in MLflow.

### Option B — Train on your own CSV

```python
import pandas as pd
from src.pipeline.automl_pipeline import AutoMLPipeline

df = pd.read_csv('data/raw/your_dataset.csv')
pipeline = AutoMLPipeline(n_hpo_trials=10, experiment_name='My-Run')
results = pipeline.run(df, target_col='your_target_column')

print(f"Champion AUC: {results['auc']:.4f}")
print(f"Top 3 models: {[r['name'] for r in results['all_results'][:3]]}")
```

### View MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open: http://localhost:5000
```

### Start FastAPI Server

> **Before starting**, apply these two fixes to `src/serving/api.py`:
>
> 1. Add global variable declarations (before `_load_last_state()`):
>    ```python
>    last_experiment_name = None
>    last_champion_version = None
>    ```
> 2. Fix the MLflow URI (line 46) to use the local database:
>    ```python
>    mlflow.set_tracking_uri('sqlite:///mlflow.db')
>    ```

```bash
uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI — you can call every endpoint directly from the browser.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health check + last experiment info |
| `POST` | `/train` | Upload CSV file → run full AutoML pipeline |
| `POST` | `/train/openml` | Train using OpenML dataset ID |
| `POST` | `/predict` | Send features as JSON → get prediction + probability |
| `GET` | `/leaderboard` | All model results from last training run, ranked by AUC |

---

### Example: Train via API

```bash
# Upload a CSV file
curl -X POST http://localhost:8000/train \
  -F "file=@data/raw/your_dataset.csv" \
  -F "target_col=your_target"
```

```bash
# Train on OpenML Electricity dataset (ID 151)
curl -X POST "http://localhost:8000/train/openml?dataset_id=151&target_col=target"
```

**Response:**
```json
{
  "status": "success",
  "best_model_name": "Champion-Ensemble",
  "auc_roc": 0.9743,
  "n_features_selected": 13,
  "top_3_models": ["LightGBM", "RandomForest", "ExtraTrees"]
}
```

---

### Example: Predict

The champion model was trained on the **Electricity dataset**. Features are electricity market readings for NSW and Victoria, Australia.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "date": 0.0,
      "day": 2,
      "period": 17,
      "nswprice": 0.056443,
      "nswdemand": 0.439155,
      "vicprice": 0.003467,
      "vicdemand": 0.422915,
      "transfer": 0.414912
    }
  }'
```

**Response:**
```json
{
  "predicted_class": 1,
  "probability": 0.8912,
  "all_probabilities": {
    "0": 0.1088,
    "1": 0.8912
  }
}
```

`predicted_class: 1` means the model predicts electricity price will go **UP**.
(`0` = DOWN, `1` = UP — LabelEncoder assigns alphabetically)

---

### Example: Leaderboard

```bash
curl http://localhost:8000/leaderboard
```

**Response** (actual results from training run on Electricity dataset):
```json
[
  {"tags.mlflow.runName": "Champion-Ensemble", "metrics.auc_roc": 0.9743, "metrics.f1_macro": null},
  {"tags.mlflow.runName": "LightGBM",          "metrics.auc_roc": 0.9723, "metrics.f1_macro": 0.9104},
  {"tags.mlflow.runName": "ExtraTrees",         "metrics.auc_roc": 0.9587, "metrics.f1_macro": 0.8886},
  {"tags.mlflow.runName": "XGBoost",            "metrics.auc_roc": 0.9566, "metrics.f1_macro": 0.8824},
  {"tags.mlflow.runName": "RandomForest",       "metrics.auc_roc": 0.9654, "metrics.f1_macro": 0.8978},
  {"tags.mlflow.runName": "GradientBoosting",   "metrics.auc_roc": 0.9227, "metrics.f1_macro": 0.8350},
  {"tags.mlflow.runName": "MLP",                "metrics.auc_roc": 0.9105, "metrics.f1_macro": 0.8236},
  {"tags.mlflow.runName": "KNN",                "metrics.auc_roc": 0.9103, "metrics.f1_macro": 0.8264}
]
```

### Example: Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "mlflow_uri": "sqlite:///mlflow.db",
  "last_experiment": "Credit-Demo",
  "last_champion_version": "5"
}
```

---

## Results

### Electricity Dataset (OpenML ID 151)

Binary classification: predict whether NSW electricity price goes UP or DOWN.
**45,312 rows · 8 features · Feature selection: all 8 kept (< 20 col threshold)**

| Model | AUC-ROC | F1-Macro | Notes |
|---|---|---|---|
| **Champion Ensemble** | **0.9743** | — | Stacking: LightGBM + RandomForest + ExtraTrees |
| LightGBM | 0.9723 | 0.9104 | Best single model |
| RandomForest | 0.9654 | 0.8978 | Default params (HPO skipped — dataset < 50k rows) |
| ExtraTrees | 0.9587 | 0.8886 | Default params |
| XGBoost | 0.9566 | 0.8824 | Default params (HPO skipped — dataset < 50k rows) |
| GradientBoosting | 0.9227 | 0.8350 | Default params |
| MLP | 0.9105 | 0.8236 | PyTorch, early stopping |
| KNN | 0.9103 | 0.8264 | Default params |
| TabNet | 0.8995 | 0.8141 | Attention-based tabular DL |
| 1D-CNN | 0.8928 | 0.8036 | Conv blocks on tabular data |
| SVM | 0.8758 | 0.7866 | RBF kernel |
| DecisionTree | 0.8375 | 0.7485 | Single tree, max_depth=5 |
| LogisticRegression | 0.8251 | 0.7395 | Linear baseline |
| ElasticNetClassifier | 0.8251 | 0.7395 | Linear baseline |

Ensemble selection: Stacking AUC=0.9724 vs Voting AUC=0.9712 → **Stacking selected**.
Champion evaluated on held-out test set: **AUC = 0.9743**.

---

### Credit-G Dataset (OpenML ID 31)

Binary classification: predict credit risk (good / bad).
**1,000 rows · 20 features · Feature selection: RFECV → 47 features after OHE expansion**

| Model | AUC-ROC | F1-Macro |
|---|---|---|
| **Champion Ensemble** | **0.8690** | — |
| RandomForest | 0.8610 | 0.6734 |
| ExtraTrees | 0.8586 | 0.7327 |
| XGBoost | 0.8537 | 0.7800 |
| LightGBM | 0.8461 | 0.7716 |
| ElasticNetClassifier | 0.8362 | 0.6943 |
| LogisticRegression | 0.8326 | 0.6795 |
| SVM | 0.8499 | 0.6686 |
| DecisionTree | 0.8289 | 0.6564 |
| GradientBoosting | 0.8254 | 0.7286 |
| KNN | 0.8010 | 0.6747 |
| MLP | 0.7926 | 0.6860 |
| 1D-CNN | 0.7372 | 0.6726 |
| TabNet | 0.4726 | 0.4402 |

Ensemble selection: Stacking AUC=0.7492 vs Voting AUC=0.7517 → **Voting selected**.

> Screenshots of MLflow UI, FastAPI /docs, and leaderboard available in `/docs/screenshots/`

---

## Future Work

| Feature | Description |
|---|---|
| Regression support | Switch loss functions, metrics, and model list for regression tasks |
| Airflow DAG | Scheduled weekly retraining with auto-promotion if AUC improves |
| Plotly Dash dashboard | Live leaderboard UI, SHAP charts, loss curves |
| Docker Compose | One-command deploy: `docker-compose up` |
| pytest test suite | Unit tests for all modules |
| Multi-GPU DL training | Faster MLP/TabNet/CNN training on large datasets |

---

## Author

**Shivang Maurya**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-shivang--maurya-blue?logo=linkedin)](https://www.linkedin.com/in/shivang-maurya-746a19322)

---
