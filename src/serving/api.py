# ─────────────────────────────────────────────────────────
# api.py — FastAPI REST Service for Model Serving
# PURPOSE: Expose trained models as HTTP endpoints.
#
# ENDPOINTS:
#   POST /train          — Upload CSV, run full AutoML pipeline
#   POST /train/openml   — Train using OpenML dataset ID
#   POST /predict        — Send features, get prediction back
#   GET  /leaderboard    — View all model results from LAST training run
#   GET  /health         — Service health check
#
# WHY FASTAPI:
#   - Automatic OpenAPI docs (visit /docs in browser)
#   - Pydantic validation: bad inputs return clear error messages
#   - Async support: handles multiple requests concurrently
#   - Type hints throughout: easier to maintain and test
#
# DESIGN DECISIONS:
#   - last_experiment_name: tracks which experiment to show in leaderboard
#   - last_champion_version: tracks which model version to use for predict
#   - Both update automatically after each /train call
# ─────────────────────────────────────────────────────────

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import io
import math
import logging

from src.pipeline.automl_pipeline import AutoMLPipeline
from src.serving.schemas import PredictRequest, PredictResponse, TrainResponse

logger = logging.getLogger(__name__)

app = FastAPI(
    title='TabularAutoML',
    description='End-to-end Tabular AutoML: upload CSV → get trained model → get predictions',
    version='1.0.0'
)

# ── Must match URI used during training ──────────────────────────────────────
mlflow.set_tracking_uri('sqlite:////home/shivang/projects/AutoML/mlflow.db')

# Add this function after global variables in api.py
def _load_last_state():
    """On startup, find the most recent experiment from MLflow."""
    global last_experiment_name, last_champion_version
    try:
        client = mlflow.tracking.MlflowClient()
        # Get latest champion version
        versions = client.get_latest_versions('AutoML-Champion')
        if versions:
            last_champion_version = versions[0].version
        # Get most recent experiment
        experiments = client.search_experiments(
            order_by=['last_update_time DESC']
        )
        if experiments:
            last_experiment_name = experiments[0].name
        logger.info(f'Loaded state: experiment={last_experiment_name}, version={last_champion_version}')
    except Exception as e:
        logger.warning(f'Could not load last state: {e}')

# Call it at startup
_load_last_state()


@app.get('/health')
async def health_check():
    return {
        'status': 'healthy',
        'mlflow_uri': mlflow.get_tracking_uri(),
        'last_experiment': last_experiment_name,
        'last_champion_version': last_champion_version
    }


@app.post('/train', response_model=TrainResponse)
async def train_model(
    file: UploadFile = File(...),
    target_col: str = 'target',
    background_tasks: BackgroundTasks = None
):
    global last_experiment_name, last_champion_version

    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail='Only CSV files accepted')

    # Read file contents into memory
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    logger.info(f'Received file: {file.filename}, shape: {df.shape}')

    # Set experiment name
    experiment_name = f'run-{file.filename}'

    # Run AutoML pipeline
    pipeline = AutoMLPipeline(experiment_name=experiment_name)
    results = pipeline.run(df, target_col=target_col)

    # Update global state — leaderboard and predict will use this run
    last_experiment_name = experiment_name
    client = mlflow.tracking.MlflowClient()
    latest = client.get_latest_versions('AutoML-Champion')[0]
    last_champion_version = latest.version

    logger.info(f'Training complete. Experiment={experiment_name}, Champion version={last_champion_version}')

    return TrainResponse(
        status='success',
        best_model_name='Champion-Ensemble',
        auc_roc=round(results['auc'], 4),
        n_features_selected=len(results['selected_features']),
        top_3_models=[r['name'] for r in results['all_results'][:3]]
    )


@app.post('/train/openml', response_model=TrainResponse)
async def train_from_openml(dataset_id: int, target_col: str = 'target'):
    """
    Train using OpenML dataset ID instead of CSV upload.
    Example: dataset_id=31  → Credit-g (1000 rows, credit risk)
             dataset_id=151 → Electricity (45k rows, binary)
    """
    global last_experiment_name, last_champion_version

    try:
        import openml
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        df = X.copy()
        df[target_col] = y
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'OpenML fetch failed: {e}')

    # Set experiment name
    experiment_name = f'openml-{dataset_id}'

    # Run AutoML pipeline
    pipeline = AutoMLPipeline(experiment_name=experiment_name)
    results = pipeline.run(df, target_col=target_col)

    # Update global state
    last_experiment_name = experiment_name
    client = mlflow.tracking.MlflowClient()
    latest = client.get_latest_versions('AutoML-Champion')[0]
    last_champion_version = latest.version

    logger.info(f'Training complete. Experiment={experiment_name}, Champion version={last_champion_version}')

    return TrainResponse(
        status='success',
        best_model_name='Champion-Ensemble',
        auc_roc=round(results['auc'], 4),
        n_features_selected=len(results['selected_features']),
        top_3_models=[r['name'] for r in results['all_results'][:3]]
    )


@app.post('/predict', response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict using the champion model from the LAST training run.
    If no training has been done via API, falls back to Production stage.
    """
    try:
        # Use last trained version if available — otherwise use Production
        if last_champion_version is not None:
            model_uri = f'models:/AutoML-Champion/{last_champion_version}'
        else:
            model_uri = 'models:/AutoML-Champion/Production'

        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f'Loaded model: {model_uri}')

    except Exception as e:
        raise HTTPException(status_code=503, detail=f'No model found: {e}')

    # Convert request features to DataFrame
    input_df = pd.DataFrame([request.features])

    # Get probabilities
    proba = model.predict_proba(input_df)[0]
    predicted_class = int(np.argmax(proba))
    confidence = float(np.max(proba))

    return PredictResponse(
        predicted_class=predicted_class,
        probability=round(confidence, 4),
        all_probabilities={str(i): round(float(p), 4) for i, p in enumerate(proba)}
    )


@app.get('/leaderboard')
async def get_leaderboard():
    """
    Returns all model results from the LAST training run only.
    Ranked by AUC descending.
    """
    try:
        # Use last experiment if available — otherwise search all
        if last_experiment_name:
            runs = mlflow.search_runs(
                experiment_names=[last_experiment_name],
                order_by=['metrics.auc_roc DESC'],
                max_results=50
            )
        else:
            runs = mlflow.search_runs(
                search_all_experiments=True,
                order_by=['metrics.auc_roc DESC'],
                max_results=50
            )

        cols = ['tags.mlflow.runName', 'metrics.auc_roc', 'metrics.f1_macro', 'start_time']
        available_cols = [c for c in cols if c in runs.columns]
        result = runs[available_cols]

        # Replace NaN and Inf — JSON cannot serialize these
        result = result.where(result.notna(), other=None)

        # Clean any remaining non-JSON-compliant float values
        records = []
        for record in result.to_dict(orient='records'):
            clean = {}
            for k, v in record.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    clean[k] = None
                else:
                    clean[k] = v
            records.append(clean)

        return records

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'MLflow error: {str(e)}')