# PURPOSE: Single entry point to run the entire AutoML workflow.
#          Ties together ALL other modules in the correct sequence.
#
# FLOW:
#   1. Load & validate data
#   2. Auto-preprocess (missing, encode, scale, outlier)
#   3. Feature engineering (feature selection)
#   4. Train/test split
#   5. Train all 13 models with HPO
#      └─ interaction features applied INSIDE loop for linear models only
#   6. Build ensemble from top-3
#   7. Log everything to MLflow
#   8. Register champion model in MLflow Model Registry



import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, log_loss
import logging

#Custom Functions
from src.ingestion.loader import DataLoader
from src.ingestion.validator import DataValidator
from src.preprocessing.missing import MissingValueHandler
from src.preprocessing.encoder import AutoEncoder
from src.preprocessing.scaler import AutoScaler
from src.preprocessing.outlier import OutlierHandler
from src.feature_engineering.selector import FeatureSelector
from src.feature_engineering.interactions import InteractionFeatureGenerator, LINEAR_MODELS
from src.models.ml_models import get_ml_models
from src.models.dl_mlp import MLPClassifier
from src.models.dl_tabnet import TabNetWrapper
from src.models.dl_cnn import CNNClassifier
from src.models.ensemble import EnsembleBuilder
from src.hpo.optuna_tuner import OptunaHPO

logger = logging.getLogger(__name__)


class AutoMLPipeline:

    def __init__(self, n_hpo_trials: int = 50, experiment_name: str = 'AutoML-Run'):

        self.loader              = DataLoader()
        self.validator           = DataValidator()
        self.missing_handler     = MissingValueHandler()
        self.encoder             = AutoEncoder()
        self.scaler              = AutoScaler()
        self.outlier_handler     = OutlierHandler()
        self.feature_selector    = FeatureSelector(method='auto')   
        self.interaction_gen     = InteractionFeatureGenerator()   
        self.hpo                 = OptunaHPO(n_trials=n_hpo_trials)
        self.ensemble_builder    = EnsembleBuilder()

        # MLflow experiment name — groups all runs together in the UI
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def run(self, df: pd.DataFrame, target_col: str) -> dict:

        logger.info('=== AutoML Pipeline Starting ===')

        # ── STEP 1: Validate 
        report = self.validator.validate(df, target_col)
        if not report.is_valid:
            raise ValueError(f'Data validation failed: {report.errors}')
        for warning in report.warnings:
            logger.warning(warning)

        # ── STEP 2: Preprocess
        # ORDER :
        #   outlier removal → missing values → encode → scale
        #   encoding before scaling: encoder outputs numbers, scaler needs numbers
        df_clean = self.outlier_handler.remove_outliers(df, target_col)
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]

        X = self.missing_handler.fit_transform(X)
        X = self.encoder.fit_transform(X, y)
        X = self.scaler.fit_transform(X)

        # ── STEP 3: Feature Selection 

        X = self.feature_selector.fit_transform(X, y)
        X.columns = [str(col).replace('[', '_')
                      .replace(']', '_')
                      .replace('<', '_')
                      .replace('>', '_')
                      .replace(' ', '_') 
             for col in X.columns]
        logger.info(
            f'Selected {X.shape[1]} features out of {df.shape[1] - 1} original '
            f'(method: {self.feature_selector.get_method_used()})'
        )

        # ── STEP 4: Train/Test Split
        # stratify=y ensures both splits have same class distribution.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # ── STEP 5: Train All 13 Models 
        all_results = []
        all_models = get_ml_models() + [
            ('MLP',    MLPClassifier()),
            ('TabNet', TabNetWrapper()),
            ('1D-CNN', CNNClassifier()),
        ]

        for name, model in all_models:
            with mlflow.start_run(run_name=name):
                logger.info(f'Training {name}...')

                if name in LINEAR_MODELS:
                    logger.info(
                        f'[{name}] Linear model — applying interaction features.'
                    )
                    X_tr_final   = self.interaction_gen.fit_transform(X_tr)
                    X_test_final = self.interaction_gen.transform(X_test)
                    X_val_final  = self.interaction_gen.transform(X_val)
                else:
                    X_tr_final   = X_tr
                    X_test_final = X_test
                    X_val_final  = X_val

                # ── HPO: find best hyperparameters using Optuna
                try:
                    best_params = self.hpo.optimize(name, X_tr_final, y_tr)
                    if best_params is not None:
                        model.set_params(**best_params)
                        mlflow.log_params(best_params)
                except (ValueError, AttributeError):
                    # Some models don't have HPO defined — use defaults
                    logger.info(f'Using default params for {name}')

                model.fit(X_tr_final, y_tr)

                y_proba_all = model.predict_proba(X_test_final)
                if y_proba_all.shape[1] == 2:
                    y_proba = y_proba_all[:, 1]
                    auc = roc_auc_score(y_test, y_proba)
                else:
                    y_proba = y_proba_all
                    auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
                f1       = f1_score(y_test, model.predict(X_test_final), average='macro')
                logloss  = log_loss(y_test, y_proba_all)

                # Log all metrics to MLflow for comparison in the UI
                mlflow.log_metric('auc_roc',   auc)
                mlflow.log_metric('f1_macro',  f1)
                mlflow.log_metric('log_loss',  logloss)
                mlflow.sklearn.log_model(model, name)

                all_results.append({
                    'name': name, 'model': model,
                    'auc': auc,
                    # store which X_val this model needs for ensemble building
                    'X_val': X_val_final
                })
                logger.info(f'{name}: AUC={auc:.4f}, F1={f1:.4f}')

        # ── STEP 6: Build Ensemble 
        # Sort by AUC, take top 3 models for stacking
        all_results.sort(key=lambda x: x['auc'], reverse=True)
        top3 = [(r['name'], r['model']) for r in all_results[:3]]
        logger.info(f'Top 3 models: {[r["name"] for r in all_results[:3]]}')

        champion = self.ensemble_builder.build_best_ensemble(
            top3, X_tr, y_tr, X_val, y_val
        )

        # Final evaluation of champion ensemble on held-out test set
        champ_proba = champion.predict_proba(X_test)
        if champ_proba.shape[1] == 2:
            champion_auc = roc_auc_score(y_test, champ_proba[:, 1])
        else:
            champion_auc = roc_auc_score(y_test, champ_proba, multi_class='ovr', average='macro')
        logger.info(f'Champion ensemble AUC: {champion_auc:.4f}')

        # ── STEP 7: Register in MLflow Model Registry 
        with mlflow.start_run(run_name='Champion-Ensemble'):
            mlflow.log_metric('auc_roc', champion_auc)
            mlflow.sklearn.log_model(champion, 'champion-model')
            # Register model first
            mlflow.register_model(
                f'runs:/{mlflow.active_run().info.run_id}/champion-model',
                'AutoML-Champion'
            )

        # THEN promote — after registration is complete
        client = mlflow.tracking.MlflowClient()
        latest = client.get_latest_versions('AutoML-Champion')[0]
        client.transition_model_version_stage(
            name='AutoML-Champion',
            version=latest.version,
            stage='Production'
        )
        logger.info(f'Champion version {latest.version} promoted to Production')

        return {
            'champion':          champion,
            'auc':               champion_auc,
            'all_results':       all_results,
            'selected_features': self.feature_selector.selected_features,
        }