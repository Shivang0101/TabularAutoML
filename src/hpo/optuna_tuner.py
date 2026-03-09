# ─────────────────────────────────────────────────────────
# optuna_tuner.py — Hyperparameter Optimization with Optuna
# PURPOSE: Automatically find optimal hyperparameters for each model.
#
# WHY OPTUNA vs GridSearchCV:
#   GridSearch: tries every combination → 10 params x 10 values = 10 billion trials
#   Optuna (TPE): uses Bayesian inference to focus on promising regions
#   → 50 trials with Optuna >> 1000 trials with GridSearch in terms of efficiency
#
# KEY CONCEPTS:
#   Trial: one evaluation of a hyperparameter configuration
#   Study: a collection of trials optimizing one objective
#   TPE Sampler: Tree-structured Parzen Estimator — the Bayesian algorithm
#   MedianPruner: kills bad trials early (before all epochs complete)
# ─────────────────────────────────────────────────────────
 
import optuna
import mlflow
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import logging
 
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)
 
 
class OptunaHPO:
 
    def __init__(self, n_trials: int = 50, cv_folds: int = 5,
                 scoring: str = 'roc_auc', timeout: int = 300):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.timeout = timeout

    def optimize(self, model_name: str, X_train, y_train) -> dict:
        logger.info(f'Starting HPO for {model_name} ({self.n_trials} trials)')

    # ── Adaptive scaling based on dataset size ──────────────────
    # Large datasets: each trial is expensive → reduce trials and folds
    # to keep total HPO time under ~30 mins on CPU
        n_samples = len(X_train)
        if n_samples > 50000:
            self.n_trials = 20
            self.timeout = 180
            self.cv_folds = 3
            logger.info(f'Large dataset ({n_samples} rows) detected → '
                        f'reduced to {self.n_trials} trials, {self.cv_folds} folds')
            if model_name == 'XGBoost':
                objective = self._xgboost_objective(X_train, y_train)
            elif model_name == 'LightGBM':
                objective = self._lgbm_objective(X_train, y_train)
            elif model_name == 'RandomForest':
                objective = self._rf_objective(X_train, y_train)
            else:
                raise ValueError(f'No HPO objective defined for {model_name}')
    
            # Create Optuna study
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
            )
    
            study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=True
            )
    
            logger.info(f'{model_name} best score: {study.best_value:.4f}')
            logger.info(f'{model_name} best params: {study.best_params}')
    
            return study.best_params
 
    def _xgboost_objective(self, X_train, y_train):
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
 
            model = XGBClassifier(
                **params,
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1
            )
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(
                model, X_train, y_train, cv=cv, scoring=self.scoring, n_jobs=-1
            )
            return scores.mean()
 
        return objective
 
    def _lgbm_objective(self, X_train, y_train):
        def objective(trial):
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
            model = LGBMClassifier(**params, verbose=-1, random_state=42)
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=self.scoring)
            return scores.mean()
        return objective
 
    def _rf_objective(self, X_train, y_train):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            }
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=self.scoring)
            return scores.mean()
        return objective
