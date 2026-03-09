# ─────────────────────────────────────────────────────────
# selector.py — Feature Selection
# PURPOSE: Identify and keep only the most informative features.
#
# FOUR METHODS (auto-selected by default based on dataset size):
#   1. RFECV:     Recursive Feature Elimination with Cross-Validation.
#                 Trains a model, removes least important feature, repeats.
#                 Optimal feature count chosen by CV score.
#                 ⚠ Slow — only used on small datasets (< 10k rows).
#   2. SelectKBest: Statistical tests (mutual information score).
#                 Fast filter method — ranks features by statistical
#                 correlation with target. No model training needed.
#                 Used on large datasets (>= 50k rows).
#   3. SHAP:      Train XGBoost, compute SHAP values, drop near-zero features.
#                 SHAP = SHapley Additive exPlanations — gold standard for
#                 feature importance. Handles multiclass automatically.
#                 Used on medium datasets (10k–50k rows).
#   4. auto:      Picks method based on dataset size (default).
#                 < 10k rows   → rfecv   (slow but most accurate)
#                 10k–50k rows → shap    (best quality/speed tradeoff)
#                 >= 50k rows  → kbest   (fastest, scales to millions)
#
# WHY auto IS THE RIGHT DEFAULT FOR AutoML:
#   An AutoML platform accepts unknown datasets of any size.
#   Hardcoding one method would either be too slow (rfecv on 500k rows)
#   or too weak (kbest on 500 rows). auto adapts to whatever comes in.
# ─────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV, SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import shap
from typing import List
import logging

logger = logging.getLogger(__name__)

# ── Dataset size thresholds for auto mode 
_RFECV_MAX_ROWS   = 10_000   
_KBEST_MIN_ROWS   = 50_000   


class FeatureSelector:

    def __init__(self, method: str = 'auto', k_best: int = 20):
        self.method = method      # 'auto', 'rfecv', 'kbest', 'shap'
        self.k_best = k_best      # used by kbest — how many features to keep
        self.selected_features: List[str] = []
        self._method_used: str = ''   # track which method auto actually picked

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        MIN_COLS_FOR_SELECTION = 20  # below this, keep everything
        if X.shape[1] < MIN_COLS_FOR_SELECTION:
            self.selected_features = X.columns.tolist()  # keep all
            self._method_used = 'none (too few columns)'
            logger.info(
                f'[FeatureSelector] Skipped — only {X.shape[1]} columns '
                f'(threshold: {MIN_COLS_FOR_SELECTION}). Keeping all features.'
            )
            return X
        if self.method == 'auto':
            n_rows = len(X)
            if n_rows < _RFECV_MAX_ROWS:
                chosen = 'rfecv'
            elif n_rows < _KBEST_MIN_ROWS:
                chosen = 'shap'
            else:
                chosen = 'kbest'
            self._method_used = chosen
            logger.info(
                f'[FeatureSelector] auto mode: {n_rows} rows → selected method={chosen}'
            )
        else:
            chosen = self.method
            self._method_used = chosen
        if chosen == 'rfecv':
            return self._rfecv(X, y)
        elif chosen == 'kbest':
            return self._kbest(X, y)
        elif chosen == 'shap':
            return self._shap_selection(X, y)
        else:
            raise ValueError(
                f"Unknown method: '{chosen}'. "
                f"Choose from: 'auto', 'rfecv', 'kbest', 'shap'."
            )

    def _rfecv(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        if len(X) > _RFECV_MAX_ROWS:
            logger.warning(
                f'[FeatureSelector] RFECV on {len(X)} rows may be very slow '
                f'(trains ~{X.shape[1] * 5} models). '
                f'Consider method=shap for faster results.'
            )

        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        selector = RFECV(
            estimator, cv=5, scoring='roc_auc',
            n_jobs=-1, min_features_to_select=10
        )
        selector.fit(X, y)
        self.selected_features = X.columns[selector.support_].tolist()
        logger.info(f'[FeatureSelector] RFECV kept {len(self.selected_features)} features.')
        return X[self.selected_features]

    def _kbest(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        k = min(self.k_best, X.shape[1])   # can't select more features than exist
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        selector.fit(X, y)
        self.selected_features = X.columns[selector.get_support()].tolist()
        logger.info(f'[FeatureSelector] KBest kept {len(self.selected_features)} features.')
        return X[self.selected_features]

    def _shap_selection(self, X: pd.DataFrame, y: pd.Series,
                        shap_threshold: float = 0.001) -> pd.DataFrame:

        xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        xgb.fit(X, y)

        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            mean_abs_shap = np.mean(
                [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
            )
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

        self.selected_features = [
            col for col, importance in zip(X.columns, mean_abs_shap)
            if importance >= shap_threshold
        ]
        logger.info(f'[FeatureSelector] SHAP kept {len(self.selected_features)} features.')
        return X[self.selected_features]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_features:
            raise RuntimeError(
                '[FeatureSelector] transform() called before fit_transform(). '
                'Run fit_transform(X_train, y_train) first.'
            )
        return X[self.selected_features]

    def get_method_used(self) -> str:
        return self._method_used