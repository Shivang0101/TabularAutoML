# ─────────────────────────────────────────────────────────
# interactions.py — Polynomial and Interaction Feature Generator
#
# PURPOSE: Generate polynomial and interaction features for LINEAR models only.
#
# WHY ONLY LINEAR MODELS:
#   Linear models (LogReg, ElasticNet) cannot learn feature interactions
#   on their own — age × salary is invisible to them unless we create it.
#   Tree models (RF, XGBoost...) learn interactions via splits naturally.
#   Neural networks (MLP, TabNet, CNN) learn interactions via layers.
#   → Applying this to non-linear models wastes memory and can hurt accuracy.
#
# MODELS THAT NEED THIS (2 out of 13):
#    LogisticRegression  — linear, can't learn interactions alone
#    ElasticNet          — linear, can't learn interactions alone
#
# MODELS THAT DON'T (11 out of 13):
#    DecisionTree, RandomForest, GradientBoosting,
#      XGBoost, LightGBM, ExtraTrees  — tree splits handle interactions
#    SVM                            — kernel handles non-linearity
#    KNN                            — more features = worse distance calc
#    MLP, TabNet, 1D-CNN            — layers/attention handle interactions
#
# FEATURE EXPLOSION WARNING:
#   10 cols → degree=2 → ~65 new columns
#   20 cols → degree=2 → ~230 new columns
#   Always guard with max_cols to prevent this.
# ─────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import logging

logger = logging.getLogger(__name__)
LINEAR_MODELS = {'LogisticRegression', 'ElasticNet'}


class InteractionFeatureGenerator:

    def __init__(self, degree: int = 2, max_cols: int = 8):
        self.degree = degree
        self.max_cols = max_cols
        self._poly = None
        self._new_feature_names = []
        self._numeric_cols = []
        self.fitted_ = False  # always initialize to False

    def needs_interactions(self, model_name: str) -> bool:
        return model_name in LINEAR_MODELS

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Reset fitted state each time fit_transform is called
        self.fitted_ = False
        self._poly = None

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self._numeric_cols = numeric_cols

        if len(numeric_cols) > self.max_cols:
            logger.warning(
                f'[InteractionFeatureGenerator] Skipped — {len(numeric_cols)} numeric '
                f'columns exceeds max_cols={self.max_cols}. '
                f'Would generate ~{len(numeric_cols)**2} new features. '
                f'Returning original DataFrame.'
            )
            # fitted_ stays False — transform() will return X as-is
            return X

        logger.info(
            f'[InteractionFeatureGenerator] Generating degree={self.degree} '
            f'interactions for {len(numeric_cols)} numeric columns.'
        )
        self._poly = PolynomialFeatures(
            degree=self.degree,
            interaction_only=False,
            include_bias=False
        )
        poly_data = self._poly.fit_transform(X[numeric_cols])
        self._new_feature_names = self._poly.get_feature_names_out(numeric_cols).tolist()
        non_numeric = X.select_dtypes(exclude=[np.number])
        poly_df = pd.DataFrame(poly_data, columns=self._new_feature_names, index=X.index)

        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1:]:
                denom = X[col2].replace(0, np.nan)
                poly_df[f'{col1}_div_{col2}'] = X[col1] / denom

        result = pd.concat([non_numeric, poly_df], axis=1).fillna(0)

        # Mark as successfully fitted
        self.fitted_ = True

        logger.info(
            f'[InteractionFeatureGenerator] '
            f'{X.shape[1]} columns → {result.shape[1]} columns after interactions.'
        )
        return result

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # If fit_transform was skipped — return original data silently
        # This handles the case where numeric cols > max_cols
        if not self.fitted_:
            return X

        non_numeric = X.select_dtypes(exclude=[np.number])
        poly_data = self._poly.transform(X[self._numeric_cols])
        poly_df = pd.DataFrame(poly_data, columns=self._new_feature_names, index=X.index)

        for i, col1 in enumerate(self._numeric_cols):
            for col2 in self._numeric_cols[i + 1:]:
                denom = X[col2].replace(0, np.nan)
                poly_df[f'{col1}_div_{col2}'] = X[col1] / denom

        return pd.concat([non_numeric, poly_df], axis=1).fillna(0)