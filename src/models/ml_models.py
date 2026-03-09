# ─────────────────────────────────────────────────────────
# ml_models.py — All 10 ML Model Definitions
# PURPOSE: Central registry of all classical ML models.
#          Returns configured model instances ready for HPO and training.
#
# ALGORITHM SELECTION RATIONALE:
#   Linear models (LR): Fast baseline, interpretable, good for linearly separable data
#   Tree models (DT, RF, ET): Handle non-linear patterns, feature interactions
#   Boosting (GBM, XGB, LGBM): Usually highest accuracy, sequential error correction
#   Kernel (SVM): Excellent for small datasets with complex boundaries
#   Instance (KNN): Non-parametric, adapts to data shape automatically
#   Regularized (ElasticNet): Regression tasks, feature selection built-in
# ─────────────────────────────────────────────────────────

from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import Dict, Any, List, Tuple
 
 
def get_ml_models(task: str = 'classification') -> List[Tuple[str, Any]]:
    if task == 'classification':
        return [
            ('LogisticRegression', LogisticRegression(
                C=1.0, solver='lbfgs', max_iter=1000, random_state=42
            )),
            ('DecisionTree', DecisionTreeClassifier(
                max_depth=5, min_samples_split=20, random_state=42
            )),
            ('RandomForest', RandomForestClassifier(
                n_estimators=200, max_features='sqrt', n_jobs=-1, random_state=42
            )),
            ('GradientBoosting', GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, subsample=0.8, random_state=42
            )),
            ('XGBoost', XGBClassifier(
                n_estimators=300, eta=0.05, max_depth=6, subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss', random_state=42, n_jobs=-1
            )),
            ('LightGBM', LGBMClassifier(
                n_estimators=300, learning_rate=0.05, num_leaves=63,
                subsample=0.8, colsample_bytree=0.8, verbose=-1, random_state=42
            )),
            ('SVM', SVC(
                C=1.0, kernel='rbf', probability=True, random_state=42
            )),
            ('KNN', KNeighborsClassifier(
                n_neighbors=7, weights='distance', n_jobs=-1
            )),
            # ElasticNet penalty inside LogisticRegression = works for classification
# l1_ratio=0.5 means equal L1 + L2 mix, same concept, correct model type
            ('ElasticNetClassifier', LogisticRegression(
                C=1.0, penalty='elasticnet', solver='saga',
                l1_ratio=0.5, max_iter=1000, random_state=42
            )),
            ('ExtraTrees', ExtraTreesClassifier(
                n_estimators=200, max_features='sqrt', n_jobs=-1, random_state=42
            )),
        ]
    else:
        raise NotImplementedError('Regression model list — add sklearn regressors here')
