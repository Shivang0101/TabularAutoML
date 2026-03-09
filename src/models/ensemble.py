# ─────────────────────────────────────────────────────────
# ensemble.py — Stacking and Voting Ensemble Builder
# PURPOSE: Combine top-3 models to build a more robust final predictor.
#
# TWO ENSEMBLE METHODS:
#   Stacking:  Base models predict on held-out folds (out-of-fold predictions).
#              A meta-learner (Logistic Regression) learns to combine those predictions.
#              WHY: Meta-learner learns each model's strengths/weaknesses.
#
#   Voting:    Average predicted probabilities from all base models (soft voting).
#              WHY: Simple but effective. Reduces variance.
#
# AUTO-SELECT: Compares both on a holdout set and picks the winner.
# CALIBRATION: Applies Platt Scaling to ensure probabilities are well-calibrated.
#
# FIXES:
#   1. Multiclass-safe AUC — binary vs multiclass handled separately
#   2. Calibration data leakage fix — val set split into select + calib halves
#   3. DL models filtered out — CNN/MLP/TabNet incompatible with sklearn ensemble
# ─────────────────────────────────────────────────────────

from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from typing import List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# DL models are excluded from ensemble — they are not compatible with
# sklearn's internal estimator validation in VotingClassifier/StackingClassifier
DL_MODELS = {'MLP', 'TabNet', '1D-CNN'}


class EnsembleBuilder:

    def build_best_ensemble(self, top_models: List[Tuple[str, Any]],
                             X_train, y_train, X_val, y_val):
        """
        Builds, compares, and calibrates Stacking vs Voting ensemble.
        Automatically selects the better one based on AUC on holdout set.

        Args:
            top_models : list of (name, fitted_model) tuples — top-3 from leaderboard
            X_train    : training features
            y_train    : training labels
            X_val      : validation features (will be split internally)
            y_val      : validation labels

        Returns:
            calibrated : best ensemble wrapped in Platt Scaling calibration
        """

        # ── Filter out DL models ────────────────────────────────────────────
        # DL models (MLP, TabNet, CNN) are not compatible with sklearn's
        # VotingClassifier/StackingClassifier internal estimator validation.
        # They still compete on the leaderboard and can win individually —
        # just not inside the stacking/voting ensemble.
        sklearn_models = [(name, model) for name, model in top_models
                          if name not in DL_MODELS]

        # Fallback: if all top 3 happened to be DL models, use original list
        if len(sklearn_models) == 0:
            logger.warning('All top-3 models are DL — using original list for ensemble')
            sklearn_models = top_models

        logger.info(f'Ensemble using models: {[name for name, _ in sklearn_models]}')

        # ── Split val set into two halves ───────────────────────────────────
        # X_val_select → used to compare stacking vs voting AUC (model selection)
        # X_val_calib  → used to fit Platt calibration (avoids data leakage)
        # WHY: Using same data for both selection AND calibration = mild leakage.
        X_val_select, X_val_calib, y_val_select, y_val_calib = train_test_split(
            X_val, y_val, test_size=0.5, random_state=42, stratify=y_val
        )

        # ── Build Stacking Ensemble ─────────────────────────────────────────
        # StackingClassifier trains base models, then trains meta-learner
        # on their out-of-fold predictions (cv=5 means 5-fold OOF predictions)
        # stack_method='predict_proba': pass probabilities to meta-learner
        stacking = StackingClassifier(
            estimators=sklearn_models,
            final_estimator=LogisticRegression(C=1.0, max_iter=1000),
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        stacking.fit(X_train, y_train)

        # ── Build Voting Ensemble ───────────────────────────────────────────
        # VotingClassifier averages predicted probabilities from all base models
        # voting='soft': use probabilities not hard class labels → better calibrated
        voting = VotingClassifier(
            estimators=sklearn_models,
            voting='soft',
            n_jobs=-1
        )
        voting.fit(X_train, y_train)

        # ── Compare AUC on selection set ───────────────────────────────────
        # Binary → use [:, 1] directly
        # Multiclass → use ovr + macro averaging
        stack_probs = stacking.predict_proba(X_val_select)
        vote_probs  = voting.predict_proba(X_val_select)

        if stack_probs.shape[1] == 2:
            stacking_auc = roc_auc_score(y_val_select, stack_probs[:, 1])
            voting_auc   = roc_auc_score(y_val_select, vote_probs[:, 1])
        else:
            stacking_auc = roc_auc_score(
                y_val_select, stack_probs,
                multi_class='ovr', average='macro'
            )
            voting_auc = roc_auc_score(
                y_val_select, vote_probs,
                multi_class='ovr', average='macro'
            )

        logger.info(f'Stacking AUC: {stacking_auc:.4f} | Voting AUC: {voting_auc:.4f}')

        # ── Select better ensemble ──────────────────────────────────────────
        best = stacking if stacking_auc >= voting_auc else voting
        best_name = 'Stacking' if stacking_auc >= voting_auc else 'Voting'
        logger.info(f'Selected {best_name} ensemble as champion')

        # ── Platt Scaling Calibration ───────────────────────────────────────
        # Raw model probabilities are often overconfident.
        # Platt Scaling fits a logistic curve to map raw scores →
        # calibrated probabilities.
        # cv='prefit': model already trained, just fit the calibration layer
        # Fitted on X_val_calib — separate from selection set → no leakage
        calibrated = CalibratedClassifierCV(
            best,
            method='sigmoid',
            cv='prefit'
        )
        calibrated.fit(X_val_calib, y_val_calib)

        logger.info(f'Platt calibration fitted on {len(X_val_calib)} samples')

        return calibrated