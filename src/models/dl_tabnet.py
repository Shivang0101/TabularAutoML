# TabNet Wrapper (sklearn-compatible)
# TabNet: Sequential attention for feature selection in tabular DL

 
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
 
class TabNetWrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"
    def __init__(self, n_steps=3, gamma=1.3, n_a=8, n_d=8,
                 learning_rate=0.02, max_epochs=200, patience=15):
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_a = n_a
        self.n_d = n_d
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
 
    def fit(self, X, y):
        self.model_ = TabNetClassifier(
            n_steps=self.n_steps,
            gamma=self.gamma,
            n_a=self.n_a,
            n_d=self.n_d,
            optimizer_params={'lr': self.learning_rate},
            verbose=0  # Suppress training output
        )
        # TabNet expects numpy arrays (not DataFrames)
        X_np = X.values if hasattr(X, 'values') else X
        y_np = y.values if hasattr(y, 'values') else y
 
        self.model_.fit(
            X_np, y_np,
            max_epochs=self.max_epochs,
            patience=self.patience,  # Early stopping
            batch_size=1024,
            virtual_batch_size=128   # TabNet's ghost batch normalization batch size
        )
        return self
 
    def predict_proba(self, X):
        X_np = X.values if hasattr(X, 'values') else X
        return self.model_.predict_proba(X_np)
 
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
