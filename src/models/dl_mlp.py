# ─────────────────────────────────────────────────────────
# PURPOSE: PyTorch MLP with batch normalization, dropout, and early stopping.
#
# ARCHITECTURE: Input → [Linear → BatchNorm → ReLU → Dropout] x 4 → Output
#
# KEY COMPONENTS:
#   BatchNorm: normalizes activations during training → faster convergence
#   Dropout:   randomly zeros neurons → prevents overfitting
#   Early Stopping: stops training when validation loss stops improving
# ─────────────────────────────────────────────────────────
 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
 
 
class MLPNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int,
                 dropout_rate: float = 0.3):
        super(MLPNet, self).__init__()
        layers = []
        prev_dim = input_dim
 
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
 
 
class MLPClassifier(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"
    def __init__(self, hidden_dims=[256, 128, 64, 32], lr=0.001,
                 epochs=100, batch_size=256, dropout=0.3, patience=10):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.patience = patience
 
    def fit(self, X, y):
        X = X.values if hasattr(X, 'values') else np.array(X)
        y = y.values if hasattr(y, 'values') else np.array(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        input_dim = X.shape[1]
 
        self.model_ = MLPNet(input_dim, self.hidden_dims, n_classes, self.dropout)
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_.to(self.device_)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
 
        X_tensor = torch.FloatTensor(X).to(self.device_)
        y_tensor = torch.LongTensor(y).to(self.device_)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
 
        best_loss = float('inf')
        patience_counter = 0
 
        for epoch in range(self.epochs):
            self.model_.train()  # Set to training mode (enables dropout, batchnorm)
            epoch_loss = 0
 
            for X_batch, y_batch in loader:
                optimizer.zero_grad()          # Clear gradients from previous step
                outputs = self.model_(X_batch)  # Forward pass
                loss = criterion(outputs, y_batch)  # Compute loss
                loss.backward()                # Backpropagation — compute gradients
                optimizer.step()               # Update weights
                epoch_loss += loss.item()
 
            avg_loss = epoch_loss / len(loader)
 
            # Early stopping: stop if loss hasn't improved for `patience` epochs
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save the best model weights
                self.best_state_ = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break  # Stop training — no more improvement
 
        # Restore the best weights found during training
        self.model_.load_state_dict(self.best_state_)
        return self
 
    def predict_proba(self, X):
        X = X.values if hasattr(X, 'values') else np.array(X)
        """Return class probabilities (needed for AUC-ROC calculation)."""
        self.model_.eval()  # Disable dropout and batchnorm for inference
        with torch.no_grad():  # Don't compute gradients during inference (saves memory)
            X_tensor = torch.FloatTensor(X).to(self.device_)
            logits = self.model_(X_tensor)
            probs = torch.softmax(logits, dim=1)  # Convert logits to probabilities
        return probs.cpu().numpy()  # Move back to CPU and convert to numpy
 
    def predict(self, X):
        X = X.values if hasattr(X, 'values') else np.array(X)
        """Return predicted class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
