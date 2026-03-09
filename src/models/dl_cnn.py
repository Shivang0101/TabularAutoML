# ─────────────────────────────────────────────────────────
# PURPOSE: PyTorch 1D-CNN for tabular classification.
#          Treats each feature as a "channel" and applies
#          convolutions to learn local feature interactions.
#
# ARCHITECTURE:
#   Input (batch, 1, n_features)
#     → [Conv1d → BatchNorm1d → ReLU → Dropout] x 3
#     → AdaptiveAvgPool1d(1)   ← collapses spatial dimension
#     → Flatten
#     → [Linear → ReLU → Dropout] x 2
#     → Output (n_classes)
#
# KEY COMPONENTS:
#   Conv1d:            slides a kernel over features → captures local interactions
#   BatchNorm1d:       normalizes conv outputs → stable, faster training
#   AdaptiveAvgPool1d: reduces any feature-length to size 1 → flexible input sizes
#   Early Stopping:    stops training when loss stops improving → prevents overfitting
# ─────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class CNNNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 num_filters: int = 64, kernel_size: int = 3,
                 dropout_rate: float = 0.3):
        super(CNNNet, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=num_filters,
                      kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2,
                      kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=num_filters * 2, out_channels=num_filters * 4,
                      kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters * 4),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_filters * 4, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, output_dim)  # final logits — one per class
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape coming in: (batch, n_features)
        x = x.unsqueeze(1)          # → (batch, 1, n_features) for Conv1d

        x = self.conv_block1(x)     # → (batch, num_filters, n_features)
        x = self.conv_block2(x)     # → (batch, num_filters*2, n_features)
        x = self.conv_block3(x)     # → (batch, num_filters*4, n_features)

        x = self.global_pool(x)     # → (batch, num_filters*4, 1)
        x = x.squeeze(-1)           # → (batch, num_filters*4)  flatten last dim

        return self.fc(x)           # → (batch, n_classes)


class CNNClassifier(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"
    def __init__(self, num_filters=64, kernel_size=3, lr=0.001,
                 epochs=100, batch_size=256, dropout=0.3, patience=10):
        self.num_filters = num_filters    # number of conv filters in first block
        self.kernel_size = kernel_size    # how many adjacent features each filter sees
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.patience = patience          # early stopping patience in epochs

    def fit(self, X, y):
        # numpy safety: accepts both DataFrames and numpy arrays
        X_np = X.values if hasattr(X, 'values') else np.array(X)
        y_np = y.values if hasattr(y, 'values') else np.array(y)

        self.classes_ = np.unique(y_np)
        n_classes = len(self.classes_)
        input_dim = X_np.shape[1]

        # build model and move to GPU if available
        self.model_ = CNNNet(input_dim, n_classes,
                             num_filters=self.num_filters,
                             kernel_size=self.kernel_size,
                             dropout_rate=self.dropout)
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_.to(self.device_)

        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        # wrap data in PyTorch DataLoader for batched training
        X_tensor = torch.FloatTensor(X_np).to(self.device_)
        y_tensor = torch.LongTensor(y_np).to(self.device_)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model_.train()   # enables dropout and batchnorm training behavior
            epoch_loss = 0

            for X_batch, y_batch in loader:
                optimizer.zero_grad()            # clear gradients from previous step
                outputs = self.model_(X_batch)   # forward pass
                loss = criterion(outputs, y_batch)  # compute cross-entropy loss
                loss.backward()                  # backpropagation — compute gradients
                optimizer.step()                 # update weights
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)

            # early stopping: save best weights, stop if no improvement
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # snapshot the best weights seen so far
                self.best_state_ = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break   # no improvement for `patience` epochs → stop training

        # restore the best weights (not necessarily the last epoch's weights)
        self.model_.load_state_dict(self.best_state_)
        return self

    def predict_proba(self, X):
        """Return class probabilities (needed for AUC-ROC and ensemble stacking)."""
        X_np = X.values if hasattr(X, 'values') else np.array(X)

        self.model_.eval()   # disables dropout and batchnorm for inference
        with torch.no_grad():   # don't compute gradients → saves memory at inference
            X_tensor = torch.FloatTensor(X_np).to(self.device_)
            logits = self.model_(X_tensor)
            probs = torch.softmax(logits, dim=1)   # convert raw logits → probabilities
        return probs.cpu().numpy()   # move back to CPU and convert to numpy

    def predict(self, X):
        """Return predicted class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)   # class with highest probability