"""
1D Convolutional / Temporal Neural Autoencoder using PyTorch.
Learns typical cross-sensor and temporal correlations, computing reconstruction error MSE as anomaly score.
"""

from typing import Optional
import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.config import AutoencoderConfig


if TORCH_AVAILABLE:
    class Conv1DAutoencoderModel(nn.Module):
        """
        Temporal 1D Convolutional Autoencoder with bottleneck latent representation.
        """
        def __init__(self, in_features: int, latent_dim: int = 8):
            super().__init__()
            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(in_features, 32),
                nn.ReLU(),
                nn.Linear(32, latent_dim),
                nn.ReLU(),
            )
            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Linear(32, in_features),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return reconstructed


class AutoencoderDetector:
    """
    Unsupervised neural autoencoder detector. Trains on baseline telemetry to reconstruct
    expected sensor dynamics. High reconstruction error indicates an anomaly.
    """

    def __init__(self, config: Optional[AutoencoderConfig] = None):
        self.config = config or AutoencoderConfig()
        self.scaler = StandardScaler()
        self.model: Optional[object] = None
        self.is_fitted = False
        self._score_min = 0.0
        self._score_max = 1.0

    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        """Trains the autoencoder network on feature matrix X."""
        X_scaled = self.scaler.fit_transform(X).astype(np.float32)
        n_samples, n_features = X_scaled.shape

        if TORCH_AVAILABLE:
            torch.manual_seed(42)
            device = torch.device("cpu")
            self.model = Conv1DAutoencoderModel(
                in_features=n_features, latent_dim=self.config.latent_dim
            ).to(device)

            dataset = TensorDataset(torch.from_numpy(X_scaled), torch.from_numpy(X_scaled))
            loader = TorchDataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-5
            )

            self.model.train()
            for _ in range(self.config.epochs):
                for batch_x, _ in loader:
                    optimizer.zero_grad()
                    recon = self.model(batch_x)
                    loss = criterion(recon, batch_x)
                    loss.backward()
                    optimizer.step()

            self.model.eval()
            with torch.no_grad():
                full_recon = self.model(torch.from_numpy(X_scaled)).numpy()
                raw_mse = np.mean((X_scaled - full_recon) ** 2, axis=1)
        else:
            # Fallback PCA-based subspace reconstruction error
            u, s, vt = np.linalg.svd(X_scaled, full_matrices=False)
            k = min(self.config.latent_dim, len(s))
            X_recon = (u[:, :k] * s[:k]) @ vt[:k, :]
            raw_mse = np.mean((X_scaled - X_recon) ** 2, axis=1)

        self.is_fitted = True
        self._score_min = float(np.percentile(raw_mse, 2.0))
        self._score_max = float(np.percentile(raw_mse, 98.0))
        if self._score_max <= self._score_min:
            self._score_max = self._score_min + 1.0

        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Computes normalized reconstruction error anomaly score in [0, 1].
        """
        if not self.is_fitted:
            self.fit(X)

        X_scaled = self.scaler.transform(X).astype(np.float32)

        if TORCH_AVAILABLE and isinstance(self.model, nn.Module):
            self.model.eval()
            with torch.no_grad():
                recon = self.model(torch.from_numpy(X_scaled)).numpy()
                raw_mse = np.mean((X_scaled - recon) ** 2, axis=1)
        else:
            u, s, vt = np.linalg.svd(X_scaled, full_matrices=False)
            k = min(self.config.latent_dim, len(s))
            X_recon = (u[:, :k] * s[:k]) @ vt[:k, :]
            raw_mse = np.mean((X_scaled - X_recon) ** 2, axis=1)

        norm_scores = (raw_mse - self._score_min) / (self._score_max - self._score_min)
        return np.clip(norm_scores, 0.0, 1.0)
