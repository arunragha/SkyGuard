"""
Statistical baseline anomaly detector implementing Rolling Z-score and Hampel filter (rolling median & MAD).
"""

from typing import List, Optional
import numpy as np
import pandas as pd

from src.config import StatisticalConfig


class StatisticalBaselineDetector:
    """
    Statistical baseline detector using rolling Z-Score and robust Hampel filtering
    (rolling median and median absolute deviation).
    Catches localized impulse spikes, step changes, and sensor excursions.
    """

    def __init__(self, config: Optional[StatisticalConfig] = None):
        self.config = config or StatisticalConfig()
        self.hampel_window = self.config.hampel_window
        self.n_sigmas = self.config.n_sigmas
        self.z_score_window = self.config.z_score_window
        self.is_fitted = True

    def fit(self, X: np.ndarray) -> "StatisticalBaselineDetector":
        # Statistical baseline is non-parametric / rolling online
        self.is_fitted = True
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Computes composite statistical anomaly score in [0, 1] across all input feature channels.
        """
        if isinstance(X, pd.DataFrame):
            arr = X.values
        else:
            arr = np.asarray(X)

        n_samples, n_channels = arr.shape
        channel_scores = np.zeros((n_samples, n_channels))

        for c in range(n_channels):
            series = pd.Series(arr[:, c], dtype=float)

            # 1. Rolling Median and MAD (Hampel Filter)
            roll_med = series.rolling(window=self.hampel_window, min_periods=1).median()
            roll_med = roll_med.bfill().ffill().fillna(series)

            abs_diff = (series - roll_med).abs()
            mad = abs_diff.rolling(window=self.hampel_window, min_periods=1).median()
            mad = mad.bfill().ffill().fillna(1.0) * 1.4826  # Normal distribution scale factor
            mad = mad.replace(0.0, 1e-5)

            hampel_score = (abs_diff / mad).values

            # 2. Rolling Z-Score
            roll_mean = series.rolling(window=self.z_score_window, min_periods=1).mean().bfill().ffill().fillna(series)
            roll_std = series.rolling(window=self.z_score_window, min_periods=1).std().bfill().ffill().fillna(1.0)
            roll_std = roll_std.replace(0.0, 1e-5)

            z_score = ((series - roll_mean).abs() / roll_std).values

            # Combined channel anomaly metric: max(hampel, z)
            comb = np.maximum(hampel_score, z_score)
            # Map through soft saturation where n_sigmas (e.g. 3.0) corresponds to ~0.50
            norm_channel = 1.0 - np.exp(-0.25 * np.maximum(0.0, comb - 1.0))
            channel_scores[:, c] = norm_channel

        # Composite score across channels: peak channel + mean channel deviation
        max_channel = np.max(channel_scores, axis=1)
        mean_channel = np.mean(channel_scores, axis=1)
        composite = 0.7 * max_channel + 0.3 * mean_channel
        composite = np.nan_to_num(composite, nan=0.0)
        return np.clip(composite, 0.0, 1.0)
