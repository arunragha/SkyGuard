"""
Ensemble meta-scorer aggregating scores across multiple unsupervised detectors.
"""

from typing import Dict, List, Optional
import numpy as np

from src.config import EnsembleConfig


class EnsembleMetaScorer:
    """
    Combines anomaly scores from Isolation Forest, Autoencoder, Statistical Baseline, and LOF
    using calibrated weighted aggregation.
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        raw_weights = self.config.weights
        total = sum(raw_weights.values())
        if total <= 0:
            total = 1.0
        self.weights = {k: v / total for k, v in raw_weights.items()}

    def combine(self, model_scores: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combines dictionary of {model_name: scores_array} into a single composite score in [0, 1].
        """
        if not model_scores:
            raise ValueError("model_scores dictionary cannot be empty")

        n_samples = len(next(iter(model_scores.values())))
        combined = np.zeros(n_samples)
        weight_sum = 0.0

        for name, scores in model_scores.items():
            w = self.weights.get(name, 0.25)
            combined += w * np.asarray(scores)
            weight_sum += w

        if weight_sum > 0:
            combined /= weight_sum

        return np.clip(combined, 0.0, 1.0)
