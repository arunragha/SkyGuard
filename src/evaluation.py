"""
Unsupervised model comparison and evaluation suite against synthetic injected ground-truth anomalies.
Reports Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and fault archetype breakdown.
"""

import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import AppConfig, load_config
from src.data.synthetic import WeatherAnomalyGenerator
from src.detector import SkyGuardDetector


class ModelEvaluator:
    """
    Evaluates unsupervised anomaly detection methods against synthetic injected sensor faults.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.generator = WeatherAnomalyGenerator(self.config, seed=42)
        self.detector = SkyGuardDetector(self.config)

    def run_benchmark(
        self,
        days: int = 14,
        anomaly_ratio: float = 0.05,
        sensitivity: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Runs complete benchmark comparison across all models.
        """
        # 1. Generate synthetic dataset
        train_df = self.generator.generate(
            stations=["STATION_ALPHA"], days=days, anomaly_ratio=0.0
        )
        test_df = self.generator.generate(
            stations=["STATION_ALPHA"], days=days, anomaly_ratio=anomaly_ratio
        )

        y_true = test_df["is_anomaly"].values
        anomaly_types = test_df["anomaly_type"].values

        # 2. Fit detector on clean baseline
        t0 = time.perf_counter()
        self.detector.fit(train_df)
        fit_time = time.perf_counter() - t0

        # 3. Predict on test set
        t0 = time.perf_counter()
        results = self.detector.predict(test_df, sensitivity=sensitivity)
        inference_time = time.perf_counter() - t0
        latency_ms_per_sample = (inference_time / len(test_df)) * 1000.0

        # Extract model scores
        models = {
            "Isolation Forest": results["score_isolation_forest"].values,
            "Conv1D Autoencoder": results["score_autoencoder"].values,
            "Statistical (Hampel/Z)": results["score_statistical"].values,
            "Local Outlier Factor": results["score_lof"].values,
            "SkyGuard Ensemble": results["anomaly_score"].values,
        }

        # 4. Compute metrics for each model
        model_metrics = {}
        for name, scores in models.items():
            thresh = self.detector.compute_dynamic_threshold(scores, sensitivity=sensitivity)
            y_pred = (scores >= thresh).astype(int)

            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            try:
                roc = roc_auc_score(y_true, scores)
            except Exception:
                roc = 0.5
            try:
                prauc = average_precision_score(y_true, scores)
            except Exception:
                prauc = 0.0

            # Fault type breakdown recall
            type_recalls = {}
            for atype in ["spike", "stuck", "drift", "dropout", "multivariate"]:
                mask = anomaly_types == atype
                if np.sum(mask) > 0:
                    type_rec = np.mean(y_pred[mask])
                    type_recalls[atype] = round(float(type_rec), 3)
                else:
                    type_recalls[atype] = 0.0

            model_metrics[name] = {
                "precision": round(float(prec), 3),
                "recall": round(float(rec), 3),
                "f1_score": round(float(f1), 3),
                "roc_auc": round(float(roc), 3),
                "pr_auc": round(float(prauc), 3),
                "threshold": round(float(thresh), 3),
                "type_recalls": type_recalls,
            }

        # Categorization accuracy on detected true anomalies
        detected_true_mask = (y_true == 1) & (results["predicted_anomaly"] == 1)
        if np.sum(detected_true_mask) > 0:
            pred_types = results.loc[detected_true_mask, "predicted_type"].values
            true_types = anomaly_types[detected_true_mask]
            cat_accuracy = float(np.mean(pred_types == true_types))
        else:
            cat_accuracy = 0.0

        return {
            "num_samples": len(test_df),
            "num_anomalies": int(np.sum(y_true)),
            "fit_time_seconds": round(fit_time, 2),
            "latency_ms_per_sample": round(latency_ms_per_sample, 3),
            "categorization_accuracy": round(cat_accuracy, 3),
            "models": model_metrics,
        }

    def print_benchmark_table(self, report: Dict[str, Any]) -> None:
        """Formats and prints summary markdown table of model comparison."""
        print("\n" + "=" * 80)
        print("[SKYGUARD] UNSUPERVISED ANOMALY DETECTION BENCHMARK REPORT")
        print("=" * 80)
        print(f"Total Samples: {report['num_samples']} | Injected Anomalies: {report['num_anomalies']}")
        print(f"Fit Time: {report['fit_time_seconds']}s | Inference Latency: {report['latency_ms_per_sample']} ms/sample")
        print(f"Root Cause Categorization Accuracy: {report['categorization_accuracy']:.1%}\n")

        print(
            f"{'Model':<24} | {'Precision':<9} | {'Recall':<7} | {'F1-Score':<8} | {'ROC-AUC':<7} | {'PR-AUC':<7} | {'Threshold':<9}"
        )
        print("-" * 88)
        for name, m in report["models"].items():
            print(
                f"{name:<24} | {m['precision']:<9.3f} | {m['recall']:<7.3f} | {m['f1_score']:<8.3f} | {m['roc_auc']:<7.3f} | {m['pr_auc']:<7.3f} | {m['threshold']:<9.3f}"
            )
        print("-" * 88)

        print("\nArchetype Recall Breakdown by Model:")
        print(
            f"{'Model':<24} | {'Spike':<7} | {'Stuck':<7} | {'Drift':<7} | {'Dropout':<7} | {'Multivariate':<12}"
        )
        print("-" * 76)
        for name, m in report["models"].items():
            tr = m["type_recalls"]
            print(
                f"{name:<24} | {tr['spike']:<7.2f} | {tr['stuck']:<7.2f} | {tr['drift']:<7.2f} | {tr['dropout']:<7.2f} | {tr['multivariate']:<12.2f}"
            )
        print("=" * 80 + "\n")


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    report = evaluator.run_benchmark(days=14, anomaly_ratio=0.05, sensitivity=0.5)
    evaluator.print_benchmark_table(report)
