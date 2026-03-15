"""
Layer 1 Miner — Model Generation

Template miner for Layer 1 of the Insignia subnet. Miners train ML models
(typically GBDTs) to predict short-horizon price movements and submit them
for validator evaluation.

This template demonstrates the full miner lifecycle:
  1. Fetch publicly available market data
  2. Engineer features from the public feature registry
  3. Train a model using the miner's chosen algorithm and hyperparameters
  4. Serialize the model and submit to validators via the L1ModelSubmission synapse

PROPRIETARY BOUNDARY:
  - Miners use publicly available data + their own sourced data
  - Validators score against proprietary data miners never see
  - The evaluation score is the only feedback miners receive
  - Miners are free to innovate on architecture, features, and HPO

Usage:
    python neurons/l1_miner.py --netuid <NETUID> --wallet.name <WALLET> --subtensor.network test
"""

from __future__ import annotations

import os
import io
import time
import hashlib
import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import joblib

try:
    import bittensor as bt
except ImportError:
    bt = None

from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [L1-Miner] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature Registry (public — miners and validators share this contract)
# ---------------------------------------------------------------------------

PUBLIC_FEATURE_REGISTRY = [
    "ret_1", "ret_5", "ret_10", "ret_30", "ret_60",
    "vol_10", "vol_30", "vol_60",
    "rsi_14", "rsi_30",
    "funding_rate", "funding_rate_ma_8h",
    "open_interest_change_1h", "open_interest_change_4h",
    "volume_ratio_1h", "volume_ratio_4h",
    "ob_imbalance_top5", "ob_imbalance_top10",
    "liquidation_intensity_1h",
    "taker_buy_sell_ratio_1h",
    "spread_bps",
    "price_range_pct_1h",
    "vwap_deviation",
    "hour_of_day_sin", "hour_of_day_cos",
    "day_of_week_sin", "day_of_week_cos",
]


# ---------------------------------------------------------------------------
# Model Training Pipeline
# ---------------------------------------------------------------------------

class L1ModelTrainer:
    """
    Encapsulates the model training pipeline for an L1 miner.

    This is a reference implementation using HistGradientBoosting. Miners
    are encouraged to replace this with their own architecture, HPO strategy,
    and feature engineering pipeline. Validators score outputs, not methodology.
    """

    def __init__(
        self,
        target_instrument: str = "BTC-USDT-PERP",
        target_horizon_minutes: int = 60,
        features: List[str] | None = None,
        n_estimators: int = 10,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        min_samples_leaf: int = 50,
        l2_regularization: float = 0.1,
        max_bins: int = 64,
        random_state: int = 42,
    ):
        self.target_instrument = target_instrument
        self.target_horizon_minutes = target_horizon_minutes
        self.features = features or PUBLIC_FEATURE_REGISTRY[:10]
        self.random_state = random_state
        self.hyperparams = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_samples_leaf": min_samples_leaf,
            "l2_regularization": l2_regularization,
            "max_bins": max_bins,
        }
        self.model: Optional[Pipeline] = None
        self.training_metrics: Dict[str, float] = {}

    def train(self, df: pd.DataFrame) -> Pipeline:
        """
        Train the model on the provided dataframe.

        The dataframe must contain columns matching self.features and a
        'target' column with forward returns or directional labels.
        """
        available = [f for f in self.features if f in df.columns]
        if not available:
            raise ValueError("No features found in dataframe")

        X = df[available].values
        y = df["target"].values

        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]

        y_cls = (y > 0).astype(int)

        # Time-series train/test split for IS/OOS evaluation
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_cls[:split_idx], y_cls[split_idx:]

        eval_model = HistGradientBoostingClassifier(
            max_iter=self.hyperparams["n_estimators"],
            max_depth=self.hyperparams["max_depth"],
            learning_rate=self.hyperparams["learning_rate"],
            min_samples_leaf=self.hyperparams["min_samples_leaf"],
            l2_regularization=self.hyperparams["l2_regularization"],
            max_bins=self.hyperparams["max_bins"],
            random_state=self.random_state,
        )
        eval_model.fit(X_train, y_train)
        in_sample_f1 = float(f1_score(y_train, eval_model.predict(X_train), zero_division=0.0))
        oos_f1 = float(f1_score(y_test, eval_model.predict(X_test), zero_division=0.0))

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", HistGradientBoostingClassifier(
                max_iter=self.hyperparams["n_estimators"],
                max_depth=self.hyperparams["max_depth"],
                learning_rate=self.hyperparams["learning_rate"],
                min_samples_leaf=self.hyperparams["min_samples_leaf"],
                l2_regularization=self.hyperparams["l2_regularization"],
                max_bins=self.hyperparams["max_bins"],
                random_state=self.random_state,
            )),
        ])
        pipeline.fit(X, y_cls)

        self.model = pipeline
        self.training_metrics = {
            "in_sample_f1": in_sample_f1,
            "out_of_sample_f1": oos_f1,
            "generalization_gap": abs(in_sample_f1 - oos_f1),
            "n_samples": int(len(X)),
            "n_features": len(available),
        }
        logger.info(
            "Training complete: IS_F1=%.4f OOS_F1=%.4f GenGap=%.4f",
            self.training_metrics["in_sample_f1"],
            self.training_metrics["out_of_sample_f1"],
            self.training_metrics["generalization_gap"],
        )
        return pipeline

    def serialize(self) -> bytes:
        """Serialize the trained model to bytes for network transmission."""
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        buf = io.BytesIO()
        joblib.dump(self.model, buf)
        return buf.getvalue()

    def get_submission_metadata(self) -> Dict[str, Any]:
        return {
            "model_type": "gbdt",
            "features_used": self.features,
            "hyperparams": self.hyperparams,
            "target_instrument": self.target_instrument,
            "target_horizon_minutes": self.target_horizon_minutes,
            "training_metrics": self.training_metrics,
        }


# ---------------------------------------------------------------------------
# Synthetic Data Generator (for demo / hackathon)
# ---------------------------------------------------------------------------

def generate_demo_data(
    n_samples: int = 5000,
    n_features: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic market data for hackathon demonstration.

    In production, miners fetch real OHLCV + derivatives data from
    public APIs (Binance, Coinbase, etc.) and compute features from
    the public feature registry.
    """
    rng = np.random.RandomState(seed)

    features = PUBLIC_FEATURE_REGISTRY[:n_features]
    data = {}

    for i, feat in enumerate(features):
        if "ret" in feat:
            data[feat] = rng.normal(0, 0.02, n_samples)
        elif "vol" in feat:
            data[feat] = np.abs(rng.normal(0.01, 0.005, n_samples))
        elif "rsi" in feat:
            data[feat] = rng.uniform(20, 80, n_samples)
        elif "funding" in feat:
            data[feat] = rng.normal(0, 0.001, n_samples)
        elif "oi" in feat or "open_interest" in feat:
            data[feat] = rng.normal(0, 0.05, n_samples)
        else:
            data[feat] = rng.normal(0, 1, n_samples)

    signal = sum(
        data[f] * w
        for f, w in zip(features[:5], [0.3, 0.2, 0.15, 0.1, 0.05])
    )
    noise = rng.normal(0, 0.01, n_samples)
    data["target"] = signal + noise

    df = pd.DataFrame(data)
    df.index = pd.date_range("2025-01-01", periods=n_samples, freq="h")
    return df


# ---------------------------------------------------------------------------
# Miner Main Loop
# ---------------------------------------------------------------------------

class L1Miner:
    """
    Layer 1 miner neuron. In production, this would:
      1. Register on the Bittensor network
      2. Listen for evaluation requests from validators
      3. Train/retrain models on each epoch
      4. Submit model artifacts via the L1ModelSubmission synapse
    """

    def __init__(self, trainer: L1ModelTrainer | None = None):
        self.trainer = trainer or L1ModelTrainer()
        self.current_epoch: int = 0
        self.last_submission_time: float = 0
        self.submission_history: List[Dict] = []

    def train_and_submit(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train a model and prepare a submission package.

        Returns a dict containing everything needed for the
        L1ModelSubmission synapse.
        """
        logger.info("Epoch %d: Training model...", self.current_epoch)
        self.trainer.train(data)
        artifact = self.trainer.serialize()
        metadata = self.trainer.get_submission_metadata()

        submission = {
            "model_artifact": artifact,
            "model_type": metadata["model_type"],
            "features_used": metadata["features_used"],
            "training_window_start": str(data.index[0]),
            "training_window_end": str(data.index[-1]),
            "hyperparams": metadata["hyperparams"],
            "preprocessing_hash": hashlib.sha256(artifact).hexdigest()[:16],
            "target_instrument": metadata["target_instrument"],
            "target_horizon_minutes": metadata["target_horizon_minutes"],
            "self_reported_generalization_gap": metadata["training_metrics"].get(
                "generalization_gap", 0.0
            ),
            "artifact_size_bytes": len(artifact),
            "artifact_hash": hashlib.sha256(artifact).hexdigest(),
        }

        self.submission_history.append({
            "epoch": self.current_epoch,
            "metrics": metadata["training_metrics"],
            "artifact_hash": submission["artifact_hash"],
        })
        self.current_epoch += 1
        self.last_submission_time = time.time()

        logger.info(
            "Submission ready: %d bytes, hash=%s",
            len(artifact),
            submission["artifact_hash"][:12],
        )
        return submission


# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------

def demo():
    """Run a standalone demonstration of the L1 miner pipeline."""
    logger.info("=" * 60)
    logger.info("Insignia L1 Miner — Demo Mode")
    logger.info("=" * 60)

    data = generate_demo_data(n_samples=5000)
    logger.info("Generated %d samples with %d features", len(data), len(data.columns) - 1)

    miner = L1Miner()
    submission = miner.train_and_submit(data)

    logger.info("--- Submission Package ---")
    for key, val in submission.items():
        if key == "model_artifact":
            logger.info("  %s: <%d bytes>", key, len(val))
        elif isinstance(val, list):
            logger.info("  %s: [%d items]", key, len(val))
        elif isinstance(val, dict):
            logger.info("  %s: {%d keys}", key, len(val))
        else:
            logger.info("  %s: %s", key, val)

    logger.info("--- Training Metrics ---")
    for key, val in miner.trainer.training_metrics.items():
        logger.info("  %s: %s", key, val)

    return submission


if __name__ == "__main__":
    demo()
