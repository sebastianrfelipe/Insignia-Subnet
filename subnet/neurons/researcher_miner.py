"""
Researcher Miner — Model Generation (single paired mechanism)

Template miner for the researcher role of the Insignia subnet. Researcher
miners train ML models (typically GBDTs) to predict short-horizon price
movements and submit them for validator evaluation.

Under the single paired mechanism the model is not promoted into a pool;
instead the validator pairs this researcher with trader miners (chain-seeded)
and evaluates the resulting ``(model, strategy)`` pairs jointly.

This template demonstrates the full miner lifecycle:
  1. Fetch publicly available market data
  2. Engineer features from the public feature registry
  3. Train a model using the miner's chosen algorithm and hyperparameters
  4. Serialize the model AND package the source code that produced/serves it
  5. Submit both to validators via the ModelSubmission synapse

CODE SUBMISSION (Metanova/NOVA-style reproducibility):
  The miner ships a signed ``code_bundle`` (a deterministic tar.gz containing
  the training source, the serialized model, and a sandbox ``inference.py``
  entrypoint) alongside the artifact. Validators re-execute the entrypoint in
  an isolated sandbox and confirm it reproduces the artifact's predictions
  before scoring. See ``insignia/code_submission.py``.

PROPRIETARY BOUNDARY:
  - Miners use publicly available data + their own sourced data
  - Validators score against proprietary data miners never see
  - The evaluation score is the only feedback miners receive
  - Miners are free to innovate on architecture, features, and HPO

Usage:
    python neurons/researcher_miner.py --netuid <NETUID> --wallet.name <WALLET> --subtensor.network test
"""

from __future__ import annotations

import io
import json
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
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from insignia.protocol import MinerRole
from insignia.code_submission import (
    DEFAULT_ENTRYPOINT,
    CodeBundle,
    build_code_bundle,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Researcher-Miner] %(message)s")
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
# Sandbox inference entrypoint
#
# This source is shipped inside every code bundle as ``inference.py``. The
# validator runs it in an isolated sandbox: it reads the feature matrix from
# ``input.json``, loads the bundled model, and writes predictions to
# ``result.json`` using EXACTLY the same convention the validator's
# ``ModelEvaluator`` applies to the artifact it deserializes. The reproduced
# predictions must match for the submission to be accepted.
# ---------------------------------------------------------------------------

INFERENCE_ENTRYPOINT_SOURCE = '''\
"""Sandbox inference entrypoint for an Insignia researcher submission.

Reads input.json -> {"features": [[...], ...], "feature_names": [...]}
Writes result.json -> {"predictions": [...], "n": <int>, "model_file": "model.joblib"}

Run as: python inference.py   (no network; cwd is the extracted bundle)
"""
import json
import os

import numpy as np
import joblib

INPUT = os.environ.get("INSIGNIA_INPUT", "input.json")
OUTPUT = os.environ.get("INSIGNIA_OUTPUT", "result.json")
MODEL_FILE = "model.joblib"


def predict(model, X):
    """Mirror the validator's ModelEvaluator prediction convention exactly."""
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(X)
        raw = np.asarray(raw)
        if raw.ndim == 2:
            return raw[:, 1] - 0.5
        return raw - 0.5
    return model.predict(X).astype(float)


def main():
    with open(INPUT, "r") as fh:
        payload = json.load(fh)

    X = np.asarray(payload.get("features", []), dtype=float)
    model = joblib.load(MODEL_FILE)

    if X.size == 0:
        predictions = []
    else:
        predictions = [float(p) for p in predict(model, X)]

    with open(OUTPUT, "w") as fh:
        json.dump(
            {"predictions": predictions, "n": len(predictions), "model_file": MODEL_FILE},
            fh,
        )


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Model Training Pipeline
# ---------------------------------------------------------------------------

class ModelTrainer:
    """
    Encapsulates the model training pipeline for a researcher miner.

    This is a reference implementation using HistGradientBoosting. Miners
    are encouraged to replace this with their own architecture, HPO strategy,
    and feature engineering pipeline. Validators score outputs, not methodology.
    """

    def __init__(
        self,
        target_instrument: str = "BTC-USDT-PERP",
        target_horizon_minutes: int = 60,
        features: List[str] | None = None,
        n_estimators: int = 2000,
        max_depth: int = 3,
        learning_rate: float = 0.003,
        min_samples_leaf: int = 60,
        l2_regularization: float = 0.5,
        max_bins: int = 64,
        random_state: int = 42,
    ):
        self.target_instrument = target_instrument
        self.target_horizon_minutes = target_horizon_minutes
        self.features = features or PUBLIC_FEATURE_REGISTRY
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
        in_sample_acc = float(accuracy_score(y_train, eval_model.predict(X_train)))
        oos_acc = float(accuracy_score(y_test, eval_model.predict(X_test)))

        # Final model: fit on all data with preprocessing pipeline
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
            "in_sample_accuracy": in_sample_acc,
            "out_of_sample_accuracy": oos_acc,
            "overfitting_gap": in_sample_acc - oos_acc,
            "n_samples": int(len(X)),
            "n_features": len(available),
        }
        logger.info(
            "Training complete: IS=%.4f OOS=%.4f Gap=%.4f",
            self.training_metrics["in_sample_accuracy"],
            self.training_metrics["out_of_sample_accuracy"],
            self.training_metrics["overfitting_gap"],
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

    def build_code_bundle(self, artifact: bytes | None = None) -> CodeBundle:
        """
        Package the source that produced/serves this model into a signed,
        reproducible code bundle.

        Contents:
          - ``inference.py``  — sandbox entrypoint the validator executes
          - ``model.joblib``  — the serialized model the entrypoint loads
          - ``train.py``      — the full training source (this module), for audit
          - ``metadata.json`` — declared model type, features, and hyperparams

        Miners are free to extend this with their real training package; the
        only hard requirement is that ``inference.py`` reproduces the artifact's
        predictions from ``input.json``.
        """
        if artifact is None:
            artifact = self.serialize()

        try:
            with open(os.path.abspath(__file__), "rb") as fh:
                train_src = fh.read()
        except OSError:
            train_src = b"# training source unavailable in this environment\n"

        metadata = self.get_submission_metadata()
        files = {
            DEFAULT_ENTRYPOINT: INFERENCE_ENTRYPOINT_SOURCE.encode("utf-8"),
            "model.joblib": artifact,
            "train.py": train_src,
            "metadata.json": json.dumps(metadata, indent=2, default=str).encode("utf-8"),
        }
        return build_code_bundle(files, entrypoint=DEFAULT_ENTRYPOINT)


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

class ResearcherMiner:
    """
    Researcher miner neuron. In production, this would:
      1. Register on the Bittensor network (declaring MinerRole.RESEARCHER)
      2. Listen for evaluation requests from validators
      3. Train/retrain models on each epoch
      4. Submit model artifacts via the ModelSubmission synapse

    The validator pairs this researcher with trader miners (chain-seeded) and
    evaluates the resulting ``(model, strategy)`` pairs jointly.
    """

    role = MinerRole.RESEARCHER

    def __init__(self, trainer: ModelTrainer | None = None):
        self.trainer = trainer or ModelTrainer()
        self.current_epoch: int = 0
        self.last_submission_time: float = 0
        self.submission_history: List[Dict] = []

    def train_and_submit(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train a model and prepare a submission package.

        Returns a dict containing everything needed for the
        ModelSubmission synapse.
        """
        logger.info("Epoch %d: Training model...", self.current_epoch)
        self.trainer.train(data)
        artifact = self.trainer.serialize()
        metadata = self.trainer.get_submission_metadata()

        bundle = self.trainer.build_code_bundle(artifact)

        submission = {
            "role": MinerRole.RESEARCHER.value,
            "model_artifact": artifact,
            "model_type": metadata["model_type"],
            "features_used": metadata["features_used"],
            "training_window_start": str(data.index[0]),
            "training_window_end": str(data.index[-1]),
            "hyperparams": metadata["hyperparams"],
            "preprocessing_hash": hashlib.sha256(artifact).hexdigest()[:16],
            "target_instrument": metadata["target_instrument"],
            "target_horizon_minutes": metadata["target_horizon_minutes"],
            "self_reported_overfitting_score": metadata["training_metrics"].get(
                "overfitting_gap", 0.0
            ),
            "artifact_size_bytes": len(artifact),
            "artifact_hash": hashlib.sha256(artifact).hexdigest(),
            # Code submission (reproducibility): ship the source alongside the model.
            **bundle.to_submission_fields(),
            "code_bundle_size_bytes": bundle.size_bytes,
        }

        self.submission_history.append({
            "epoch": self.current_epoch,
            "metrics": metadata["training_metrics"],
            "artifact_hash": submission["artifact_hash"],
            "code_bundle_hash": bundle.bundle_hash,
        })
        self.current_epoch += 1
        self.last_submission_time = time.time()

        logger.info(
            "Submission ready: model=%d bytes (hash=%s), code=%d bytes (hash=%s, entry=%s)",
            len(artifact),
            submission["artifact_hash"][:12],
            bundle.size_bytes,
            bundle.bundle_hash[:12],
            bundle.entrypoint,
        )
        return submission


# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------

def demo():
    """Run a standalone demonstration of the researcher miner pipeline."""
    logger.info("=" * 60)
    logger.info("Insignia Researcher Miner — Demo Mode")
    logger.info("=" * 60)

    data = generate_demo_data(n_samples=5000)
    logger.info("Generated %d samples with %d features", len(data), len(data.columns) - 1)

    miner = ResearcherMiner()
    submission = miner.train_and_submit(data)

    logger.info("--- Submission Package ---")
    for key, val in submission.items():
        if isinstance(val, (bytes, bytearray)):
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
