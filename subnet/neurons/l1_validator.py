"""
Layer 1 Validator — Model Evaluation

Template validator for Layer 1 of the Insignia subnet. Validators receive
model submissions from miners, run them against a proprietary benchmark
dataset, and assign composite scores used for Yuma consensus weight-setting.

PROPRIETARY BOUNDARY:
  The validator's benchmark dataset (enterprise tick-by-tick data) is the
  core competitive moat. It is NEVER exposed to miners. The scoring
  framework itself is fully transparent — only the data and the proprietary
  overfitting detector implementation are private.

  In this template, the ProprietaryBenchmark class demonstrates the
  interface. The actual production implementation loads real tick data
  from the subnet owner's enterprise feed.

Evaluation Flow:
  1. Receive L1ModelSubmission from miner
  2. Deserialize model artifact
  3. Validate model compatibility (ONNX inference test)
  4. Run model against proprietary holdout window
  5. Compute multi-metric score vector
  6. Apply anti-gaming checks (fingerprinting, rate limits)
  7. Return composite score; set weights for Yuma consensus

Usage:
    python neurons/l1_validator.py --netuid <NETUID> --wallet.name <WALLET> --subtensor.network test
"""

from __future__ import annotations

import io
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import joblib

try:
    import bittensor as bt
except ImportError:
    bt = None

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from insignia.scoring import (
    CompositeScorer,
    ScoreVector,
    WeightConfig,
    OverfittingDetector,
    ReferenceOverfittingDetector,
)
from insignia.incentive import (
    SubmissionRateLimit,
    ModelFingerprinter,
    CrossLayerFeedbackEngine,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [L1-Validator] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Proprietary Benchmark Interface
# ---------------------------------------------------------------------------

class BenchmarkDataProvider(ABC):
    """
    Abstract interface for the validator's benchmark data.

    Production validators implement this with real enterprise tick data.
    The hackathon demo uses synthetic data that mimics the statistical
    properties of real markets without exposing actual data.
    """

    @abstractmethod
    def get_holdout_window(self, epoch_id: int) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Return (feature_df, actual_returns) for the current holdout window.

        The window changes each epoch to prevent miners from overfitting
        to a specific time period. Windows cover diverse market regimes.
        """
        ...

    @abstractmethod
    def get_regime_label(self, epoch_id: int) -> str:
        """Return the market regime of the current window (for logging)."""
        ...


class DemoBenchmarkProvider(BenchmarkDataProvider):
    """
    Synthetic benchmark for hackathon demonstration.

    Generates holdout windows with controlled statistical properties
    across different market regimes. In production, this is replaced
    by the subnet owner's enterprise tick data feed.
    """

    REGIMES = ["trending_up", "trending_down", "ranging", "high_volatility", "low_volatility"]

    def __init__(self, n_samples: int = 1000, seed: int = 12345):
        self.n_samples = n_samples
        self.seed = seed

    def get_holdout_window(self, epoch_id: int) -> Tuple[pd.DataFrame, np.ndarray]:
        rng = np.random.RandomState(self.seed + epoch_id)
        regime = self.REGIMES[epoch_id % len(self.REGIMES)]

        from neurons.l1_miner import PUBLIC_FEATURE_REGISTRY

        features = PUBLIC_FEATURE_REGISTRY[:15]
        data = {}

        drift, vol_scale = {
            "trending_up": (0.001, 1.0),
            "trending_down": (-0.001, 1.0),
            "ranging": (0.0, 0.5),
            "high_volatility": (0.0, 3.0),
            "low_volatility": (0.0, 0.3),
        }[regime]

        for feat in features:
            if "ret" in feat:
                data[feat] = rng.normal(drift, 0.02 * vol_scale, self.n_samples)
            elif "vol" in feat:
                data[feat] = np.abs(rng.normal(0.01 * vol_scale, 0.005, self.n_samples))
            else:
                data[feat] = rng.normal(0, 1, self.n_samples)

        actuals = sum(
            data[f] * w for f, w in zip(features[:5], [0.25, 0.20, 0.15, 0.10, 0.05])
        ) + rng.normal(0, 0.005, self.n_samples)

        df = pd.DataFrame(data)
        return df, actuals

    def get_regime_label(self, epoch_id: int) -> str:
        return self.REGIMES[epoch_id % len(self.REGIMES)]


# ---------------------------------------------------------------------------
# Model Evaluator
# ---------------------------------------------------------------------------

class ModelEvaluator:
    """
    Evaluates a deserialized model against benchmark data and returns
    a ScoreVector.

    This is the core evaluation loop that runs on every validator for
    every miner submission. Deterministic execution ensures all
    validators converge on the same scores (required for consensus).
    """

    def __init__(
        self,
        scorer: CompositeScorer | None = None,
        benchmark: BenchmarkDataProvider | None = None,
    ):
        self.scorer = scorer or CompositeScorer()
        self.benchmark = benchmark or DemoBenchmarkProvider()

    def evaluate(
        self,
        model_artifact: bytes,
        features_used: List[str],
        epoch_id: int = 0,
    ) -> Tuple[ScoreVector, Dict[str, Any]]:
        """
        Full evaluation pipeline for one miner's model.

        Returns (score_vector, diagnostics_dict).
        """
        model = self._deserialize(model_artifact)
        holdout_df, actuals = self.benchmark.get_holdout_window(epoch_id)

        available_features = [f for f in features_used if f in holdout_df.columns]
        if not available_features:
            return ScoreVector(composite=0.0), {"error": "no matching features"}

        X = holdout_df[available_features].values
        mask = ~np.isnan(X).any(axis=1)
        X_clean = X[mask]
        actuals_clean = actuals[mask]

        t0 = time.perf_counter()
        if hasattr(model, "predict_proba"):
            raw_preds = model.predict_proba(X_clean)
            if raw_preds.ndim == 2:
                predictions = raw_preds[:, 1] - 0.5
            else:
                predictions = raw_preds - 0.5
        else:
            predictions = model.predict(X_clean).astype(float)
        inference_ms = (time.perf_counter() - t0) * 1000

        position_returns = predictions * actuals_clean
        equity_curve = np.cumsum(position_returns) + 1.0

        in_sample_acc = float(np.mean(np.sign(predictions[:len(predictions)//2]) ==
                                      np.sign(actuals_clean[:len(actuals_clean)//2])))
        out_sample_acc = float(np.mean(np.sign(predictions[len(predictions)//2:]) ==
                                       np.sign(actuals_clean[len(actuals_clean)//2:])))

        model_complexity = self._extract_complexity(model)

        score = self.scorer.score_l1(
            predictions=predictions,
            actuals=actuals_clean,
            equity_curve=equity_curve,
            n_features=len(available_features),
            inference_ms=inference_ms,
            in_sample_accuracy=in_sample_acc,
            out_of_sample_accuracy=out_sample_acc,
            model_complexity=model_complexity,
        )

        diagnostics = {
            "regime": self.benchmark.get_regime_label(epoch_id),
            "n_samples_evaluated": int(len(X_clean)),
            "inference_ms": round(inference_ms, 2),
            "in_sample_accuracy": round(in_sample_acc, 4),
            "out_of_sample_accuracy": round(out_sample_acc, 4),
            "model_complexity": model_complexity,
        }

        return score, diagnostics

    def _deserialize(self, artifact: bytes):
        buf = io.BytesIO(artifact)
        return joblib.load(buf)

    @staticmethod
    def _extract_complexity(model) -> Dict[str, Any]:
        """Extract complexity metrics from the model for overfitting detection."""
        complexity = {}
        actual_model = model
        if hasattr(model, "named_steps"):
            actual_model = model.named_steps.get("model", model)

        if hasattr(actual_model, "n_iter_"):
            complexity["n_estimators"] = int(actual_model.n_iter_)
        elif hasattr(actual_model, "n_estimators"):
            complexity["n_estimators"] = int(actual_model.n_estimators)

        if hasattr(actual_model, "max_depth"):
            complexity["max_depth"] = int(actual_model.max_depth or 6)

        if hasattr(actual_model, "min_samples_leaf"):
            complexity["min_samples_leaf"] = int(actual_model.min_samples_leaf)

        return complexity


# ---------------------------------------------------------------------------
# L1 Validator Neuron
# ---------------------------------------------------------------------------

class L1Validator:
    """
    Layer 1 Validator neuron.

    Manages the full evaluation cycle:
      - Receives miner model submissions
      - Applies anti-gaming checks
      - Runs model evaluation against proprietary benchmark
      - Computes weights for Yuma consensus
      - Promotes top models to Layer 2 pool

    The validator maintains state across epochs: miner scores, model
    fingerprints, promotion history, and cross-layer feedback.
    """

    def __init__(
        self,
        evaluator: ModelEvaluator | None = None,
        weights: WeightConfig | None = None,
        top_n_promote: int = 10,
    ):
        self.evaluator = evaluator or ModelEvaluator()
        self.rate_limiter = SubmissionRateLimit()
        self.fingerprinter = ModelFingerprinter()
        self.feedback_engine = CrossLayerFeedbackEngine()
        self.top_n_promote = top_n_promote

        self.current_epoch: int = 0
        self.miner_scores: Dict[str, ScoreVector] = {}
        self.promoted_models: List[Dict] = []
        self.epoch_history: List[Dict] = []

    def process_submission(
        self,
        miner_uid: str,
        model_artifact: bytes,
        features_used: List[str],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process a single miner's model submission.

        Returns a result dict with score, acceptance status, and diagnostics.
        """
        if not self.rate_limiter.check(miner_uid):
            return {
                "accepted": False,
                "rejection_reason": "Rate limited: 1 submission per epoch",
                "composite_score": 0.0,
            }

        fingerprint = self.fingerprinter.compute_fingerprint(model_artifact)
        if self.fingerprinter.is_exact_duplicate(miner_uid, fingerprint):
            return {
                "accepted": False,
                "rejection_reason": "Duplicate model detected",
                "composite_score": 0.0,
            }

        score, diagnostics = self.evaluator.evaluate(
            model_artifact=model_artifact,
            features_used=features_used,
            epoch_id=self.current_epoch,
        )

        l2_adjustment = self.feedback_engine.compute_adjustment(miner_uid)
        adjusted_composite = score.composite * l2_adjustment

        self.fingerprinter.register(miner_uid, fingerprint, np.array([adjusted_composite]))
        self.rate_limiter.record(miner_uid)
        self.miner_scores[miner_uid] = ScoreVector(
            raw=score.raw,
            normalized=score.normalized,
            composite=adjusted_composite,
        )

        correlated = self.fingerprinter.find_correlated_miners(miner_uid)

        result = {
            "accepted": True,
            "composite_score": round(adjusted_composite, 6),
            "score_breakdown": {k: round(v, 4) for k, v in score.normalized.items()},
            "raw_metrics": {k: round(v, 4) for k, v in score.raw.items()},
            "diagnostics": diagnostics,
            "l2_feedback_multiplier": round(l2_adjustment, 4),
            "correlated_miners": len(correlated),
        }

        logger.info(
            "Miner %s: score=%.4f (regime=%s, l2_adj=%.2f)",
            miner_uid,
            adjusted_composite,
            diagnostics.get("regime", "unknown"),
            l2_adjustment,
        )
        return result

    def run_epoch(self, submissions: Dict[str, Dict], force: bool = False) -> Dict[str, Any]:
        """
        Process all submissions for the current epoch, compute rankings,
        and promote top models.

        Set force=True to bypass rate limiting (useful for demos).
        """
        logger.info("=" * 50)
        logger.info("Epoch %d — Processing %d submissions", self.current_epoch, len(submissions))
        logger.info("Regime: %s", self.evaluator.benchmark.get_regime_label(self.current_epoch))

        if force:
            self.rate_limiter._last_submission.clear()

        results = {}
        for miner_uid, sub in submissions.items():
            results[miner_uid] = self.process_submission(
                miner_uid=miner_uid,
                model_artifact=sub["model_artifact"],
                features_used=sub.get("features_used", []),
                metadata=sub.get("metadata", {}),
            )

        ranked = sorted(
            [(uid, r) for uid, r in results.items() if r["accepted"]],
            key=lambda x: x[1]["composite_score"],
            reverse=True,
        )

        for rank, (uid, r) in enumerate(ranked):
            r["rank"] = rank + 1
            r["total_miners"] = len(ranked)
            r["promoted_to_l2"] = rank < self.top_n_promote

        promoted = []
        for rank, (uid, r) in enumerate(ranked[: self.top_n_promote]):
            promoted.append({
                "model_id": f"epoch{self.current_epoch}_rank{rank+1}_{uid}",
                "miner_uid": uid,
                "composite_score": r["composite_score"],
                "epoch": self.current_epoch,
            })
        self.promoted_models = promoted

        weights = self._compute_consensus_weights(ranked)

        epoch_summary = {
            "epoch": self.current_epoch,
            "n_submissions": len(submissions),
            "n_accepted": len(ranked),
            "n_promoted": len(promoted),
            "top_score": ranked[0][1]["composite_score"] if ranked else 0.0,
            "median_score": ranked[len(ranked) // 2][1]["composite_score"] if ranked else 0.0,
            "regime": self.evaluator.benchmark.get_regime_label(self.current_epoch),
            "weights": weights,
        }
        self.epoch_history.append(epoch_summary)
        self.current_epoch += 1

        logger.info(
            "Epoch complete: %d accepted, %d promoted, top=%.4f",
            len(ranked),
            len(promoted),
            epoch_summary["top_score"],
        )
        return {"results": results, "promoted": promoted, "summary": epoch_summary}

    def _compute_consensus_weights(
        self, ranked: List[Tuple[str, Dict]]
    ) -> Dict[str, float]:
        """
        Compute normalized weights for Yuma consensus.

        Weights are proportional to composite scores. In production,
        these are submitted to the Bittensor network via set_weights().
        """
        if not ranked:
            return {}

        scores = np.array([r["composite_score"] for _, r in ranked])
        if scores.sum() < 1e-12:
            return {uid: 1.0 / len(ranked) for uid, _ in ranked}

        normalized = scores / scores.sum()
        return {uid: float(w) for (uid, _), w in zip(ranked, normalized)}

    def get_promoted_models(self) -> List[Dict]:
        return self.promoted_models


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    """Run a standalone L1 validator demonstration with multiple miners."""
    from neurons.l1_miner import L1Miner, L1ModelTrainer, generate_demo_data

    logger.info("=" * 60)
    logger.info("Insignia L1 Validator — Demo Mode")
    logger.info("=" * 60)

    miners = {}
    for i in range(5):
        data = generate_demo_data(n_samples=3000, seed=42 + i)
        trainer = L1ModelTrainer(
            n_estimators=200 + i * 50,
            max_depth=4 + i,
            learning_rate=0.03 + i * 0.01,
            random_state=42 + i,
        )
        miner = L1Miner(trainer=trainer)
        submission = miner.train_and_submit(data)
        miners[f"miner_{i}"] = submission
        logger.info("Miner %d trained: IS=%.4f OOS=%.4f",
                     i, miner.trainer.training_metrics["in_sample_accuracy"],
                     miner.trainer.training_metrics["out_of_sample_accuracy"])

    validator = L1Validator(top_n_promote=3)

    for epoch in range(3):
        logger.info("\n--- Epoch %d ---", epoch)
        epoch_result = validator.run_epoch(miners)

        for uid, result in epoch_result["results"].items():
            if result["accepted"]:
                logger.info(
                    "  %s: rank=%d score=%.4f %s",
                    uid,
                    result["rank"],
                    result["composite_score"],
                    "[PROMOTED]" if result.get("promoted_to_l2") else "",
                )

    logger.info("\n--- Promoted Models ---")
    for m in validator.get_promoted_models():
        logger.info("  %s (score=%.4f)", m["model_id"], m["composite_score"])

    return validator


if __name__ == "__main__":
    demo()
