"""
Model Validator — Researcher-side Model Evaluation

Template validator for the researcher (model) side of the Insignia subnet.
Validators receive model submissions from researcher miners, run them against a
proprietary benchmark dataset, and assign composite scores used for Yuma
consensus weight-setting.

Under the single paired mechanism, the unified ``PairedValidator``
(``neurons/validator.py``) drives the per-epoch lifecycle; this module provides
the reusable ``ModelEvaluator`` it calls to score the researcher half of each
pair. The standalone ``ModelValidator`` below is the legacy single-layer
evaluator retained for the emulator and demos.

PROPRIETARY BOUNDARY:
  The validator's benchmark dataset (enterprise tick-by-tick data) is the
  core competitive moat. It is NEVER exposed to miners. The scoring
  framework itself is fully transparent — only the data and the proprietary
  overfitting detector implementation are private.

  In this template, the ProprietaryBenchmark class demonstrates the
  interface. The actual production implementation loads real tick data
  from the subnet owner's enterprise feed.

Evaluation Flow:
  1. Receive ModelSubmission from researcher miner
  2. Deserialize model artifact
  3. Validate model compatibility (ONNX inference test)
  4. Run model against proprietary holdout window
  5. Compute multi-metric score vector
  6. Apply anti-gaming checks (fingerprinting, rate limits)
  7. Return composite score; set weights for Yuma consensus

Usage:
    python neurons/model_validator.py --netuid <NETUID> --wallet.name <WALLET> --subtensor.network test
"""

from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

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
from insignia.code_submission import (
    CodeBundleConfig,
    CodeBundleVerifier,
    CodeFingerprinter,
    ReproducibilityChecker,
    SandboxConfig,
    SandboxRunner,
)
from insignia.safe_model_loader import safe_load_model, UnsafeArtifactError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Model-Validator] %(message)s")
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

        from neurons.researcher_miner import PUBLIC_FEATURE_REGISTRY

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
        capture: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ScoreVector, Dict[str, Any]]:
        """
        Full evaluation pipeline for one miner's model.

        Returns (score_vector, diagnostics_dict).

        When ``capture`` is provided it is populated with the cleaned feature
        matrix, the features actually used, and the artifact's predictions so
        the caller can drive an independent reproducibility check on the miner's
        submitted code without re-deserializing or re-windowing.
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
        predictions = self.predict(model, X_clean)
        inference_ms = (time.perf_counter() - t0) * 1000

        if capture is not None:
            capture["X_clean"] = X_clean
            capture["available_features"] = available_features
            capture["predictions"] = predictions

        position_returns = predictions * actuals_clean
        equity_curve = np.cumsum(position_returns) + 1.0

        in_sample_acc = float(np.mean(np.sign(predictions[:len(predictions)//2]) ==
                                      np.sign(actuals_clean[:len(actuals_clean)//2])))
        out_sample_acc = float(np.mean(np.sign(predictions[len(predictions)//2:]) ==
                                       np.sign(actuals_clean[len(actuals_clean)//2:])))

        model_complexity = self._extract_complexity(model)

        score = self.scorer.score_model(
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

    @staticmethod
    def predict(model, X: np.ndarray) -> np.ndarray:
        """
        Canonical prediction convention shared by the validator and the
        sandboxed ``inference.py`` entrypoint (see
        ``neurons.researcher_miner.INFERENCE_ENTRYPOINT_SOURCE``). Keeping the
        two in sync is what makes the reproducibility comparison meaningful.
        """
        if hasattr(model, "predict_proba"):
            raw_preds = np.asarray(model.predict_proba(X))
            if raw_preds.ndim == 2:
                return raw_preds[:, 1] - 0.5
            return raw_preds - 0.5
        return model.predict(X).astype(float)

    def _deserialize(self, artifact: bytes):
        # SECURITY: ``artifact`` is untrusted miner-supplied bytes. Plain
        # ``joblib.load`` (pickle) would execute arbitrary code on this
        # validator while reconstructing a hostile artifact. ``safe_load_model``
        # restricts deserialization to an allowlist of numpy/sklearn classes and
        # raises ``UnsafeArtifactError`` on anything else, which the submission
        # pipeline turns into a rejection. See ``insignia.safe_model_loader``.
        return safe_load_model(artifact)

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
# Code Submission Validator
# ---------------------------------------------------------------------------

class CodeSubmissionValidator:
    """
    Validates the code half of a researcher submission.

    Pipeline per submission:
      1. **Structure + static safety** — ``CodeBundleVerifier`` confirms the
         bundle hash, manifest, entrypoint, and that no source contains a
         sandbox-escaping pattern.
      2. **Plagiarism** — ``CodeFingerprinter`` flags miners shipping verbatim
         or lightly-edited copies of another miner's source.
      3. **Reproducibility** — ``ReproducibilityChecker`` re-runs the bundle's
         entrypoint in an isolated sandbox over the validator's evaluation
         features and verifies the reproduced predictions match the artifact's.

    The validator can then gate scoring on reproducibility so opaque,
    hard-coded, or tampered artifacts earn nothing.
    """

    def __init__(
        self,
        verifier: CodeBundleVerifier | None = None,
        reproducibility: ReproducibilityChecker | None = None,
        fingerprinter: CodeFingerprinter | None = None,
        bundle_config: CodeBundleConfig | None = None,
        sandbox_config: SandboxConfig | None = None,
        reject_duplicates: bool = False,
    ):
        self.bundle_config = bundle_config or CodeBundleConfig()
        self.verifier = verifier or CodeBundleVerifier(self.bundle_config)
        self.reproducibility = reproducibility or ReproducibilityChecker(
            runner=SandboxRunner(sandbox_config or SandboxConfig(), self.bundle_config)
        )
        self.fingerprinter = fingerprinter or CodeFingerprinter()
        # By default, identical code is *reported* (so a validator can apply
        # reward-sharing like ``ModelFingerprinter`` does for correlated models)
        # rather than hard-rejected — many honest miners legitimately start from
        # the public reference pipeline. Set True to reject verbatim copies.
        self.reject_duplicates = reject_duplicates

    def validate(
        self,
        miner_uid: str,
        code_bundle: bytes,
        code_entrypoint: str,
        code_bundle_hash: str = "",
        code_manifest: Dict[str, Any] | None = None,
        repro_features: Optional[np.ndarray] = None,
        repro_feature_names: Optional[List[str]] = None,
        reference_predictions: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "code_verified": False,
            "code_reproducible": False,
            "reproducibility_score": 0.0,
            "code_duplicate_of": [],
            "code_rejection_reason": "",
        }

        if not code_bundle:
            result["code_rejection_reason"] = "no code bundle submitted"
            return result

        verification = self.verifier.verify(
            archive=code_bundle,
            entrypoint=code_entrypoint,
            declared_hash=code_bundle_hash,
            manifest=code_manifest,
        )
        if not verification.ok:
            result["code_rejection_reason"] = verification.reason
            return result
        result["code_verified"] = True

        fingerprint = self.fingerprinter.compute(code_bundle)
        duplicates = self.fingerprinter.find_duplicates(miner_uid, fingerprint)
        self.fingerprinter.register(miner_uid, fingerprint)
        result["code_fingerprint"] = fingerprint[:16]
        if duplicates:
            result["code_duplicate_of"] = duplicates
            if self.reject_duplicates:
                result["code_rejection_reason"] = (
                    f"code duplicates miner(s): {', '.join(duplicates)}"
                )
                return result

        if repro_features is None or reference_predictions is None:
            result["code_rejection_reason"] = "no evaluation features for reproducibility"
            return result

        repro = self.reproducibility.check(
            archive=code_bundle,
            entrypoint=code_entrypoint,
            features=repro_features.tolist() if hasattr(repro_features, "tolist") else repro_features,
            feature_names=repro_feature_names or [],
            reference_predictions=(
                reference_predictions.tolist()
                if hasattr(reference_predictions, "tolist")
                else reference_predictions
            ),
        )
        result["reproducibility_score"] = repro.score
        result["code_reproducible"] = repro.ok
        result["repro_n_compared"] = repro.n_compared
        if not repro.ok:
            result["code_rejection_reason"] = repro.reason
        return result


# ---------------------------------------------------------------------------
# Model Validator Neuron (legacy single-layer evaluator)
# ---------------------------------------------------------------------------

class ModelValidator:
    """
    Model Validator neuron (legacy single-layer evaluator).

    Manages the full evaluation cycle:
      - Receives researcher miner model submissions
      - Applies anti-gaming checks
      - Runs model evaluation against proprietary benchmark
      - Computes weights for Yuma consensus
      - (Legacy) promotes top models for downstream trading use

    The validator maintains state across epochs: miner scores, model
    fingerprints, promotion history, and cross-layer feedback. Under the
    single paired mechanism this role is subsumed by ``PairedValidator``.
    """

    def __init__(
        self,
        evaluator: ModelEvaluator | None = None,
        weights: WeightConfig | None = None,
        top_n_promote: int = 10,
        code_validator: CodeSubmissionValidator | None = None,
        require_code: bool = False,
        gate_on_reproducibility: bool = True,
    ):
        self.evaluator = evaluator or ModelEvaluator()
        self.rate_limiter = SubmissionRateLimit()
        self.fingerprinter = ModelFingerprinter()
        self.feedback_engine = CrossLayerFeedbackEngine()
        self.top_n_promote = top_n_promote
        # Code-submission verification (reproducibility). ``require_code``
        # rejects artifact-only submissions; ``gate_on_reproducibility`` zeroes
        # the score of any submission whose code is present but fails to
        # reproduce the artifact.
        self.code_validator = code_validator or CodeSubmissionValidator()
        self.require_code = require_code
        self.gate_on_reproducibility = gate_on_reproducibility

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
        code_bundle: Optional[bytes] = None,
        code_entrypoint: str = "",
        code_bundle_hash: str = "",
        code_manifest: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a single miner's model submission.

        Returns a result dict with score, acceptance status, and diagnostics.

        If a ``code_bundle`` is supplied the validator verifies it, screens it
        for plagiarism, and re-runs it in a sandbox to confirm it reproduces the
        artifact's predictions. With ``gate_on_reproducibility`` set, a
        non-reproducible submission is scored zero; with ``require_code`` set, an
        artifact-only submission is rejected outright.
        """
        if not self.rate_limiter.check(miner_uid):
            return {
                "accepted": False,
                "rejection_reason": "Rate limited: 1 submission per epoch",
                "composite_score": 0.0,
            }

        if self.require_code and not code_bundle:
            return {
                "accepted": False,
                "rejection_reason": "Code submission required (no code_bundle provided)",
                "composite_score": 0.0,
            }

        fingerprint = self.fingerprinter.compute_fingerprint(model_artifact)
        if self.fingerprinter.is_exact_duplicate(miner_uid, fingerprint):
            return {
                "accepted": False,
                "rejection_reason": "Duplicate model detected",
                "composite_score": 0.0,
            }

        capture: Dict[str, Any] = {}
        try:
            score, diagnostics = self.evaluator.evaluate(
                model_artifact=model_artifact,
                features_used=features_used,
                epoch_id=self.current_epoch,
                capture=capture if code_bundle else None,
            )
        except UnsafeArtifactError as exc:
            # Hostile or unloadable artifact: reject and score 0 rather than
            # letting a malicious pickle run on the validator.
            self.rate_limiter.record(miner_uid)
            logger.warning("Miner %s: artifact rejected by safe loader (%s)", miner_uid, exc)
            return {
                "accepted": False,
                "composite_score": 0.0,
                "rejection_reason": f"Unsafe/invalid model artifact: {exc}",
            }

        # --- Code submission: verify + reproduce in sandbox ---
        code_result: Dict[str, Any] = {}
        if code_bundle:
            code_result = self.code_validator.validate(
                miner_uid=miner_uid,
                code_bundle=code_bundle,
                code_entrypoint=code_entrypoint,
                code_bundle_hash=code_bundle_hash,
                code_manifest=code_manifest,
                repro_features=capture.get("X_clean"),
                repro_feature_names=capture.get("available_features"),
                reference_predictions=capture.get("predictions"),
            )
            if self.gate_on_reproducibility and not code_result.get("code_reproducible"):
                self.rate_limiter.record(miner_uid)
                logger.info(
                    "Miner %s: code submission rejected (%s) — scored 0",
                    miner_uid, code_result.get("code_rejection_reason", "unverified"),
                )
                return {
                    "accepted": False,
                    "composite_score": 0.0,
                    "rejection_reason": "Code submission failed: "
                    + code_result.get("code_rejection_reason", "not reproducible"),
                    "code_submission": code_result,
                }

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
        if code_result:
            result["code_submission"] = code_result

        logger.info(
            "Miner %s: score=%.4f (regime=%s, l2_adj=%.2f%s)",
            miner_uid,
            adjusted_composite,
            diagnostics.get("regime", "unknown"),
            l2_adjustment,
            ", code reproduced repro=%.3f" % code_result["reproducibility_score"]
            if code_result.get("code_reproducible") else "",
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
                code_bundle=sub.get("code_bundle"),
                code_entrypoint=sub.get("code_entrypoint", ""),
                code_bundle_hash=sub.get("code_bundle_hash", ""),
                code_manifest=sub.get("code_manifest"),
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
    """Run a standalone model validator demonstration with multiple miners."""
    from neurons.researcher_miner import ResearcherMiner, ModelTrainer, generate_demo_data

    logger.info("=" * 60)
    logger.info("Insignia Model Validator — Demo Mode")
    logger.info("=" * 60)

    miners = {}
    for i in range(5):
        data = generate_demo_data(n_samples=3000, seed=42 + i)
        trainer = ModelTrainer(
            n_estimators=200 + i * 50,
            max_depth=4 + i,
            learning_rate=0.03 + i * 0.01,
            random_state=42 + i,
        )
        miner = ResearcherMiner(trainer=trainer)
        submission = miner.train_and_submit(data)
        miners[f"miner_{i}"] = submission
        logger.info("Miner %d trained: IS=%.4f OOS=%.4f",
                     i, miner.trainer.training_metrics["in_sample_accuracy"],
                     miner.trainer.training_metrics["out_of_sample_accuracy"])

    # require_code=True: artifact-only submissions are rejected; every accepted
    # submission must ship code that reproduces its artifact in the sandbox.
    validator = ModelValidator(top_n_promote=3, require_code=True)

    for epoch in range(3):
        logger.info("\n--- Epoch %d ---", epoch)
        epoch_result = validator.run_epoch(miners)

        for uid, result in epoch_result["results"].items():
            if result["accepted"]:
                code = result.get("code_submission", {})
                logger.info(
                    "  %s: rank=%d score=%.4f repro=%.3f %s",
                    uid,
                    result["rank"],
                    result["composite_score"],
                    code.get("reproducibility_score", 0.0),
                    "[PROMOTED]" if result.get("promoted_to_l2") else "",
                )
            else:
                logger.info("  %s: REJECTED (%s)", uid, result.get("rejection_reason", ""))

    logger.info("\n--- Promoted Models ---")
    for m in validator.get_promoted_models():
        logger.info("  %s (score=%.4f)", m["model_id"], m["composite_score"])

    return validator


if __name__ == "__main__":
    demo()
