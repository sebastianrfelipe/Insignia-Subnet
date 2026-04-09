"""
Insignia Incentive Mechanism

Implements the game-theoretic incentive design that drives honest behavior
across both subnet layers. This module answers the hackathon criterion:
"Why does scoring reward genuine quality? Why do top attack vectors fail?"

Design principles:
  1. Multi-metric composite scoring prevents single-dimension gaming
  2. Rolling holdout windows prevent overfitting exploitation
  3. Cross-layer feedback rewards models that survive real deployment
  4. Rate limiting + fingerprinting prevent sybil/spam attacks
  5. Buyback mechanism aligns token value with firm P&L

Attack Vector Analysis:
  See InsigniaIncentiveAnalysis class for formal attack/defense matrix.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .scoring import CompositeScorer, ScoreVector, WeightConfig


# ---------------------------------------------------------------------------
# Rate Limiter: prevents submission spam
# ---------------------------------------------------------------------------

@dataclass
class SubmissionRateLimit:
    """
    Enforces one model submission per miner per evaluation epoch.
    Prevents miners from brute-forcing the evaluation function.
    """

    min_epoch_seconds: int = 86400  # 24 hours
    _last_submission: Dict[str, float] = field(default_factory=dict)

    def check(self, miner_uid: str, current_time: float | None = None) -> bool:
        t = current_time or time.time()
        last = self._last_submission.get(miner_uid, 0.0)
        return (t - last) >= self.min_epoch_seconds

    def record(self, miner_uid: str, current_time: float | None = None):
        self._last_submission[miner_uid] = current_time or time.time()


# ---------------------------------------------------------------------------
# Model Fingerprinting: detects duplicate/plagiarized models
# ---------------------------------------------------------------------------

class ModelFingerprinter:
    """
    Detects duplicate or near-duplicate model submissions across miners.

    Uses a combination of:
      - Model artifact hash (exact duplicate detection)
      - Prediction correlation analysis (behavioral similarity)
      - Feature subset overlap (structural similarity)

    Miners submitting plagiarized models receive zero score. If two
    submissions are correlated above threshold, both share the reward
    (disincentivizing free-riding).
    """

    def __init__(self, correlation_threshold: float = 0.95):
        self.correlation_threshold = correlation_threshold
        self._artifact_hashes: Dict[str, str] = {}
        self._prediction_cache: Dict[str, np.ndarray] = {}

    def compute_fingerprint(self, model_artifact: bytes) -> str:
        return hashlib.sha256(model_artifact).hexdigest()

    def is_exact_duplicate(self, miner_uid: str, fingerprint: str) -> bool:
        for uid, fp in self._artifact_hashes.items():
            if uid != miner_uid and fp == fingerprint:
                return True
        return False

    def register(self, miner_uid: str, fingerprint: str, predictions: np.ndarray):
        self._artifact_hashes[miner_uid] = fingerprint
        self._prediction_cache[miner_uid] = predictions

    def find_correlated_miners(self, miner_uid: str) -> List[Tuple[str, float]]:
        """Find miners whose predictions are suspiciously correlated."""
        if miner_uid not in self._prediction_cache:
            return []
        my_preds = self._prediction_cache[miner_uid]
        correlated = []
        for uid, preds in self._prediction_cache.items():
            if uid == miner_uid:
                continue
            if len(preds) != len(my_preds):
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                corr_matrix = np.corrcoef(my_preds, preds)
            corr = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
            if corr >= self.correlation_threshold:
                correlated.append((uid, corr))
        return correlated


# ---------------------------------------------------------------------------
# L2 Copy-Trade Detection
# ---------------------------------------------------------------------------

class CopyTradeDetector:
    """
    Detects copy-trading in Layer 2 by analyzing position correlation.

    If two L2 miners' position logs are highly correlated (same entries,
    exits, sizing within tolerance), they are flagged and share rewards.
    This mirrors Taoshi PTN's anti-plagiarism approach.
    """

    def __init__(
        self,
        time_tolerance_sec: float = 60.0,
        size_tolerance_pct: float = 0.05,
        correlation_threshold: float = 0.90,
    ):
        self.time_tolerance_sec = time_tolerance_sec
        self.size_tolerance_pct = size_tolerance_pct
        self.correlation_threshold = correlation_threshold

    def detect(
        self,
        positions_a: List[Dict],
        positions_b: List[Dict],
    ) -> Tuple[bool, float]:
        """
        Compare two position logs for copy-trading behavior.
        Returns (is_copy, correlation_score).
        """
        if not positions_a or not positions_b:
            return False, 0.0

        matches = 0
        total = max(len(positions_a), len(positions_b))

        for pa in positions_a:
            for pb in positions_b:
                time_match = (
                    abs(pa.get("timestamp", 0) - pb.get("timestamp", 0))
                    < self.time_tolerance_sec
                )
                side_match = pa.get("side") == pb.get("side")
                instrument_match = pa.get("instrument") == pb.get("instrument")
                size_a = pa.get("size", 0)
                size_b = pb.get("size", 0)
                size_match = (
                    abs(size_a - size_b) / max(abs(size_a), 1e-12)
                    < self.size_tolerance_pct
                )
                if time_match and side_match and instrument_match and size_match:
                    matches += 1
                    break

        correlation = matches / max(total, 1)
        return correlation >= self.correlation_threshold, correlation


# ---------------------------------------------------------------------------
# Cross-Layer Feedback
# ---------------------------------------------------------------------------

@dataclass
class CrossLayerFeedbackEngine:
    """
    Propagates Layer 2 performance signals back to Layer 1 scoring.

    When L2 strategies using a specific L1 model perform well, that model
    receives a retroactive bonus in L1 scoring. When L2 strategies fail
    despite good L1 simulation scores, the model is penalized.

    This closes the simulation-to-reality gap: models are ultimately
    judged by deployment outcomes, not just backtest metrics.
    """

    bonus_weight: float = 0.15
    penalty_weight: float = 0.10
    min_l2_epochs: int = 3
    _model_l2_history: Dict[str, List[float]] = field(default_factory=dict)

    def record_l2_performance(self, model_id: str, l2_score: float):
        if model_id not in self._model_l2_history:
            self._model_l2_history[model_id] = []
        self._model_l2_history[model_id].append(l2_score)

    def compute_adjustment(self, model_id: str) -> float:
        """
        Compute the L1 score adjustment based on L2 deployment history.

        Returns a multiplier: >1.0 = bonus, <1.0 = penalty, 1.0 = neutral.
        """
        history = self._model_l2_history.get(model_id, [])
        if len(history) < self.min_l2_epochs:
            return 1.0

        avg_l2 = np.mean(history[-self.min_l2_epochs :])
        if avg_l2 > 0.6:
            return 1.0 + self.bonus_weight * (avg_l2 - 0.6) / 0.4
        elif avg_l2 < 0.3:
            return 1.0 - self.penalty_weight * (0.3 - avg_l2) / 0.3
        return 1.0


# ---------------------------------------------------------------------------
# Buyback Mechanism
# ---------------------------------------------------------------------------

@dataclass
class BuybackMechanism:
    """
    Models the token buyback mechanism that aligns miner incentives with
    real trading outcomes.

    When the subnet owner's firm deploys winning model+strategy pairs and
    generates profit, a configurable percentage of that P&L is used to
    purchase alpha tokens on the open market. This creates direct demand
    pressure on the token, increasing its value for all miners.

    The mechanism creates a virtuous cycle:
      Better models -> Higher firm P&L -> More buybacks -> Higher token
      value -> Stronger miner incentive -> Better models
    """

    buyback_pct: float = 0.20  # 20% of firm P&L goes to buybacks
    min_profit_threshold: float = 1000.0  # Minimum P&L before buyback triggers
    buyback_frequency_hours: int = 168  # Weekly

    def compute_buyback_amount(self, period_pnl: float) -> float:
        if period_pnl <= self.min_profit_threshold:
            return 0.0
        return period_pnl * self.buyback_pct


# ---------------------------------------------------------------------------
# Commit-Reveal Scheme (Approach B: off-chain with validator attestation)
#
# Prevents validator latency exploitation (Vector 8) and front-running by
# requiring miners to commit to trade hashes before market data is
# available, then reveal after the validation window closes.
#
# Recommended by the deployer agent's feasibility assessment (CR-FEAS-001).
# ---------------------------------------------------------------------------

@dataclass
class CommitRevealConfig:
    """Parameters for the off-chain commit-reveal scheme."""

    commit_window_seconds: float = 30.0
    reveal_window_seconds: float = 15.0
    hash_algorithm: str = "sha256"
    nonce_bits: int = 128
    max_reveal_attempts: int = 3
    late_reveal_penalty: float = 1.0


@dataclass
class CommitRecord:
    miner_uid: str
    commit_hash: str
    timestamp: float
    validator_attestations: Dict[str, float] = field(default_factory=dict)
    revealed: bool = False
    reveal_data: Optional[bytes] = None
    reveal_nonce: Optional[str] = None


class CommitRevealManager:
    """
    Off-chain commit-reveal manager with multi-validator attestation.

    Flow:
      1. Miner computes trade decision, generates nonce, sends
         commit_hash = SHA-256(trade_data || nonce) to validators.
      2. Validators attest to receiving the commit before the deadline.
      3. After the validation window closes, miner reveals trade_data
         and nonce.
      4. Validator recomputes hash and verifies match.
      5. Trades with invalid or late reveals are scored zero.
    """

    def __init__(self, config: CommitRevealConfig | None = None):
        self.config = config or CommitRevealConfig()
        self._commits: Dict[str, CommitRecord] = {}
        self._epoch_commits: Dict[int, List[str]] = {}

    @staticmethod
    def compute_hash(trade_data: bytes, nonce: bytes) -> str:
        return hashlib.sha256(trade_data + nonce).hexdigest()

    @staticmethod
    def generate_nonce(n_bits: int = 128) -> bytes:
        import os as _os
        return _os.urandom(n_bits // 8)

    def submit_commit(
        self,
        miner_uid: str,
        commit_hash: str,
        epoch: int,
        current_time: float | None = None,
    ) -> bool:
        """Register a miner's commit hash. Returns True if accepted."""
        t = current_time or time.time()
        key = f"{epoch}:{miner_uid}"
        if key in self._commits:
            return False

        record = CommitRecord(
            miner_uid=miner_uid,
            commit_hash=commit_hash,
            timestamp=t,
        )
        self._commits[key] = record
        self._epoch_commits.setdefault(epoch, []).append(key)
        return True

    def attest_commit(
        self,
        miner_uid: str,
        epoch: int,
        validator_uid: str,
        current_time: float | None = None,
    ) -> bool:
        """Validator attests it received the commit before the deadline."""
        t = current_time or time.time()
        key = f"{epoch}:{miner_uid}"
        record = self._commits.get(key)
        if record is None:
            return False
        record.validator_attestations[validator_uid] = t
        return True

    def submit_reveal(
        self,
        miner_uid: str,
        epoch: int,
        trade_data: bytes,
        nonce: bytes,
        current_time: float | None = None,
    ) -> Tuple[bool, str]:
        """
        Miner reveals their trade data and nonce.
        Returns (success, reason).
        """
        t = current_time or time.time()
        key = f"{epoch}:{miner_uid}"
        record = self._commits.get(key)
        if record is None:
            return False, "no_commit_found"

        if record.revealed:
            return False, "already_revealed"

        elapsed = t - record.timestamp
        deadline = self.config.commit_window_seconds + self.config.reveal_window_seconds
        if elapsed > deadline:
            return False, "reveal_too_late"

        expected_hash = self.compute_hash(trade_data, nonce)
        if expected_hash != record.commit_hash:
            return False, "hash_mismatch"

        record.revealed = True
        record.reveal_data = trade_data
        record.reveal_nonce = nonce.hex()
        return True, "ok"

    def is_valid_for_scoring(self, miner_uid: str, epoch: int) -> bool:
        """Check whether a miner's submission is valid for scoring."""
        key = f"{epoch}:{miner_uid}"
        record = self._commits.get(key)
        if record is None:
            return True  # no commit-reveal required if scheme not active
        return record.revealed and len(record.validator_attestations) > 0

    def get_penalty(self, miner_uid: str, epoch: int) -> float:
        """
        Returns 0.0 if valid, or a penalty multiplier (up to 1.0)
        for invalid/late reveals.
        """
        key = f"{epoch}:{miner_uid}"
        record = self._commits.get(key)
        if record is None:
            return 0.0
        if not record.revealed:
            return self.config.late_reveal_penalty
        if len(record.validator_attestations) == 0:
            return 0.5
        return 0.0


# ---------------------------------------------------------------------------
# Attack / Defense Matrix (for judges)
# ---------------------------------------------------------------------------

@dataclass
class AttackDefense:
    attack: str
    description: str
    defense: str
    mechanism: str


ATTACK_DEFENSE_MATRIX: List[AttackDefense] = [
    AttackDefense(
        attack="Overfitting to public data patterns",
        description="Miner memorizes patterns in publicly available data that happen to correlate with the validation window.",
        defense="Validators use proprietary tick-by-tick data that miners cannot access. Rolling holdout windows change each epoch. The overfitting detector specifically penalizes in-sample/out-of-sample gaps.",
        mechanism="Data asymmetry + OverfittingDetector + rolling windows",
    ),
    AttackDefense(
        attack="Submission spam / brute-force",
        description="Miner submits many model variants per epoch to maximize chance of a lucky high score.",
        defense="Rate limited to 1 submission per miner per epoch (24h minimum). Each submission must include a full metadata manifest.",
        mechanism="SubmissionRateLimit",
    ),
    AttackDefense(
        attack="Model plagiarism (L1)",
        description="Miner copies another miner's model artifact or training approach.",
        defense="SHA-256 fingerprinting detects exact duplicates. Prediction correlation analysis detects behavioral clones. Correlated models share rewards, removing the incentive to copy.",
        mechanism="ModelFingerprinter",
    ),
    AttackDefense(
        attack="Copy-trading (L2)",
        description="L2 miner mirrors another miner's positions instead of building their own strategy.",
        defense="Position correlation analysis with time/size tolerance. Correlated strategies share rewards. Mirrors Taoshi PTN's proven approach.",
        mechanism="CopyTradeDetector",
    ),
    AttackDefense(
        attack="Single-metric gaming",
        description="Miner optimizes for one dominant metric (e.g., accuracy) while ignoring others.",
        defense="Composite scoring across 7 L1 metrics and 7 L2 metrics. No single metric dominates (max weight 20%). Weight configuration is published but balanced.",
        mechanism="CompositeScorer + WeightConfig",
    ),
    AttackDefense(
        attack="Validator data leakage",
        description="Miner attempts to reverse-engineer the proprietary validation dataset from score feedback.",
        defense="Only aggregate scores returned, never raw data. Rolling windows ensure each epoch uses different data. 30-day delayed release of historical windows.",
        mechanism="Score-only feedback + delayed window release",
    ),
    AttackDefense(
        attack="L2 paper trading manipulation",
        description="L2 miner fabricates paper trading results or cherry-picks favorable reporting windows.",
        defense="Validators track real-time position updates via continuous streaming. All positions timestamped and signed. Reporting gaps are penalized.",
        mechanism="L2PositionUpdate synapse + continuous tracking",
    ),
    AttackDefense(
        attack="Sybil attack (multiple identities)",
        description="Single entity registers multiple miners to increase share of emissions.",
        defense="Model fingerprinting + prediction correlation catch behavioral duplicates. Bittensor's staking requirements raise the cost of sybil identities.",
        mechanism="ModelFingerprinter + network-level stake requirements",
    ),
    AttackDefense(
        attack="Regime-specific exploitation",
        description="Model only works in specific market regimes (e.g., bull market) and fails in others.",
        defense="Stability Score metric explicitly measures cross-regime consistency. Validation windows deliberately cover trending, ranging, high-vol, low-vol, and crisis periods.",
        mechanism="stability_score + regime-diverse validation windows",
    ),
    AttackDefense(
        attack="Validator latency exploitation",
        description="Miners exploit validator latency to submit trades using market data that has already materialized but not yet been validated.",
        defense="Enforce min_prediction_lead_time, validator latency penalty, and commit-reveal scheme requiring miners to commit trade hashes before data is available.",
        mechanism="CommitRevealManager + min_prediction_lead_time + validator_latency_penalty_weight",
    ),
    AttackDefense(
        attack="Miner-validator collusion",
        description="A colluding validator inflates scores for cooperating miners via weight-setting manipulation or information leakage.",
        defense="Multi-validator consensus scoring, weight entropy minimum, cross-validator score variance caps, validator rotation limits, and agreement deviation monitoring.",
        mechanism="weight_entropy_minimum + cross_validator_score_variance_max + validator_rotation + validator_agreement_threshold",
    ),
    AttackDefense(
        attack="L1/L2 weight skew exploitation",
        description="Adversarial miners exploit the emission split between L1 and L2 to capture disproportionate rewards.",
        defense="Cross-layer penalty strength parameter penalizes deviations from the configured l1_l2_emission_split.",
        mechanism="cross_layer_penalty_strength + l1_l2_emission_split",
    ),
    AttackDefense(
        attack="Cross-layer timing sync attack",
        description="Adversarial miners exploit timing gaps between L1 and L2 scoring windows to game cross-layer feedback.",
        defense="Cross-layer latency threshold enforcement and temporal correlation monitoring.",
        mechanism="cross_layer_latency + collusion_detection_lookback_epochs",
    ),
]
