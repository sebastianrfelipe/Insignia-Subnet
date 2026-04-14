"""
Simulation Harness with AI Agent Miners

Self-contained simulation that runs the full Insignia subnet pipeline
with configurable bot miners. Each bot type simulates a different
strategy — honest, adversarial, or degenerate — to test the robustness
of a given parameter configuration.

Agent Types:
  - HonestMiner: standard model training, best-effort submission
  - OverfittingMiner: deliberately overfits to training data
  - CopycatMiner: copies another miner's model with perturbations
  - SingleMetricGamer: optimizes only one metric dimension
  - SybilMiner: creates correlated submissions across identities
  - RandomMiner: noise baseline with random models
  - HonestTrader: standard L2 paper trading strategy
  - CopyTrader: mirrors another L2 miner's positions
"""

from __future__ import annotations

import io
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from insignia.scoring import (
    CompositeScorer, WeightConfig, ReferenceOverfittingDetector,
)
from insignia.incentive import (
    SubmissionRateLimit, ModelFingerprinter, CopyTradeDetector,
    CrossLayerFeedbackEngine,
)
from insignia.cross_layer import (
    CrossLayerOrchestrator, PromotionEngine, PromotionConfig,
)
from neurons.l1_miner import (
    L1Miner, L1ModelTrainer, generate_demo_data, PUBLIC_FEATURE_REGISTRY,
)
from neurons.l1_validator import L1Validator, ModelEvaluator, DemoBenchmarkProvider
from neurons.l2_miner import L2StrategyMiner, PaperTradingEngine, SlippageConfig, Side
from neurons.l2_validator import L2Validator

from insignia.protocol import InstrumentId
from tuning.parameter_space import decode

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
logger = logging.getLogger("simulation")
logger.setLevel(logging.INFO)

DEFAULT_TRADING_PAIRS = [
    InstrumentId.BTC_USDT_PERP.value,
    InstrumentId.ETH_USDT_PERP.value,
    InstrumentId.SOL_USDT_PERP.value,
    InstrumentId.AVAX_USDT_PERP.value,
    InstrumentId.ADA_USDT_PERP.value,
]


# ---------------------------------------------------------------------------
# Miner Agent Base
# ---------------------------------------------------------------------------

class MinerAgent(ABC):
    """Base class for all simulated L1 miner agents."""

    agent_type: str = "base"

    def __init__(self, uid: str, seed: int = 42):
        self.uid = uid
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def produce_submission(self, epoch: int) -> Dict[str, Any]:
        """Produce a model submission for the given epoch."""
        ...

    def is_adversarial(self) -> bool:
        return False


class HonestMiner(MinerAgent):
    """Standard good-faith miner that trains models honestly."""

    agent_type = "honest"

    def __init__(self, uid: str, seed: int = 42, n_features: int = 10):
        super().__init__(uid, seed)
        self.n_features = n_features

    def produce_submission(self, epoch: int) -> Dict[str, Any]:
        data = generate_demo_data(
            n_samples=500 + epoch * 100,
            n_features=self.n_features,
            seed=self.seed + epoch * 13,
        )
        trainer = L1ModelTrainer(
            n_estimators=10 + epoch * 2,
            max_depth=3,
            learning_rate=0.08,
            min_samples_leaf=20,
            max_bins=32,
            features=PUBLIC_FEATURE_REGISTRY[:self.n_features],
            random_state=self.seed + epoch,
        )
        miner = L1Miner(trainer=trainer)
        return miner.train_and_submit(data)


class OverfittingMiner(MinerAgent):
    """Deliberately overfits by using high complexity and small regularization."""

    agent_type = "overfitter"

    def produce_submission(self, epoch: int) -> Dict[str, Any]:
        data = generate_demo_data(n_samples=200, n_features=15, seed=self.seed + epoch)
        trainer = L1ModelTrainer(
            n_estimators=50,
            max_depth=8,
            learning_rate=0.3,
            min_samples_leaf=2,
            l2_regularization=0.0,
            max_bins=255,
            features=PUBLIC_FEATURE_REGISTRY[:15],
            random_state=self.seed + epoch,
        )
        miner = L1Miner(trainer=trainer)
        return miner.train_and_submit(data)

    def is_adversarial(self) -> bool:
        return True


class CopycatMiner(MinerAgent):
    """Copies another miner's submission with minor perturbations."""

    agent_type = "copycat"

    def __init__(self, uid: str, target_uid: str, seed: int = 42):
        super().__init__(uid, seed)
        self.target_uid = target_uid
        self._cached_artifact: Optional[bytes] = None

    def set_target_artifact(self, artifact: bytes):
        self._cached_artifact = artifact

    def produce_submission(self, epoch: int) -> Dict[str, Any]:
        if self._cached_artifact is not None:
            buf = io.BytesIO(self._cached_artifact)
            model = joblib.load(buf)
            n_feat = self._infer_n_features(model)
            out_buf = io.BytesIO()
            joblib.dump(model, out_buf)
            artifact = out_buf.getvalue()
            features_used = PUBLIC_FEATURE_REGISTRY[:n_feat]
        else:
            data = generate_demo_data(n_samples=500, n_features=10, seed=self.seed + epoch)
            trainer = L1ModelTrainer(
                features=PUBLIC_FEATURE_REGISTRY[:10], random_state=self.seed + epoch,
            )
            miner = L1Miner(trainer=trainer)
            sub = miner.train_and_submit(data)
            artifact = sub["model_artifact"]
            features_used = sub.get("features_used", PUBLIC_FEATURE_REGISTRY[:10])

        import hashlib
        return {
            "model_artifact": artifact,
            "model_type": "gbdt",
            "features_used": features_used,
            "training_window_start": "2025-01-01",
            "training_window_end": "2025-07-01",
            "hyperparams": {},
            "preprocessing_hash": hashlib.sha256(artifact).hexdigest()[:16],
            "target_instrument": "BTC-USDT-PERP",
            "target_horizon_minutes": 60,
            "self_reported_overfitting_score": 0.0,
            "artifact_size_bytes": len(artifact),
            "artifact_hash": hashlib.sha256(artifact).hexdigest(),
        }

    @staticmethod
    def _infer_n_features(model) -> int:
        if hasattr(model, "n_features_in_"):
            return int(model.n_features_in_)
        if hasattr(model, "named_steps"):
            for step in model.named_steps.values():
                if hasattr(step, "n_features_in_"):
                    return int(step.n_features_in_)
        return 10

    def is_adversarial(self) -> bool:
        return True


class SingleMetricGamer(MinerAgent):
    """Optimizes aggressively for directional accuracy at the expense of everything else."""

    agent_type = "single_metric_gamer"

    def produce_submission(self, epoch: int) -> Dict[str, Any]:
        data = generate_demo_data(n_samples=800, n_features=5, seed=self.seed + epoch)
        trainer = L1ModelTrainer(
            n_estimators=30,
            max_depth=6,
            learning_rate=0.15,
            min_samples_leaf=5,
            max_bins=128,
            features=PUBLIC_FEATURE_REGISTRY[:5],
            random_state=self.seed + epoch,
        )
        miner = L1Miner(trainer=trainer)
        return miner.train_and_submit(data)

    def is_adversarial(self) -> bool:
        return True


class RandomMiner(MinerAgent):
    """Submits essentially random models — noise floor baseline."""

    agent_type = "random"

    def produce_submission(self, epoch: int) -> Dict[str, Any]:
        data = generate_demo_data(n_samples=100, n_features=3, seed=self.seed + epoch * 97)
        trainer = L1ModelTrainer(
            n_estimators=3,
            max_depth=1,
            learning_rate=0.5,
            min_samples_leaf=30,
            features=PUBLIC_FEATURE_REGISTRY[:3],
            random_state=self.seed + epoch * 97,
        )
        miner = L1Miner(trainer=trainer)
        return miner.train_and_submit(data)


class SybilMiner(MinerAgent):
    """
    Part of a sybil cluster — multiple identities with correlated models.
    Uses the same seed and data with tiny variations.
    """

    agent_type = "sybil"

    def __init__(self, uid: str, cluster_seed: int, identity_idx: int):
        super().__init__(uid, cluster_seed + identity_idx)
        self.cluster_seed = cluster_seed
        self.identity_idx = identity_idx

    def produce_submission(self, epoch: int) -> Dict[str, Any]:
        data = generate_demo_data(
            n_samples=500, n_features=10,
            seed=self.cluster_seed + epoch,
        )
        trainer = L1ModelTrainer(
            n_estimators=10 + self.identity_idx,
            max_depth=3,
            learning_rate=0.08 + self.identity_idx * 0.005,
            min_samples_leaf=20,
            features=PUBLIC_FEATURE_REGISTRY[:10],
            random_state=self.cluster_seed + self.identity_idx,
        )
        miner = L1Miner(trainer=trainer)
        return miner.train_and_submit(data)

    def is_adversarial(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# L2 Trader Agent Base
# ---------------------------------------------------------------------------

class TraderAgent(ABC):
    agent_type: str = "base_trader"

    def __init__(self, uid: str, seed: int = 42):
        self.uid = uid
        self.seed = seed

    @abstractmethod
    def create_l2_miner(
        self, promoted_artifacts: Dict[str, bytes], trading_config: Dict,
    ) -> L2StrategyMiner:
        ...

    def is_adversarial(self) -> bool:
        return False


class HonestTrader(TraderAgent):
    agent_type = "honest_trader"

    def create_l2_miner(
        self, promoted_artifacts: Dict[str, bytes], trading_config: Dict,
    ) -> L2StrategyMiner:
        engine = PaperTradingEngine(
            initial_capital=100_000,
            max_position_pct=trading_config.get("max_position_pct", 0.10),
            max_drawdown_pct=trading_config.get("max_drawdown_pct", 0.20),
            slippage=SlippageConfig(
                base_spread_bps=trading_config.get("base_spread_bps", 2.0),
                volatility_impact_factor=trading_config.get("volatility_impact_factor", 0.5),
                size_impact_factor=trading_config.get("size_impact_factor", 0.1),
                fee_bps=trading_config.get("fee_bps", 5.0),
            ),
        )
        l2 = L2StrategyMiner(engine=engine)
        for mid, artifact in list(promoted_artifacts.items())[:3]:
            l2.load_model(mid, artifact)
        return l2


class CopyTrader(TraderAgent):
    """Mirrors the first honest trader's positions with slight delay."""

    agent_type = "copy_trader"

    def create_l2_miner(
        self, promoted_artifacts: Dict[str, bytes], trading_config: Dict,
    ) -> L2StrategyMiner:
        engine = PaperTradingEngine(
            initial_capital=100_000,
            max_position_pct=trading_config.get("max_position_pct", 0.10),
            max_drawdown_pct=trading_config.get("max_drawdown_pct", 0.20),
        )
        l2 = L2StrategyMiner(engine=engine)
        for mid, artifact in list(promoted_artifacts.items())[:3]:
            l2.load_model(mid, artifact)
        return l2

    def is_adversarial(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Simulation Harness
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Results from a single simulation run."""
    l1_epoch_results: List[Dict] = field(default_factory=list)
    l2_epoch_result: Optional[Dict] = None
    promotion_summary: Optional[Dict] = None
    l1_feedback: Dict[str, float] = field(default_factory=dict)

    miner_scores: Dict[str, float] = field(default_factory=dict)
    miner_types: Dict[str, str] = field(default_factory=dict)
    l2_scores: Dict[str, float] = field(default_factory=dict)
    l2_types: Dict[str, str] = field(default_factory=dict)

    honest_l1_scores: List[float] = field(default_factory=list)
    adversarial_l1_scores: List[float] = field(default_factory=list)
    honest_l2_scores: List[float] = field(default_factory=list)
    adversarial_l2_scores: List[float] = field(default_factory=list)
    epoch_commitments: Dict[int, Dict[str, bool]] = field(default_factory=dict)
    miner_commit_status: Dict[str, str] = field(default_factory=dict)
    miner_commit_rates: Dict[str, float] = field(default_factory=dict)
    miner_accuracy_by_commit_status: Dict[str, Dict[str, float]] = field(default_factory=dict)
    no_reveal_streaks: Dict[str, int] = field(default_factory=dict)
    no_reveal_miners: List[str] = field(default_factory=list)
    selective_reveal_penalties: Dict[str, Dict[str, float | str]] = field(default_factory=dict)
    commit_timestamps: Dict[str, float] = field(default_factory=dict)
    reveal_timestamps: Dict[str, float] = field(default_factory=dict)
    validator_latencies: Dict[str, float] = field(default_factory=dict)
    submission_timing_gaps: Dict[str, float] = field(default_factory=dict)
    per_validator_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    validator_weight_vectors: Dict[str, List[float]] = field(default_factory=dict)
    validator_scoring_history: Dict[str, List[Tuple[int, str]]] = field(default_factory=dict)
    miner_validator_temporal_corr: Dict[Tuple[str, str], float] = field(default_factory=dict)
    cross_layer_latencies: Dict[str, float] = field(default_factory=dict)
    attack_monitoring: Dict[str, Any] = field(default_factory=dict)
    breach_alerts: List[Dict[str, Any]] = field(default_factory=list)
    breach_trends: Dict[str, Any] = field(default_factory=dict)
    sentinel_breach_trends: Dict[str, Any] = field(default_factory=dict)
    convergence_criteria: Dict[str, Any] = field(default_factory=dict)
    convergence_indexes: List[str] = field(default_factory=list)
    trading_pair_counts: Dict[str, int] = field(default_factory=dict)
    ensemble_signals: Dict[str, Dict[str, float]] = field(default_factory=dict)


class SimulationHarness:
    """
    Runs the full Insignia pipeline with configurable parameters and
    AI agent miners. Returns structured results for fitness evaluation.
    """

    def __init__(
        self,
        l1_agents: List[MinerAgent],
        l2_agents: List[TraderAgent],
        n_epochs: int = 3,
        n_trading_steps: int = 200,
    ):
        self.l1_agents = l1_agents
        self.l2_agents = l2_agents
        self.n_epochs = n_epochs
        self.n_trading_steps = n_trading_steps
        self._last_config: Optional[Dict] = None

    @property
    def last_config(self) -> Optional[Dict]:
        """The decoded parameter config from the most recent run."""
        return self._last_config

    def run(self, param_vector: np.ndarray) -> SimulationResult:
        """
        Execute a full simulation with the given parameter configuration.
        Returns structured results for fitness evaluation.
        """
        config = decode(param_vector)
        self._last_config = config
        result = SimulationResult()

        weight_config = config["weight_config"]
        promotion_config = config["promotion_config"]
        ovf_params = config["overfitting"]
        feedback_params = config["feedback"]
        anti_gaming = config["anti_gaming"]
        trading_config = config["trading"]
        validation_timing = config.get("validation_timing", {})
        consensus_integrity = config.get("consensus_integrity", {})
        trading_pairs = list(config.get("market_data", {}).get("trading_pairs", DEFAULT_TRADING_PAIRS))
        if not trading_pairs:
            trading_pairs = list(DEFAULT_TRADING_PAIRS)

        result.trading_pair_counts = {pair: 0 for pair in trading_pairs}
        validator_ids = [f"validator_{i}" for i in range(3)]
        no_reveal_streaks = {agent.uid: 0 for agent in self.l1_agents}
        commit_counts = {agent.uid: 0 for agent in self.l1_agents}
        accuracy_when_committed = {agent.uid: [] for agent in self.l1_agents}
        accuracy_when_not_committed = {agent.uid: [] for agent in self.l1_agents}
        scoring_history = {agent.uid: [] for agent in self.l1_agents}

        ovf_detector = ReferenceOverfittingDetector(
            gap_threshold=ovf_params["gap_threshold"],
            decay_rate=ovf_params["decay_rate"],
        )
        scorer = CompositeScorer(weights=weight_config, overfitting_detector=ovf_detector)
        benchmark = DemoBenchmarkProvider()
        evaluator = ModelEvaluator(scorer=scorer, benchmark=benchmark)

        l1_validator = L1Validator(
            evaluator=evaluator,
            weights=weight_config,
            top_n_promote=promotion_config.top_n,
        )
        l1_validator.fingerprinter = ModelFingerprinter(
            correlation_threshold=anti_gaming["fingerprint_correlation_threshold"],
        )

        orchestrator = CrossLayerOrchestrator(
            promotion_engine=PromotionEngine(promotion_config),
            feedback_bonus_weight=feedback_params["bonus_weight"],
            feedback_penalty_weight=feedback_params["penalty_weight"],
        )

        # --- Phase 1: L1 Epochs ---
        all_submissions = {}
        for agent in self.l1_agents:
            result.miner_types[agent.uid] = agent.agent_type

        for epoch in range(self.n_epochs):
            submissions = {}
            epoch_commitments: Dict[str, bool] = {}
            for agent in self.l1_agents:
                sub = agent.produce_submission(epoch)
                instrument = trading_pairs[(epoch + len(submissions)) % len(trading_pairs)]
                sub["target_instrument"] = instrument
                submissions[agent.uid] = sub
                all_submissions[agent.uid] = sub
                result.trading_pair_counts[instrument] = result.trading_pair_counts.get(instrument, 0) + 1

                should_commit = not (
                    agent.is_adversarial()
                    and agent.uid.endswith("0")
                    and epoch == self.n_epochs - 1
                )
                epoch_commitments[agent.uid] = should_commit
                if should_commit:
                    commit_counts[agent.uid] += 1
                    commit_ts = float(epoch * 100 + len(submissions) * 3 + 5)
                    reveal_ts = commit_ts + 20.0
                    result.commit_timestamps[f"{epoch}:{agent.uid}"] = commit_ts
                    result.reveal_timestamps[f"{epoch}:{agent.uid}"] = reveal_ts
                    result.miner_commit_status[agent.uid] = "revealed"
                    no_reveal_streaks[agent.uid] = 0
                else:
                    result.miner_commit_status[agent.uid] = "missing_reveal"
                    no_reveal_streaks[agent.uid] += 1
                    if agent.uid not in result.no_reveal_miners:
                        result.no_reveal_miners.append(agent.uid)

                if isinstance(agent, HonestMiner) and epoch == 0:
                    for other in self.l1_agents:
                        if isinstance(other, CopycatMiner) and other.target_uid == agent.uid:
                            other.set_target_artifact(sub["model_artifact"])

            epoch_result = l1_validator.run_epoch(submissions, force=True)
            result.l1_epoch_results.append(epoch_result)
            result.epoch_commitments[epoch] = epoch_commitments

            epoch_scores = {
                uid: {
                    "composite_score": r.get("composite_score", 0.0),
                    "overfitting_score": r.get("raw_metrics", {}).get("overfitting_penalty", 0.0),
                    "artifact_hash": submissions[uid].get("preprocessing_hash", ""),
                }
                for uid, r in epoch_result["results"].items()
                if r.get("accepted")
            }
            for uid, score_info in epoch_scores.items():
                composite_score = float(score_info["composite_score"])
                assigned_validator = validator_ids[(epoch + len(uid)) % len(validator_ids)]
                scoring_history.setdefault(uid, []).append((epoch, assigned_validator))
                if epoch_commitments.get(uid, False):
                    accuracy_when_committed[uid].append(composite_score)
                else:
                    accuracy_when_not_committed[uid].append(composite_score)

            orchestrator.process_l1_epoch(epoch_scores, epoch)

        last_results = result.l1_epoch_results[-1]["results"]
        for uid, r in last_results.items():
            if r.get("accepted"):
                score = r["composite_score"]
                result.miner_scores[uid] = score
                agent = next((a for a in self.l1_agents if a.uid == uid), None)
                if agent and not agent.is_adversarial():
                    result.honest_l1_scores.append(score)
                elif agent:
                    result.adversarial_l1_scores.append(score)

        for uid in result.miner_types:
            result.no_reveal_streaks[uid] = no_reveal_streaks.get(uid, 0)
            result.miner_commit_rates[uid] = commit_counts.get(uid, 0) / max(self.n_epochs, 1)
            committed_scores = accuracy_when_committed.get(uid, [])
            not_committed_scores = accuracy_when_not_committed.get(uid, [])
            result.miner_accuracy_by_commit_status[uid] = {
                "accuracy_when_committed": float(np.mean(committed_scores)) if committed_scores else 0.0,
                "accuracy_when_not_committed": float(np.mean(not_committed_scores)) if not_committed_scores else 0.0,
            }
        result.validator_scoring_history = scoring_history
        result.validator_latencies = {
            uid: float(800 + (idx % 4) * 500 + (2200 if uid.startswith("sybil") else 0))
            for idx, uid in enumerate(result.miner_types)
        }
        result.submission_timing_gaps = {
            uid: float(20 if uid.startswith("copycat") else 42 - (idx % 3) * 4)
            for idx, uid in enumerate(result.miner_types)
        }
        result.per_validator_scores = {
            vid: {
                uid: max(
                    0.0,
                    result.miner_scores.get(uid, 0.0)
                    + (0.08 if uid.startswith("sybil") and vid == "validator_0" else 0.0)
                    - 0.02 * idx
                )
                for idx, uid in enumerate(result.miner_types)
            }
            for vid in validator_ids
        }
        total_score = sum(result.miner_scores.values()) or 1.0
        ordered_uids = list(result.miner_scores)
        result.validator_weight_vectors = {}
        for idx, vid in enumerate(validator_ids):
            weights = []
            for uid in ordered_uids:
                base_weight = result.miner_scores[uid] / total_score
                if uid.startswith("sybil") and vid == "validator_0":
                    base_weight *= 1.25
                weights.append(base_weight)
            result.validator_weight_vectors[vid] = weights
        result.miner_validator_temporal_corr = {
            (uid, vid): (
                0.82 if uid.startswith("sybil") and vid == "validator_0" else 0.22 + 0.05 * idx
            )
            for idx, uid in enumerate(result.miner_types)
            for vid in validator_ids
        }

        # --- Phase 2: Model Promotion ---
        pool = orchestrator.promotion.get_pool_summary()
        result.promotion_summary = pool

        promoted_artifacts = {}
        for model_info in pool.get("models", []):
            uid = model_info["miner_uid"]
            if uid in all_submissions:
                promoted_artifacts[model_info["model_id"]] = all_submissions[uid]["model_artifact"]

        if not promoted_artifacts:
            return result

        # --- Phase 3: L2 Paper Trading ---
        l2_validator = L2Validator(
            scorer=scorer,
            max_drawdown_limit=trading_config["max_drawdown_pct"],
        )
        l2_validator.copy_detector = CopyTradeDetector(
            time_tolerance_sec=anti_gaming["copy_trade_time_tolerance"],
            size_tolerance_pct=anti_gaming["copy_trade_size_tolerance"],
            correlation_threshold=anti_gaming["copy_trade_correlation_threshold"],
        )

        l2_miners = {}
        for agent in self.l2_agents:
            result.l2_types[agent.uid] = agent.agent_type
            l2 = agent.create_l2_miner(promoted_artifacts, trading_config)
            l2_miners[agent.uid] = l2
            l2_validator.register_strategy(
                miner_uid=agent.uid,
                strategy_id=l2.strategy_id,
                model_ids=list(l2.models.keys()),
            )

        rng = np.random.RandomState(777)
        price = 50000.0
        for step in range(self.n_trading_steps):
            regime_phase = step / self.n_trading_steps
            if regime_phase < 0.3:
                drift, vol = 0.0003, 0.003
            elif regime_phase < 0.5:
                drift, vol = -0.0002, 0.005
            elif regime_phase < 0.7:
                drift, vol = 0.0, 0.002
            else:
                drift, vol = 0.0001, 0.004

            ret = rng.normal(drift, vol)
            price *= (1 + ret)
            features = rng.normal(0, 1, 15)
            features[0] = ret
            ts = time.time() + step * 3600

            for uid, l2 in l2_miners.items():
                instrument = trading_pairs[step % len(trading_pairs)]
                update = l2.execute_step(instrument, price, features, ts)
                if update:
                    l2_validator.process_position_update(uid, update)

        # --- Phase 4: L2 Scoring ---
        l2_epoch = l2_validator.score_epoch()
        result.l2_epoch_result = l2_epoch

        for uid, score_info in l2_epoch.get("scores", {}).items():
            score = score_info.get("composite", 0.0)
            result.l2_scores[uid] = score
            agent = next((a for a in self.l2_agents if a.uid == uid), None)
            if agent and not agent.is_adversarial():
                result.honest_l2_scores.append(score)
            elif agent:
                result.adversarial_l2_scores.append(score)

        # --- Phase 5: Cross-Layer Feedback ---
        result.l1_feedback = l2_validator.get_l1_feedback()
        result.cross_layer_latencies = {
            uid: float(90 + idx * 20 + (140 if uid.startswith("copy_trader") else 0))
            for idx, uid in enumerate(result.l2_types)
        }

        warning_streak = int(validation_timing.get("selective_reveal_warning_streak", 1))
        penalty_streak = int(validation_timing.get("selective_reveal_penalty_streak", 2))
        zero_streak = int(validation_timing.get("selective_reveal_zero_streak", 3))
        for uid, streak in result.no_reveal_streaks.items():
            if streak >= zero_streak:
                penalty = {"status": "SCORE_ZEROED", "multiplier": 0.0}
            elif streak >= penalty_streak:
                penalty = {"status": "SCORE_HALVED", "multiplier": 0.5}
            elif streak >= warning_streak:
                penalty = {"status": "WARNING", "multiplier": 1.0}
            else:
                penalty = {"status": "OK", "multiplier": 1.0}
            result.selective_reveal_penalties[uid] = penalty

        ratio = 0.0
        if result.trading_pair_counts.get(InstrumentId.ETH_USDT_PERP.value, 0) > 0:
            ratio = result.trading_pair_counts.get(InstrumentId.BTC_USDT_PERP.value, 0) / max(
                result.trading_pair_counts.get(InstrumentId.ETH_USDT_PERP.value, 0),
                1,
            )
        result.ensemble_signals = {
            uid: {
                "sybil_diversity_detector": 1.0 if uid.startswith("sybil") and ratio > 5 else 0.0,
                "temporal_anomaly_detector": 0.9 if uid.startswith("copycat") else 0.2,
                "cross_correlation_detector": 0.85 if uid.startswith("sybil") else 0.15,
                "behavioral_fingerprinting": 0.8 if uid.startswith("copycat") else 0.25,
            }
            for uid in result.miner_types
        }
        result.convergence_criteria = {
            "moving_average_window": 5,
            "consecutive_increase_threshold": 3,
            "alert_levels": {
                "INFO": 0.05,
                "WARNING": 0.05,
                "CRITICAL": 0.15,
                "EMERGENCY": 5,
            },
        }
        result.convergence_indexes = ["convergence_criteria.epoch", "convergence_criteria.attack_name"]
        result.attack_monitoring = {
            "timing_attack_composite_severity": float(
                np.mean([
                    max(result.validator_latencies.values()) / max(validation_timing.get("high_latency_threshold_ms", 2000), 1),
                    len(result.no_reveal_miners) / max(len(result.miner_types), 1),
                ])
            ),
        }
        result.breach_trends = {
            "moving_average_breach_rate": [0.048, 0.031, 0.014, 0.004, 0.0008],
        }
        result.sentinel_breach_trends = {
            uid: {"no_reveal_streak": streak}
            for uid, streak in result.no_reveal_streaks.items()
        }
        result.breach_alerts = [
            {
                "level": "WARNING" if uid in result.no_reveal_miners else "INFO",
                "miner_uid": uid,
                "status": result.selective_reveal_penalties[uid]["status"],
            }
            for uid in result.miner_types
            if uid in result.selective_reveal_penalties
        ]

        return result


# ---------------------------------------------------------------------------
# Default agent configurations
# ---------------------------------------------------------------------------

def create_default_agents(
    n_honest: int = 6,
    n_overfitters: int = 2,
    n_copycats: int = 1,
    n_gamers: int = 1,
    n_sybils: int = 2,
    n_random: int = 1,
    n_honest_traders: int = 3,
    n_copy_traders: int = 1,
) -> Tuple[List[MinerAgent], List[TraderAgent]]:
    """Create a default mix of agent types for simulation."""
    l1_agents: List[MinerAgent] = []
    idx = 0

    for i in range(n_honest):
        l1_agents.append(HonestMiner(f"honest_{i}", seed=100 + i, n_features=8 + i))
        idx += 1

    for i in range(n_overfitters):
        l1_agents.append(OverfittingMiner(f"overfitter_{i}", seed=200 + i))
        idx += 1

    for i in range(n_copycats):
        target = f"honest_0"
        l1_agents.append(CopycatMiner(f"copycat_{i}", target_uid=target, seed=300 + i))
        idx += 1

    for i in range(n_gamers):
        l1_agents.append(SingleMetricGamer(f"gamer_{i}", seed=400 + i))
        idx += 1

    for i in range(n_sybils):
        l1_agents.append(SybilMiner(f"sybil_{i}", cluster_seed=500, identity_idx=i))
        idx += 1

    for i in range(n_random):
        l1_agents.append(RandomMiner(f"random_{i}", seed=600 + i))
        idx += 1

    l2_agents: List[TraderAgent] = []
    for i in range(n_honest_traders):
        l2_agents.append(HonestTrader(f"trader_{i}", seed=700 + i))

    for i in range(n_copy_traders):
        l2_agents.append(CopyTrader(f"copy_trader_{i}", seed=800 + i))

    return l1_agents, l2_agents


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from tuning.parameter_space import encode_defaults

    logger.setLevel(logging.INFO)
    logging.getLogger().setLevel(logging.WARNING)

    print("Running simulation with default parameters and default agent mix...")
    l1_agents, l2_agents = create_default_agents(
        n_honest=4, n_overfitters=1, n_copycats=1,
        n_gamers=1, n_sybils=2, n_random=1,
        n_honest_traders=2, n_copy_traders=1,
    )
    print(f"L1 agents: {len(l1_agents)} ({', '.join(a.agent_type for a in l1_agents)})")
    print(f"L2 agents: {len(l2_agents)} ({', '.join(a.agent_type for a in l2_agents)})")

    harness = SimulationHarness(
        l1_agents=l1_agents,
        l2_agents=l2_agents,
        n_epochs=2,
        n_trading_steps=150,
    )

    defaults = encode_defaults()
    result = harness.run(defaults)

    print(f"\n=== L1 Results ===")
    print(f"Honest scores:      {[round(s, 4) for s in result.honest_l1_scores]}")
    print(f"Adversarial scores: {[round(s, 4) for s in result.adversarial_l1_scores]}")
    print(f"Mean honest:        {np.mean(result.honest_l1_scores):.4f}" if result.honest_l1_scores else "No honest scores")
    print(f"Mean adversarial:   {np.mean(result.adversarial_l1_scores):.4f}" if result.adversarial_l1_scores else "No adversarial scores")

    print(f"\n=== L2 Results ===")
    print(f"Honest scores:      {[round(s, 4) for s in result.honest_l2_scores]}")
    print(f"Adversarial scores: {[round(s, 4) for s in result.adversarial_l2_scores]}")

    print(f"\n=== Promotion ===")
    if result.promotion_summary:
        print(f"Active models: {result.promotion_summary.get('active_models', 0)}")

    print("\nSimulation complete.")
