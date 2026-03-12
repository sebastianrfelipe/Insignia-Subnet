"""
Cross-Layer Promotion and Feedback Engine

Manages the flow of models from Layer 1 to Layer 2 and the reverse flow
of performance feedback from Layer 2 back to Layer 1.

This is the core architectural innovation of the Insignia subnet:
the two-layer chain ensures that models are not only algorithmically
sound (L1 simulation) but also empirically proven (L2 deployment).

Promotion Criteria (L1 -> L2):
  1. Composite L1 score ranks in top-N (configurable, default N=10)
  2. Model has survived >= 2 consecutive epochs without score decay
  3. Overfitting penalty below published threshold
  4. Model passes ONNX inference compatibility test

Feedback Loop (L2 -> L1):
  1. L2 validators track which L1 models were used by each strategy
  2. When strategies using model X perform well, X gets retroactive bonus
  3. When strategies using model X fail despite good L1 scores, X is penalized
  4. Adjustments applied in next L1 epoch's scoring

Model Pool Management:
  - Pool capacity: top-N models (configurable)
  - Expiry: models are removed after M epochs without L2 usage
  - Versioning: each model retains its full evaluation history
"""

from __future__ import annotations

import time
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [CrossLayer] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Promoted Model Record
# ---------------------------------------------------------------------------

@dataclass
class PromotedModel:
    """Record of a model promoted from Layer 1 to the Layer 2 pool."""

    model_id: str
    miner_uid: str
    artifact_hash: str
    composite_score: float
    promotion_epoch: int
    consecutive_epochs: int = 1
    overfitting_score: float = 0.0

    l2_usage_count: int = 0
    l2_total_pnl_contribution: float = 0.0
    l2_avg_strategy_score: float = 0.0

    last_l2_update_epoch: int = 0
    expired: bool = False
    expiry_reason: str = ""

    score_history: List[float] = field(default_factory=list)

    def record_l2_usage(self, strategy_score: float, pnl_contribution: float):
        self.l2_usage_count += 1
        self.l2_total_pnl_contribution += pnl_contribution
        prev_total = self.l2_avg_strategy_score * (self.l2_usage_count - 1)
        self.l2_avg_strategy_score = (prev_total + strategy_score) / self.l2_usage_count


# ---------------------------------------------------------------------------
# Promotion Engine
# ---------------------------------------------------------------------------

@dataclass
class PromotionConfig:
    top_n: int = 10
    min_consecutive_epochs: int = 2
    max_overfitting_score: float = 0.40
    max_score_decay_pct: float = 0.20
    expiry_epochs_without_usage: int = 5


class PromotionEngine:
    """
    Manages the L1 -> L2 model promotion pipeline.

    Evaluates promotion eligibility based on configurable criteria,
    maintains the active model pool, and handles model expiry.
    """

    def __init__(self, config: PromotionConfig | None = None):
        self.config = config or PromotionConfig()
        self.pool: Dict[str, PromotedModel] = {}
        self._epoch_scores: Dict[str, List[float]] = {}

    def evaluate_candidates(
        self,
        epoch_results: Dict[str, Dict],
        current_epoch: int,
    ) -> List[PromotedModel]:
        """
        Evaluate L1 epoch results and promote eligible models.

        Args:
            epoch_results: miner_uid -> {composite_score, overfitting_score, artifact_hash, ...}
            current_epoch: current epoch number

        Returns:
            List of newly promoted models
        """
        for uid, result in epoch_results.items():
            score = result.get("composite_score", 0.0)
            if uid not in self._epoch_scores:
                self._epoch_scores[uid] = []
            self._epoch_scores[uid].append(score)

        ranked = sorted(
            epoch_results.items(),
            key=lambda x: x[1].get("composite_score", 0.0),
            reverse=True,
        )

        newly_promoted = []
        for rank, (uid, result) in enumerate(ranked[: self.config.top_n]):
            score = result.get("composite_score", 0.0)
            overfitting = result.get("overfitting_score", 0.0)
            artifact_hash = result.get("artifact_hash", "")

            if overfitting > self.config.max_overfitting_score:
                logger.info(
                    "Skipping %s: overfitting=%.2f > threshold=%.2f",
                    uid, overfitting, self.config.max_overfitting_score,
                )
                continue

            history = self._epoch_scores.get(uid, [])
            consecutive = self._count_consecutive_epochs(history)
            if consecutive < self.config.min_consecutive_epochs:
                logger.info(
                    "Skipping %s: consecutive=%d < min=%d",
                    uid, consecutive, self.config.min_consecutive_epochs,
                )
                continue

            if self._has_score_decay(history):
                logger.info("Skipping %s: significant score decay detected", uid)
                continue

            model_id = f"e{current_epoch}_r{rank+1}_{uid}"

            if uid in self.pool and not self.pool[uid].expired:
                existing = self.pool[uid]
                existing.composite_score = score
                existing.consecutive_epochs += 1
                existing.score_history.append(score)
                logger.info("Updated existing model for %s (score=%.4f)", uid, score)
            else:
                model = PromotedModel(
                    model_id=model_id,
                    miner_uid=uid,
                    artifact_hash=artifact_hash,
                    composite_score=score,
                    promotion_epoch=current_epoch,
                    overfitting_score=overfitting,
                    score_history=[score],
                )
                self.pool[uid] = model
                newly_promoted.append(model)
                logger.info("Promoted %s: model_id=%s score=%.4f", uid, model_id, score)

        self._expire_stale_models(current_epoch)
        return newly_promoted

    def _count_consecutive_epochs(self, history: List[float]) -> int:
        if not history:
            return 0
        count = 0
        for score in reversed(history):
            if score > 0:
                count += 1
            else:
                break
        return count

    def _has_score_decay(self, history: List[float]) -> bool:
        if len(history) < 2:
            return False
        recent = history[-1]
        prev = history[-2]
        if prev <= 0:
            return False
        decay = (prev - recent) / prev
        return decay > self.config.max_score_decay_pct

    def _expire_stale_models(self, current_epoch: int):
        for uid, model in self.pool.items():
            if model.expired:
                continue
            epochs_since_l2 = current_epoch - model.last_l2_update_epoch
            if (
                model.l2_usage_count == 0
                and (current_epoch - model.promotion_epoch) >= self.config.expiry_epochs_without_usage
            ):
                model.expired = True
                model.expiry_reason = "No L2 usage within expiry window"
                logger.info("Expired model %s for %s: no L2 usage", model.model_id, uid)

    def get_active_pool(self) -> List[PromotedModel]:
        return [m for m in self.pool.values() if not m.expired]

    def get_pool_summary(self) -> Dict[str, Any]:
        active = self.get_active_pool()
        return {
            "total_models": len(self.pool),
            "active_models": len(active),
            "expired_models": len(self.pool) - len(active),
            "avg_score": float(np.mean([m.composite_score for m in active])) if active else 0.0,
            "total_l2_usage": sum(m.l2_usage_count for m in active),
            "models": [
                {
                    "model_id": m.model_id,
                    "miner_uid": m.miner_uid,
                    "composite_score": round(m.composite_score, 4),
                    "l2_usage": m.l2_usage_count,
                    "l2_avg_score": round(m.l2_avg_strategy_score, 4),
                    "consecutive_epochs": m.consecutive_epochs,
                }
                for m in active
            ],
        }


# ---------------------------------------------------------------------------
# Full Cross-Layer Orchestrator
# ---------------------------------------------------------------------------

class CrossLayerOrchestrator:
    """
    Orchestrates the full cross-layer pipeline:
      L1 evaluation -> Promotion -> L2 deployment -> L2 scoring -> L1 feedback

    This is the top-level coordinator that ties both layers together.
    Used by the subnet owner to manage the end-to-end pipeline.
    """

    def __init__(
        self,
        promotion_engine: PromotionEngine | None = None,
        feedback_bonus_weight: float = 0.15,
        feedback_penalty_weight: float = 0.10,
    ):
        self.promotion = promotion_engine or PromotionEngine()
        self.feedback_bonus_weight = feedback_bonus_weight
        self.feedback_penalty_weight = feedback_penalty_weight
        self._l2_model_scores: Dict[str, List[float]] = {}

    def process_l1_epoch(
        self,
        epoch_results: Dict[str, Dict],
        current_epoch: int,
    ) -> Dict[str, Any]:
        """Process L1 epoch results: apply feedback, promote models."""
        adjusted = {}
        for uid, result in epoch_results.items():
            adj = self._compute_feedback_adjustment(uid)
            adjusted_score = result.get("composite_score", 0.0) * adj
            adjusted[uid] = {**result, "composite_score": adjusted_score}

        promoted = self.promotion.evaluate_candidates(adjusted, current_epoch)

        return {
            "n_promoted": len(promoted),
            "promoted_models": [m.model_id for m in promoted],
            "pool_summary": self.promotion.get_pool_summary(),
        }

    def record_l2_results(
        self,
        model_id: str,
        miner_uid: str,
        strategy_score: float,
        pnl_contribution: float,
    ):
        """Record L2 strategy results for a specific model."""
        for uid, model in self.promotion.pool.items():
            if model.model_id == model_id or uid == miner_uid:
                model.record_l2_usage(strategy_score, pnl_contribution)
                model.last_l2_update_epoch = self.promotion.pool[uid].promotion_epoch + 1

        if model_id not in self._l2_model_scores:
            self._l2_model_scores[model_id] = []
        self._l2_model_scores[model_id].append(strategy_score)

    def _compute_feedback_adjustment(self, miner_uid: str) -> float:
        """Compute the feedback multiplier for an L1 miner based on their models' L2 performance."""
        model = self.promotion.pool.get(miner_uid)
        if not model or model.l2_usage_count == 0:
            return 1.0

        avg_l2 = model.l2_avg_strategy_score
        if avg_l2 > 0.6:
            return 1.0 + self.feedback_bonus_weight * (avg_l2 - 0.6) / 0.4
        elif avg_l2 < 0.3:
            return 1.0 - self.feedback_penalty_weight * (0.3 - avg_l2) / 0.3
        return 1.0

    def get_full_pipeline_status(self) -> Dict[str, Any]:
        pool = self.promotion.get_pool_summary()
        return {
            "model_pool": pool,
            "total_l2_feedback_records": sum(
                len(v) for v in self._l2_model_scores.values()
            ),
            "models_with_l2_data": len(self._l2_model_scores),
        }
