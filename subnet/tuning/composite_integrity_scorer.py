"""
Composite integrity scoring utilities.

This module packages the report's EXP-023 composite integrity score into a
small reusable helper that can be consumed by the validator, sentinel, or
autoresearch tooling without pulling in the full scoring stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Iterable, Mapping, Sequence


@dataclass
class CompositeIntegrityConfig:
    honest_weight: float = 0.6
    complexity_bonus: float = 0.3
    consistency_weight: float = 0.1
    lookback_window: int = 10


class CompositeIntegrityScorer:
    """
    EXP-023 composite integrity score proposed during orchestration.

    The score blends breach minimization, detector responsiveness, and
    temporal stability into a normalized [0, 1] value.
    """

    def __init__(self, config: CompositeIntegrityConfig | None = None):
        self.config = config or CompositeIntegrityConfig()

    def normalize_breach_rate(
        self,
        breach_rate: float,
        max_breach_rate_observed: float,
    ) -> float:
        if max_breach_rate_observed <= 0:
            return 0.0
        normalized = breach_rate / max_breach_rate_observed
        return max(0.0, min(1.0, normalized))

    def calculate_temporal_stability_score(
        self,
        performance_history: Sequence[float] | Iterable[float],
    ) -> float:
        history = list(performance_history)[-self.config.lookback_window :]
        if len(history) < 2:
            return 1.0

        perf_mean = mean(history)
        if abs(perf_mean) < 1e-12:
            return 0.0

        perf_std = pstdev(history)
        stability = 1.0 - (perf_std / abs(perf_mean))
        return max(0.0, min(1.0, stability))

    def calculate_composite_integrity_score(
        self,
        miner_data: Mapping[str, float | Sequence[float]],
    ) -> float:
        breach_rate = float(miner_data.get("breach_rate", 0.0))
        max_breach_rate_observed = float(
            miner_data.get("max_breach_rate_observed", max(breach_rate, 1.0))
        )
        detection_delay = float(miner_data.get("detection_delay_normalized", 0.0))
        if detection_delay < 0:
            detection_delay = 0.0

        temporal_history = miner_data.get("performance_history", [])
        temporal_stability = self.calculate_temporal_stability_score(
            temporal_history if isinstance(temporal_history, Sequence) else []
        )
        normalized_breach_rate = self.normalize_breach_rate(
            breach_rate,
            max_breach_rate_observed,
        )

        composite = (
            self.config.honest_weight * (1.0 - normalized_breach_rate)
            + self.config.complexity_bonus * (1.0 / (1.0 + detection_delay))
            + self.config.consistency_weight * temporal_stability
        )
        return max(0.0, min(1.0, composite))
