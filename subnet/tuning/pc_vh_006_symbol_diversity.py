"""
PC-VH-006 Symbol Diversity Enforcement

Implements the symbol-diversity defense recommended by the third orchestration
run to reduce residual Sybil pressure driven by BTCUSDT:ETHUSDT concentration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class SymbolDiversityConfig:
    min_trading_pairs: int = 3
    max_symbol_dominance: float = 0.60
    warning_ratio: float = 1.35
    critical_ratio: float = 2.0
    penalty_base: float = 0.10
    penalty_escalation: float = 1.5
    penalty_max: float = 0.50
    grace_generations: int = 2


@dataclass
class SymbolDiversityReport:
    unique_pairs: int
    max_symbol_dominance: float
    btc_eth_ratio: float
    symbol_diversity_score: float
    severity: float
    penalty: float
    penalty_state: str
    consecutive_breach_generations: int
    triggered_rules: List[str] = field(default_factory=list)
    projected_sybil_reduction: float = 0.0

    def to_dict(self) -> Dict[str, float | int | str | List[str]]:
        return {
            "unique_pairs": self.unique_pairs,
            "max_symbol_dominance": round(self.max_symbol_dominance, 4),
            "btc_eth_ratio": round(self.btc_eth_ratio, 4),
            "symbol_diversity_score": round(self.symbol_diversity_score, 4),
            "severity": round(self.severity, 4),
            "penalty": round(self.penalty, 4),
            "penalty_state": self.penalty_state,
            "consecutive_breach_generations": self.consecutive_breach_generations,
            "triggered_rules": list(self.triggered_rules),
            "projected_sybil_reduction": round(self.projected_sybil_reduction, 4),
        }


class SymbolDiversityEnforcer:
    """Evaluate pair-distribution concentration and derive PC-VH-006 penalties."""

    def __init__(self, config: SymbolDiversityConfig | None = None):
        self.config = config or SymbolDiversityConfig()

    @staticmethod
    def _btc_eth_ratio(trading_pair_counts: Dict[str, int]) -> float:
        btc = float(
            trading_pair_counts.get("BTCUSDT", 0)
            + trading_pair_counts.get("BTC-USDT-PERP", 0)
        )
        eth = float(
            trading_pair_counts.get("ETHUSDT", 0)
            + trading_pair_counts.get("ETH-USDT-PERP", 0)
        )
        if eth <= 0:
            return float("inf") if btc > 0 else 0.0
        return btc / eth

    @staticmethod
    def _symbol_diversity_score(trading_pair_counts: Dict[str, int]) -> float:
        total = float(sum(trading_pair_counts.values()))
        if total <= 0 or len(trading_pair_counts) <= 1:
            return 0.0
        probs = [count / total for count in trading_pair_counts.values() if count > 0]
        if len(probs) <= 1:
            return 0.0
        entropy = -sum(p * math.log(p) for p in probs)
        return entropy / max(math.log(len(trading_pair_counts)), 1e-12)

    def evaluate(
        self,
        trading_pair_counts: Dict[str, int],
        consecutive_breach_generations: int = 0,
    ) -> SymbolDiversityReport:
        counts = {symbol: int(max(count, 0)) for symbol, count in trading_pair_counts.items()}
        total = float(sum(counts.values()))
        unique_pairs = sum(1 for count in counts.values() if count > 0)
        dominance = (max(counts.values()) / total) if total > 0 and counts else 0.0
        ratio = self._btc_eth_ratio(counts)
        score = self._symbol_diversity_score(counts)

        triggered: List[str] = []
        min_pair_deficit = max(0.0, (self.config.min_trading_pairs - unique_pairs) / max(self.config.min_trading_pairs, 1))
        if unique_pairs < self.config.min_trading_pairs:
            triggered.append("min_trading_pairs")

        dominance_excess = max(
            0.0,
            (dominance - self.config.max_symbol_dominance) / max(1.0 - self.config.max_symbol_dominance, 1e-12),
        )
        if dominance > self.config.max_symbol_dominance:
            triggered.append("max_symbol_dominance")

        ratio_severity = 0.0
        if ratio >= self.config.critical_ratio:
            ratio_severity = 1.0
            triggered.append("critical_ratio")
        elif ratio >= self.config.warning_ratio:
            ratio_severity = min(
                1.0,
                (ratio - self.config.warning_ratio)
                / max(self.config.critical_ratio - self.config.warning_ratio, 1e-12),
            )
            triggered.append("warning_ratio")

        severity = min(
            1.0,
            0.35 * min_pair_deficit + 0.35 * dominance_excess + 0.30 * ratio_severity,
        )

        effective_streak = max(0, consecutive_breach_generations - self.config.grace_generations)
        if not triggered:
            penalty = 0.0
            penalty_state = "OK"
        else:
            penalty = min(
                self.config.penalty_max,
                self.config.penalty_base
                * max(severity, 0.5)
                * (self.config.penalty_escalation ** effective_streak),
            )
            if ratio >= self.config.critical_ratio:
                penalty_state = "CRITICAL"
            elif ratio >= self.config.warning_ratio or dominance > self.config.max_symbol_dominance:
                penalty_state = "WARNING"
            else:
                penalty_state = "INFO"

        projected_sybil_reduction = min(
            0.70,
            max(0.0, 0.35 * score + 0.35 * (1.0 - penalty / max(self.config.penalty_max, 1e-12))),
        )

        return SymbolDiversityReport(
            unique_pairs=unique_pairs,
            max_symbol_dominance=dominance,
            btc_eth_ratio=ratio,
            symbol_diversity_score=score,
            severity=severity,
            penalty=penalty,
            penalty_state=penalty_state,
            consecutive_breach_generations=consecutive_breach_generations,
            triggered_rules=triggered,
            projected_sybil_reduction=projected_sybil_reduction,
        )
