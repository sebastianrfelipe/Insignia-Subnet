"""
Sentinel symbol-diversity coordination for PC-VH-006.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from tuning.pc_vh_006_symbol_diversity import (
    SymbolDiversityConfig,
    SymbolDiversityEnforcer,
    SymbolDiversityReport,
)


@dataclass
class SentinelSymbolAssessment:
    report: SymbolDiversityReport
    alert_level: str
    recommended_action: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "report": self.report.to_dict(),
            "alert_level": self.alert_level,
            "recommended_action": self.recommended_action,
        }


class SentinelSymbolMonitor:
    """Turn PC-VH-006 measurements into sentinel-friendly assessments."""

    def __init__(self, config: SymbolDiversityConfig | None = None):
        self.enforcer = SymbolDiversityEnforcer(config)

    def assess(
        self,
        trading_pair_counts: Dict[str, int],
        consecutive_breach_generations: int = 0,
    ) -> SentinelSymbolAssessment:
        report = self.enforcer.evaluate(
            trading_pair_counts,
            consecutive_breach_generations=consecutive_breach_generations,
        )

        if report.penalty_state == "CRITICAL":
            alert_level = "CRITICAL"
            action = "Enforce PC-VH-006 penalties immediately and rebalance symbol exposure."
        elif report.penalty_state == "WARNING":
            alert_level = "WARNING"
            action = "Maintain PC-VH-006 monitoring and bias optimization toward pair diversification."
        elif report.penalty_state == "INFO":
            alert_level = "INFO"
            action = "Track symbol concentration through the grace period."
        else:
            alert_level = "INFO"
            action = "Symbol diversity is within target bounds."

        return SentinelSymbolAssessment(
            report=report,
            alert_level=alert_level,
            recommended_action=action,
        )
