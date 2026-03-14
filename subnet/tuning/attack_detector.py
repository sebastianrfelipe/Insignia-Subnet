"""
Attack Vector Detection System

Automatically detects whether each of the 9 documented attack vectors
has been breached under a given parameter configuration.

For each attack, defines:
  - A breach condition (boolean)
  - A severity score (0.0 = fully defended, 1.0 = fully exploited)
  - A human-readable description of the breach

The detector takes SimulationResult as input and produces a structured
BreachReport used by the optimizer as a fitness dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tuning.simulation import SimulationResult


@dataclass
class AttackBreach:
    attack_name: str
    breached: bool
    severity: float  # 0.0 = defended, 1.0 = fully exploited
    description: str
    details: Dict = field(default_factory=dict)


@dataclass
class BreachReport:
    breaches: List[AttackBreach] = field(default_factory=list)

    @property
    def n_breached(self) -> int:
        return sum(1 for b in self.breaches if b.breached)

    @property
    def total_attacks(self) -> int:
        return len(self.breaches)

    @property
    def breach_rate(self) -> float:
        if not self.breaches:
            return 0.0
        return self.n_breached / self.total_attacks

    @property
    def mean_severity(self) -> float:
        if not self.breaches:
            return 0.0
        return float(np.mean([b.severity for b in self.breaches]))

    @property
    def max_severity(self) -> float:
        if not self.breaches:
            return 0.0
        return float(max(b.severity for b in self.breaches))

    def to_dict(self) -> Dict:
        return {
            "n_breached": int(self.n_breached),
            "total_attacks": int(self.total_attacks),
            "breach_rate": round(float(self.breach_rate), 4),
            "mean_severity": round(float(self.mean_severity), 4),
            "max_severity": round(float(self.max_severity), 4),
            "breaches": [
                {
                    "attack": b.attack_name,
                    "breached": bool(b.breached),
                    "severity": round(float(b.severity), 4),
                    "description": b.description,
                }
                for b in self.breaches
            ],
        }

    def summary(self) -> str:
        lines = [f"Breach Report: {self.n_breached}/{self.total_attacks} attacks breached"]
        for b in self.breaches:
            status = "BREACHED" if b.breached else "DEFENDED"
            lines.append(f"  [{status}] {b.attack_name}: severity={b.severity:.2f} — {b.description}")
        return "\n".join(lines)


class AttackDetector:
    """
    Evaluates a SimulationResult against all known attack vectors
    and produces a BreachReport.
    """

    def evaluate(self, result: SimulationResult) -> BreachReport:
        report = BreachReport()
        report.breaches.append(self._check_overfitting(result))
        report.breaches.append(self._check_plagiarism(result))
        report.breaches.append(self._check_single_metric_gaming(result))
        report.breaches.append(self._check_sybil(result))
        report.breaches.append(self._check_copy_trading(result))
        report.breaches.append(self._check_random_baseline(result))
        report.breaches.append(self._check_adversarial_dominance(result))
        report.breaches.append(self._check_honest_separation(result))
        report.breaches.append(self._check_score_concentration(result))
        return report

    def _check_overfitting(self, result: SimulationResult) -> AttackBreach:
        """Check if overfitting miners score higher than honest miners."""
        overfitter_scores = [
            result.miner_scores[uid]
            for uid, t in result.miner_types.items()
            if t == "overfitter" and uid in result.miner_scores
        ]
        honest_scores = result.honest_l1_scores

        if not overfitter_scores or not honest_scores:
            return AttackBreach(
                "overfitting_exploitation", False, 0.0,
                "No overfitting miners or no honest miners in simulation",
            )

        mean_overfitter = np.mean(overfitter_scores)
        mean_honest = np.mean(honest_scores)
        max_honest = max(honest_scores) if honest_scores else 0

        breached = mean_overfitter > mean_honest
        if mean_honest > 0:
            severity = max(0.0, min(1.0, (mean_overfitter - mean_honest) / mean_honest))
        else:
            severity = 1.0 if mean_overfitter > 0 else 0.0

        return AttackBreach(
            "overfitting_exploitation", breached, severity,
            f"Overfitters avg={mean_overfitter:.4f} vs honest avg={mean_honest:.4f}",
            {"overfitter_scores": overfitter_scores, "honest_mean": mean_honest},
        )

    def _check_plagiarism(self, result: SimulationResult) -> AttackBreach:
        """Check if copycat miners evaded detection and scored independently."""
        copycat_scores = [
            result.miner_scores[uid]
            for uid, t in result.miner_types.items()
            if t == "copycat" and uid in result.miner_scores
        ]

        if not copycat_scores:
            return AttackBreach(
                "model_plagiarism", False, 0.0,
                "No copycat miners in simulation",
            )

        honest_scores = result.honest_l1_scores
        mean_honest = np.mean(honest_scores) if honest_scores else 0

        # A copycat succeeds if they score comparably to honest miners
        # (meaning fingerprinting didn't catch them or didn't penalize them)
        high_scoring = [s for s in copycat_scores if s > mean_honest * 0.8]
        breached = len(high_scoring) > 0
        severity = len(high_scoring) / max(len(copycat_scores), 1)

        return AttackBreach(
            "model_plagiarism", breached, severity,
            f"{len(high_scoring)}/{len(copycat_scores)} copycats scored above 80% of honest mean",
        )

    def _check_single_metric_gaming(self, result: SimulationResult) -> AttackBreach:
        """Check if single-metric gamers rank in the top half."""
        gamer_scores = [
            result.miner_scores[uid]
            for uid, t in result.miner_types.items()
            if t == "single_metric_gamer" and uid in result.miner_scores
        ]

        if not gamer_scores:
            return AttackBreach(
                "single_metric_gaming", False, 0.0,
                "No single-metric gamers in simulation",
            )

        all_scores = sorted(result.miner_scores.values(), reverse=True)
        median_score = np.median(all_scores) if all_scores else 0

        high_ranking = [s for s in gamer_scores if s > median_score]
        breached = len(high_ranking) > 0
        severity = len(high_ranking) / max(len(gamer_scores), 1)

        return AttackBreach(
            "single_metric_gaming", breached, severity,
            f"{len(high_ranking)}/{len(gamer_scores)} gamers ranked above median",
        )

    def _check_sybil(self, result: SimulationResult) -> AttackBreach:
        """Check if sybil identities collectively earn disproportionate share."""
        sybil_scores = [
            result.miner_scores[uid]
            for uid, t in result.miner_types.items()
            if t == "sybil" and uid in result.miner_scores
        ]
        honest_scores = result.honest_l1_scores

        if not sybil_scores or not honest_scores:
            return AttackBreach(
                "sybil_attack", False, 0.0,
                "No sybil miners or no honest miners in simulation",
            )

        sybil_total = sum(sybil_scores)
        honest_per_miner = np.mean(honest_scores)

        # Sybils succeed if their collective share exceeds 2x a single honest miner
        n_sybils = len(sybil_scores)
        fair_share = honest_per_miner * n_sybils
        breached = sybil_total > fair_share * 1.5
        if fair_share > 0:
            severity = max(0.0, min(1.0, (sybil_total - fair_share) / fair_share))
        else:
            severity = 1.0 if sybil_total > 0 else 0.0

        return AttackBreach(
            "sybil_attack", breached, severity,
            f"Sybil cluster total={sybil_total:.4f} vs fair_share={fair_share:.4f}",
        )

    def _check_copy_trading(self, result: SimulationResult) -> AttackBreach:
        """Check if copy-traders evaded L2 detection."""
        copy_trader_scores = [
            result.l2_scores[uid]
            for uid, t in result.l2_types.items()
            if t == "copy_trader" and uid in result.l2_scores
        ]
        honest_l2 = result.honest_l2_scores

        if not copy_trader_scores:
            return AttackBreach(
                "copy_trading", False, 0.0,
                "No copy-traders in L2 simulation",
            )

        mean_honest = np.mean(honest_l2) if honest_l2 else 0
        high_scoring = [s for s in copy_trader_scores if s > mean_honest * 0.7]
        breached = len(high_scoring) > 0
        severity = len(high_scoring) / max(len(copy_trader_scores), 1)

        return AttackBreach(
            "copy_trading", breached, severity,
            f"{len(high_scoring)}/{len(copy_trader_scores)} copy-traders scored above 70% of honest L2 mean",
        )

    def _check_random_baseline(self, result: SimulationResult) -> AttackBreach:
        """
        Check that random miners don't score well — if they do,
        the scoring mechanism isn't discriminating enough.
        """
        random_scores = [
            result.miner_scores[uid]
            for uid, t in result.miner_types.items()
            if t == "random" and uid in result.miner_scores
        ]
        honest_scores = result.honest_l1_scores

        if not random_scores or not honest_scores:
            return AttackBreach(
                "random_baseline_discrimination", False, 0.0,
                "No random miners or no honest miners",
            )

        mean_random = np.mean(random_scores)
        mean_honest = np.mean(honest_scores)

        # Breach if random miners get >60% of honest score (not enough discrimination)
        breached = mean_random > mean_honest * 0.6
        if mean_honest > 0:
            severity = max(0.0, min(1.0, mean_random / mean_honest))
        else:
            severity = 0.0

        return AttackBreach(
            "random_baseline_discrimination", breached, severity,
            f"Random avg={mean_random:.4f} vs honest avg={mean_honest:.4f} (ratio={mean_random/max(mean_honest,1e-12):.2f})",
        )

    def _check_adversarial_dominance(self, result: SimulationResult) -> AttackBreach:
        """Check if any adversarial miner is the top scorer overall."""
        if not result.miner_scores:
            return AttackBreach(
                "adversarial_dominance", False, 0.0,
                "No miner scores available",
            )

        top_uid = max(result.miner_scores, key=result.miner_scores.get)
        top_type = result.miner_types.get(top_uid, "unknown")
        is_adversarial = top_type in ("overfitter", "copycat", "single_metric_gamer", "sybil")

        adversarial_in_top3 = 0
        ranked = sorted(result.miner_scores.items(), key=lambda x: x[1], reverse=True)
        for uid, _ in ranked[:3]:
            t = result.miner_types.get(uid, "unknown")
            if t in ("overfitter", "copycat", "single_metric_gamer", "sybil"):
                adversarial_in_top3 += 1

        severity = adversarial_in_top3 / 3.0

        return AttackBreach(
            "adversarial_dominance", is_adversarial, severity,
            f"Top miner: {top_uid} ({top_type}), adversarial in top 3: {adversarial_in_top3}/3",
        )

    def _check_honest_separation(self, result: SimulationResult) -> AttackBreach:
        """
        Check that there's meaningful score separation between honest
        and adversarial miners. If they're clustered together, the
        mechanism isn't providing enough differentiation signal.
        """
        honest = result.honest_l1_scores
        adversarial = result.adversarial_l1_scores

        if not honest or not adversarial:
            return AttackBreach(
                "insufficient_separation", False, 0.0,
                "Need both honest and adversarial scores for separation check",
            )

        mean_honest = np.mean(honest)
        mean_adversarial = np.mean(adversarial)
        gap = mean_honest - mean_adversarial

        # Breach if adversarial miners are within 10% of honest mean
        if mean_honest > 0:
            relative_gap = gap / mean_honest
        else:
            relative_gap = 0.0

        breached = relative_gap < 0.10
        severity = max(0.0, min(1.0, 1.0 - relative_gap * 5))

        return AttackBreach(
            "insufficient_separation", breached, severity,
            f"Gap: honest={mean_honest:.4f} - adversarial={mean_adversarial:.4f} = {gap:.4f} ({relative_gap:.1%})",
        )

    def _check_score_concentration(self, result: SimulationResult) -> AttackBreach:
        """
        Check that emission weights aren't overly concentrated.
        Extreme concentration means one miner captures most value.
        """
        scores = list(result.miner_scores.values())
        if len(scores) < 2:
            return AttackBreach(
                "score_concentration", False, 0.0,
                "Not enough miners for concentration analysis",
            )

        scores_arr = np.array(scores)
        total = scores_arr.sum()
        if total < 1e-12:
            return AttackBreach(
                "score_concentration", False, 0.0,
                "All scores near zero",
            )

        shares = scores_arr / total
        hhi = float(np.sum(shares ** 2))

        # HHI > 0.25 indicates high concentration (1/n for uniform)
        fair_hhi = 1.0 / len(scores)
        breached = hhi > max(0.25, fair_hhi * 3)
        severity = max(0.0, min(1.0, (hhi - fair_hhi) / (1.0 - fair_hhi)))

        return AttackBreach(
            "score_concentration", breached, severity,
            f"HHI={hhi:.4f} (fair={fair_hhi:.4f}, threshold={max(0.25, fair_hhi*3):.4f})",
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from tuning.parameter_space import encode_defaults
    from tuning.simulation import SimulationHarness, create_default_agents

    print("Running attack detection with default parameters...\n")

    l1_agents, l2_agents = create_default_agents(
        n_honest=4, n_overfitters=1, n_copycats=1,
        n_gamers=1, n_sybils=2, n_random=1,
        n_honest_traders=2, n_copy_traders=1,
    )

    harness = SimulationHarness(
        l1_agents=l1_agents, l2_agents=l2_agents,
        n_epochs=2, n_trading_steps=150,
    )

    defaults = encode_defaults()
    sim_result = harness.run(defaults)

    detector = AttackDetector()
    report = detector.evaluate(sim_result)
    print(report.summary())
    print(f"\nOverall: {report.n_breached}/{report.total_attacks} breached, "
          f"mean severity={report.mean_severity:.4f}")
