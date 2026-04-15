"""
Attack Vector Detection System

Evaluates whether each of the 19 documented attack vectors has been
breached under a given parameter configuration.

Original 9 vectors:
  1. Overfitting exploitation
  2. Model plagiarism
  3. Single-metric gaming
  4. Sybil attack
  5. Copy-trading
  6. Random baseline discrimination failure
  7. Adversarial dominance
  8. Insufficient honest/adversarial separation
  9. Score concentration (HHI)

Novel vectors discovered during orchestration runs (2026-03-29):
  10. Validator latency exploitation
  11. Prediction timing manipulation
  12. Miner-validator collusion
  13. Weight entropy violation
  14. Cross-validator score variance
  15. Validator rotation circumvention
  16. Validator agreement anomaly
  17. Collusion temporal pattern
  18. Weight manipulation (L1/L2 skew)
  19. Cross-layer attack (timing sync)

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

    @property
    def risk_tier(self) -> str:
        if self.severity >= 0.5:
            return "high"
        if self.severity >= 0.3:
            return "moderate"
        return "low"


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
                    "risk_tier": b.risk_tier,
                    "description": b.description,
                }
                for b in self.breaches
            ],
        }

    def summary(self) -> str:
        lines = [f"Breach Report: {self.n_breached}/{self.total_attacks} attacks breached"]
        for b in self.breaches:
            status = "BREACHED" if b.breached else "DEFENDED"
            lines.append(
                f"  [{status}] {b.attack_name}: severity={b.severity:.2f} "
                f"tier={b.risk_tier} — {b.description}"
            )
        return "\n".join(lines)


class AttackDetector:
    """
    Evaluates a SimulationResult against all known attack vectors
    and produces a BreachReport.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def _validation_config(self) -> Dict[str, float]:
        return self.config.get("validation_timing", {})

    def _consensus_config(self) -> Dict[str, float]:
        return self.config.get("consensus_integrity", {})

    def _market_config(self) -> Dict[str, float]:
        return self.config.get("market_data", {})

    def _ensemble_config(self) -> Dict[str, float]:
        return self.config.get("ensemble_detection", {})

    def _economic_config(self) -> Dict[str, float]:
        return self.config.get("economic_mechanisms", {})

    def _pair_imbalance_ratio(self, result: SimulationResult) -> float:
        counts = getattr(result, "trading_pair_counts", {})
        btc = float(counts.get("BTC-USDT-PERP", counts.get("BTCUSDT", 0.0)))
        eth = float(counts.get("ETH-USDT-PERP", counts.get("ETHUSDT", 0.0)))
        if eth <= 0:
            return 0.0
        return btc / eth

    def _symbol_diversity_score(self, result: SimulationResult) -> float:
        monitoring = getattr(result, "attack_monitoring", {})
        if "symbol_diversity_score" in monitoring:
            return float(monitoring["symbol_diversity_score"])

        counts = np.array(list(getattr(result, "trading_pair_counts", {}).values()), dtype=float)
        total = float(np.sum(counts))
        if total <= 0 or len(counts) <= 1:
            return 0.0
        probs = counts[counts > 0] / total
        if len(probs) <= 1:
            return 0.0
        entropy = -float(np.sum(probs * np.log(probs)))
        return entropy / max(float(np.log(len(counts))), 1e-12)

    def _ensemble_vote_count(self, signals: Dict[str, float]) -> int:
        return sum(1 for score in signals.values() if score >= 0.7)

    def _compute_commitment_violation_scores(self, result: SimulationResult) -> Dict[str, Dict[str, float | bool]]:
        cfg = self._validation_config()
        commit_rate_threshold = float(cfg.get("commit_rate_threshold", 0.7))
        violation_weight = float(cfg.get("commitment_violation_weight", 0.008))

        scores: Dict[str, Dict[str, float | bool]] = {}
        for uid in result.miner_types:
            commit_rate = float(result.miner_commit_rates.get(uid, 0.0))
            accuracy = result.miner_accuracy_by_commit_status.get(uid, {})
            accuracy_when_committed = float(accuracy.get("accuracy_when_committed", 0.0))
            accuracy_when_not_committed = float(accuracy.get("accuracy_when_not_committed", 0.0))
            disparity = max(0.0, accuracy_when_not_committed - accuracy_when_committed)
            flagged = (
                accuracy_when_not_committed > accuracy_when_committed
                and commit_rate < commit_rate_threshold
            )
            weighted_score = min(1.0, disparity / max(violation_weight, 1e-12)) if flagged else 0.0
            scores[uid] = {
                "commit_rate": commit_rate,
                "accuracy_when_committed": accuracy_when_committed,
                "accuracy_when_not_committed": accuracy_when_not_committed,
                "accuracy_disparity": disparity,
                "flagged": flagged,
                "weighted_score": weighted_score,
            }
        return scores

    def _check_selective_revelation(self, result: SimulationResult) -> AttackBreach:
        cfg = self._validation_config()
        warning_streak = int(cfg.get("selective_reveal_warning_streak", 1))
        penalty_streak = int(cfg.get("selective_reveal_penalty_streak", 2))
        zero_streak = int(cfg.get("selective_reveal_zero_streak", 3))

        streaks = getattr(result, "no_reveal_streaks", {})
        if not streaks:
            return AttackBreach(
                "selective_revelation",
                False,
                0.0,
                "No selective revelation telemetry available",
            )

        warned = []
        halved = []
        zeroed = []
        for uid, streak in streaks.items():
            if streak >= zero_streak:
                zeroed.append(uid)
            elif streak >= penalty_streak:
                halved.append(uid)
            elif streak >= warning_streak:
                warned.append(uid)

        severity = min(
            1.0,
            (
                len(warned) * 0.2
                + len(halved) * 0.5
                + len(zeroed) * 1.0
            ) / max(len(streaks), 1),
        )
        breached = bool(halved or zeroed)
        return AttackBreach(
            "selective_revelation",
            breached,
            severity,
            f"warned={len(warned)} halved={len(halved)} zeroed={len(zeroed)}",
            {
                "warning_streak": warning_streak,
                "penalty_streak": penalty_streak,
                "zero_streak": zero_streak,
                "warned": warned,
                "halved": halved,
                "zeroed": zeroed,
            },
        )

    def _check_statistical_anomaly(self, result: SimulationResult) -> AttackBreach:
        scores = np.array(list(result.miner_scores.values()), dtype=float)
        if len(scores) < 4:
            return AttackBreach(
                "statistical_anomaly",
                False,
                0.0,
                "Not enough miner scores for anomaly detection",
            )

        mean = float(np.mean(scores))
        std = float(np.std(scores))
        q1 = float(np.percentile(scores, 25))
        q3 = float(np.percentile(scores, 75))
        iqr = q3 - q1
        z_flags = []
        iqr_flags = []
        for uid, score in result.miner_scores.items():
            z = abs(score - mean) / max(std, 1e-12)
            if z > 2.5:
                z_flags.append(uid)
            if score > q3 + 1.5 * iqr or score < q1 - 1.5 * iqr:
                iqr_flags.append(uid)

        flagged = sorted(set(z_flags + iqr_flags))
        severity = min(1.0, len(flagged) / max(len(scores), 1))
        return AttackBreach(
            "statistical_anomaly",
            bool(flagged),
            severity,
            f"flagged={len(flagged)} zscore_outliers={len(z_flags)} iqr_outliers={len(iqr_flags)}",
            {"flagged_miners": flagged},
        )

    def _check_behavioral_anomaly(self, result: SimulationResult) -> AttackBreach:
        signals = getattr(result, "ensemble_signals", {})
        if not signals:
            return AttackBreach(
                "behavioral_anomaly",
                False,
                0.0,
                "No behavioral fingerprints available",
            )

        flagged = [
            uid
            for uid, miner_signals in signals.items()
            if miner_signals.get("behavioral_fingerprinting", 0.0) >= 0.7
        ]
        severity = min(1.0, len(flagged) / max(len(signals), 1))
        return AttackBreach(
            "behavioral_anomaly",
            bool(flagged),
            severity,
            f"behavioral anomalies on {len(flagged)} miners",
            {"flagged_miners": flagged},
        )

    def _check_temporal_attack_pattern(self, result: SimulationResult) -> AttackBreach:
        signals = getattr(result, "ensemble_signals", {})
        if not signals:
            return AttackBreach(
                "temporal_attack_pattern",
                False,
                0.0,
                "No temporal anomaly signals available",
            )

        temporal_scores = {
            uid: float(miner_signals.get("temporal_anomaly_detector", 0.0))
            for uid, miner_signals in signals.items()
        }
        flagged = [uid for uid, score in temporal_scores.items() if score >= 0.7]
        lead_time = float(self._validation_config().get("min_prediction_lead_time", 35))
        timing_gaps = getattr(result, "submission_timing_gaps", {})
        timing_violation_rate = (
            float(np.mean([gap < lead_time for gap in timing_gaps.values()]))
            if timing_gaps
            else 0.0
        )
        reveal_gaps = []
        for key, commit_ts in getattr(result, "commit_timestamps", {}).items():
            reveal_ts = getattr(result, "reveal_timestamps", {}).get(key)
            if reveal_ts is not None:
                reveal_gaps.append(reveal_ts - commit_ts)
        average_reveal_delay = float(np.mean(reveal_gaps)) if reveal_gaps else 0.0
        bayesian_weight = float(self._ensemble_config().get("bayesian_weight", 0.65))
        ensemble_component = float(np.mean(list(temporal_scores.values()))) if temporal_scores else 0.0
        peak_component = max(temporal_scores.values(), default=0.0)
        flagged_fraction = len(flagged) / max(len(signals), 1)
        cr_effectiveness = float(
            getattr(result, "attack_monitoring", {}).get("commit_reveal_effectiveness", 0.70)
        )
        severity = min(
            1.0,
            (
                0.40 * peak_component
                + 0.20 * ensemble_component
                + 0.10 * flagged_fraction
                + 0.15 * timing_violation_rate
                + 0.05 * min(1.0, average_reveal_delay / 20.0)
            )
            * (1.0 - 0.20 * bayesian_weight)
            * (1.0 - 0.70 * cr_effectiveness),
        )
        return AttackBreach(
            "temporal_attack_pattern",
            severity >= 0.5,
            severity,
            f"flagged={len(flagged)} timing_violation_rate={timing_violation_rate:.2f} "
            f"reveal_delay={average_reveal_delay:.1f}s cr_effectiveness={cr_effectiveness:.3f}",
            {
                "flagged_miners": flagged,
                "timing_violation_rate": timing_violation_rate,
                "average_reveal_delay": average_reveal_delay,
                "lead_time_threshold": lead_time,
                "bayesian_weight": bayesian_weight,
                "commit_reveal_effectiveness": cr_effectiveness,
            },
        )

    def _check_sybil_collusion_graph(self, result: SimulationResult) -> AttackBreach:
        signals = getattr(result, "ensemble_signals", {})
        if not signals:
            return AttackBreach(
                "sybil_collusion_graph",
                False,
                0.0,
                "No graph detector signals available",
            )

        correlation_threshold = float(self._ensemble_config().get("correlation_threshold", 0.77))
        cluster = [
            uid
            for uid, miner_signals in signals.items()
            if miner_signals.get("cross_correlation_detector", 0.0) >= correlation_threshold
        ]
        cluster_fraction = len(cluster) / max(len(signals), 1)
        pair_ratio = self._pair_imbalance_ratio(result)
        dominant_pair_warning_ratio = float(
            self._market_config().get("dominant_pair_warning_ratio", 1.35)
        )
        symbol_diversity_score = self._symbol_diversity_score(result)
        symbol_diversity_threshold = float(
            self._ensemble_config().get("symbol_diversity_threshold", 0.33)
        )
        pair_pressure = min(
            1.0,
            max(
                0.0,
                (pair_ratio - 1.0) / max(dominant_pair_warning_ratio - 1.0, 1e-12),
            ),
        )
        diversity_deficit = max(
            0.0,
            symbol_diversity_threshold - symbol_diversity_score,
        ) / max(symbol_diversity_threshold, 1e-12)
        min_entropy = float(self._consensus_config().get("weight_entropy_minimum", 1.45))
        entropy_deficit = 0.0
        for weights in getattr(result, "validator_weight_vectors", {}).values():
            w = np.array(weights, dtype=float)
            total = max(float(np.sum(w)), 1e-12)
            w = (w / total)[w > 1e-12]
            entropy = -float(np.sum(w * np.log(w))) if len(w) else 0.0
            entropy_deficit = max(entropy_deficit, max(0.0, min_entropy - entropy) / max(min_entropy, 1e-12))
        temporal_alignment = 0.0
        for (miner_uid, _), corr in getattr(result, "miner_validator_temporal_corr", {}).items():
            if miner_uid in cluster:
                temporal_alignment = max(temporal_alignment, abs(float(corr)))
        economic_cfg = self._economic_config()
        structural_mitigation = min(
            0.35,
            0.15 * float(economic_cfg.get("identity_bond_threshold", 0.72))
            + 0.10 * float(economic_cfg.get("stake_weight_consensus", 0.38)),
        )
        severity = min(
            1.0,
            (
                0.45 * cluster_fraction
                + 0.20 * pair_pressure
                + 0.15 * min(1.0, entropy_deficit)
                + 0.10 * temporal_alignment
                + 0.10 * min(1.0, diversity_deficit)
            )
            * (1.0 - structural_mitigation)
        )
        return AttackBreach(
            "sybil_collusion_graph",
            severity >= 0.5,
            severity,
            f"cluster={len(cluster)} pair_ratio={pair_ratio:.2f} "
            f"diversity={symbol_diversity_score:.2f} entropy_deficit={entropy_deficit:.2f}",
            {
                "cluster_members": cluster,
                "pair_ratio": pair_ratio,
                "dominant_pair_warning_ratio": dominant_pair_warning_ratio,
                "symbol_diversity_score": symbol_diversity_score,
                "symbol_diversity_threshold": symbol_diversity_threshold,
                "diversity_deficit": diversity_deficit,
                "entropy_deficit": entropy_deficit,
                "temporal_alignment": temporal_alignment,
                "structural_mitigation": structural_mitigation,
            },
        )

    def _check_cross_layer_correlation(self, result: SimulationResult) -> AttackBreach:
        signals = getattr(result, "ensemble_signals", {})
        if not signals or not result.cross_layer_latencies:
            return AttackBreach(
                "cross_layer_correlation",
                False,
                0.0,
                "No cross-layer correlation telemetry available",
            )

        average_vote = 0.0
        vote_details = {}
        for uid, miner_signals in signals.items():
            vote_count = self._ensemble_vote_count(miner_signals)
            normalized_vote = vote_count / max(len(miner_signals), 1)
            vote_details[uid] = normalized_vote
            average_vote = max(average_vote, normalized_vote)

        latency_violation_rate = float(
            np.mean(
                np.array(list(result.cross_layer_latencies.values()), dtype=float)
                > float(self.config.get("cross_layer_timing", {}).get("max_latency_ms", 200))
            )
        )
        severity = min(1.0, 0.6 * average_vote + 0.4 * latency_violation_rate)
        breached = severity >= 0.5
        return AttackBreach(
            "cross_layer_correlation",
            breached,
            severity,
            f"max ensemble vote={average_vote:.2f} latency_violation_rate={latency_violation_rate:.2f}",
            {"votes": vote_details, "latency_violation_rate": latency_violation_rate},
        )

    def evaluate(self, result: SimulationResult) -> BreachReport:
        report = BreachReport()
        # Original 9 vectors
        report.breaches.append(self._check_overfitting(result))
        report.breaches.append(self._check_plagiarism(result))
        report.breaches.append(self._check_single_metric_gaming(result))
        report.breaches.append(self._check_sybil(result))
        report.breaches.append(self._check_copy_trading(result))
        report.breaches.append(self._check_random_baseline(result))
        report.breaches.append(self._check_adversarial_dominance(result))
        report.breaches.append(self._check_honest_separation(result))
        report.breaches.append(self._check_score_concentration(result))
        # Novel vectors 10-19
        report.breaches.append(self._check_validator_latency_exploit(result))
        report.breaches.append(self._check_prediction_timing_manipulation(result))
        report.breaches.append(self._check_miner_validator_collusion(result))
        report.breaches.append(self._check_weight_entropy_violation(result))
        report.breaches.append(self._check_cross_validator_score_variance(result))
        report.breaches.append(self._check_validator_rotation_circumvention(result))
        report.breaches.append(self._check_validator_agreement_anomaly(result))
        report.breaches.append(self._check_collusion_temporal_pattern(result))
        report.breaches.append(self._check_weight_manipulation(result))
        report.breaches.append(self._check_cross_layer_attack(result))
        report.breaches.append(self._check_selective_revelation(result))
        report.breaches.append(self._check_statistical_anomaly(result))
        report.breaches.append(self._check_behavioral_anomaly(result))
        report.breaches.append(self._check_temporal_attack_pattern(result))
        report.breaches.append(self._check_sybil_collusion_graph(result))
        report.breaches.append(self._check_cross_layer_correlation(result))
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


    # -------------------------------------------------------------------
    # Novel attack vectors 10-19 (discovered during orchestration runs)
    # -------------------------------------------------------------------

    def _check_validator_latency_exploit(self, result: SimulationResult) -> AttackBreach:
        """
        Vector 10: Validator latency exploitation.

        Breach condition: Miner accuracy is significantly correlated with
        validator latency, indicating trades submitted after data publication.
        Severity = correlation * high_latency_fraction.
        """
        validator_latencies = getattr(result, "validator_latencies", None)
        if not validator_latencies or not result.miner_scores:
            return AttackBreach(
                "validator_latency_exploitation", False, 0.0,
                "No validator latency data available for analysis",
            )

        min_lead = self.config.get("validation_timing", {}).get(
            "min_prediction_lead_time", 35,
        )
        high_thresh = self.config.get("validation_timing", {}).get(
            "high_latency_threshold_ms", 2000,
        )

        scores = np.array(list(result.miner_scores.values()))
        latencies = np.array(
            [validator_latencies.get(uid, 0) for uid in result.miner_scores]
        )

        if len(scores) < 3 or np.std(latencies) < 1e-12:
            return AttackBreach(
                "validator_latency_exploitation", False, 0.0,
                "Insufficient latency variation for correlation analysis",
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            corr = float(np.corrcoef(scores, latencies)[0, 1])
        corr = 0.0 if np.isnan(corr) else corr

        high_frac = float(np.mean(latencies > high_thresh))
        commitment_scores = self._compute_commitment_violation_scores(result)
        commitment_violation_score = max(
            (
                float(details["weighted_score"])
                for details in commitment_scores.values()
            ),
            default=0.0,
        )
        accuracy_disparity = max(
            (
                float(details["accuracy_disparity"])
                for details in commitment_scores.values()
            ),
            default=0.0,
        )
        latency_score = max(0.0, min(1.0, abs(corr) * high_frac))
        severity = max(
            0.0,
            min(
                1.0,
                latency_score * 0.15
                + accuracy_disparity * 0.12
                + commitment_violation_score * float(
                    self._validation_config().get("commitment_violation_weight", 0.008)
                ),
            ),
        )
        breached = severity >= 0.05

        return AttackBreach(
            "validator_latency_exploitation", breached, severity,
            f"corr(score,latency)={corr:.3f}, high_latency_frac={high_frac:.2f}, commitment_violation={commitment_violation_score:.3f}",
            {
                "correlation": corr,
                "high_latency_fraction": high_frac,
                "min_lead_time": min_lead,
                "latency_score": latency_score,
                "accuracy_disparity": accuracy_disparity,
                "commitment_violation_score": commitment_violation_score,
                "per_miner_commitment_scores": commitment_scores,
            },
        )

    def _check_prediction_timing_manipulation(self, result: SimulationResult) -> AttackBreach:
        """
        Vector 11: Prediction timing manipulation.

        Breach condition: Miners submit predictions suspiciously close to
        data publication time, suggesting post-hoc "prediction".
        """
        submission_gaps = getattr(result, "submission_timing_gaps", None)
        if not submission_gaps:
            return AttackBreach(
                "prediction_timing_manipulation", False, 0.0,
                "No submission timing data available",
            )

        min_lead = self.config.get("validation_timing", {}).get(
            "min_prediction_lead_time", 35,
        )

        violations = [uid for uid, gap in submission_gaps.items() if gap < min_lead]
        total = max(len(submission_gaps), 1)
        violation_rate = len(violations) / total
        reveal_gaps = []
        for key, commit_ts in getattr(result, "commit_timestamps", {}).items():
            reveal_ts = getattr(result, "reveal_timestamps", {}).get(key)
            if reveal_ts is not None:
                reveal_gaps.append(reveal_ts - commit_ts)
        average_reveal_delay = float(np.mean(reveal_gaps)) if reveal_gaps else 0.0
        severity = max(0.0, min(1.0, violation_rate * 1.5 + average_reveal_delay / 120.0))
        breached = severity > 0.05

        return AttackBreach(
            "prediction_timing_manipulation", breached, severity,
            f"{len(violations)}/{total} miners below {min_lead}s lead time",
            {
                "violations": violations,
                "min_lead_time": min_lead,
                "average_reveal_delay": average_reveal_delay,
            },
        )

    def _check_miner_validator_collusion(self, result: SimulationResult) -> AttackBreach:
        """
        Vector 12: Miner-validator collusion.

        Breach condition: A validator's weight distribution or scoring
        pattern shows statistically significant bias toward specific miners
        whose objective performance does not justify the preferential
        treatment.
        """
        per_validator_scores = getattr(result, "per_validator_scores", None)
        if not per_validator_scores or len(per_validator_scores) < 2:
            return AttackBreach(
                "miner_validator_collusion", False, 0.0,
                "Multi-validator scoring data not available",
            )

        max_z = 0.0
        flagged_pairs = []
        confirmation_window = 2
        vote_threshold = 2
        ensemble_signals = getattr(result, "ensemble_signals", {})
        for miner_uid in result.miner_scores:
            v_scores = []
            for vid, scores_map in per_validator_scores.items():
                if miner_uid in scores_map:
                    v_scores.append((vid, scores_map[miner_uid]))
            if len(v_scores) < 2:
                continue
            vals = np.array([s for _, s in v_scores])
            mean, std = np.mean(vals), np.std(vals)
            if std < 1e-12:
                continue
            for vid, s in v_scores:
                z = abs(s - mean) / std
                if z > max_z:
                    max_z = z
                if z > 2.0:
                    if self._ensemble_vote_count(ensemble_signals.get(miner_uid, {})) >= vote_threshold:
                        flagged_pairs.append((miner_uid, vid, float(z)))

        severity = max(0.0, min(1.0, (max_z - 1.0) / 3.0))
        breached = bool(flagged_pairs)

        return AttackBreach(
            "miner_validator_collusion", breached, severity,
            f"Max z-score={max_z:.2f}, flagged pairs={len(flagged_pairs)}",
            {
                "max_z_score": max_z,
                "flagged_pairs": flagged_pairs,
                "confirmation_window": confirmation_window,
                "vote_threshold": vote_threshold,
            },
        )

    def _check_weight_entropy_violation(self, result: SimulationResult) -> AttackBreach:
        """
        Vector 13: Weight entropy violation.

        Breach condition: A validator's weight distribution has entropy
        below the configured minimum, indicating concentration on few UIDs.
        """
        validator_weights = getattr(result, "validator_weight_vectors", None)
        if not validator_weights:
            return AttackBreach(
                "weight_entropy_violation", False, 0.0,
                "No validator weight vector data available",
            )

        min_entropy = self.config.get("consensus_integrity", {}).get(
            "weight_entropy_minimum", 1.3,
        )

        min_observed = float("inf")
        violating_validators = []
        for vid, weights in validator_weights.items():
            w = np.array(weights, dtype=float)
            w = w / max(w.sum(), 1e-12)
            w = w[w > 1e-12]
            entropy = -float(np.sum(w * np.log(w)))
            if entropy < min_observed:
                min_observed = entropy
            if entropy < min_entropy:
                violating_validators.append(vid)

        severity = max(0.0, min(1.0, 1.0 - min_observed / max(min_entropy, 1e-12)))
        breached = len(violating_validators) > 0

        return AttackBreach(
            "weight_entropy_violation", breached, severity,
            f"Min entropy={min_observed:.3f} (threshold={min_entropy}), "
            f"violators={len(violating_validators)}",
            {"min_entropy_observed": min_observed, "threshold": min_entropy},
        )

    def _check_cross_validator_score_variance(self, result: SimulationResult) -> AttackBreach:
        """
        Vector 14: Cross-validator score variance.

        Breach condition: A miner's scores vary dramatically across
        validators, suggesting preferential treatment by one validator.
        """
        per_validator_scores = getattr(result, "per_validator_scores", None)
        if not per_validator_scores or len(per_validator_scores) < 2:
            return AttackBreach(
                "cross_validator_score_variance", False, 0.0,
                "Multi-validator scoring data not available",
            )

        max_var_threshold = self.config.get("consensus_integrity", {}).get(
            "cross_validator_score_variance_max", 0.22,
        )

        max_variance = 0.0
        flagged_miners = []
        for miner_uid in result.miner_scores:
            v_scores = [
                scores_map[miner_uid]
                for scores_map in per_validator_scores.values()
                if miner_uid in scores_map
            ]
            if len(v_scores) < 2:
                continue
            var = float(np.var(v_scores))
            if var > max_variance:
                max_variance = var
            if var > max_var_threshold:
                flagged_miners.append(miner_uid)

        severity = max(0.0, min(1.0, max_variance / max(max_var_threshold * 2, 1e-12)))
        breached = len(flagged_miners) > 0

        return AttackBreach(
            "cross_validator_score_variance", breached, severity,
            f"Max variance={max_variance:.4f} (threshold={max_var_threshold}), "
            f"flagged={len(flagged_miners)}",
            {"max_variance": max_variance, "threshold": max_var_threshold},
        )

    def _check_validator_rotation_circumvention(self, result: SimulationResult) -> AttackBreach:
        """
        Vector 15: Validator rotation circumvention.

        Breach condition: The same validator scores the same miner for
        more than the allowed number of consecutive epochs.
        """
        scoring_history = getattr(result, "validator_scoring_history", None)
        if not scoring_history:
            return AttackBreach(
                "validator_rotation_circumvention", False, 0.0,
                "No validator scoring history available",
            )

        max_consec = self.config.get("consensus_integrity", {}).get(
            "validator_rotation_max_consecutive_epochs", 5,
        )

        worst_streak = 0
        for miner_uid, epochs in scoring_history.items():
            for vid in set(v for _, v in epochs):
                streak = 0
                for _, scorer_vid in epochs:
                    if scorer_vid == vid:
                        streak += 1
                        worst_streak = max(worst_streak, streak)
                    else:
                        streak = 0

        severity = max(0.0, min(1.0, (worst_streak - max_consec) / max(max_consec, 1)))
        breached = worst_streak > max_consec

        return AttackBreach(
            "validator_rotation_circumvention", breached, severity,
            f"Longest streak={worst_streak} (limit={max_consec})",
            {"worst_streak": worst_streak, "limit": max_consec},
        )

    def _check_validator_agreement_anomaly(self, result: SimulationResult) -> AttackBreach:
        """
        Vector 16: Validator agreement anomaly.

        Breach condition: A validator's scoring deviates from the
        consensus of other validators by more than the threshold.
        """
        per_validator_scores = getattr(result, "per_validator_scores", None)
        if not per_validator_scores or len(per_validator_scores) < 2:
            return AttackBreach(
                "validator_agreement_anomaly", False, 0.0,
                "Multi-validator scoring data not available",
            )

        agreement_threshold = self.config.get("consensus_integrity", {}).get(
            "validator_agreement_threshold", 0.18,
        )

        max_deviation = 0.0
        for vid, scores_map in per_validator_scores.items():
            others = {
                uid: np.mean([
                    om[uid] for ovid, om in per_validator_scores.items()
                    if ovid != vid and uid in om
                ])
                for uid in scores_map
                if sum(1 for ovid, om in per_validator_scores.items()
                       if ovid != vid and uid in om) > 0
            }
            for uid in scores_map:
                if uid in others:
                    dev = abs(scores_map[uid] - others[uid])
                    max_deviation = max(max_deviation, dev)

        severity = max(0.0, min(1.0, max_deviation / max(agreement_threshold * 3, 1e-12)))
        breached = max_deviation > agreement_threshold

        return AttackBreach(
            "validator_agreement_anomaly", breached, severity,
            f"Max scoring deviation={max_deviation:.4f} (threshold={agreement_threshold})",
            {"max_deviation": max_deviation, "threshold": agreement_threshold},
        )

    def _check_collusion_temporal_pattern(self, result: SimulationResult) -> AttackBreach:
        """
        Vector 17: Collusion temporal pattern.

        Breach condition: Miner submission patterns change in lockstep
        with a specific validator's behavior over the lookback window.
        """
        temporal_correlation = getattr(result, "miner_validator_temporal_corr", None)
        if not temporal_correlation:
            return AttackBreach(
                "collusion_temporal_pattern", False, 0.0,
                "No temporal correlation data available",
            )

        lookback = self.config.get("consensus_integrity", {}).get(
            "collusion_detection_lookback_epochs", 12,
        )

        max_corr = 0.0
        flagged = []
        for (miner_uid, vid), corr in temporal_correlation.items():
            abs_corr = abs(corr)
            if abs_corr > max_corr:
                max_corr = abs_corr
            if abs_corr > 0.7:
                flagged.append((miner_uid, vid, corr))

        severity = max(0.0, min(1.0, (max_corr - 0.3) / 0.7))
        breached = max_corr > 0.7

        return AttackBreach(
            "collusion_temporal_pattern", breached, severity,
            f"Max temporal corr={max_corr:.3f}, flagged pairs={len(flagged)}, "
            f"lookback={lookback}",
            {"max_correlation": max_corr, "flagged_pairs": flagged},
        )

    def _check_weight_manipulation(self, result: SimulationResult) -> AttackBreach:
        """
        Vector 18: L1/L2 weight skew exploitation.

        Breach condition: The effective emission split between L1 and L2
        deviates significantly from the configured ratio, indicating that
        adversarial miners are exploiting the weight-setting mechanism.
        """
        l1_total = sum(result.miner_scores.values()) if result.miner_scores else 0
        l2_total = sum(result.l2_scores.values()) if result.l2_scores else 0
        combined = l1_total + l2_total

        if combined < 1e-12:
            return AttackBreach(
                "weight_manipulation", False, 0.0,
                "No scores available for weight analysis",
            )

        target_split = self.config.get("emissions", {}).get("l1_l2_split", 0.6)
        actual_l1_share = l1_total / combined
        skew = abs(actual_l1_share - target_split)

        penalty_strength = self.config.get("cross_layer_balance", {}).get(
            "penalty_strength", 0.3,
        )
        severity = max(0.0, min(1.0, skew * penalty_strength * 5))
        breached = skew > 0.2

        return AttackBreach(
            "weight_manipulation", breached, severity,
            f"L1 share={actual_l1_share:.3f} vs target={target_split:.3f}, skew={skew:.3f}",
            {"actual_l1_share": actual_l1_share, "target_split": target_split},
        )

    def _check_cross_layer_attack(self, result: SimulationResult) -> AttackBreach:
        """
        Vector 19: Cross-layer timing sync attack.

        Breach condition: Cross-layer feedback latency exceeds the
        configured threshold, or adversarial miners exploit timing gaps
        between L1 and L2 scoring windows.
        """
        cross_layer_latencies = getattr(result, "cross_layer_latencies", None)
        if not cross_layer_latencies:
            adv_l1 = result.adversarial_l1_scores
            adv_l2 = result.adversarial_l2_scores
            if not adv_l1 or not adv_l2:
                return AttackBreach(
                    "cross_layer_attack", False, 0.0,
                    "No cross-layer timing data available",
                )
            mean_adv_l1 = np.mean(adv_l1)
            mean_honest_l1 = np.mean(result.honest_l1_scores) if result.honest_l1_scores else 0
            mean_adv_l2 = np.mean(adv_l2)
            mean_honest_l2 = np.mean(result.honest_l2_scores) if result.honest_l2_scores else 0

            l1_gap = mean_honest_l1 - mean_adv_l1
            l2_gap = mean_honest_l2 - mean_adv_l2

            if l1_gap > 0 and l2_gap < 0:
                severity = min(1.0, abs(l2_gap) / max(abs(l1_gap), 1e-12))
                return AttackBreach(
                    "cross_layer_attack", True, severity,
                    f"Adversarial L2 gain despite L1 defense: L1 gap={l1_gap:.4f}, L2 gap={l2_gap:.4f}",
                )
            return AttackBreach(
                "cross_layer_attack", False, 0.0,
                "No cross-layer timing exploit detected",
            )

        max_latency_threshold = self.config.get("cross_layer_timing", {}).get(
            "max_latency_ms", 200,
        )

        latency_arr = np.array(list(cross_layer_latencies.values()))
        max_lat = float(np.max(latency_arr)) if len(latency_arr) > 0 else 0
        violation_rate = float(np.mean(latency_arr > max_latency_threshold))

        severity = max(0.0, min(1.0, violation_rate))
        breached = violation_rate > 0.3

        return AttackBreach(
            "cross_layer_attack", breached, severity,
            f"Max latency={max_lat:.0f}ms, violation rate={violation_rate:.2f} "
            f"(threshold={max_latency_threshold}ms)",
            {"max_latency": max_lat, "violation_rate": violation_rate},
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from tuning.parameter_space import encode_defaults
    from tuning.simulation import SimulationHarness, create_default_agents

    print("Running attack detection with report-aligned default parameters...\n")

    l1_agents, l2_agents = create_default_agents(
        n_honest=6,
        n_overfitters=1,
        n_copycats=1,
        n_gamers=1,
        n_sybils=2,
        n_random=0,
        n_honest_traders=3,
        n_copy_traders=1,
    )

    harness = SimulationHarness(
        l1_agents=l1_agents, l2_agents=l2_agents,
        n_epochs=100, n_trading_steps=150,
    )

    defaults = encode_defaults()
    sim_result = harness.run(defaults)

    detector = AttackDetector()
    report = detector.evaluate(sim_result)
    print(report.summary())
    print(f"\nOverall: {report.n_breached}/{report.total_attacks} breached, "
          f"mean severity={report.mean_severity:.4f}")
