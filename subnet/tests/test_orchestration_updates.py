import sys
import types
import unittest
from dataclasses import dataclass, field

simulation_stub = types.ModuleType("tuning.simulation")


@dataclass
class SimulationResult:
    miner_types: dict = field(default_factory=dict)
    miner_scores: dict = field(default_factory=dict)
    miner_commit_rates: dict = field(default_factory=dict)
    miner_accuracy_by_commit_status: dict = field(default_factory=dict)
    no_reveal_streaks: dict = field(default_factory=dict)
    honest_l1_scores: list = field(default_factory=list)
    adversarial_l1_scores: list = field(default_factory=list)
    l2_scores: dict = field(default_factory=dict)
    l2_types: dict = field(default_factory=dict)
    honest_l2_scores: list = field(default_factory=list)
    adversarial_l2_scores: list = field(default_factory=list)
    validator_latencies: dict = field(default_factory=dict)
    submission_timing_gaps: dict = field(default_factory=dict)
    per_validator_scores: dict = field(default_factory=dict)
    validator_weight_vectors: dict = field(default_factory=dict)
    validator_scoring_history: dict = field(default_factory=dict)
    miner_validator_temporal_corr: dict = field(default_factory=dict)
    cross_layer_latencies: dict = field(default_factory=dict)
    ensemble_signals: dict = field(default_factory=dict)
    trading_pair_counts: dict = field(default_factory=dict)


simulation_stub.SimulationResult = SimulationResult
sys.modules.setdefault("tuning.simulation", simulation_stub)

from tuning.attack_detector import AttackDetector
from tuning.composite_integrity_scorer import CompositeIntegrityScorer


class OrchestrationUpdateTests(unittest.TestCase):
    def test_commitment_violation_and_selective_revelation(self):
        result = SimulationResult(
            miner_types={"miner_a": "honest", "miner_b": "sybil"},
            miner_scores={"miner_a": 0.8, "miner_b": 0.6},
            miner_commit_rates={"miner_a": 0.9, "miner_b": 0.4},
            miner_accuracy_by_commit_status={
                "miner_a": {
                    "accuracy_when_committed": 0.8,
                    "accuracy_when_not_committed": 0.7,
                },
                "miner_b": {
                    "accuracy_when_committed": 0.4,
                    "accuracy_when_not_committed": 0.9,
                },
            },
            no_reveal_streaks={"miner_a": 0, "miner_b": 3},
        )
        detector = AttackDetector(
            {
                "validation_timing": {
                    "commit_rate_threshold": 0.7,
                    "commitment_violation_weight": 0.008,
                    "selective_reveal_warning_streak": 1,
                    "selective_reveal_penalty_streak": 2,
                    "selective_reveal_zero_streak": 3,
                }
            }
        )

        scores = detector._compute_commitment_violation_scores(result)
        self.assertTrue(scores["miner_b"]["flagged"])
        self.assertFalse(scores["miner_a"]["flagged"])

        selective = detector._check_selective_revelation(result)
        self.assertTrue(selective.breached)
        self.assertIn("zeroed=1", selective.description)

    def test_composite_integrity_score_is_normalized(self):
        scorer = CompositeIntegrityScorer()
        score = scorer.calculate_composite_integrity_score(
            {
                "breach_rate": 0.000049,
                "max_breach_rate_observed": 0.05,
                "detection_delay_normalized": 0.2,
                "performance_history": [0.94, 0.95, 0.96, 0.97, 0.9705],
            }
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreater(score, 0.8)


if __name__ == "__main__":
    unittest.main()
