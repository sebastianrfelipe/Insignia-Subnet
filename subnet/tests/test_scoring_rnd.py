"""
Tests for the scoring-function R&D loop and exploit signal collector.

Covers:
  - RetrospectiveValidator computes correct discrimination metrics.
  - Promotion gate keeps variants that improve separation/AUC.
  - Promotion gate drops variants that regress the honest floor.
  - ExploitSignalCollector detects metric concentration.
  - ExploitSignalCollector detects size bias across a population.
  - ExploitSignalCollector detects oracle blind spots.
  - ScoringRNDLoop proposes, evaluates, and promotes kept experiments.
  - ScoringRNDLoop summary reports kept/dropped counts.
"""

import unittest

import numpy as np

from insignia.scoring import WeightConfig
from insignia.scoring_rnd import (
    DiscriminationMetrics,
    ExploitSignal,
    ExploitSignalCollector,
    RetrospectiveValidator,
    ScoringExperiment,
    ScoringRNDLoop,
)
from insignia.scoring import ScoreVector


class RetrospectiveValidatorTests(unittest.TestCase):
    def setUp(self):
        self.validator = RetrospectiveValidator(
            promotion_separation_delta=0.02,
            promotion_auc_delta=0.02,
            max_honest_floor_regression=0.01,
        )

    def test_compute_metrics_perfect_separation(self):
        honest = [0.9, 0.92, 0.88]
        adversarial = [0.1, 0.2, 0.05]
        m = self.validator.compute_metrics(honest, adversarial)
        self.assertGreater(m.separation, 0.7)
        self.assertEqual(m.auc, 1.0)
        self.assertEqual(m.adversary_leak_rate, 0.0)

    def test_compute_metrics_random_overlap(self):
        honest = [0.5, 0.5, 0.5]
        adversarial = [0.5, 0.5, 0.5]
        m = self.validator.compute_metrics(honest, adversarial)
        self.assertEqual(m.separation, 0.0)
        self.assertEqual(m.auc, 0.5)
        # Adversaries equal the honest floor; strict > means no leak.
        self.assertEqual(m.adversary_leak_rate, 0.0)
        self.assertEqual(m.adversary_ceiling, 0.5)

    def test_should_promote_on_separation_improvement(self):
        baseline = DiscriminationMetrics(separation=0.50, auc=0.70, honest_floor=0.80)
        candidate = DiscriminationMetrics(separation=0.60, auc=0.72, honest_floor=0.80)
        kept, reason = self.validator.should_promote(baseline, candidate)
        self.assertTrue(kept)

    def test_should_promote_on_auc_improvement(self):
        baseline = DiscriminationMetrics(separation=0.50, auc=0.70, honest_floor=0.80)
        candidate = DiscriminationMetrics(separation=0.50, auc=0.80, honest_floor=0.80)
        kept, reason = self.validator.should_promote(baseline, candidate)
        self.assertTrue(kept)

    def test_should_drop_on_floor_regression(self):
        baseline = DiscriminationMetrics(separation=0.50, auc=0.70, honest_floor=0.80)
        candidate = DiscriminationMetrics(separation=0.60, auc=0.80, honest_floor=0.70)
        kept, reason = self.validator.should_promote(baseline, candidate)
        self.assertFalse(kept)
        self.assertIn("floor regressed", reason)

    def test_should_drop_on_no_improvement(self):
        baseline = DiscriminationMetrics(separation=0.50, auc=0.70, honest_floor=0.80)
        candidate = DiscriminationMetrics(separation=0.505, auc=0.701, honest_floor=0.80)
        kept, reason = self.validator.should_promote(baseline, candidate)
        self.assertFalse(kept)
        self.assertIn("no improvement", reason)


class ExploitSignalCollectorTests(unittest.TestCase):
    def setUp(self):
        self.collector = ExploitSignalCollector()

    def test_metric_concentration_detected(self):
        # One metric dominates: F1 normalized = 0.9, others sum to 0.1.
        sv = ScoreVector(
            raw={},
            normalized={
                "penalized_f1": 0.95,
                "penalized_sharpe": 0.10,
                "max_drawdown": 0.50,
                "variance_score": 0.30,
                "overfitting_penalty": 0.40,
                "feature_efficiency": 0.20,
                "latency": 0.10,
            },
            composite=0.5,
        )
        signal = self.collector.record_metric_concentration(
            "r0", epoch=0, score_vector=sv, concentration_threshold=0.30
        )
        self.assertIsNotNone(signal)
        self.assertEqual(signal.signal_type, "metric_concentration")
        self.assertEqual(signal.affected_metric, "penalized_f1")

    def test_metric_concentration_not_detected_when_balanced(self):
        sv = ScoreVector(
            raw={},
            normalized={f"m{i}": 0.5 for i in range(7)},
            composite=0.5,
        )
        signal = self.collector.record_metric_concentration(
            "r0", epoch=0, score_vector=sv, concentration_threshold=0.60
        )
        self.assertIsNone(signal)

    def test_size_bias_detected_across_population(self):
        composites = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        sizes =      [1,   2,   3,   4,   5,   6,   7]
        signal = self.collector.evaluate_size_bias_batch(
            composites, sizes, epoch=0, correlation_threshold=0.70
        )
        self.assertIsNotNone(signal)
        self.assertEqual(signal.signal_type, "size_bias")

    def test_size_bias_not_detected_when_uncorrelated(self):
        composites = [0.7, 0.2, 0.8, 0.3, 0.6, 0.1, 0.9]
        sizes =      [1,    2,   3,   4,   5,   6,   7]
        signal = self.collector.evaluate_size_bias_batch(
            composites, sizes, epoch=0, correlation_threshold=0.70
        )
        self.assertIsNone(signal)

    def test_oracle_blind_spot_detected(self):
        signal = self.collector.record_oracle_blind_spot(
            "r0", epoch=0, composite=0.90, orthogonal_check=0.40,
            orthogonal_name="deployment_pnl", gap_threshold=0.30,
        )
        self.assertIsNotNone(signal)
        self.assertEqual(signal.signal_type, "oracle_blind_spot")

    def test_propose_revisions_dedupes(self):
        sv = ScoreVector(
            raw={},
            normalized={"penalized_f1": 0.95, "other": 0.10},
            composite=0.5,
        )
        self.collector.record_metric_concentration("r0", 0, sv, 0.40)
        self.collector.record_metric_concentration("r1", 0, sv, 0.40)
        proposals = self.collector.propose_revisions()
        self.assertEqual(len(proposals), 1)


class ScoringRNDLoopTests(unittest.TestCase):
    def setUp(self):
        self.rnd = ScoringRNDLoop()

    def test_propose_creates_experiment(self):
        exp = self.rnd.propose(
            description="reduce F1 weight",
            weights=WeightConfig(model_penalized_f1=0.15),
            motivation="metric concentration signal",
            target_vector="single_metric_gaming",
        )
        self.assertTrue(exp.experiment_id.startswith("EXP-SCR-"))
        self.assertEqual(exp.description, "reduce F1 weight")

    def test_evaluate_keeps_improving_variant(self):
        exp = self.rnd.propose(
            description="improved weights",
            weights=WeightConfig(),
        )
        kept, reason = self.rnd.evaluate(
            exp,
            honest_scores_baseline=[0.80, 0.82, 0.78],
            adversarial_scores_baseline=[0.40, 0.50, 0.30],
            honest_scores_candidate=[0.85, 0.87, 0.83],
            adversarial_scores_candidate=[0.20, 0.25, 0.15],
        )
        self.assertTrue(kept)
        self.assertTrue(exp.kept)
        self.assertEqual(self.rnd.kept_count, 1)

    def test_evaluate_drops_non_improving_variant(self):
        exp = self.rnd.propose(
            description="no improvement",
            weights=WeightConfig(),
        )
        kept, reason = self.rnd.evaluate(
            exp,
            honest_scores_baseline=[0.80, 0.82, 0.78],
            adversarial_scores_baseline=[0.40, 0.50, 0.30],
            honest_scores_candidate=[0.80, 0.82, 0.78],
            adversarial_scores_candidate=[0.40, 0.50, 0.30],
        )
        self.assertFalse(kept)
        self.assertFalse(exp.kept)
        self.assertEqual(self.rnd.dropped_count, 1)

    def test_promote_updates_current_scorer(self):
        new_weights = WeightConfig(model_penalized_f1=0.15)
        exp = self.rnd.propose("promote me", new_weights)
        self.rnd.evaluate(
            exp,
            honest_scores_baseline=[0.80, 0.82],
            adversarial_scores_baseline=[0.40, 0.50],
            honest_scores_candidate=[0.85, 0.87],
            adversarial_scores_candidate=[0.20, 0.25],
        )
        if exp.kept:
            scorer = self.rnd.promote(exp)
            self.assertEqual(scorer.weights.model_penalized_f1, 0.15)

    def test_promote_rejects_unkept_experiment(self):
        exp = self.rnd.propose("not kept", WeightConfig())
        self.rnd.evaluate(
            exp,
            honest_scores_baseline=[0.8],
            adversarial_scores_baseline=[0.4],
            honest_scores_candidate=[0.8],
            adversarial_scores_candidate=[0.4],
        )
        with self.assertRaises(ValueError):
            self.rnd.promote(exp)

    def test_summary_reports_counts(self):
        exp1 = self.rnd.propose("keep", WeightConfig())
        self.rnd.evaluate(
            exp1,
            honest_scores_baseline=[0.8, 0.82],
            adversarial_scores_baseline=[0.4, 0.5],
            honest_scores_candidate=[0.85, 0.87],
            adversarial_scores_candidate=[0.2, 0.25],
        )
        exp2 = self.rnd.propose("drop", WeightConfig())
        self.rnd.evaluate(
            exp2,
            honest_scores_baseline=[0.8, 0.82],
            adversarial_scores_baseline=[0.4, 0.5],
            honest_scores_candidate=[0.8, 0.82],
            adversarial_scores_candidate=[0.4, 0.5],
        )
        summary = self.rnd.summary()
        self.assertEqual(summary["total_experiments"], 2)
        self.assertEqual(summary["kept"], 1)
        self.assertEqual(summary["dropped"], 1)


if __name__ == "__main__":
    unittest.main()
