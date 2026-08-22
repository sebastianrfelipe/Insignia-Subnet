"""
Tests for the ground-truth enrichment feedback loop.

Covers:
  - EnrichmentTracker computes correct hit rates and enrichment factor.
  - Enrichment factor = 1.0 when promoted and baseline hit rates are equal.
  - Sim-vs-live Spearman correlation captures ranking quality.
  - Sample-size confidence shrinkage at small N.
  - Time-decay window prunes old outcomes.
  - ScoringRNDLoop enrichment gate blocks promotion when EF is too low.
  - compute_fitness includes the 5th objective when enrichment is provided.
  - compute_fitness returns 0.0 for the 5th objective when enrichment is absent.
  - ExploitSignalCollector.record_sim_live_gap fires an oracle_blind_spot.
"""

import unittest

import numpy as np

from insignia.enrichment import (
    EnrichmentConfig,
    EnrichmentTracker,
    LiveEnrichmentMetrics,
    PairOutcome,
)
from insignia.scoring import WeightConfig
from insignia.scoring_rnd import (
    ExploitSignalCollector,
    ScoringRNDLoop,
)


class EnrichmentTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = EnrichmentTracker()

    def test_hit_rate_computation(self):
        """4/5 promoted pairs hit, 1/5 baseline pairs hit -> EF = 4.0."""
        # 5 promoted pairs: 4 hits (PnL > 0), 1 miss.
        for i in range(4):
            self.tracker.record_outcome(
                f"promoted_{i}", epoch=0, sim_composite=0.8,
                live_pnl=0.05, live_sharpe=1.5, deployed=True,
            )
        self.tracker.record_outcome(
            "promoted_4", epoch=0, sim_composite=0.8,
            live_pnl=-0.02, live_sharpe=-0.5, deployed=True,
        )
        # 5 baseline pairs: 1 hit, 4 misses.
        self.tracker.record_outcome(
            "baseline_0", epoch=0, sim_composite=0.3,
            live_pnl=0.01, live_sharpe=0.3, deployed=False,
        )
        for i in range(4):
            self.tracker.record_outcome(
                f"baseline_{i+1}", epoch=0, sim_composite=0.3,
                live_pnl=-0.03, live_sharpe=-0.8, deployed=False,
            )

        m = self.tracker.compute_enrichment(current_epoch=0)
        self.assertAlmostEqual(m.promoted_hit_rate, 0.8, places=3)
        self.assertAlmostEqual(m.baseline_hit_rate, 0.2, places=3)
        self.assertAlmostEqual(m.enrichment_factor, 4.0, places=3)
        self.assertEqual(m.n_promoted, 5)
        self.assertEqual(m.n_baseline, 5)

    def test_enrichment_factor_one_when_random(self):
        """Equal hit rates -> EF = 1.0 (no enrichment)."""
        for i in range(5):
            self.tracker.record_outcome(
                f"promoted_{i}", epoch=0, sim_composite=0.8,
                live_pnl=0.02, deployed=True,
            )
            self.tracker.record_outcome(
                f"baseline_{i}", epoch=0, sim_composite=0.3,
                live_pnl=0.02, deployed=False,
            )
        m = self.tracker.compute_enrichment(current_epoch=0)
        self.assertAlmostEqual(m.promoted_hit_rate, 1.0)
        self.assertAlmostEqual(m.baseline_hit_rate, 1.0)
        self.assertAlmostEqual(m.enrichment_factor, 1.0, places=3)

    def test_sim_live_correlation(self):
        """Sim scores that rank-predict live P&L -> high Spearman."""
        # Sim composite perfectly predicts live P&L ordering.
        sim_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        live_pnls = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        for i, (s, p) in enumerate(zip(sim_scores, live_pnls)):
            self.tracker.record_outcome(
                f"pair_{i}", epoch=0, sim_composite=s,
                live_pnl=p, deployed=True,
            )
        m = self.tracker.compute_enrichment(current_epoch=0)
        self.assertGreater(m.sim_vs_live_correlation, 0.95)

    def test_shrinkage_at_small_n(self):
        """Few observations -> shrinkage pulls enrichment toward 0."""
        # 2 promoted pairs, both hits; 2 baseline pairs, both misses.
        # Raw EF would be infinite, but shrinkage should pull it down.
        tracker = EnrichmentTracker(
            EnrichmentConfig(confidence_k=30.0)
        )
        tracker.record_outcome("p0", epoch=0, sim_composite=0.9,
                               live_pnl=0.05, deployed=True)
        tracker.record_outcome("p1", epoch=0, sim_composite=0.9,
                               live_pnl=0.03, deployed=True)
        tracker.record_outcome("b0", epoch=0, sim_composite=0.2,
                               live_pnl=-0.02, deployed=False)
        tracker.record_outcome("b1", epoch=0, sim_composite=0.2,
                               live_pnl=-0.04, deployed=False)
        m = tracker.compute_enrichment(current_epoch=0)
        # Shrinkage at N=4 with k=30: sqrt(4/34) ≈ 0.343
        expected_shrink = np.sqrt(4 / 34)
        self.assertAlmostEqual(m.shrinkage, expected_shrink, places=3)
        # Shrunk EF should be much less than raw EF.
        self.assertLess(m.shrunk_enrichment_factor, m.enrichment_factor)

    def test_time_decay_prunes_old(self):
        """Aged outcomes leave the window."""
        self.tracker.record_outcome("old", epoch=0, sim_composite=0.8,
                                    live_pnl=0.05, deployed=True)
        self.tracker.record_outcome("new", epoch=20, sim_composite=0.8,
                                    live_pnl=0.03, deployed=True)
        self.tracker.time_decay(current_epoch=20)
        # After pruning with window=12, epoch-0 entry is gone (cutoff=8).
        self.assertEqual(len(self.tracker._outcomes), 1)
        self.assertEqual(self.tracker._outcomes[0].pair_id, "new")

    def test_empty_tracker_returns_defaults(self):
        """No outcomes -> EF=1.0, shrinkage=0.0 (graceful degradation)."""
        m = self.tracker.compute_enrichment(current_epoch=0)
        self.assertAlmostEqual(m.enrichment_factor, 1.0)
        self.assertAlmostEqual(m.shrinkage, 0.0)
        self.assertEqual(m.n_promoted, 0)


class ScoringRNDEnrichmentGateTests(unittest.TestCase):
    def test_rnd_loop_enrichment_gate_blocks_low_ef(self):
        """A variant that improves sim separation but has EF below the
        floor must not be promoted."""
        rnd = ScoringRNDLoop(min_enrichment_factor=1.5)
        exp = rnd.propose(
            description="test variant",
            weights=WeightConfig(),
            motivation="test",
        )
        # Baseline: separation 0.10. Candidate: separation 0.15 (improves).
        # But enrichment factor is 1.0 with shrinkage 1.0 (lots of data).
        enrichment = LiveEnrichmentMetrics(
            enrichment_factor=1.0,
            promoted_hit_rate=0.2,
            baseline_hit_rate=0.2,
            n_promoted=100,
            n_baseline=100,
            shrinkage=1.0,
        )
        kept, reason = rnd.evaluate(
            exp,
            honest_scores_baseline=[0.5, 0.5, 0.5],
            adversarial_scores_baseline=[0.4, 0.4, 0.4],
            honest_scores_candidate=[0.6, 0.6, 0.6],
            adversarial_scores_candidate=[0.3, 0.3, 0.3],
            enrichment_metrics=enrichment,
        )
        self.assertFalse(kept)
        self.assertIn("enrichment gate", reason)

    def test_rnd_loop_promotes_when_ef_above_floor(self):
        """When enrichment is above the floor and sim improves, promote."""
        rnd = ScoringRNDLoop(min_enrichment_factor=1.5)
        exp = rnd.propose(
            description="test variant",
            weights=WeightConfig(),
            motivation="test",
        )
        enrichment = LiveEnrichmentMetrics(
            enrichment_factor=3.0,
            promoted_hit_rate=0.6,
            baseline_hit_rate=0.2,
            n_promoted=100,
            n_baseline=100,
            shrinkage=1.0,
        )
        kept, reason = rnd.evaluate(
            exp,
            honest_scores_baseline=[0.5, 0.5, 0.5],
            adversarial_scores_baseline=[0.4, 0.4, 0.4],
            honest_scores_candidate=[0.6, 0.6, 0.6],
            adversarial_scores_candidate=[0.3, 0.3, 0.3],
            enrichment_metrics=enrichment,
        )
        self.assertTrue(kept)

    def test_rnd_loop_without_enrichment_works_as_before(self):
        """No enrichment data -> only sim gate applies (backward compat)."""
        rnd = ScoringRNDLoop()
        exp = rnd.propose(
            description="test variant",
            weights=WeightConfig(),
            motivation="test",
        )
        kept, reason = rnd.evaluate(
            exp,
            honest_scores_baseline=[0.5, 0.5, 0.5],
            adversarial_scores_baseline=[0.4, 0.4, 0.4],
            honest_scores_candidate=[0.6, 0.6, 0.6],
            adversarial_scores_candidate=[0.3, 0.3, 0.3],
        )
        self.assertTrue(kept)
        self.assertIsNone(exp.enrichment_factor)


class ComputeFitnessTests(unittest.TestCase):
    def test_fitness_with_enrichment(self):
        """compute_fitness with enrichment returns 5 objectives."""
        from tuning.optimizer import compute_fitness, OBJECTIVE_NAMES

        # Build a minimal SimulationResult mock.
        class MockSimResult:
            honest_researcher_scores = [0.5, 0.6, 0.7]
            adversarial_researcher_scores = [0.2, 0.3]

        class MockBreachReport:
            breach_rate = 0.01

        enrichment = LiveEnrichmentMetrics(
            enrichment_factor=2.5,
            shrinkage=0.8,
        )
        fitness = compute_fitness(
            MockSimResult(), MockBreachReport(), enrichment
        )
        self.assertEqual(len(fitness), 5)
        self.assertEqual(len(OBJECTIVE_NAMES), 5)
        # 5th objective = -EF * shrinkage = -2.5 * 0.8 = -2.0
        self.assertAlmostEqual(fitness[4], -2.0, places=3)

    def test_fitness_without_enrichment(self):
        """Without enrichment, 5th objective is 0.0 (backward compatible)."""
        from tuning.optimizer import compute_fitness

        class MockSimResult:
            honest_researcher_scores = [0.5, 0.6, 0.7]
            adversarial_researcher_scores = [0.2, 0.3]

        class MockBreachReport:
            breach_rate = 0.01

        fitness = compute_fitness(MockSimResult(), MockBreachReport())
        self.assertEqual(len(fitness), 5)
        self.assertAlmostEqual(fitness[4], 0.0)


class ExploitSignalSimLiveGapTests(unittest.TestCase):
    def test_exploit_signal_sim_live_gap(self):
        """Sim-high/live-low pair fires an oracle_blind_spot signal."""
        collector = ExploitSignalCollector()
        signal = collector.record_sim_live_gap(
            pair_id="pair_42",
            epoch=5,
            sim_composite=0.85,
            live_pnl_rank=0.15,
            gap_threshold=0.30,
        )
        self.assertIsNotNone(signal)
        self.assertEqual(signal.signal_type, "oracle_blind_spot")
        self.assertEqual(signal.affected_metric, "live_pnl_rank")
        self.assertGreater(signal.severity, 0.30)

    def test_exploit_signal_sim_live_gap_no_signal_when_aligned(self):
        """When sim and live agree, no signal fires."""
        collector = ExploitSignalCollector()
        signal = collector.record_sim_live_gap(
            pair_id="pair_43",
            epoch=5,
            sim_composite=0.50,
            live_pnl_rank=0.55,
            gap_threshold=0.30,
        )
        self.assertIsNone(signal)


if __name__ == "__main__":
    unittest.main()
