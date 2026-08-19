"""
Tests for the novelty tracker and diversity scoring.

Covers:
  - First-time submissions score full novelty (1.0) and zero duplicate.
  - Re-submissions decay novelty across epochs.
  - Exact cross-miner duplicates are detected and penalized/invalidated.
  - Near-duplicates via feature-set Jaccard and prediction correlation.
  - Trading-style clone detection via position correlation.
  - Time-decay window pruning.
  - Style signature derivation.
  - Scoring engine apply_novelty_adjustment integrates with ScoreVector.
"""

import unittest

import numpy as np

from insignia.novelty import (
    NoveltyTracker,
    NoveltyConfig,
    compute_style_signature,
    feature_set_signature,
)
from insignia.scoring import ScoreVector, CompositeScorer


class NoveltyTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = NoveltyTracker(
            NoveltyConfig(
                novelty_bonus_weight=0.10,
                novelty_decay_epochs=4,
                duplicate_penalty=0.50,
                feature_novelty_threshold=0.85,
                prediction_correlation_threshold=0.90,
                style_correlation_threshold=0.85,
                history_window_epochs=12,
                invalidate_exact_duplicates=False,
            )
        )

    def test_first_submission_is_fully_novel(self):
        fp = "abc123"
        novelty, dup = self.tracker.model_novelty(
            "r0", fp, ["f1", "f2", "f3"], None, epoch=0
        )
        self.assertEqual(novelty, 1.0)
        self.assertEqual(dup, 0.0)

    def test_resubmission_decays_novelty(self):
        fp = "abc123"
        features = ["f1", "f2", "f3"]
        self.tracker.register_model("r0", fp, features, None, epoch=0)

        # Epoch 0 (first registration): novelty should decay from first seen.
        novelty_e0, _ = self.tracker.model_novelty("r0", fp, features, None, epoch=0)
        # Same epoch as first seen -> 2^0 = 1.0
        self.assertAlmostEqual(novelty_e0, 1.0, places=5)

        # Epoch 4: 2^(4/4) = 0.5
        novelty_e4, _ = self.tracker.model_novelty("r0", fp, features, None, epoch=4)
        self.assertAlmostEqual(novelty_e4, 0.5, places=2)

        # Epoch 8: 2^(8/4) = 0.25
        novelty_e8, _ = self.tracker.model_novelty("r0", fp, features, None, epoch=8)
        self.assertAlmostEqual(novelty_e8, 0.25, places=2)

    def test_exact_cross_miner_duplicate(self):
        fp = "shared_fp"
        features = ["f1", "f2"]
        self.tracker.register_model("r0", fp, features, None, epoch=0)

        # Different miner, same fingerprint -> duplicate score 1.0
        novelty, dup = self.tracker.model_novelty("r1", fp, features, None, epoch=1)
        self.assertEqual(dup, 1.0)

    def test_exact_duplicate_invalidates_when_configured(self):
        tracker = NoveltyTracker(
            NoveltyConfig(invalidate_exact_duplicates=True)
        )
        fp = "shared_fp"
        features = ["f1", "f2"]
        tracker.register_model("r0", fp, features, None, epoch=0)
        novelty, dup = tracker.model_novelty("r1", fp, features, None, epoch=1)
        self.assertEqual(novelty, 0.0)
        self.assertEqual(dup, 1.0)

    def test_near_duplicate_feature_jaccard(self):
        features_a = ["f1", "f2", "f3", "f4"]
        features_b = ["f1", "f2", "f3", "f5"]  # 3/5 overlap = 0.6 jaccard
        self.tracker.register_model("r0", "fp_a", features_a, None, epoch=0)

        # Below threshold: no duplicate.
        novelty, dup = self.tracker.model_novelty("r1", "fp_b", features_b, None, epoch=1)
        self.assertEqual(dup, 0.0)

        # High overlap (above threshold): duplicate detected.
        features_c = ["f1", "f2", "f3", "f4", "f5", "f6"]
        features_d = ["f1", "f2", "f3", "f4", "f5", "f7"]  # 5/7 = 0.71 jaccard
        tracker2 = NoveltyTracker(
            NoveltyConfig(feature_novelty_threshold=0.65)
        )
        tracker2.register_model("r0", "fp_c", features_c, None, epoch=0)
        novelty, dup = tracker2.model_novelty("r1", "fp_d", features_d, None, epoch=1)
        self.assertGreater(dup, 0.0)

    def test_near_duplicate_prediction_correlation(self):
        preds_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        preds_b = np.array([1.1, 2.1, 2.9, 4.1, 5.1])  # very correlated
        self.tracker.register_model("r0", "fp_a", ["f1"], preds_a, epoch=0)

        novelty, dup = self.tracker.model_novelty(
            "r1", "fp_b", ["f2"], preds_b, epoch=1
        )
        self.assertGreater(dup, 0.0)

    def test_trading_style_clone_detection(self):
        instruments = ["BTCUSDT", "ETHUSDT"]
        sig, vec = compute_style_signature(
            [
                {"instrument": "BTCUSDT", "side": "long", "size": 1.0, "holding_seconds": 30},
                {"instrument": "ETHUSDT", "side": "short", "size": 2.0, "holding_seconds": 120},
            ],
            instruments,
        )
        self.assertNotEqual(sig, "empty")
        self.assertEqual(len(vec), 2)

        self.tracker.register_trading("t0", sig, vec, epoch=0)
        # Same signature from a different miner -> duplicate.
        novelty, dup = self.tracker.trading_novelty("t1", sig, vec, epoch=1)
        self.assertEqual(dup, 1.0)

    def test_time_decay_prunes_old_entries(self):
        self.tracker.register_model("r0", "fp_old", ["f1"], None, epoch=0)
        # Prune everything older than epoch 12 - 12 = 0 boundary.
        self.tracker.time_decay(current_epoch=13)
        # After pruning, the old entry should be gone, so re-submit is novel.
        novelty, _ = self.tracker.model_novelty("r0", "fp_old", ["f1"], None, epoch=13)
        self.assertEqual(novelty, 1.0)

    def test_reset_clears_history(self):
        self.tracker.register_model("r0", "fp", ["f1"], None, epoch=0)
        self.tracker.reset()
        novelty, dup = self.tracker.model_novelty("r0", "fp", ["f1"], None, epoch=1)
        self.assertEqual(novelty, 1.0)
        self.assertEqual(dup, 0.0)

    def test_feature_set_signature_deterministic(self):
        sig_a = feature_set_signature(["f1", "f2", "f3"])
        sig_b = feature_set_signature(["f3", "f2", "f1"])  # same set, diff order
        sig_c = feature_set_signature(["f1", "f2", "f4"])
        self.assertEqual(sig_a, sig_b)
        self.assertNotEqual(sig_a, sig_c)


class ScoringNoveltyIntegrationTests(unittest.TestCase):
    def test_apply_novelty_adjustment_boosts_novel(self):
        sv = ScoreVector(
            raw={}, normalized={}, composite=0.80,
        )
        CompositeScorer.apply_novelty_adjustment(
            sv, novelty_score=1.0, duplicate_score=0.0,
            novelty_bonus_weight=0.10, duplicate_penalty=0.50,
        )
        # 0.80 * (1 + 0.10 - 0) = 0.88
        self.assertAlmostEqual(sv.composite, 0.88, places=3)
        self.assertEqual(sv.base_composite, 0.80)
        self.assertAlmostEqual(sv.novelty_bonus, 0.10, places=3)

    def test_apply_novelty_adjustment_penalizes_duplicate(self):
        sv = ScoreVector(
            raw={}, normalized={}, composite=0.80,
        )
        CompositeScorer.apply_novelty_adjustment(
            sv, novelty_score=0.0, duplicate_score=1.0,
            novelty_bonus_weight=0.10, duplicate_penalty=0.50,
        )
        # 0.80 * (1 + 0 - 0.50) = 0.40
        self.assertAlmostEqual(sv.composite, 0.40, places=3)
        self.assertAlmostEqual(sv.duplicate_penalty, 0.50, places=3)

    def test_apply_novelty_adjustment_neutral_when_both_zero(self):
        sv = ScoreVector(
            raw={}, normalized={}, composite=0.70,
        )
        CompositeScorer.apply_novelty_adjustment(
            sv, novelty_score=0.0, duplicate_score=0.0,
        )
        self.assertAlmostEqual(sv.composite, 0.70, places=3)
        self.assertEqual(sv.novelty_bonus, 0.0)
        self.assertEqual(sv.duplicate_penalty, 0.0)

    def test_scorevector_to_dict_includes_novelty_fields(self):
        sv = ScoreVector(composite=0.5, novelty_bonus=0.05, duplicate_penalty=0.1, base_composite=0.5)
        d = sv.to_dict()
        self.assertIn("novelty_bonus", d)
        self.assertIn("duplicate_penalty", d)
        self.assertIn("base_composite", d)


if __name__ == "__main__":
    unittest.main()
