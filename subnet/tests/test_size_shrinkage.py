"""
Tests for sample-size confidence shrinkage on ratio metrics.

Verifies that omega_ratio, sharpe_ratio, sortino_ratio, and
penalized_sharpe all apply the √(N / (N + k)) shrinkage so that:
  - thin-sample luck cannot dominate (small N is shrunk toward 0),
  - large samples are unaffected (N >> k → factor ~1),
  - the shrinkage is monotonic in N for identical per-period edge,
  - the confidence_k parameter controls the shrinkage strength.
"""

import math
import unittest

import numpy as np

from insignia.scoring import (
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
    penalized_sharpe,
)


def _constant_edge_returns(n: int, edge: float = 0.01, noise: float = 0.005,
                          seed: int = 42) -> np.ndarray:
    """Generate n returns with a fixed positive edge and stable noise."""
    rng = np.random.default_rng(seed)
    return rng.normal(edge, noise, n)


class OmegaShrinkageTests(unittest.TestCase):
    def test_small_n_shrunk_below_large_n(self):
        small = np.array([0.02, 0.01, -0.005, 0.015, 0.01])
        large = np.tile(small, 20)  # 100 returns, same distribution
        s = omega_ratio(small)
        l = omega_ratio(large)
        self.assertGreater(l, s)

    def test_large_n_approaches_unshrunk(self):
        returns = np.array([0.02, 0.01, -0.005, 0.015, 0.01] * 100)
        val = omega_ratio(returns, confidence_k=30.0)
        # With N=500, shrinkage factor ~ sqrt(500/530) ≈ 0.972
        self.assertGreater(val, 0.9)

    def test_confidence_k_controls_strength(self):
        returns = np.array([0.02, 0.01, -0.005, 0.015, 0.01])
        weak = omega_ratio(returns, confidence_k=1.0)
        strong = omega_ratio(returns, confidence_k=1000.0)
        self.assertGreater(weak, strong)

    def test_no_loss_shrinkage_not_full_cap(self):
        # All gains, no losses: previously hit cap 10.0, now shrunk.
        returns = np.array([0.01, 0.02, 0.015])
        val = omega_ratio(returns)
        self.assertLess(val, 10.0)

    def test_zero_k_no_shrinkage(self):
        returns = np.array([0.02, 0.01, -0.005, 0.015, 0.01])
        val = omega_ratio(returns, confidence_k=0.0)
        # With k=0, shrinkage = sqrt(N/N) = 1.0, so no shrinkage.
        gains = returns[returns > 0]
        losses = -returns[returns <= 0]
        expected = min(10.0, float(np.sum(gains) / np.sum(losses)))
        self.assertAlmostEqual(val, expected, places=5)


class SharpeShrinkageTests(unittest.TestCase):
    def test_small_n_shrunk_below_large_n(self):
        # Use small edge so values don't hit the [−5, 10] cap, letting
        # the shrinkage difference be visible.
        small = _constant_edge_returns(5, edge=0.001, noise=0.01)
        large = _constant_edge_returns(200, edge=0.001, noise=0.01)
        s = sharpe_ratio(small)
        l = sharpe_ratio(large)
        self.assertGreater(l, s)

    def test_large_n_approaches_unshrunk(self):
        returns = _constant_edge_returns(500, edge=0.001, noise=0.01)
        val = sharpe_ratio(returns, confidence_k=30.0)
        # With N=500, shrinkage ≈ 0.972
        self.assertGreater(val, 0.9)

    def test_confidence_k_controls_strength(self):
        # Use high edge / low noise so Sharpe is clearly positive.
        returns = _constant_edge_returns(10, edge=0.01, noise=0.005)
        weak = sharpe_ratio(returns, confidence_k=1.0)
        strong = sharpe_ratio(returns, confidence_k=1000.0)
        self.assertGreater(weak, strong)

    def test_zero_k_no_shrinkage(self):
        returns = _constant_edge_returns(50, edge=0.001, noise=0.01)
        val = sharpe_ratio(returns, confidence_k=0.0)
        rf_daily = 0.0
        excess = returns - rf_daily
        expected = float(
            np.mean(excess) / np.std(excess) * np.sqrt(365)
        )
        expected = max(-5.0, min(10.0, expected))
        self.assertAlmostEqual(val, expected, places=3)


class SortinoShrinkageTests(unittest.TestCase):
    def test_small_n_shrunk_below_large_n(self):
        small = _constant_edge_returns(5)
        large = _constant_edge_returns(200)
        s = sortino_ratio(small)
        l = sortino_ratio(large)
        self.assertGreater(l, s)

    def test_confidence_k_controls_strength(self):
        returns = _constant_edge_returns(10)
        weak = sortino_ratio(returns, confidence_k=1.0)
        strong = sortino_ratio(returns, confidence_k=1000.0)
        self.assertGreater(weak, strong)

    def test_no_downside_shrinkage_not_full_cap(self):
        # All positive returns, no downside: previously hit 10.0, now shrunk.
        returns = np.array([0.01, 0.02, 0.015, 0.005])
        val = sortino_ratio(returns)
        self.assertLess(val, 10.0)

    def test_zero_k_no_shrinkage(self):
        returns = _constant_edge_returns(50)
        val = sortino_ratio(returns, confidence_k=0.0)
        excess = returns
        downside = np.minimum(excess, 0.0)
        downside_dev = float(np.sqrt(np.mean(downside ** 2)))
        if downside_dev < 1e-12:
            expected = 10.0
        else:
            expected = float(
                np.mean(excess) / downside_dev * np.sqrt(365)
            )
            expected = max(-5.0, min(15.0, expected))
        self.assertAlmostEqual(val, expected, places=3)


class PenalizedSharpeShrinkageTests(unittest.TestCase):
    def test_shrinkage_reduces_score_at_small_n(self):
        # With a fixed dataset, higher k → more shrinkage → lower score.
        # This directly tests the shrinkage mechanism without conflating
        # it with the sub-window variance penalty (which is non-monotonic
        # in N).
        rng = np.random.default_rng(123)
        preds = rng.normal(0.01, 0.02, 50)
        actuals = rng.normal(0.0, 0.02, 50)
        weak = penalized_sharpe(preds, actuals, confidence_k=1.0)
        strong = penalized_sharpe(preds, actuals, confidence_k=1000.0)
        self.assertGreater(weak, strong)

    def test_shrinkage_negligible_at_large_n(self):
        # At N >> k, the shrinkage factor ≈ 1, so k has little effect.
        rng = np.random.default_rng(123)
        preds = rng.normal(0.01, 0.02, 2000)
        actuals = rng.normal(0.0, 0.02, 2000)
        default = penalized_sharpe(preds, actuals)
        unshrunk = penalized_sharpe(preds, actuals, confidence_k=0.0)
        self.assertAlmostEqual(default, unshrunk, places=1)

    def test_zero_k_no_shrinkage(self):
        # With k=0, shrinkage factor = 1.0, so the only penalty is the
        # sub-window variance penalty. Verify against the full formula.
        rng = np.random.default_rng(123)
        preds = rng.normal(0.01, 0.02, 50)
        actuals = rng.normal(0.0, 0.02, 50)
        val = penalized_sharpe(preds, actuals, confidence_k=0.0)
        # Reproduce the full formula including sub-window penalty.
        position_returns = preds * actuals
        excess = position_returns
        annualization = math.sqrt(365 * 24)
        sharpe = float(annualization * np.mean(excess) / np.std(excess))
        n_windows = 5
        chunk = len(position_returns) // n_windows
        window_sharpes = []
        for i in range(n_windows):
            s, e = i * chunk, (i + 1) * chunk
            w = position_returns[s:e]
            if np.std(w) < 1e-12:
                window_sharpes.append(0.0)
            else:
                window_sharpes.append(float(np.mean(w) / np.std(w)))
        penalty = 0.3 * float(np.std(window_sharpes))
        expected = float(sharpe - penalty)
        self.assertAlmostEqual(val, expected, places=3)


if __name__ == "__main__":
    unittest.main()
