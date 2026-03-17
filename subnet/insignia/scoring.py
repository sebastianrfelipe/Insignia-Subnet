"""
Insignia Scoring Engine

Defines the composite scoring framework used by validators to evaluate
miner submissions across both layers. The framework is designed with
pluggable metric functions so that proprietary evaluation components
(e.g., the overfitting detection metric, benchmark dataset) remain
private while the scoring *structure* is fully transparent.

Architecture:
    MetricFn  ->  ScoreVector  ->  CompositeScorer  ->  float (0..1)
                                        |
                                  WeightConfig (configurable)

Each metric is a callable that takes model predictions + ground truth
and returns a float. The CompositeScorer normalizes, weights, and
aggregates them into a single score used for Yuma consensus.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Score Vector: the multi-dimensional evaluation result
# ---------------------------------------------------------------------------

@dataclass
class ScoreVector:
    """
    A named vector of metric scores produced by evaluating one miner.

    Each entry maps a metric name to its raw value and its normalized
    (0-1) value. The composite score is the weighted sum of normalized
    values.
    """

    raw: Dict[str, float] = field(default_factory=dict)
    normalized: Dict[str, float] = field(default_factory=dict)
    composite: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "raw": self.raw,
            "normalized": self.normalized,
            "composite": round(self.composite, 6),
        }


# ---------------------------------------------------------------------------
# Metric Definitions (Layer 1)
# ---------------------------------------------------------------------------

def penalized_f1(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Penalized F1: directional prediction quality with penalty for
    inconsistency across rolling sub-windows.

    Computes F1 score for the binary up/down classification, then
    applies a variance penalty (mean - lambda*std) across 5 rolling
    windows to reward cross-regime consistency.
    """
    if len(predictions) == 0:
        return 0.0
    pred_dir = (predictions > 0).astype(int)
    actual_dir = (actuals > 0).astype(int)

    tp = float(np.sum((pred_dir == 1) & (actual_dir == 1)))
    fp = float(np.sum((pred_dir == 1) & (actual_dir == 0)))
    fn = float(np.sum((pred_dir == 0) & (actual_dir == 1)))

    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    if precision + recall < 1e-12:
        return 0.0
    f1 = 2.0 * precision * recall / (precision + recall)

    n_windows = 5
    if len(predictions) < n_windows * 2:
        return f1

    chunk = len(predictions) // n_windows
    window_f1s = []
    for i in range(n_windows):
        s, e = i * chunk, (i + 1) * chunk
        p_w = (predictions[s:e] > 0).astype(int)
        a_w = (actuals[s:e] > 0).astype(int)
        tp_w = float(np.sum((p_w == 1) & (a_w == 1)))
        fp_w = float(np.sum((p_w == 1) & (a_w == 0)))
        fn_w = float(np.sum((p_w == 0) & (a_w == 1)))
        prec_w = tp_w / max(tp_w + fp_w, 1e-12)
        rec_w = tp_w / max(tp_w + fn_w, 1e-12)
        f1_w = 2.0 * prec_w * rec_w / max(prec_w + rec_w, 1e-12)
        window_f1s.append(f1_w)

    lam = 0.5
    penalty = lam * float(np.std(window_f1s))
    return float(max(0.0, f1 - penalty))


def penalized_sharpe(
    predictions: np.ndarray,
    actuals: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization: float = np.sqrt(365 * 24),
) -> float:
    """
    Penalized Sharpe Ratio: risk-adjusted returns with a consistency
    penalty across rolling sub-windows.

    Computes the annualized Sharpe ratio on a simulated portfolio, then
    penalizes for high variance of sub-window Sharpe ratios. This prevents
    models that achieve a high aggregate Sharpe through a single lucky
    window while being inconsistent elsewhere.
    """
    position_returns = predictions * actuals
    if len(position_returns) < 2 or np.std(position_returns) < 1e-12:
        return 0.0
    excess = position_returns - risk_free_rate
    sharpe = float(annualization * np.mean(excess) / np.std(excess))

    n_windows = 5
    if len(position_returns) < n_windows * 2:
        return sharpe

    chunk = len(position_returns) // n_windows
    window_sharpes = []
    for i in range(n_windows):
        s, e = i * chunk, (i + 1) * chunk
        w = position_returns[s:e]
        if np.std(w) < 1e-12:
            window_sharpes.append(0.0)
        else:
            window_sharpes.append(float(np.mean(w) / np.std(w)))

    lam = 0.3
    penalty = lam * float(np.std(window_sharpes))
    return float(sharpe - penalty)


def max_drawdown_score(equity_curve: np.ndarray) -> float:
    """
    Maximum drawdown of an equity curve. Returns a value in [0, 1] where
    0 = no drawdown and 1 = total loss. Validators penalize higher values.
    """
    if len(equity_curve) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / np.maximum(peak, 1e-12)
    return float(np.max(drawdown))


def variance_score(
    predictions: np.ndarray,
    actuals: np.ndarray,
    n_windows: int = 5,
) -> float:
    """
    Variance Score: measures cross-regime consistency of model performance.

    Computes directional accuracy in each rolling sub-window and returns
    1 - coefficient_of_variation. Higher = more stable across market regimes.
    Models that only work in one regime (e.g., trending markets) score poorly.
    """
    if len(predictions) < n_windows * 2:
        return 0.0
    chunk_size = len(predictions) // n_windows
    accuracies = []
    for i in range(n_windows):
        start = i * chunk_size
        end = start + chunk_size
        p, a = predictions[start:end], actuals[start:end]
        acc = float(np.mean(np.sign(p) == np.sign(a)))
        accuracies.append(acc)
    arr = np.array(accuracies)
    if np.mean(arr) < 1e-12:
        return 0.0
    cv = np.std(arr) / np.mean(arr)
    return float(max(0.0, 1.0 - cv))


def feature_efficiency(n_features_used: int, max_features: int = 200) -> float:
    """
    Penalizes models that require an excessive number of features.
    Returns value in (0, 1] — fewer features = higher score.
    """
    if n_features_used <= 0:
        return 0.0
    return float(1.0 / (1.0 + math.log(max(1, n_features_used) / 10)))


def latency_score(inference_ms: float, target_ms: float = 50.0) -> float:
    """
    Scores inference speed. Returns 1.0 if at or below target, decays
    exponentially above it.
    """
    if inference_ms <= target_ms:
        return 1.0
    return float(math.exp(-(inference_ms - target_ms) / target_ms))


class OverfittingDetector(ABC):
    """
    Abstract base for overfitting detection.

    The subnet owner's proprietary overfitting metric implements this
    interface. The metric itself is never exposed — only the resulting
    score (0 = no overfitting, 1 = severe overfitting) is published.

    For the hackathon demo, a reference implementation is provided that
    uses the gap between in-sample and out-of-sample performance.
    """

    @abstractmethod
    def evaluate(
        self,
        in_sample_score: float,
        out_of_sample_score: float,
        model_complexity: Dict,
    ) -> float:
        """Return overfitting score in [0, 1]."""
        ...


class ReferenceOverfittingDetector(OverfittingDetector):
    """
    Reference (non-proprietary) overfitting detector.

    Uses the normalized gap between in-sample and out-of-sample accuracy
    as a proxy for overfitting severity. The production validator replaces
    this with a proprietary metric tuned for GBDTs on financial time-series.
    """

    def __init__(self, gap_threshold: float = 0.15, decay_rate: float = 5.0):
        self.gap_threshold = gap_threshold
        self.decay_rate = decay_rate

    def evaluate(
        self,
        in_sample_score: float,
        out_of_sample_score: float,
        model_complexity: Dict,
    ) -> float:
        gap = max(0.0, in_sample_score - out_of_sample_score)
        if gap <= self.gap_threshold:
            return 0.0
        excess = gap - self.gap_threshold
        penalty = 1.0 - math.exp(-self.decay_rate * excess)

        n_trees = model_complexity.get("n_estimators", 100)
        depth = model_complexity.get("max_depth", 6)
        complexity_factor = min(1.0, (n_trees * depth) / 5000)
        return float(min(1.0, penalty * (0.7 + 0.3 * complexity_factor)))


# ---------------------------------------------------------------------------
# Layer 2 Metrics
# ---------------------------------------------------------------------------

def realized_pnl_score(pnl: float, baseline: float = 0.0) -> float:
    """Normalize realized P&L relative to a baseline (e.g., buy-and-hold)."""
    if pnl <= baseline:
        return 0.0
    return float(min(1.0, (pnl - baseline) / max(abs(baseline), 1.0)))


def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    Omega ratio: probability-weighted gains / probability-weighted losses.
    Captures the full return distribution, not just mean/variance.
    """
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    if len(losses) == 0 or np.sum(losses) < 1e-12:
        return 10.0  # cap
    return float(min(10.0, np.sum(gains) / np.sum(losses)))


def win_rate(trades: List[float]) -> float:
    """Fraction of trades that were profitable."""
    if not trades:
        return 0.0
    return float(sum(1 for t in trades if t > 0) / len(trades))


def consistency_score(
    daily_returns: np.ndarray,
    window_days: int = 7,
) -> float:
    """
    Rolling sub-window analysis. Penalizes strategies that spike then
    collapse. Returns 1.0 for perfectly consistent positive returns.
    """
    if len(daily_returns) < window_days * 2:
        return 0.0
    n_windows = len(daily_returns) // window_days
    window_sharpes = []
    for i in range(n_windows):
        start = i * window_days
        w = daily_returns[start : start + window_days]
        if np.std(w) < 1e-12:
            window_sharpes.append(0.0)
        else:
            window_sharpes.append(float(np.mean(w) / np.std(w)))
    arr = np.array(window_sharpes)
    positive_frac = np.mean(arr > 0)
    if np.mean(np.abs(arr)) < 1e-12:
        return 0.0
    cv = np.std(arr) / np.mean(np.abs(arr))
    return float(positive_frac * max(0.0, 1.0 - cv))


# ---------------------------------------------------------------------------
# Composite Scorer
# ---------------------------------------------------------------------------

@dataclass
class WeightConfig:
    """
    Configurable weights for the composite score. Weights are published
    so miners know the evaluation priorities. Exact values are tuned per
    epoch and may be adjusted by the subnet owner.
    """

    # Layer 1 weights (must sum to 1.0)
    l1_penalized_f1: float = 0.20
    l1_penalized_sharpe: float = 0.20
    l1_max_drawdown: float = 0.15
    l1_variance_score: float = 0.15
    l1_overfitting_penalty: float = 0.15
    l1_feature_efficiency: float = 0.05
    l1_latency: float = 0.10

    # Layer 2 weights (must sum to 1.0)
    l2_realized_pnl: float = 0.25
    l2_omega: float = 0.20
    l2_max_drawdown: float = 0.15
    l2_win_rate: float = 0.10
    l2_consistency: float = 0.20
    l2_model_attribution: float = 0.10


class CompositeScorer:
    """
    Aggregates per-metric scores into a single composite score for
    Yuma consensus weight-setting.

    The scorer is deterministic: given the same inputs and weights, it
    always produces the same output. This is critical for validator
    consensus — all validators running the same scorer on the same data
    must agree on miner rankings.
    """

    def __init__(
        self,
        weights: WeightConfig | None = None,
        overfitting_detector: OverfittingDetector | None = None,
    ):
        self.weights = weights or WeightConfig()
        self.overfitting_detector = overfitting_detector or ReferenceOverfittingDetector()

    def score_l1(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        equity_curve: np.ndarray,
        n_features: int,
        inference_ms: float,
        in_sample_accuracy: float,
        out_of_sample_accuracy: float,
        model_complexity: Dict,
    ) -> ScoreVector:
        """
        Compute the Layer 1 composite score for a single miner's model.

        All metric computations are transparent. The only proprietary component
        is the OverfittingDetector implementation, which is pluggable.
        """
        raw = {
            "penalized_f1": penalized_f1(predictions, actuals),
            "penalized_sharpe": penalized_sharpe(predictions, actuals),
            "max_drawdown": max_drawdown_score(equity_curve),
            "variance_score": variance_score(predictions, actuals),
            "overfitting_penalty": self.overfitting_detector.evaluate(
                in_sample_accuracy, out_of_sample_accuracy, model_complexity
            ),
            "feature_efficiency": feature_efficiency(n_features),
            "latency": latency_score(inference_ms),
        }

        normalized = self._normalize_l1(raw)

        w = self.weights
        composite = (
            w.l1_penalized_f1 * normalized["penalized_f1"]
            + w.l1_penalized_sharpe * normalized["penalized_sharpe"]
            + w.l1_max_drawdown * normalized["max_drawdown"]
            + w.l1_variance_score * normalized["variance_score"]
            + w.l1_overfitting_penalty * normalized["overfitting_penalty"]
            + w.l1_feature_efficiency * normalized["feature_efficiency"]
            + w.l1_latency * normalized["latency"]
        )

        return ScoreVector(raw=raw, normalized=normalized, composite=composite)

    def score_l2(
        self,
        realized_pnl: float,
        returns: np.ndarray,
        max_dd: float,
        trades: List[float],
        daily_returns: np.ndarray,
        model_attribution_score: float,
        baseline_pnl: float = 0.0,
    ) -> ScoreVector:
        """
        Compute the Layer 2 composite score for a strategy miner.

        All inputs are derived from real or paper trading outcomes — no
        simulation involved. This is the empirical proof layer.
        """
        raw = {
            "realized_pnl": realized_pnl_score(realized_pnl, baseline_pnl),
            "omega": omega_ratio(returns),
            "max_drawdown": max_dd,
            "win_rate": win_rate(trades),
            "consistency": consistency_score(daily_returns),
            "model_attribution": model_attribution_score,
        }

        normalized = self._normalize_l2(raw)

        w = self.weights
        composite = (
            w.l2_realized_pnl * normalized["realized_pnl"]
            + w.l2_omega * normalized["omega"]
            + w.l2_max_drawdown * normalized["max_drawdown"]
            + w.l2_win_rate * normalized["win_rate"]
            + w.l2_consistency * normalized["consistency"]
            + w.l2_model_attribution * normalized["model_attribution"]
        )

        return ScoreVector(raw=raw, normalized=normalized, composite=composite)

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_l1(raw: Dict[str, float]) -> Dict[str, float]:
        """
        Map raw L1 metrics to [0, 1] where 1 is best.

        Penalized F1: already in [0, 1]
        Penalized Sharpe: sigmoid transform centered at 1.0
        Max drawdown: inverted (lower drawdown = higher score)
        Variance score: already in [0, 1]
        Overfitting penalty: inverted (lower penalty = higher score)
        Feature efficiency: already in (0, 1]
        Latency: already in (0, 1]
        """
        sharpe_raw = raw["penalized_sharpe"]
        sharpe_norm = 1.0 / (1.0 + math.exp(-0.5 * (sharpe_raw - 1.0)))

        return {
            "penalized_f1": min(1.0, max(0.0, raw["penalized_f1"])),
            "penalized_sharpe": sharpe_norm,
            "max_drawdown": max(0.0, 1.0 - raw["max_drawdown"]),
            "variance_score": min(1.0, max(0.0, raw["variance_score"])),
            "overfitting_penalty": max(0.0, 1.0 - raw["overfitting_penalty"]),
            "feature_efficiency": min(1.0, max(0.0, raw["feature_efficiency"])),
            "latency": min(1.0, max(0.0, raw["latency"])),
        }

    @staticmethod
    def _normalize_l2(raw: Dict[str, float]) -> Dict[str, float]:
        """Map raw L2 metrics to [0, 1] where 1 is best."""
        omega_norm = min(1.0, raw["omega"] / 3.0)

        return {
            "realized_pnl": min(1.0, max(0.0, raw["realized_pnl"])),
            "omega": omega_norm,
            "max_drawdown": max(0.0, 1.0 - raw["max_drawdown"]),
            "win_rate": min(1.0, max(0.0, raw["win_rate"])),
            "consistency": min(1.0, max(0.0, raw["consistency"])),
            "model_attribution": min(1.0, max(0.0, raw["model_attribution"])),
        }
