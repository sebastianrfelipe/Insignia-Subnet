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

L1 Metric Design Philosophy:
    Metrics are computed across K rolling out-of-sample windows, producing
    a distribution of per-window scores. The validator-facing metric is:

        penalized_metric = mean(scores) - variance_penalty * std(scores)

    This formulation (borrowed from robust GBDT HPO) simultaneously rewards:
      - High absolute performance (mean)
      - Low variance / generalization across market regimes (std penalty)

    A separate generalization_gap = |in_sample - out_of_sample| metric
    directly measures overfitting without needing a complex detector.

L1 Scoring Vector (6 dimensions):
    1. penalized_f1:    mean_f1 - λ·std_f1         (directional quality + stability)
    2. penalized_sharpe: mean_sharpe - λ·std_sharpe (risk-adjusted returns + stability)
    3. max_drawdown:     worst drawdown across all windows
    4. generalization_gap: |train_f1 - val_f1|      (overfitting — minimize)
    5. feature_efficiency: penalizes excessive feature count
    6. latency:           inference speed
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score as sklearn_f1_score


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
# Rolling Window Helpers
# ---------------------------------------------------------------------------

def _split_rolling_windows(
    predictions: np.ndarray,
    actuals: np.ndarray,
    n_windows: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split predictions and actuals into K non-overlapping rolling windows."""
    if len(predictions) < n_windows * 2:
        return [(predictions, actuals)]
    chunk = len(predictions) // n_windows
    windows = []
    for i in range(n_windows):
        s = i * chunk
        e = s + chunk
        windows.append((predictions[s:e], actuals[s:e]))
    return windows


def window_f1(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Binary F1 score: prediction direction vs actual direction."""
    if len(predictions) == 0:
        return 0.0
    pred_cls = (np.sign(predictions) > 0).astype(int)
    actual_cls = (np.sign(actuals) > 0).astype(int)
    if len(np.unique(actual_cls)) < 2:
        return float(np.mean(pred_cls == actual_cls))
    return float(sklearn_f1_score(actual_cls, pred_cls, zero_division=0.0))


def window_sharpe(
    predictions: np.ndarray,
    actuals: np.ndarray,
    annualization: float = np.sqrt(365 * 24),
) -> float:
    """Sharpe ratio of a position-return series within one window."""
    position_returns = predictions * actuals
    if len(position_returns) < 2 or np.std(position_returns) < 1e-12:
        return 0.0
    return float(annualization * np.mean(position_returns) / np.std(position_returns))


# ---------------------------------------------------------------------------
# Layer 1 Metric Definitions
# ---------------------------------------------------------------------------

def penalized_f1(
    predictions: np.ndarray,
    actuals: np.ndarray,
    n_windows: int = 5,
    variance_penalty: float = 0.5,
) -> Dict[str, float]:
    """
    Variance-penalized F1 across rolling out-of-sample windows.

    Returns dict with mean_f1, std_f1, and the penalized score.
    The formulation mean - λ·std simultaneously rewards high directional
    accuracy and low variance across market regimes.
    """
    windows = _split_rolling_windows(predictions, actuals, n_windows)
    f1_scores = [window_f1(p, a) for p, a in windows]
    arr = np.array(f1_scores)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    penalized = mean_val - variance_penalty * std_val
    return {
        "mean_f1": mean_val,
        "std_f1": std_val,
        "penalized_f1": max(0.0, penalized),
        "window_f1_scores": f1_scores,
    }


def penalized_sharpe(
    predictions: np.ndarray,
    actuals: np.ndarray,
    n_windows: int = 5,
    variance_penalty: float = 0.5,
) -> Dict[str, float]:
    """
    Variance-penalized Sharpe ratio across rolling out-of-sample windows.

    Same mean - λ·std formulation applied to per-window Sharpe ratios.
    Promotes models with consistently good risk-adjusted returns rather
    than models that spike in one regime and collapse in another.
    """
    windows = _split_rolling_windows(predictions, actuals, n_windows)
    sharpe_scores = [window_sharpe(p, a) for p, a in windows]
    arr = np.array(sharpe_scores)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    penalized = mean_val - variance_penalty * std_val
    return {
        "mean_sharpe": mean_val,
        "std_sharpe": std_val,
        "penalized_sharpe": penalized,
        "window_sharpe_scores": sharpe_scores,
    }


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


def generalization_gap(
    in_sample_f1: float,
    out_of_sample_f1: float,
) -> float:
    """
    Direct measure of overfitting: |train_f1 - val_f1|.

    Simpler and more interpretable than the previous OverfittingDetector.
    A model that memorizes training data will have a large gap. A model
    that generalizes will have a small gap. This is the single most
    important metric for GBDTs on financial time-series.

    Returns value in [0, 1] — lower is better.
    """
    return float(min(1.0, abs(in_sample_f1 - out_of_sample_f1)))


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


# Legacy aliases kept for backward compatibility
def directional_accuracy(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Fraction of correct directional predictions. Use penalized_f1 for scoring."""
    if len(predictions) == 0:
        return 0.0
    return float(np.mean(np.sign(predictions) == np.sign(actuals)))


def simulated_sharpe(
    predictions: np.ndarray,
    actuals: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization: float = np.sqrt(365 * 24),
) -> float:
    """Legacy single-window Sharpe. Use penalized_sharpe for scoring."""
    position_returns = predictions * actuals
    if len(position_returns) < 2 or np.std(position_returns) < 1e-12:
        return 0.0
    excess = position_returns - risk_free_rate
    return float(annualization * np.mean(excess) / np.std(excess))


def stability_score(
    predictions: np.ndarray,
    actuals: np.ndarray,
    n_windows: int = 5,
) -> float:
    """Legacy stability metric. Subsumed by the mean-std formulation in penalized_f1/sharpe."""
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


class OverfittingDetector(ABC):
    """
    Abstract base for overfitting detection. Kept for backward compatibility.
    The new scoring uses generalization_gap() directly instead.
    """

    @abstractmethod
    def evaluate(
        self,
        in_sample_score: float,
        out_of_sample_score: float,
        model_complexity: Dict,
    ) -> float:
        ...


class ReferenceOverfittingDetector(OverfittingDetector):
    """
    Legacy overfitting detector kept for backward compatibility.
    New scoring path uses generalization_gap() which is simpler
    and directly matches the GBDT HPO objective: minimize |train - val|.
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

    L1 Metric Vector (6 dimensions):
      penalized_f1:      mean_f1 - λ·std_f1 across rolling OOS windows
      penalized_sharpe:  mean_sharpe - λ·std_sharpe across rolling OOS windows
      max_drawdown:      worst peak-to-trough loss
      generalization_gap: |train_f1 - val_f1| — direct overfitting measure
      feature_efficiency: penalizes models needing too many features
      latency:           inference speed for deployment viability
    """

    # Layer 1 weights (6 params, must sum to 1.0)
    l1_penalized_f1: float = 0.25
    l1_penalized_sharpe: float = 0.25
    l1_max_drawdown: float = 0.15
    l1_generalization_gap: float = 0.20
    l1_feature_efficiency: float = 0.05
    l1_latency: float = 0.10

    # Layer 2 weights (must sum to 1.0)
    l2_realized_pnl: float = 0.25
    l2_omega: float = 0.20
    l2_max_drawdown: float = 0.15
    l2_win_rate: float = 0.10
    l2_consistency: float = 0.20
    l2_model_attribution: float = 0.10

    # Variance penalty coefficient (λ) for mean - λ·std formulation
    variance_penalty: float = 0.5
    # Number of rolling windows for mean/std computation
    n_eval_windows: int = 5


class CompositeScorer:
    """
    Aggregates per-metric scores into a single composite score for
    Yuma consensus weight-setting.

    The scorer is deterministic: given the same inputs and weights, it
    always produces the same output. This is critical for validator
    consensus — all validators running the same scorer on the same data
    must agree on miner rankings.

    L1 scoring uses the GBDT-optimized mean - λ·std formulation:
      - Metrics are computed per rolling OOS window
      - Mean rewards absolute performance
      - Std penalty rewards generalization across regimes
      - Generalization gap directly penalizes overfitting
    """

    def __init__(
        self,
        weights: WeightConfig | None = None,
        overfitting_detector: OverfittingDetector | None = None,
    ):
        self.weights = weights or WeightConfig()
        # Legacy detector kept for backward compat; new path uses generalization_gap
        self.overfitting_detector = overfitting_detector or ReferenceOverfittingDetector()

    def score_l1(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        equity_curve: np.ndarray,
        n_features: int,
        inference_ms: float,
        in_sample_f1: float,
        out_of_sample_f1: float,
        model_complexity: Dict,
    ) -> ScoreVector:
        """
        Compute the Layer 1 composite score for a single miner's model.

        Uses the variance-penalized mean - λ·std formulation for F1 and
        Sharpe, and a direct generalization gap for overfitting detection.
        """
        w = self.weights

        f1_result = penalized_f1(
            predictions, actuals,
            n_windows=w.n_eval_windows,
            variance_penalty=w.variance_penalty,
        )
        sharpe_result = penalized_sharpe(
            predictions, actuals,
            n_windows=w.n_eval_windows,
            variance_penalty=w.variance_penalty,
        )
        max_dd = max_drawdown_score(equity_curve)
        gen_gap = generalization_gap(in_sample_f1, out_of_sample_f1)
        feat_eff = feature_efficiency(n_features)
        latency = latency_score(inference_ms)

        raw = {
            "penalized_f1": f1_result["penalized_f1"],
            "mean_f1": f1_result["mean_f1"],
            "std_f1": f1_result["std_f1"],
            "penalized_sharpe": sharpe_result["penalized_sharpe"],
            "mean_sharpe": sharpe_result["mean_sharpe"],
            "std_sharpe": sharpe_result["std_sharpe"],
            "max_drawdown": max_dd,
            "generalization_gap": gen_gap,
            "feature_efficiency": feat_eff,
            "latency": latency,
        }

        normalized = self._normalize_l1(raw)

        composite = (
            w.l1_penalized_f1 * normalized["penalized_f1"]
            + w.l1_penalized_sharpe * normalized["penalized_sharpe"]
            + w.l1_max_drawdown * normalized["max_drawdown"]
            + w.l1_generalization_gap * normalized["generalization_gap"]
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

        penalized_f1:       already in [0, 1] (clamped)
        penalized_sharpe:   sigmoid transform centered at 1.0
        max_drawdown:       inverted (lower drawdown = higher score)
        generalization_gap: inverted (smaller gap = higher score)
        feature_efficiency: already in (0, 1]
        latency:            already in (0, 1]
        """
        sharpe_raw = raw["penalized_sharpe"]
        sharpe_norm = 1.0 / (1.0 + math.exp(-0.5 * (sharpe_raw - 1.0)))

        return {
            "penalized_f1": min(1.0, max(0.0, raw["penalized_f1"])),
            "penalized_sharpe": sharpe_norm,
            "max_drawdown": max(0.0, 1.0 - raw["max_drawdown"]),
            "generalization_gap": max(0.0, 1.0 - raw["generalization_gap"]),
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
