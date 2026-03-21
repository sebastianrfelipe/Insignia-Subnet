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
#
# L2 scoring evaluates real/paper trading outcomes rather than model
# predictions. The seven metrics below capture complementary dimensions
# of strategy quality:
#
#   1. Realized P&L (20%)        — raw profitability vs. baseline
#   2. Omega Ratio (15%)         — full-distribution risk (tail behavior)
#   3. Max Drawdown (15%)        — peak-to-trough loss; hard elimination threshold
#   4. Win Rate (10%)            — signal precision
#   5. Consistency (15%)         — rolling sub-window steadiness
#   6. Model Attribution (10%)   — credit for using strong L1 models
#   7. Execution Quality (15%)   — latency, reliability, and slippage
#
# Max Drawdown reuses the L1 `max_drawdown_score` function applied to the
# L2 equity curve. In L2 scoring it additionally serves as a hard ceiling:
# strategies that breach the drawdown limit (default 20%) are immediately
# eliminated and receive a composite score of zero.
#
# Model Attribution is computed externally by `ModelAttributionEngine` in
# `neurons/l2_validator.py` and passed into `score_l2` as a pre-computed
# float in [0, 1]. It rewards L2 miners who select and combine L1 models
# with strong deployment track records, creating incentive alignment
# between the two layers.
#
# Execution Quality evaluates the strategy's infrastructure health —
# how cleanly and efficiently it interacts with the exchange. Strategies
# with high latency, frequent order rejects, or excessive slippage are
# penalized even if their P&L looks good, because poor execution quality
# signals fragility that will degrade under real market conditions.
# ---------------------------------------------------------------------------

def realized_pnl_score(pnl: float, baseline: float = 0.0) -> float:
    """
    Realized P&L Score — absolute return quality relative to a baseline.

    Measures the strategy's raw profitability by comparing its realized
    profit/loss against a baseline return (typically buy-and-hold or zero).
    This is the most direct measure of whether a strategy generates
    economic value.

    Calculation:
        score = clamp((pnl - baseline) / max(|baseline|, 1.0), 0, 1)

    Strategies at or below the baseline receive a score of zero. The
    denominator scales by the absolute baseline value so that the metric
    is meaningful across different capital levels and market conditions.
    The floor of 1.0 in the denominator prevents division-by-zero when
    the baseline is zero (e.g., when scoring absolute P&L without a
    benchmark).

    Args:
        pnl: Realized profit/loss of the strategy in quote currency.
        baseline: Reference return to beat (default 0.0 = absolute profit).

    Returns:
        Float in [0, 1] where 0 = at or below baseline, 1 = strong
        outperformance.

    Weight: 20% of L2 composite score (highest single weight).
    """
    if pnl <= baseline:
        return 0.0
    return float(min(1.0, (pnl - baseline) / max(abs(baseline), 1.0)))


def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    Omega Ratio — full-distribution risk measure capturing tail behavior.

    Unlike Sharpe ratio (which only considers mean and variance), the Omega
    ratio captures the *entire* return distribution, including skewness and
    kurtosis (fat tails). This is critical for crypto trading strategies
    where returns are rarely normally distributed and tail risk is the
    primary destroyer of capital.

    Calculation:
        Omega = sum(max(r_i - threshold, 0)) / sum(max(threshold - r_i, 0))

    In plain terms: the ratio of probability-weighted gains above the
    threshold to probability-weighted losses below it. An Omega > 1
    indicates the strategy's gain mass exceeds its loss mass at the
    chosen threshold.

    The raw value is capped at 10.0 to prevent degenerate cases (e.g.,
    a strategy with a single winning trade and no losses) from dominating
    the composite score.

    Args:
        returns: Array of per-trade or per-period returns (as decimals,
            e.g., 0.02 for a 2% return).
        threshold: Minimum acceptable return threshold (default 0.0).

    Returns:
        Float in [0, 10] where higher = better risk-adjusted gains.
        During normalization, this is scaled to [0, 1] by dividing by 3.0
        (so Omega >= 3.0 maps to a perfect normalized score).

    Weight: 15% of L2 composite score.
    """
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    if len(losses) == 0 or np.sum(losses) < 1e-12:
        return 10.0  # cap
    return float(min(10.0, np.sum(gains) / np.sum(losses)))


def win_rate(trades: List[float]) -> float:
    """
    Win Rate — signal precision measuring the fraction of profitable trades.

    A straightforward but important metric that captures the strategy's
    ability to generate positive-expectancy signals. While a high win rate
    alone doesn't guarantee profitability (a strategy can win 90% of
    trades but lose money if losses are large), it penalizes low-conviction
    noise trading where a strategy enters and exits positions without
    meaningful directional skill.

    In the composite score, win rate carries a deliberately lower weight
    (10%) because profitable strategies can legitimately have moderate
    win rates (e.g., trend-following with ~40% wins but large risk/reward).
    Its primary role is to filter out strategies that generate excessive
    churn without directional edge.

    Calculation:
        win_rate = count(trade_pnl > 0) / total_trades

    Args:
        trades: List of per-trade P&L values (positive = profit,
            negative = loss).

    Returns:
        Float in [0, 1] where 0 = no winning trades, 1 = all trades
        profitable.

    Weight: 10% of L2 composite score.
    """
    if not trades:
        return 0.0
    return float(sum(1 for t in trades if t > 0) / len(trades))


def consistency_score(
    daily_returns: np.ndarray,
    window_days: int = 7,
) -> float:
    """
    Consistency Score — rolling sub-window analysis penalizing spike-then-collapse.

    Measures whether a strategy performs *steadily* over time rather than
    generating returns through a single lucky streak. This is the second-
    highest weighted L2 metric (20%) because consistency is the strongest
    predictor of a strategy's viability in live deployment.

    The metric divides the return history into non-overlapping weekly
    windows and computes a Sharpe-like ratio (mean/std) for each window.
    It then scores two properties:

    1. **Positive fraction**: What fraction of windows have positive
       risk-adjusted returns? Higher = more consistently profitable.
    2. **Stability (1 - CV)**: How stable are the per-window Sharpe
       ratios? Lower coefficient of variation = more predictable
       performance across time periods.

    The final score is the product of these two factors:
        consistency = positive_frac * max(0, 1 - CV)

    This product structure means both properties must be present: a
    strategy that is profitable in all windows but wildly variable still
    scores moderately, and a stable strategy that is consistently flat
    or negative also scores poorly.

    Args:
        daily_returns: Array of daily return values (as decimals).
        window_days: Size of each rolling sub-window in days (default 7).

    Returns:
        Float in [0, 1] where 0 = inconsistent/insufficient data,
        1 = perfectly consistent positive returns across all windows.

    Weight: 15% of L2 composite score.
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


@dataclass
class ExecutionMetrics:
    """
    Aggregated execution telemetry for an L2 strategy over one epoch.

    These fields mirror the operational metrics a production trading system
    tracks: latency breakdowns across the order lifecycle, reliability
    counters for infrastructure failures, and execution performance
    measures that capture real-world slippage and cost.

    Validators populate this from the continuous position-update stream
    and exchange-level telemetry reported by L2 miners.
    """

    # Latency (milliseconds) — measured across the order lifecycle
    ws_message_lag_ms: float = 0.0
    decision_to_submit_ms: float = 0.0
    submit_to_ack_ms: float = 0.0
    ack_to_fill_ms: float = 0.0
    end_to_end_intent_ms: float = 0.0

    # Reliability — infrastructure failure counters
    order_reject_count: int = 0
    cancel_count: int = 0
    partial_fill_count: int = 0
    stuck_order_count: int = 0
    reconnect_count: int = 0
    total_orders: int = 0

    # Performance — execution cost and quality
    slippage_bps: float = 0.0
    realized_fees_pct: float = 0.0
    turnover: float = 0.0


def execution_quality_score(metrics: ExecutionMetrics) -> float:
    """
    Execution Quality Score — composite measure of infrastructure health.

    Evaluates three orthogonal dimensions of execution quality and combines
    them into a single score:

    1. **Latency sub-score (40%)**: Measures how quickly the strategy
       completes the full order lifecycle (decision -> submit -> ack ->
       fill). Uses the end-to-end intent latency as the primary signal,
       with an exponential decay above a target threshold. Lower latency
       is critical because stale signals degrade P&L in fast-moving
       crypto markets.

       Target: <= 200ms end-to-end. Decays exponentially above target.

    2. **Reliability sub-score (30%)**: Measures infrastructure
       stability via failure rates across the epoch. Penalizes order
       rejects, stuck orders, partial fills, and WebSocket reconnects
       relative to total order volume. A high reject or reconnect rate
       signals fragile infrastructure that will break under load.

       failure_rate = (rejects + stuck + partials + reconnects) / orders
       reliability = max(0, 1 - failure_rate * 5)

       The 5x multiplier means a 20% failure rate zeroes the sub-score.

    3. **Slippage sub-score (30%)**: Measures realized execution cost
       in basis points. Lower slippage means better execution routing,
       smarter order sizing, and less market impact. Excessive slippage
       erodes theoretical edge.

       Target: <= 5 bps. Decays exponentially above target.

    The three sub-scores are combined as a weighted average:
        execution_quality = 0.40 * latency + 0.30 * reliability + 0.30 * slippage

    Args:
        metrics: ExecutionMetrics dataclass containing aggregated
            telemetry for the scoring epoch.

    Returns:
        Float in [0, 1] where 0 = poor execution infrastructure,
        1 = clean, fast, reliable execution.

    Weight: 15% of L2 composite score.
    """
    e2e = metrics.end_to_end_intent_ms
    target_latency_ms = 200.0
    if e2e <= target_latency_ms:
        latency_sub = 1.0
    else:
        latency_sub = math.exp(-(e2e - target_latency_ms) / target_latency_ms)

    total = max(metrics.total_orders, 1)
    failures = (
        metrics.order_reject_count
        + metrics.stuck_order_count
        + metrics.partial_fill_count
        + metrics.reconnect_count
    )
    failure_rate = failures / total
    reliability_sub = max(0.0, 1.0 - failure_rate * 5.0)

    target_slippage_bps = 5.0
    slip = abs(metrics.slippage_bps)
    if slip <= target_slippage_bps:
        slippage_sub = 1.0
    else:
        slippage_sub = math.exp(-(slip - target_slippage_bps) / target_slippage_bps)

    return float(
        0.40 * latency_sub + 0.30 * reliability_sub + 0.30 * slippage_sub
    )


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
    l2_realized_pnl: float = 0.20
    l2_omega: float = 0.15
    l2_max_drawdown: float = 0.15
    l2_win_rate: float = 0.10
    l2_consistency: float = 0.15
    l2_model_attribution: float = 0.10
    l2_execution_quality: float = 0.15


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
        execution_metrics: ExecutionMetrics | None = None,
        baseline_pnl: float = 0.0,
    ) -> ScoreVector:
        """
        Compute the Layer 2 composite score for a strategy miner.

        All inputs are derived from real or paper trading outcomes — no
        simulation involved. This is the empirical proof layer that closes
        the gap between backtested model quality (L1) and deployment
        viability.

        The composite score is a weighted sum of seven normalized metrics:

            composite = 0.20 * realized_pnl
                      + 0.15 * omega
                      + 0.15 * max_drawdown
                      + 0.10 * win_rate
                      + 0.15 * consistency
                      + 0.10 * model_attribution
                      + 0.15 * execution_quality

        Args:
            realized_pnl: Total P&L of the strategy in quote currency
                over the scoring epoch.
            returns: Per-trade or per-period returns as a numpy array
                (decimals, e.g., 0.02 for 2%). Used to compute the
                Omega ratio.
            max_dd: Maximum drawdown observed over the epoch, as a
                fraction in [0, 1]. Strategies breaching the hard
                limit (default 0.20) are eliminated before scoring.
            trades: List of per-trade P&L values. Used to compute win
                rate.
            daily_returns: Array of daily returns. Used to compute the
                consistency score via rolling sub-window analysis.
            model_attribution_score: Pre-computed attribution score in
                [0, 1] from ModelAttributionEngine, reflecting the
                quality of L1 models the strategy relies on.
            execution_metrics: Aggregated execution telemetry for the
                epoch (latency, reliability, slippage). If None, a
                default ExecutionMetrics() is used, which yields a
                perfect execution quality score — appropriate for
                paper trading where execution infrastructure is
                simulated.
            baseline_pnl: Reference P&L for the realized_pnl metric
                (default 0.0 = absolute profitability).

        Returns:
            ScoreVector containing raw metric values, normalized values
            (all in [0, 1]), and the weighted composite score.
        """
        exec_metrics = execution_metrics or ExecutionMetrics()

        raw = {
            "realized_pnl": realized_pnl_score(realized_pnl, baseline_pnl),
            "omega": omega_ratio(returns),
            "max_drawdown": max_dd,
            "win_rate": win_rate(trades),
            "consistency": consistency_score(daily_returns),
            "model_attribution": model_attribution_score,
            "execution_quality": execution_quality_score(exec_metrics),
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
            + w.l2_execution_quality * normalized["execution_quality"]
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
        """
        Map raw L2 metrics to [0, 1] where 1 is best.

        Normalization transforms per metric:
          - realized_pnl: Already in [0, 1] from `realized_pnl_score()`.
            Clamped as a safety guard.
          - omega: Divided by 3.0, so an Omega ratio of 3+ maps to a
            perfect 1.0. This threshold reflects that Omega >= 3 is
            exceptional for crypto strategies and avoids letting
            degenerate edge cases (Omega = 10) dominate.
          - max_drawdown: Inverted (1 - dd) so lower drawdown = higher
            score. A 0% drawdown scores 1.0; a 100% drawdown scores 0.0.
          - win_rate: Already in [0, 1] from `win_rate()`. Clamped as
            a safety guard.
          - consistency: Already in [0, 1] from `consistency_score()`.
            Clamped as a safety guard.
          - model_attribution: Already in [0, 1] from
            `ModelAttributionEngine`. Clamped as a safety guard.
          - execution_quality: Already in [0, 1] from
            `execution_quality_score()`. The sub-scores (latency,
            reliability, slippage) are each individually normalized
            before combination. Clamped as a safety guard.
        """
        omega_norm = min(1.0, raw["omega"] / 3.0)

        return {
            "realized_pnl": min(1.0, max(0.0, raw["realized_pnl"])),
            "omega": omega_norm,
            "max_drawdown": max(0.0, 1.0 - raw["max_drawdown"]),
            "win_rate": min(1.0, max(0.0, raw["win_rate"])),
            "consistency": min(1.0, max(0.0, raw["consistency"])),
            "model_attribution": min(1.0, max(0.0, raw["model_attribution"])),
            "execution_quality": min(1.0, max(0.0, raw["execution_quality"])),
        }
