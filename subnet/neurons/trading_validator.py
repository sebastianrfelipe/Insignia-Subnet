"""
Trading Validator — Strategy Performance Scoring

Validates trader miner strategies by scoring their real/paper trading
outcomes. Trading validators track positions in real-time, compute
risk-adjusted performance metrics, and assign scores for Yuma consensus.

Scoring Dimensions (9):
  - Realized P&L (absolute returns)
  - Omega Ratio (full distribution measure, captures tail behavior)
  - Max Drawdown (hard ceiling — breach = elimination)
  - Win Rate & Trade Quality (signal precision)
  - Consistency (rolling sub-window analysis)
  - Execution Quality (latency, reliability, slippage)
  - Annualized Volatility (cumulative realized volatility — lower = better)
  - Sharpe Ratio (risk-adjusted return per unit total volatility)
  - Sortino Ratio (risk-adjusted return per unit downside volatility)

Under the single paired mechanism, the unified ``PairedValidator``
(``neurons/validator.py``) scores the trading half of each pair using
``CompositeScorer.score_trading``; this module is the legacy standalone
trading validator retained for the emulator and demos. The cross-layer
feedback engine it references is legacy two-layer machinery.

Usage:
    python neurons/trading_validator.py --netuid <NETUID> --wallet.name <WALLET> --subtensor.network test
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

try:
    import bittensor as bt
except ImportError:
    bt = None

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from insignia.scoring import CompositeScorer, ExecutionMetrics, ScoreVector, WeightConfig
from insignia.incentive import (
    CopyTradeDetector,
    CrossLayerFeedbackEngine,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Trading-Validator] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy Performance Tracker
# ---------------------------------------------------------------------------

@dataclass
class StrategyTracker:
    """
    Tracks a single trader miner's strategy in real-time.

    Maintains position state, equity curve, execution telemetry, and all
    metrics needed for scoring. Updated incrementally as position updates
    and execution events stream in.
    """

    strategy_id: str
    miner_uid: str
    model_ids_used: List[str] = field(default_factory=list)

    # State
    positions: List[Dict] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)

    # Aggregates
    total_pnl: float = 0.0
    peak_equity: float = 100_000.0
    max_drawdown: float = 0.0
    start_time: float = 0.0
    last_update: float = 0.0

    # Execution telemetry accumulators
    _latency_samples: List[float] = field(default_factory=list)
    _order_reject_count: int = 0
    _cancel_count: int = 0
    _partial_fill_count: int = 0
    _stuck_order_count: int = 0
    _reconnect_count: int = 0
    _total_orders: int = 0
    _slippage_samples: List[float] = field(default_factory=list)
    _realized_fees: List[float] = field(default_factory=list)
    _turnover: float = 0.0

    # Flags
    eliminated: bool = False
    elimination_reason: str = ""

    def record_trade(self, trade: Dict):
        pnl = trade.get("pnl", 0.0)
        self.trades.append(trade)
        self.total_pnl += pnl

        equity = self.peak_equity + self.total_pnl
        self.equity_curve.append(equity)

        if equity > self.peak_equity:
            self.peak_equity = equity
        dd = (self.peak_equity - equity) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, dd)
        self.last_update = trade.get("timestamp", time.time())

        self._total_orders += 1
        if "slippage_bps" in trade:
            self._slippage_samples.append(trade["slippage_bps"])
        if "fee_pct" in trade:
            self._realized_fees.append(trade["fee_pct"])
        if "notional" in trade:
            self._turnover += abs(trade["notional"])

    def record_execution_event(self, event: Dict):
        """
        Record an execution telemetry event from the trader miner.

        Supported event types:
          - "latency": records end-to-end intent latency in ms
          - "reject": order rejected by exchange
          - "cancel": order cancelled
          - "partial_fill": order partially filled
          - "stuck": order stuck without response
          - "reconnect": WebSocket reconnection
        """
        etype = event.get("type", "")
        if etype == "latency":
            self._latency_samples.append(event.get("e2e_ms", 0.0))
        elif etype == "reject":
            self._order_reject_count += 1
        elif etype == "cancel":
            self._cancel_count += 1
        elif etype == "partial_fill":
            self._partial_fill_count += 1
        elif etype == "stuck":
            self._stuck_order_count += 1
        elif etype == "reconnect":
            self._reconnect_count += 1

    def build_execution_metrics(self) -> ExecutionMetrics:
        """Aggregate accumulated telemetry into an ExecutionMetrics snapshot."""
        avg_latency = (
            float(np.mean(self._latency_samples))
            if self._latency_samples
            else 0.0
        )
        avg_slippage = (
            float(np.mean(self._slippage_samples))
            if self._slippage_samples
            else 0.0
        )
        avg_fees = (
            float(np.mean(self._realized_fees))
            if self._realized_fees
            else 0.0
        )
        return ExecutionMetrics(
            end_to_end_intent_ms=avg_latency,
            order_reject_count=self._order_reject_count,
            cancel_count=self._cancel_count,
            partial_fill_count=self._partial_fill_count,
            stuck_order_count=self._stuck_order_count,
            reconnect_count=self._reconnect_count,
            total_orders=self._total_orders,
            slippage_bps=avg_slippage,
            realized_fees_pct=avg_fees,
            turnover=self._turnover,
        )

    def record_daily_return(self, ret: float):
        self.daily_returns.append(ret)

    def check_elimination(self, max_dd_limit: float = 0.20) -> bool:
        if self.max_drawdown >= max_dd_limit:
            self.eliminated = True
            self.elimination_reason = (
                f"Drawdown {self.max_drawdown:.2%} exceeds limit {max_dd_limit:.2%}"
            )
            return True
        return False


# ---------------------------------------------------------------------------
# Trading Validator (legacy standalone trading validator)
# ---------------------------------------------------------------------------

class TradingValidator:
    """
    Trading Validator neuron (legacy standalone trading validator).

    Manages continuous strategy tracking, periodic scoring, and
    cross-layer feedback for all registered trader miners.

    The validator:
      1. Receives position updates from trader miners (streaming)
      2. Tracks equity curves and risk metrics in real-time
      3. Computes scores at epoch boundaries (e.g., daily or weekly)
      4. Eliminates strategies that breach drawdown limits
      5. (Legacy) feeds performance data back via CrossLayerFeedbackEngine
      6. Computes Yuma consensus weights

    Under the single paired mechanism this role is subsumed by
    ``PairedValidator``.
    """

    def __init__(
        self,
        scorer: CompositeScorer | None = None,
        max_drawdown_limit: float = 0.20,
    ):
        self.scorer = scorer or CompositeScorer()
        self.max_drawdown_limit = max_drawdown_limit
        self.copy_detector = CopyTradeDetector()
        self.feedback_engine = CrossLayerFeedbackEngine()

        self.trackers: Dict[str, StrategyTracker] = {}
        self.epoch_scores: Dict[str, ScoreVector] = {}
        self.current_epoch: int = 0
        self.epoch_history: List[Dict] = []

    def register_strategy(
        self,
        miner_uid: str,
        strategy_id: str,
        model_ids: List[str],
    ):
        self.trackers[miner_uid] = StrategyTracker(
            strategy_id=strategy_id,
            miner_uid=miner_uid,
            model_ids_used=model_ids,
            start_time=time.time(),
        )
        logger.info("Registered strategy %s for miner %s", strategy_id, miner_uid)

    def process_position_update(self, miner_uid: str, update: Dict):
        """Process a real-time position update from a trader miner."""
        tracker = self.trackers.get(miner_uid)
        if not tracker:
            logger.warning("Unknown miner %s", miner_uid)
            return

        if tracker.eliminated:
            return

        if update.get("type") == "close":
            tracker.record_trade(update)
        elif update.get("type") in (
            "latency", "reject", "cancel", "partial_fill", "stuck", "reconnect",
        ):
            tracker.record_execution_event(update)

        if tracker.check_elimination(self.max_drawdown_limit):
            logger.warning(
                "Strategy %s ELIMINATED: %s",
                tracker.strategy_id,
                tracker.elimination_reason,
            )

    def score_epoch(self) -> Dict[str, Any]:
        """
        Score all active strategies for the current epoch.

        Returns epoch summary with per-miner scores and rankings.
        """
        logger.info("=" * 50)
        logger.info("Trading Epoch %d — Scoring %d strategies", self.current_epoch, len(self.trackers))

        scores = {}
        for miner_uid, tracker in self.trackers.items():
            if tracker.eliminated:
                scores[miner_uid] = ScoreVector(composite=0.0)
                continue

            trades_pnl = [t.get("pnl", 0.0) for t in tracker.trades]
            returns = np.array(trades_pnl) / 100_000 if trades_pnl else np.array([0.0])
            daily_ret = np.array(tracker.daily_returns) if tracker.daily_returns else returns

            exec_metrics = tracker.build_execution_metrics()

            score = self.scorer.score_trading(
                realized_pnl=tracker.total_pnl,
                returns=returns,
                max_dd=tracker.max_drawdown,
                trades=trades_pnl,
                daily_returns=daily_ret,
                execution_metrics=exec_metrics,
            )
            scores[miner_uid] = score
            self.epoch_scores[miner_uid] = score

            for mid in tracker.model_ids_used:
                self.feedback_engine.record_l2_performance(mid, score.composite)

        ranked = sorted(scores.items(), key=lambda x: x[1].composite, reverse=True)
        weights = self._compute_weights(ranked)

        # Copy-trade detection
        miner_uids = list(self.trackers.keys())
        copy_flags = {}
        for i, uid_a in enumerate(miner_uids):
            for uid_b in miner_uids[i + 1:]:
                ta = self.trackers[uid_a]
                tb = self.trackers[uid_b]
                is_copy, corr = self.copy_detector.detect(ta.positions, tb.positions)
                if is_copy:
                    copy_flags[(uid_a, uid_b)] = corr
                    logger.warning(
                        "COPY DETECTED: %s <-> %s (corr=%.2f)",
                        uid_a, uid_b, corr,
                    )

        epoch_summary = {
            "epoch": self.current_epoch,
            "n_strategies": len(self.trackers),
            "n_active": sum(1 for t in self.trackers.values() if not t.eliminated),
            "n_eliminated": sum(1 for t in self.trackers.values() if t.eliminated),
            "n_copy_flags": len(copy_flags),
            "scores": {
                uid: {
                    "composite": round(sv.composite, 4),
                    "breakdown": {k: round(v, 4) for k, v in sv.normalized.items()},
                }
                for uid, sv in ranked
            },
            "weights": weights,
        }

        self.epoch_history.append(epoch_summary)
        self.current_epoch += 1

        for rank, (uid, sv) in enumerate(ranked):
            status = "ELIMINATED" if self.trackers[uid].eliminated else f"score={sv.composite:.4f}"
            logger.info("  Rank %d: %s — %s", rank + 1, uid, status)

        return epoch_summary

    def _compute_weights(
        self, ranked: List[Tuple[str, ScoreVector]]
    ) -> Dict[str, float]:
        active = [(uid, sv) for uid, sv in ranked if sv.composite > 0]
        if not active:
            return {}
        scores = np.array([sv.composite for _, sv in active])
        if scores.sum() < 1e-12:
            return {uid: 1.0 / len(active) for uid, _ in active}
        normalized = scores / scores.sum()
        return {uid: float(w) for (uid, _), w in zip(active, normalized)}

    def get_l1_feedback(self) -> Dict[str, float]:
        """
        Get cross-layer feedback adjustments for L1 model scores.

        Returns model_id -> adjustment_multiplier mapping.
        """
        adjustments = {}
        for tracker in self.trackers.values():
            for mid in tracker.model_ids_used:
                adj = self.feedback_engine.compute_adjustment(mid)
                adjustments[mid] = adj
        return adjustments


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    """Demonstrate trading validator scoring with multiple trader miners."""
    logger.info("=" * 60)
    logger.info("Insignia Trading Validator — Demo Mode")
    logger.info("=" * 60)

    from neurons.trader_miner import TraderMiner, PaperTradingEngine, Side
    from neurons.researcher_miner import ResearcherMiner, generate_demo_data

    logger.info("Step 1: Researchers train models...")
    models = {}
    for i in range(3):
        data = generate_demo_data(n_samples=3000, seed=42 + i)
        miner = ResearcherMiner()
        sub = miner.train_and_submit(data)
        model_id = f"model_{i}"
        models[model_id] = sub["model_artifact"]
        logger.info("  Trained model %s", model_id)

    logger.info("Step 2: Initialize trader miners with different strategies...")
    validator = TradingValidator()
    traders = {}

    for j in range(4):
        engine = PaperTradingEngine(
            initial_capital=100_000,
            max_position_pct=0.05 + j * 0.02,
            max_drawdown_pct=0.20,
        )
        trader = TraderMiner(engine=engine)
        for mid, artifact in list(models.items())[:2 + (j % 2)]:
            trader.load_assigned_model(mid, artifact)

        miner_uid = f"trader_{j}"
        traders[miner_uid] = trader
        validator.register_strategy(
            miner_uid=miner_uid,
            strategy_id=trader.strategy_id,
            model_ids=list(trader.models.keys()),
        )

    logger.info("Step 3: Simulate paper trading across all miners...")
    rng = np.random.RandomState(99)
    price = 50000.0

    for step in range(300):
        ret = rng.normal(0.0001, 0.003)
        price *= (1 + ret)
        features = rng.normal(0, 1, 15)
        features[0] = ret
        ts = time.time() + step * 3600

        for uid, trader in traders.items():
            update = trader.execute_step("BTC-USDT-PERP", price, features, ts)
            if update:
                validator.process_position_update(uid, update)

    logger.info("Step 4: Score epoch...")
    epoch_result = validator.score_epoch()

    logger.info("\n--- Epoch Summary ---")
    logger.info("  Active: %d", epoch_result["n_active"])
    logger.info("  Eliminated: %d", epoch_result["n_eliminated"])

    logger.info("\n--- Cross-layer Feedback Adjustments (legacy) ---")
    for mid, adj in validator.get_l1_feedback().items():
        logger.info("  %s: %.4f", mid, adj)

    return epoch_result


if __name__ == "__main__":
    demo()
