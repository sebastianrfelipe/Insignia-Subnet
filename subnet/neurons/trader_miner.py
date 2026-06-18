"""
Trader Miner — Trading Operations / Strategy Deployment (Paper Trading)

Template miner for the trader role of the Insignia subnet. Trader miners build
complete trading strategies — position sizing, entry/exit logic, risk limits,
execution timing — around a model's signals, then run those strategies against
real market conditions via paper trading.

Under the single paired mechanism the trader does NOT self-select models from a
promotion pool. Instead the validator assigns a researcher partner
(chain-seeded) and the trader runs its strategy on that paired model. This
removes the self-selection surface that previously enabled researcher/trader
collusion.

Strategy Scope:
  - Single-model strategies: use the assigned model directly
  - Multi-model ensembles: combine signals from multiple assigned models
  - Custom risk management: stop-losses, position sizing, drawdown limits
  - Execution timing: when to act on model signals

Paper Trading Engine:
  Uses real-time exchange mid-price + configurable slippage model.
  Conservative fill assumptions prevent paper trading manipulation.

Usage:
    python neurons/trader_miner.py --netuid <NETUID> --wallet.name <WALLET> --subtensor.network test
"""

from __future__ import annotations

import io
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np
import joblib

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from insignia.protocol import MinerRole

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Trader-Miner] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Position & Trade Types
# ---------------------------------------------------------------------------

class Side(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Position:
    instrument: str
    side: Side
    size: float
    entry_price: float
    entry_time: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0

    def update_price(self, price: float):
        self.current_price = price
        if self.side == Side.LONG:
            self.unrealized_pnl = (price - self.entry_price) * self.size
        elif self.side == Side.SHORT:
            self.unrealized_pnl = (self.entry_price - price) * self.size


@dataclass
class Trade:
    trade_id: str
    instrument: str
    side: str
    size: float
    entry_price: float
    exit_price: float
    entry_time: float
    exit_time: float
    pnl: float
    slippage_cost: float
    fees: float
    model_ids_used: List[str] = field(default_factory=list)

    @property
    def net_pnl(self) -> float:
        return self.pnl - self.slippage_cost - self.fees


# ---------------------------------------------------------------------------
# Slippage Model
# ---------------------------------------------------------------------------

@dataclass
class SlippageConfig:
    """
    Conservative slippage model for paper trading.

    Uses spread + volatility-based impact to simulate realistic fills.
    Intentionally conservative to prevent paper trading manipulation.
    """

    base_spread_bps: float = 2.0
    volatility_impact_factor: float = 0.5
    size_impact_factor: float = 0.1
    fee_bps: float = 5.0  # taker fee

    def compute_slippage(
        self,
        price: float,
        size: float,
        volatility: float = 0.02,
    ) -> float:
        spread_cost = price * (self.base_spread_bps / 10000) * size
        vol_cost = price * volatility * self.volatility_impact_factor * size * 0.01
        size_cost = price * (size ** 1.5) * self.size_impact_factor * 0.0001
        return spread_cost + vol_cost + size_cost

    def compute_fees(self, price: float, size: float) -> float:
        return price * size * (self.fee_bps / 10000)


# ---------------------------------------------------------------------------
# Paper Trading Engine
# ---------------------------------------------------------------------------

class PaperTradingEngine:
    """
    Simulates trade execution with realistic fill assumptions.

    Maintains a portfolio state, tracks positions, computes P&L,
    and enforces risk limits. All state changes are logged for
    validator audit.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        max_position_pct: float = 0.10,
        max_drawdown_pct: float = 0.20,
        slippage: SlippageConfig | None = None,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.slippage = slippage or SlippageConfig()

        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []
        self.equity_history: List[Tuple[float, float]] = []  # (timestamp, equity)
        self.peak_equity: float = initial_capital
        self.current_drawdown: float = 0.0
        self._killed: bool = False

    @property
    def equity(self) -> float:
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return self.capital + unrealized

    @property
    def realized_pnl(self) -> float:
        return sum(t.net_pnl for t in self.trade_history)

    def open_position(
        self,
        instrument: str,
        side: Side,
        price: float,
        size_pct: float,
        model_ids: List[str],
        timestamp: float | None = None,
    ) -> Optional[Position]:
        """Open a new position with slippage and risk checks."""
        if self._killed:
            logger.warning("Kill switch active — no new positions")
            return None

        if instrument in self.positions:
            logger.warning("Already have position in %s", instrument)
            return None

        size_pct = min(size_pct, self.max_position_pct)
        notional = self.equity * size_pct
        size = notional / price

        slippage_cost = self.slippage.compute_slippage(price, size)
        fill_price = price + (slippage_cost / size if side == Side.LONG else -slippage_cost / size)

        position = Position(
            instrument=instrument,
            side=side,
            size=size,
            entry_price=fill_price,
            entry_time=timestamp or time.time(),
            current_price=fill_price,
        )
        self.positions[instrument] = position
        self._record_equity(timestamp)

        logger.info(
            "OPEN %s %s %.4f @ %.2f (slippage=%.2f)",
            side.value, instrument, size, fill_price, slippage_cost,
        )
        return position

    def close_position(
        self,
        instrument: str,
        price: float,
        model_ids: List[str],
        timestamp: float | None = None,
    ) -> Optional[Trade]:
        """Close an existing position."""
        if instrument not in self.positions:
            return None

        pos = self.positions.pop(instrument)
        slippage_cost = self.slippage.compute_slippage(price, pos.size)
        fees = self.slippage.compute_fees(price, pos.size)

        if pos.side == Side.LONG:
            raw_pnl = (price - pos.entry_price) * pos.size
        else:
            raw_pnl = (pos.entry_price - price) * pos.size

        trade = Trade(
            trade_id=str(uuid.uuid4())[:8],
            instrument=instrument,
            side=pos.side.value,
            size=pos.size,
            entry_price=pos.entry_price,
            exit_price=price,
            entry_time=pos.entry_time,
            exit_time=timestamp or time.time(),
            pnl=raw_pnl,
            slippage_cost=slippage_cost,
            fees=fees,
            model_ids_used=model_ids,
        )

        self.capital += trade.net_pnl
        self.trade_history.append(trade)
        self._check_drawdown()
        self._record_equity(timestamp)

        logger.info(
            "CLOSE %s %s %.4f @ %.2f -> PnL=%.2f (net=%.2f)",
            trade.side, instrument, trade.size, price, raw_pnl, trade.net_pnl,
        )
        return trade

    def _check_drawdown(self):
        eq = self.equity
        if eq > self.peak_equity:
            self.peak_equity = eq
        self.current_drawdown = (self.peak_equity - eq) / self.peak_equity
        if self.current_drawdown >= self.max_drawdown_pct:
            self._killed = True
            logger.warning(
                "KILL SWITCH: drawdown %.2f%% exceeds limit %.2f%%",
                self.current_drawdown * 100,
                self.max_drawdown_pct * 100,
            )

    def _record_equity(self, timestamp: float | None = None):
        self.equity_history.append((timestamp or time.time(), self.equity))

    def get_performance_summary(self) -> Dict[str, Any]:
        """Compute performance summary for validator submission."""
        trades_pnl = [t.net_pnl for t in self.trade_history]
        if not trades_pnl:
            return {
                "realized_pnl": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "omega_ratio": 0.0,
            }

        returns = np.array(trades_pnl) / self.initial_capital
        winning = sum(1 for p in trades_pnl if p > 0)

        sharpe = 0.0
        if len(returns) > 1 and np.std(returns) > 1e-12:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))

        gains = returns[returns > 0]
        losses = np.abs(returns[returns <= 0])
        omega = float(np.sum(gains) / max(np.sum(losses), 1e-12))

        return {
            "realized_pnl": float(sum(trades_pnl)),
            "return_pct": float(sum(trades_pnl) / self.initial_capital * 100),
            "total_trades": len(trades_pnl),
            "win_rate": winning / len(trades_pnl),
            "max_drawdown_pct": float(self.current_drawdown),
            "sharpe_ratio": sharpe,
            "omega_ratio": omega,
            "avg_trade_pnl": float(np.mean(trades_pnl)),
            "total_fees": float(sum(t.fees for t in self.trade_history)),
            "total_slippage": float(sum(t.slippage_cost for t in self.trade_history)),
        }


# ---------------------------------------------------------------------------
# Trader Miner Strategy
# ---------------------------------------------------------------------------

class TraderMiner:
    """
    Trader miner. Runs its trading strategy on the model assigned by the
    validator's chain-seeded pairing, executing via the paper trading engine.

    The strategy logic here is a reference implementation. In production,
    miners are free to build any strategy — the validator only scores
    outcomes, not methodology.
    """

    role = MinerRole.TRADER

    def __init__(
        self,
        strategy_id: str | None = None,
        engine: PaperTradingEngine | None = None,
    ):
        self.strategy_id = strategy_id or str(uuid.uuid4())[:12]
        self.engine = engine or PaperTradingEngine()
        self.models: Dict[str, Any] = {}
        self.model_n_features: Dict[str, int] = {}
        self.position_log: List[Dict] = []

    def load_model(self, model_id: str, model_artifact: bytes):
        buf = io.BytesIO(model_artifact)
        model = joblib.load(buf)
        self.models[model_id] = model
        n_feat = self._infer_n_features(model)
        self.model_n_features[model_id] = n_feat
        logger.info("Loaded model %s (%d features)", model_id, n_feat)

    def load_assigned_model(self, model_id: str, model_artifact: bytes):
        """
        Load the model assigned by the validator's pairing for this generation.

        Unlike the old self-selection flow, the trader cannot pick which model
        it runs — it is the researcher half of the chain-seeded pair.
        """
        self.load_model(model_id, model_artifact)

    @staticmethod
    def _infer_n_features(model) -> int:
        if hasattr(model, "n_features_in_"):
            return int(model.n_features_in_)
        if hasattr(model, "named_steps"):
            for step in model.named_steps.values():
                if hasattr(step, "n_features_in_"):
                    return int(step.n_features_in_)
        return 15

    def generate_signal(
        self,
        model_id: str,
        features: np.ndarray,
    ) -> Tuple[Side, float]:
        """
        Generate a trading signal from a model's prediction.

        Returns (side, confidence) where confidence is in [0, 1].
        """
        model = self.models[model_id]
        n_feat = self.model_n_features.get(model_id, len(features))
        feat_slice = features[:n_feat]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(feat_slice.reshape(1, -1))
            if proba.ndim == 2:
                bull_prob = float(proba[0, 1])
            else:
                bull_prob = float(proba[0])
        else:
            pred = float(model.predict(feat_slice.reshape(1, -1))[0])
            bull_prob = 0.5 + pred * 0.5

        if bull_prob > 0.6:
            return Side.LONG, bull_prob
        elif bull_prob < 0.4:
            return Side.SHORT, 1.0 - bull_prob
        return Side.FLAT, 0.0

    def ensemble_signal(
        self,
        features: np.ndarray,
    ) -> Tuple[Side, float]:
        """
        Combine signals from all loaded models via simple averaging.

        Miners are free to implement more sophisticated ensemble methods
        (weighted voting, stacking, etc.).
        """
        if not self.models:
            return Side.FLAT, 0.0

        signals = []
        for model_id in self.models:
            side, conf = self.generate_signal(model_id, features)
            if side == Side.LONG:
                signals.append(conf)
            elif side == Side.SHORT:
                signals.append(-conf)
            else:
                signals.append(0.0)

        avg = np.mean(signals)
        if avg > 0.15:
            return Side.LONG, float(avg)
        elif avg < -0.15:
            return Side.SHORT, float(-avg)
        return Side.FLAT, 0.0

    def execute_step(
        self,
        instrument: str,
        current_price: float,
        features: np.ndarray,
        timestamp: float | None = None,
    ) -> Optional[Dict]:
        """
        Execute one strategy step: generate signal, manage position.

        Returns a position update dict for validator streaming.
        """
        side, confidence = self.ensemble_signal(features)
        model_ids = list(self.models.keys())

        if instrument in self.engine.positions:
            pos = self.engine.positions[instrument]
            pos.update_price(current_price)

            should_close = (
                (pos.side == Side.LONG and side == Side.SHORT)
                or (pos.side == Side.SHORT and side == Side.LONG)
                or side == Side.FLAT
            )
            if should_close:
                trade = self.engine.close_position(instrument, current_price, model_ids, timestamp)
                if trade:
                    update = {
                        "type": "close",
                        "timestamp": timestamp or time.time(),
                        "instrument": instrument,
                        "side": "flat",
                        "pnl": trade.net_pnl,
                    }
                    self.position_log.append(update)
                    return update

        elif side != Side.FLAT and confidence > 0.2:
            size_pct = min(0.05, confidence * 0.08)
            pos = self.engine.open_position(
                instrument, side, current_price, size_pct, model_ids, timestamp,
            )
            if pos:
                update = {
                    "type": "open",
                    "timestamp": timestamp or time.time(),
                    "instrument": instrument,
                    "side": side.value,
                    "size": pos.size,
                    "entry_price": pos.entry_price,
                }
                self.position_log.append(update)
                return update

        return None

    def get_submission(self) -> Dict[str, Any]:
        """Prepare the TradingStrategySubmission data."""
        perf = self.engine.get_performance_summary()
        return {
            "role": MinerRole.TRADER.value,
            "strategy_id": self.strategy_id,
            "model_ids_used": list(self.models.keys()),
            "trading_mode": "paper",
            "instrument_scope": list(set(
                t.instrument for t in self.engine.trade_history
            )),
            "position_log": self.position_log,
            "realized_pnl": perf["realized_pnl"],
            "max_drawdown_pct": perf["max_drawdown_pct"],
            "sharpe_ratio": perf["sharpe_ratio"],
            "omega_ratio": perf["omega_ratio"],
            "win_rate": perf["win_rate"],
            "total_trades": perf["total_trades"],
            "performance_summary": perf,
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    """Demonstrate paper trading on a paired researcher model."""
    logger.info("=" * 60)
    logger.info("Insignia Trader Miner — Demo Mode (Paper Trading)")
    logger.info("=" * 60)

    from neurons.researcher_miner import ResearcherMiner, generate_demo_data

    logger.info("Step 1: Researcher trains a model (paired by the validator)...")
    model_data = generate_demo_data(n_samples=3000, seed=42)
    researcher = ResearcherMiner()
    submission = researcher.train_and_submit(model_data)
    model_id = "paired_model_001"

    logger.info("Step 2: Initializing trader miner...")
    trader = TraderMiner()
    trader.load_assigned_model(model_id, submission["model_artifact"])

    logger.info("Step 3: Running paper trading simulation...")
    rng = np.random.RandomState(99)
    price = 50000.0
    n_steps = 500

    for step in range(n_steps):
        ret = rng.normal(0.0001, 0.003)
        price *= (1 + ret)

        features = rng.normal(0, 1, 15)
        features[0] = ret  # ret_1

        ts = time.time() + step * 3600
        trader.execute_step("BTC-USDT-PERP", price, features, ts)

    perf = trader.engine.get_performance_summary()
    logger.info("--- Paper Trading Results ---")
    for k, v in perf.items():
        logger.info("  %s: %s", k, round(v, 4) if isinstance(v, float) else v)

    sub = trader.get_submission()
    logger.info("--- Trading Submission ---")
    logger.info("  role: %s", sub["role"])
    logger.info("  strategy_id: %s", sub["strategy_id"])
    logger.info("  total_trades: %d", sub["total_trades"])
    logger.info("  realized_pnl: %.2f", sub["realized_pnl"])
    logger.info("  position_log entries: %d", len(sub["position_log"]))

    return sub


if __name__ == "__main__":
    demo()
