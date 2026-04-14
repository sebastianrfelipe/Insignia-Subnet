"""
Insignia Subnet Protocol

Defines the Synapse (request/response) types for communication between
validators and miners across both Layer 1 (model generation) and
Layer 2 (strategy deployment).

Bittensor subnets communicate via Synapses — typed request/response
objects that travel over the network. Each Synapse defines what a
validator asks for and what a miner returns.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import bittensor as bt


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ModelType(str, Enum):
    GBDT = "gbdt"
    RANDOM_FOREST = "random_forest"
    NEURAL = "neural"
    ENSEMBLE = "ensemble"
    OTHER = "other"


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE_CAPITAL = "live_capital"


class InstrumentId(str, Enum):
    BTC_USDT_PERP = "BTC-USDT-PERP"
    ETH_USDT_PERP = "ETH-USDT-PERP"
    SOL_USDT_PERP = "SOL-USDT-PERP"
    AVAX_USDT_PERP = "AVAX-USDT-PERP"
    ADA_USDT_PERP = "ADA-USDT-PERP"


# ---------------------------------------------------------------------------
# Layer 1: Model Generation Synapses
# ---------------------------------------------------------------------------

class L1ModelSubmission(bt.Synapse):
    """
    Layer 1 miner -> validator: Submit a trained ML model for evaluation.

    The model artifact is serialized (ONNX or joblib) and sent with metadata
    describing the training configuration. Validators deserialize, run the
    model against their proprietary benchmark dataset, and return a composite
    score vector.

    Privacy note: the validator's benchmark dataset is never exposed to miners.
    Only the aggregate score is returned.
    """

    # --- Request fields (miner fills these) ---
    model_artifact: Optional[bytes] = None
    model_type: str = ModelType.GBDT.value
    features_used: List[str] = []
    training_window_start: str = ""
    training_window_end: str = ""
    hyperparams: Dict[str, Any] = {}
    preprocessing_hash: str = ""
    target_instrument: str = InstrumentId.BTC_USDT_PERP.value
    target_horizon_minutes: int = 60
    self_reported_overfitting_score: float = 0.0

    # --- Response fields (validator fills these) ---
    composite_score: Optional[float] = None
    score_breakdown: Dict[str, float] = {}
    accepted: bool = False
    rejection_reason: str = ""

    class Config:
        arbitrary_types_allowed = True


class L1EvaluationRequest(bt.Synapse):
    """
    Validator -> miner: Request current model for evaluation epoch.

    Validators periodically request the miner's latest model. This is the
    pull-based alternative to push-based submission. The miner returns its
    best model artifact and metadata.
    """

    epoch_id: int = 0
    target_instrument: str = InstrumentId.BTC_USDT_PERP.value
    evaluation_window_hint: str = ""

    # Response
    model_artifact: Optional[bytes] = None
    model_metadata: Dict[str, Any] = {}


class L1ScoreReport(bt.Synapse):
    """
    Validator -> miner: Report evaluation scores back to the miner.

    After scoring, validators inform miners of their composite score and
    per-metric breakdown so miners can iterate. The proprietary benchmark
    data is never shared — only the resulting scores.
    """

    epoch_id: int = 0
    composite_score: float = 0.0
    metric_breakdown: Dict[str, float] = {}
    rank: int = -1
    total_miners: int = 0
    promoted_to_l2: bool = False


# ---------------------------------------------------------------------------
# Layer 2: Strategy Deployment Synapses
# ---------------------------------------------------------------------------

class L2StrategySubmission(bt.Synapse):
    """
    Layer 2 miner -> validator: Submit live/paper trading strategy results.

    L2 miners receive top-performing L1 models and build trading strategies
    around them. They submit periodic performance snapshots — position logs,
    P&L, and risk metrics — for validator scoring.
    """

    strategy_id: str = ""
    model_ids_used: List[str] = []
    trading_mode: str = TradingMode.PAPER.value
    instrument_scope: List[str] = []
    position_log: List[Dict[str, Any]] = []
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    omega_ratio: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    evaluation_window_hours: int = 720  # 30 days default

    # Response
    composite_score: Optional[float] = None
    score_breakdown: Dict[str, float] = {}
    accepted: bool = False


class L2ModelPool(bt.Synapse):
    """
    L2 miner -> validator: Request the current pool of promoted L1 models.

    L2 miners need access to top-performing L1 models to build strategies.
    This synapse returns metadata and download references for the current
    model pool (top-N from Layer 1).
    """

    # Request
    requested_instruments: List[str] = []

    # Response: list of available models with metadata (no raw artifacts)
    available_models: List[Dict[str, Any]] = []
    pool_updated_at: str = ""


class L2PositionUpdate(bt.Synapse):
    """
    L2 miner -> validator: Real-time position update for continuous scoring.

    Paper/live traders push position changes as they happen so validators
    can track P&L in real-time without waiting for epoch boundaries.
    """

    strategy_id: str = ""
    timestamp: float = 0.0
    instrument: str = ""
    side: str = ""  # "long" | "short" | "flat"
    size: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    pnl_delta: float = 0.0


# ---------------------------------------------------------------------------
# Cross-Layer Synapses
# ---------------------------------------------------------------------------

class CrossLayerFeedback(bt.Synapse):
    """
    Internal synapse for cross-layer feedback propagation.

    When L2 strategies perform well (or poorly) using specific L1 models,
    this feedback adjusts L1 model scores retroactively. Models that
    translate simulation quality into real-world performance get boosted;
    models that fail in deployment get penalized.
    """

    model_id: str = ""
    l2_performance_delta: float = 0.0
    l1_score_adjustment: float = 0.0
    feedback_epoch: int = 0
    evidence: Dict[str, Any] = {}
