"""
Insignia Subnet Protocol

Defines the Synapse (request/response) types for communication between
validators and miners for both the researcher (model generation) and trader
(strategy deployment) roles.

Bittensor subnets communicate via Synapses — typed request/response
objects that travel over the network. Each Synapse defines what a
validator asks for and what a miner returns.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

try:
    import bittensor as bt
    _SynapseBase = bt.Synapse
except ImportError:
    # bittensor is only required by the live neurons. The scoring/tuning/
    # simulation stack imports this module purely for the enums (MinerRole,
    # InstrumentId, ...), so degrade gracefully when the chain SDK is absent.
    bt = None

    class _SynapseBase:  # minimal stand-in so the Synapse classes still define
        """Fallback base used when bittensor is unavailable."""

        class Config:
            arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MinerRole(str, Enum):
    """
    Role a miner declares at registration/commit in the single paired
    mechanism. Both roles share one subnet UID space and one weight vector.
    """

    RESEARCHER = "researcher"  # submits ML model artifacts
    TRADER = "trader"          # submits trading-operation logic consuming a model


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
# Model Generation Synapses (researcher role)
# ---------------------------------------------------------------------------

class ModelSubmission(_SynapseBase):
    """
    Researcher miner -> validator: Submit a trained ML model for evaluation.

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


class ModelEvaluationRequest(_SynapseBase):
    """
    Validator -> researcher miner: Request current model for evaluation epoch.

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


class ModelScoreReport(_SynapseBase):
    """
    Validator -> researcher miner: Report evaluation scores back to the miner.

    After scoring, validators inform miners of their composite score and
    per-metric breakdown so miners can iterate. The proprietary benchmark
    data is never shared — only the resulting scores.
    """

    epoch_id: int = 0
    composite_score: float = 0.0
    metric_breakdown: Dict[str, float] = {}
    rank: int = -1
    total_miners: int = 0
    promoted: bool = False


# ---------------------------------------------------------------------------
# Strategy Deployment Synapses (trader role)
# ---------------------------------------------------------------------------

class TradingStrategySubmission(_SynapseBase):
    """
    Trader miner -> validator: Submit live/paper trading strategy results.

    Trader miners run a strategy on the model assigned by the chain-seeded
    pairing. They submit periodic performance snapshots — position logs,
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


class ModelPool(_SynapseBase):
    """
    DEPRECATED (legacy two-layer promotion). trader miner -> validator:
    request the current pool of promoted models.

    Superseded by chain-seeded pairing, which assigns a researcher model to
    each trader rather than letting traders self-select from a promoted pool.
    """

    # Request
    requested_instruments: List[str] = []

    # Response: list of available models with metadata (no raw artifacts)
    available_models: List[Dict[str, Any]] = []
    pool_updated_at: str = ""


class TradingPositionUpdate(_SynapseBase):
    """
    Trader miner -> validator: Real-time position update for continuous scoring.

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
# Cross-Layer Synapses (DEPRECATED — retained for backward compatibility)
#
# The single paired mechanism replaces L1->promotion->L2 with chain-seeded
# pairing + joint evaluation, so cross-layer retroactive feedback is no longer
# part of the active protocol. See the Pairing Synapses below.
# ---------------------------------------------------------------------------

class CrossLayerFeedback(_SynapseBase):
    """
    DEPRECATED. Internal synapse for cross-layer feedback propagation in the
    old two-layer design. Superseded by joint pair evaluation.
    """

    model_id: str = ""
    l2_performance_delta: float = 0.0
    l1_score_adjustment: float = 0.0
    feedback_epoch: int = 0
    evidence: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Pairing Synapses (single paired mechanism)
# ---------------------------------------------------------------------------

class PairAssignment(_SynapseBase):
    """
    Validator -> miners: announce the chain-seeded pairing for an epoch.

    The assignment is derived deterministically from chain block state so it is
    not chooseable by miners or validators, and the partner identity is only
    revealed at evaluation time (latency-arbitrage and collusion defense).
    """

    epoch_id: int = 0
    generation: int = 0
    pairing_seed: str = ""
    researcher_uid: str = ""
    trader_uid: str = ""
    target_instrument: str = InstrumentId.BTC_USDT_PERP.value


class PairEvaluationRequest(_SynapseBase):
    """
    Validator -> researcher/trader: request the artifact/strategy for a pair.

    ``role`` tells the recipient which side of the pair it is being queried for.
    The researcher returns a model artifact; the trader returns its trading
    strategy parameters and (post-window) its revealed position log.
    """

    epoch_id: int = 0
    generation: int = 0
    role: str = MinerRole.RESEARCHER.value
    researcher_uid: str = ""
    trader_uid: str = ""
    pairing_seed: str = ""

    # Response (researcher side)
    model_artifact: Optional[bytes] = None
    model_metadata: Dict[str, Any] = {}

    # Response (trader side)
    strategy_id: str = ""
    strategy_params: Dict[str, Any] = {}
    position_log: List[Dict[str, Any]] = []

    class Config:
        arbitrary_types_allowed = True


class PairScoreReport(_SynapseBase):
    """
    Validator -> miners: report a pair's joint score and the per-miner credit.

    Both members of the pair receive the same pair-level breakdown; the
    ``miner_credit`` reflects the variance-penalized marginal contribution that
    feeds the single Yuma weight vector.
    """

    epoch_id: int = 0
    generation: int = 0
    researcher_uid: str = ""
    trader_uid: str = ""
    model_composite: float = 0.0
    trading_composite: float = 0.0
    pair_composite: float = 0.0
    objectives: List[float] = []
    pareto_rank: int = -1
    selection_score: float = 0.0
    collusion_flagged: bool = False
    miner_credit: float = 0.0
