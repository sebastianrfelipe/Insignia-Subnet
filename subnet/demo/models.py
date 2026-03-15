"""Pydantic models for miner-validator communication.

Mirrors the Synapse definitions in insignia/protocol.py but as plain
Pydantic models — no bittensor dependency required.
"""
from __future__ import annotations

from pydantic import BaseModel


class EvaluationRequest(BaseModel):
    """Validator → Miner: request the miner's current model."""
    epoch_id: int = 0
    target_instrument: str = "BTC-USDT-PERP"


class ModelSubmission(BaseModel):
    """Miner → Validator: the model artifact + metadata."""
    model_artifact_b64: str        # base64-encoded joblib bytes
    model_type: str = "gbdt"
    features_used: list[str] = []
    hyperparams: dict = {}
    target_instrument: str = "BTC-USDT-PERP"
    target_horizon_minutes: int = 120
    generalization_gap: float = 0.0
    artifact_hash: str = ""
    artifact_size_bytes: int = 0


class ScoreReport(BaseModel):
    """Validator → Miner: evaluation results."""
    epoch_id: int
    composite_score: float
    metric_breakdown: dict[str, float] = {}
    raw_metrics: dict[str, float] = {}
    rank: int = 1
    total_miners: int = 1


class MinerStatus(BaseModel):
    uid: str = "miner-0"
    model_loaded: bool = False
    artifact_size_bytes: int = 0
    n_features: int = 0
    model_type: str = "gbdt"


class ValidatorStatus(BaseModel):
    epoch: int = 0
    n_miners: int = 1
    n_epochs_completed: int = 0
    last_composite_score: float | None = None
