"""L1 Miner — FastAPI server (the axon).

Loads a pre-trained model and serves it to validators on request.
In a real subnet this would be a bt.axon; here it's a plain FastAPI app.

Usage:
    uvicorn demo.miner:app --port 8001
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from demo.models import EvaluationRequest, MinerStatus, ModelSubmission

logger = logging.getLogger("miner")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

app = FastAPI(title="Insignia L1 Miner", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- State ---
_model_bytes: bytes = b""
_meta: dict = {}
_artifact_hash: str = ""


@app.on_event("startup")
def load_model() -> None:
    global _model_bytes, _meta, _artifact_hash

    model_path = DATA_DIR / "model_v1.joblib"
    meta_path = DATA_DIR / "model_v1_meta.json"

    if not model_path.exists():
        logger.warning("No model found at %s — miner will serve empty responses", model_path)
        return

    _model_bytes = model_path.read_bytes()
    _artifact_hash = hashlib.sha256(_model_bytes).hexdigest()
    _meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    logger.info("Loaded model: %d KB, hash: %s", len(_model_bytes) // 1024, _artifact_hash[:16])


@app.get("/status")
def status() -> MinerStatus:
    return MinerStatus(
        uid="miner-0",
        model_loaded=len(_model_bytes) > 0,
        artifact_size_bytes=len(_model_bytes),
        n_features=len(_meta.get("feature_cols", [])),
        model_type="gbdt",
    )


@app.post("/forward")
def forward(request: EvaluationRequest) -> ModelSubmission:
    """The miner's forward function — returns the current model."""
    logger.info("Epoch %d: received evaluation request for %s", request.epoch_id, request.target_instrument)

    return ModelSubmission(
        model_artifact_b64=base64.b64encode(_model_bytes).decode(),
        model_type="gbdt",
        features_used=_meta.get("feature_cols", []),
        hyperparams={
            "max_iter": 3000,
            "max_depth": 6,
            "learning_rate": 0.01,
            "min_samples_leaf": 30,
            "l2_regularization": 0.2,
        },
        target_instrument=request.target_instrument,
        target_horizon_minutes=_meta.get("horizon_bars", 24) * 5,
        generalization_gap=round(
            _meta.get("train_acc", 0) - _meta.get("test_acc", 0), 4
        ),
        artifact_hash=_artifact_hash,
        artifact_size_bytes=len(_model_bytes),
    )
