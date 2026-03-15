"""L1 Validator — FastAPI server with live dashboard.

Queries the miner, evaluates the model, scores it, and serves
a real-time dashboard showing the results.

Usage:
    uvicorn demo.validator:app --port 8000
"""
from __future__ import annotations

import base64
import io
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sklearn.metrics import f1_score

from demo.models import EvaluationRequest, ScoreReport, ValidatorStatus

logger = logging.getLogger("validator")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VIZ_DIR = Path(__file__).resolve().parent.parent / "viz"
MINER_URL = "http://localhost:8001"
HORIZON = 24
N_WINDOWS = 5
LAMBDA = 1.0

app = FastAPI(title="Insignia L1 Validator", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- State ---
REGIME_NAMES = ["Trending Up", "Ranging", "High Volatility", "Trending Down", "Low Volatility"]

_windows: list[dict] = []  # per-regime test windows
_feature_cols: list[str] = []
_epoch: int = 0
_epoch_results: list[dict] = []
_comm_log: list[dict] = []


def _log_event(event: str) -> None:
    entry = {"time": datetime.now(timezone.utc).isoformat(), "event": event}
    _comm_log.append(entry)
    logger.info(event)


@app.on_event("startup")
def load_benchmark() -> None:
    global _windows, _feature_cols

    csv_path = DATA_DIR / "features_v1.csv"
    if not csv_path.exists():
        logger.warning("No benchmark data at %s", csv_path)
        return

    df = pd.read_csv(csv_path)
    df = df.drop(columns=["spot_cvd_agg_buy", "spot_taker_buy_ratio"], errors="ignore")

    raw_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    _feature_cols = [c for c in df.columns if c not in raw_cols]
    df = df.replace([np.inf, -np.inf], np.nan)
    bad = [c for c in _feature_cols if df[c].isna().sum() > len(df) * 0.05]
    _feature_cols = [c for c in _feature_cols if c not in bad]

    df["target"] = df["close"].pct_change(HORIZON).shift(-HORIZON)
    df = df.dropna(subset=_feature_cols + ["target"])

    # Split test data into 5 regime windows (each ~3.6 days of 5-min bars)
    split = int(len(df) * 0.7)
    test_df = df.iloc[split:]
    chunk = len(test_df) // len(REGIME_NAMES)

    for i, name in enumerate(REGIME_NAMES):
        s, e = i * chunk, (i + 1) * chunk
        window = test_df.iloc[s:e]
        _windows.append({
            "name": name,
            "X": window[_feature_cols].values,
            "y": (window["target"].values > 0).astype(int),
            "y_raw": window["target"].values,
            "n_rows": len(window),
            "date_start": window["timestamp"].iloc[0][:10] if "timestamp" in window.columns else "",
            "date_end": window["timestamp"].iloc[-1][:10] if "timestamp" in window.columns else "",
        })
    logger.info("Loaded benchmark: %d windows, %d features", len(_windows), len(_feature_cols))


def _score_model(pipe: object, window: dict) -> dict:
    """Score a deserialized model on a specific benchmark window."""
    X, y, y_raw = window["X"], window["y"], window["y_raw"]
    preds = pipe.predict(X)  # type: ignore

    signed_preds = np.where(preds == 1, 1.0, -1.0)
    signed_actuals = np.where(y == 1, 1.0, -1.0)
    n = len(preds)
    chunk = n // N_WINDOWS

    # Per-window F1 and Sharpe
    window_f1s, window_sharpes = [], []
    for i in range(N_WINDOWS):
        s, e = i * chunk, (i + 1) * chunk
        wf1 = float(f1_score(y[s:e], preds[s:e], zero_division=0.0))
        window_f1s.append(wf1)
        pos_ret = signed_preds[s:e] * signed_actuals[s:e]
        std = float(np.std(pos_ret))
        ws = float(np.mean(pos_ret) / std * math.sqrt(365 * 24)) if std > 1e-12 else 0.0
        window_sharpes.append(ws)

    mean_f1, std_f1 = float(np.mean(window_f1s)), float(np.std(window_f1s))
    mean_sh, std_sh = float(np.mean(window_sharpes)), float(np.std(window_sharpes))

    pen_f1 = mean_f1 - LAMBDA * std_f1
    pen_sharpe = mean_sh - LAMBDA * std_sh

    # Equity + drawdown
    equity = np.cumprod(1 + signed_preds * y_raw / HORIZON)
    peak = np.maximum.accumulate(equity)
    raw_dd = float(np.max((peak - equity) / np.maximum(peak, 1e-12)))

    # Generalization gap
    test_f1 = float(f1_score(y, preds, zero_division=0.0))

    # Feature efficiency
    n_feat = len(_feature_cols)
    raw_eff = 1.0 / (1.0 + math.log(max(1, n_feat) / 10))

    # Latency
    row = X[:1]
    t0 = time.perf_counter()
    for _ in range(50):
        pipe.predict(row)  # type: ignore
    lat_ms = (time.perf_counter() - t0) / 50 * 1000

    # Normalize
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    norm = {
        "Penalized F1": max(0.0, min(1.0, pen_f1)),
        "Penalized Sharpe": sigmoid(0.5 * (pen_sharpe - 1.0)),
        "Max Drawdown": 1.0 - raw_dd,
        "Generalization Gap": max(0.0, 1.0 - abs(0.983 - test_f1)),
        "Feature Efficiency": max(0.0, min(1.0, raw_eff)),
        "Latency": 1.0 if lat_ms <= 50 else math.exp(-(lat_ms - 50) / 50),
    }
    weights = {
        "Penalized F1": 0.25, "Penalized Sharpe": 0.25, "Max Drawdown": 0.15,
        "Generalization Gap": 0.20, "Feature Efficiency": 0.05, "Latency": 0.10,
    }
    composite = sum(weights[k] * norm[k] for k in weights)

    raw = {
        "Penalized F1": round(pen_f1, 4),
        "Penalized Sharpe": round(pen_sharpe, 2),
        "Max Drawdown": round(raw_dd, 4),
        "Generalization Gap": round(abs(0.983 - test_f1), 4),
        "Feature Efficiency": round(raw_eff, 4),
        "Latency": round(lat_ms, 1),
    }

    return {
        "composite": round(composite, 4),
        "normalized": {k: round(v, 4) for k, v in norm.items()},
        "raw": raw,
        "weights": weights,
        "window_f1s": [round(f, 4) for f in window_f1s],
        "window_sharpes": [round(s, 2) for s in window_sharpes],
    }


@app.get("/status")
def status() -> ValidatorStatus:
    return ValidatorStatus(
        epoch=_epoch,
        n_miners=1,
        n_epochs_completed=len(_epoch_results),
        last_composite_score=_epoch_results[-1]["composite"] if _epoch_results else None,
    )


@app.post("/run-epoch")
async def run_epoch() -> ScoreReport:
    global _epoch
    _epoch += 1
    # Pick regime window (rotates through 5 market conditions)
    window = _windows[(_epoch - 1) % len(_windows)]
    regime = window["name"]
    _log_event(f"Epoch {_epoch}: starting evaluation — regime: {regime} ({window['date_start']} to {window['date_end']})")

    # 1. Query miner
    _log_event(f"Epoch {_epoch}: sending EvaluationRequest to miner at {MINER_URL}")
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(
                f"{MINER_URL}/forward",
                json=EvaluationRequest(epoch_id=_epoch).model_dump(),
            )
            resp.raise_for_status()
        except Exception as e:
            _log_event(f"Epoch {_epoch}: FAILED to reach miner — {e}")
            raise HTTPException(502, f"Cannot reach miner: {e}")

    submission = resp.json()
    artifact_size = submission.get("artifact_size_bytes", 0)
    _log_event(f"Epoch {_epoch}: received model ({artifact_size // 1024} KB, hash: {submission.get('artifact_hash', '')[:16]})")

    # 2. Deserialize
    artifact_bytes = base64.b64decode(submission["model_artifact_b64"])
    pipe = joblib.load(io.BytesIO(artifact_bytes))
    _log_event(f"Epoch {_epoch}: model deserialized, running inference on {window['n_rows']} rows ({regime})")

    # 3. Score on this epoch's regime window
    metrics = _score_model(pipe, window)
    _log_event(f"Epoch {_epoch}: scored — composite: {metrics['composite'] * 100:.1f}% (regime: {regime})")

    result = {**metrics, "epoch_id": _epoch, "regime": regime}
    _epoch_results.append(result)

    return ScoreReport(
        epoch_id=_epoch,
        composite_score=metrics["composite"],
        metric_breakdown=metrics["normalized"],
        raw_metrics=metrics["raw"],
    )


@app.get("/scores")
def scores() -> list[dict]:
    return _epoch_results


@app.get("/log")
def comm_log() -> list[dict]:
    return _comm_log


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    return (Path(__file__).parent / "dashboard.html").read_text()
