"""Build the scoring system visualization (new 6-metric L1 scoring)."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

VIZ_DIR = Path(__file__).resolve().parent

WEIGHTS = {
    "Penalized F1": 0.25,
    "Penalized Sharpe": 0.25,
    "Max Drawdown": 0.15,
    "Generalization Gap": 0.20,
    "Feature Efficiency": 0.05,
    "Latency": 0.10,
}

DESCRIPTIONS = {
    "Penalized F1": "F1 score across rolling windows, penalized by variance. Rewards both quality and consistency.",
    "Penalized Sharpe": "Risk-adjusted returns across rolling windows, penalized by variance. Smooth equity = high score.",
    "Max Drawdown": "Worst peak-to-trough loss. Penalizes fragile models with tail risk.",
    "Generalization Gap": "Gap between training and test F1. Lower = better. Directly measures overfitting.",
    "Feature Efficiency": "Penalizes models requiring excessive features that may not be available in production.",
    "Latency": "Inference speed. Critical for short-horizon trading where milliseconds matter.",
}

ATTACK_CONTEXT = {
    "Penalized F1": "The variance penalty (mean − λ·std) prevents models that spike in one market regime and collapse in another. A model must be consistently good, not just occasionally lucky.",
    "Penalized Sharpe": "Accuracy alone doesn't capture risk. A model right 55% of the time but losing big on the wrong 45% is dangerous. Sharpe ensures the subnet rewards steady, reliable edge.",
    "Max Drawdown": "Prevents models that achieve high returns through extreme risk-taking. Strategies breaching 20% drawdown are eliminated entirely — mirroring institutional prop trading standards.",
    "Generalization Gap": "The most common failure mode in ML trading. Our model has a large gap (98% train → 54% test). Validators use proprietary data miners can't access, so memorizing training patterns doesn't help.",
    "Feature Efficiency": "Discourages models dependent on exotic data sources. A model using 10 features that performs as well as one using 200 is preferred — simpler is more robust.",
    "Latency": "Short-horizon trading requires fast inference. A model that takes 500ms to predict misses the trade window entirely.",
}

LAMBDA = 1.0  # variance penalty coefficient
N_WINDOWS = 5


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def build(pipe: Pipeline, model_info: dict) -> None:
    """Compute all 6 L1 metrics and build scoring_built.html."""
    y_test = model_info["y_test"]
    y_raw_test = model_info["y_raw_test"]
    preds = model_info["preds"]
    ts_test = model_info["ts_test"]
    n = len(y_test)

    # --- Per-window F1 and Sharpe ---
    chunk = n // N_WINDOWS
    window_f1s: list[float] = []
    window_sharpes: list[float] = []
    window_labels: list[str] = []

    signed_preds = np.where(preds == 1, 1.0, -1.0)
    signed_actuals = np.where(y_test == 1, 1.0, -1.0)

    for i in range(N_WINDOWS):
        s, e = i * chunk, (i + 1) * chunk
        # F1 per window
        wf1 = float(f1_score(y_test[s:e], preds[s:e], zero_division=0.0))
        window_f1s.append(wf1)
        # Sharpe per window
        pos_ret = signed_preds[s:e] * signed_actuals[s:e]
        if np.std(pos_ret) > 1e-12:
            ws = float(np.mean(pos_ret) / np.std(pos_ret)) * math.sqrt(365 * 24)
        else:
            ws = 0.0
        window_sharpes.append(ws)
        window_labels.append(f"{ts_test[s][:10]} — {ts_test[e - 1][:10]}")

    # 1. Penalized F1
    mean_f1 = float(np.mean(window_f1s))
    std_f1 = float(np.std(window_f1s))
    raw_pen_f1 = mean_f1 - LAMBDA * std_f1
    norm_pen_f1 = max(0.0, min(1.0, raw_pen_f1))

    # 2. Penalized Sharpe
    mean_sharpe = float(np.mean(window_sharpes))
    std_sharpe = float(np.std(window_sharpes))
    raw_pen_sharpe = mean_sharpe - LAMBDA * std_sharpe
    norm_pen_sharpe = _sigmoid(0.5 * (raw_pen_sharpe - 1.0))

    # 3. Max Drawdown
    equity = np.cumprod(1 + signed_preds * y_raw_test / 24)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / np.maximum(peak, 1e-12)
    raw_dd = float(np.max(drawdown))
    norm_dd = 1.0 - raw_dd

    # 4. Generalization Gap
    # Approximate train F1 ≈ train accuracy (close for near-balanced classes)
    train_f1_approx = model_info["train_acc"]
    test_f1 = float(f1_score(y_test, preds, zero_division=0.0))
    raw_gen_gap = abs(train_f1_approx - test_f1)
    norm_gen_gap = max(0.0, 1.0 - raw_gen_gap)  # lower gap = higher score

    # 5. Feature Efficiency
    n_features = len(model_info.get("feature_cols", []))
    raw_efficiency = 1.0 / (1.0 + math.log(max(1, n_features) / 10))
    norm_efficiency = max(0.0, min(1.0, raw_efficiency))

    # 6. Latency
    scaler = pipe.named_steps["scaler"]
    row = scaler.transform(np.zeros((1, n_features)))
    start_t = time.perf_counter()
    for _ in range(100):
        pipe.named_steps["model"].predict(row)
    elapsed_ms = (time.perf_counter() - start_t) / 100 * 1000
    raw_latency = elapsed_ms
    norm_latency = 1.0 if raw_latency <= 50.0 else math.exp(-(raw_latency - 50.0) / 50.0)

    metrics = [
        {"name": "Penalized F1", "raw": round(raw_pen_f1, 4), "normalized": round(norm_pen_f1, 4),
         "raw_display": f"{raw_pen_f1:.3f} (mean {mean_f1:.3f} − {LAMBDA}×std {std_f1:.3f})"},
        {"name": "Penalized Sharpe", "raw": round(raw_pen_sharpe, 2), "normalized": round(norm_pen_sharpe, 4),
         "raw_display": f"{raw_pen_sharpe:.2f} (mean {mean_sharpe:.2f} − {LAMBDA}×std {std_sharpe:.2f})"},
        {"name": "Max Drawdown", "raw": round(raw_dd, 4), "normalized": round(norm_dd, 4),
         "raw_display": f"{raw_dd * 100:.1f}%"},
        {"name": "Generalization Gap", "raw": round(raw_gen_gap, 4), "normalized": round(norm_gen_gap, 4),
         "raw_display": f"{raw_gen_gap * 100:.1f}% (train F1: {train_f1_approx * 100:.1f}% → test F1: {test_f1 * 100:.1f}%)"},
        {"name": "Feature Efficiency", "raw": round(raw_efficiency, 4), "normalized": round(norm_efficiency, 4),
         "raw_display": f"{n_features} features"},
        {"name": "Latency", "raw": round(raw_latency, 2), "normalized": round(norm_latency, 4),
         "raw_display": f"{raw_latency:.1f}ms"},
    ]

    for m in metrics:
        m["weight"] = WEIGHTS[m["name"]]
        m["desc"] = DESCRIPTIONS[m["name"]]
        m["attack"] = ATTACK_CONTEXT[m["name"]]

    composite = sum(m["weight"] * m["normalized"] for m in metrics)

    # Chart data
    step = 6
    equity_ds = [float(equity[i]) for i in range(0, n, step)]
    peak_ds = [float(peak[i]) for i in range(0, n, step)]
    ts_ds = [ts_test[i] for i in range(0, n, step)]

    data_json = {
        "composite": round(composite, 4),
        "metrics": metrics,
        "window_labels": window_labels,
        "window_f1s": [round(f, 4) for f in window_f1s],
        "window_sharpes": [round(s, 2) for s in window_sharpes],
        "equity_ts": ts_ds,
        "equity": equity_ds,
        "equity_peak": peak_ds,
        "train_f1": round(train_f1_approx * 100, 1),
        "test_f1": round(test_f1 * 100, 1),
        "n_features": n_features,
        "latency_ms": round(raw_latency, 1),
    }

    template = (VIZ_DIR / "scoring.html").read_text()
    html = template.replace("__DATA_JSON__", json.dumps(data_json))
    output_path = VIZ_DIR / "scoring_built.html"
    output_path.write_text(html)
    print(f"Built: {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")
