"""Build the model performance visualization."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HORIZON = 24  # 24 bars × 5 min = 2 hours

VIZ_DIR = Path(__file__).resolve().parent
STEP = 6  # downsample: 1 point every 30 min


def build(model_info: dict) -> None:
    """Inject data into model_perf.html template → model_perf_built.html."""
    y_test = model_info["y_test"]
    y_raw_test = model_info["y_raw_test"]
    preds = model_info["preds"]
    probs = model_info["probs"]
    ts_test = model_info["ts_test"]
    close_test = model_info["close_test"]

    n = len(y_test)

    # --- Price line (downsampled) ---
    ts_ds = [ts_test[i] for i in range(0, n, STEP)]
    close_ds = [close_test[i] for i in range(0, n, STEP)]

    # --- 2-hour prediction blocks ---
    blocks: list[dict[str, object]] = []
    block_probs: list[float] = []
    for i in range(0, n - HORIZON, HORIZON):
        end_i = min(i + HORIZON, n - 1)
        blocks.append({
            "x0": ts_test[i],
            "x1": ts_test[end_i],
            "pred": int(preds[i]),
            "correct": bool(preds[i] == y_test[i]),
        })
        block_probs.append(round(float(probs[i]), 4))

    # --- Trading metrics (for stat badges) ---
    strategy_returns = np.where(preds == 1, y_raw_test, -y_raw_test)
    equity = np.cumprod(1 + strategy_returns / HORIZON)

    total_return = round((float(equity[-1]) - 1) * 100, 2)
    win_rate = round(float((strategy_returns > 0).mean()) * 100, 1)

    bars_per_year = 288 * 365
    mean_ret = float(np.mean(strategy_returns / HORIZON))
    std_ret = float(np.std(strategy_returns / HORIZON))
    sharpe = round(mean_ret / std_ret * np.sqrt(bars_per_year), 2) if std_ret > 0 else 0.0

    running_max = np.maximum.accumulate(equity)
    max_dd = round(float(((equity - running_max) / running_max).min()) * 100, 2)

    data_json = {
        "test_acc": model_info["test_acc"],
        "edge": model_info["edge"],
        "total_return": total_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "ts": ts_ds,
        "close": close_ds,
        "blocks": blocks,
        "block_probs": block_probs,
    }

    template = (VIZ_DIR / "model_perf.html").read_text()
    html = template.replace("__DATA_JSON__", json.dumps(data_json))
    output_path = VIZ_DIR / "model_perf_built.html"
    output_path.write_text(html)
    print(f"Built: {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")
