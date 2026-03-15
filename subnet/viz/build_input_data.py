"""Build the training data explorer visualization."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

VIZ_DIR = Path(__file__).resolve().parent
STEP = 6  # downsample: 1 point every 30 min (6 × 5-min bars)


def build(df: pd.DataFrame, model_info: dict) -> None:
    """Inject data into input_data.html template → input_data_built.html."""
    ts = df["timestamp"].values[::STEP].tolist()
    split_idx = model_info["split_idx"] // STEP

    data_json = {
        "n_rows": len(df),
        "date_start": df["timestamp"].iloc[0],
        "date_end": df["timestamp"].iloc[-1],
        "n_months": round(len(df) / (24 * 12 * 30), 1),
        "test_acc": round(model_info["test_acc"] * 100, 1),
        "edge": round(model_info["edge"] * 100, 2),
        "pct_up": round(float((df["target"] > 0).mean()) * 100, 1),
        "split_idx": split_idx,
        "ts": ts,
        "close": df["close"].values[::STEP].tolist(),
        "funding_rate": df["funding_rate"].values[::STEP].tolist(),
        "oi_close": df["oi_close"].values[::STEP].tolist(),
        "oi_change_30min": df["oi_change_30min"].values[::STEP].tolist(),
        "oi_change_1h": df["oi_change_1h"].values[::STEP].tolist(),
        "oi_change_4h": df["oi_change_4h"].values[::STEP].tolist(),
        "fut_ob_imbalance": df["fut_ob_imbalance"].values[::STEP].tolist(),
        "cg_depth_ratio": df["cg_depth_ratio"].values[::STEP].tolist(),
        "liq_long_usd": df["liq_long_usd"].values[::STEP].tolist(),
        "liq_short_usd": df["liq_short_usd"].values[::STEP].tolist(),
        "taker_buy_usd": df["taker_buy_usd"].values[::STEP].tolist(),
        "taker_sell_usd": df["taker_sell_usd"].values[::STEP].tolist(),
        "spot_cvd": df["spot_cvd"].values[::STEP].tolist(),
        "spot_cvd_z_30m": df["spot_cvd_z_30m"].values[::STEP].tolist(),
        "spot_cvd_z_1h": df["spot_cvd_z_1h"].values[::STEP].tolist(),
        "spot_cvd_z_4h": df["spot_cvd_z_4h"].values[::STEP].tolist(),
        "target": df["target"].values[::STEP].tolist(),
        "importance": model_info["importance"],
    }

    template = (VIZ_DIR / "input_data.html").read_text()
    html = template.replace("__DATA_JSON__", json.dumps(data_json))
    output_path = VIZ_DIR / "input_data_built.html"
    output_path.write_text(html)
    print(f"Built: {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")
