"""Data loading, model training, and caching."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HORIZON = 24  # 24 bars × 5 min = 2 hours


def load_and_prepare() -> tuple[pd.DataFrame, list[str]]:
    """Load features_v1.csv and return clean DataFrame + feature column names."""
    df = pd.read_csv(DATA_DIR / "features_v1.csv")
    df = df.drop(columns=["spot_cvd_agg_buy", "spot_taker_buy_ratio"], errors="ignore")

    raw_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    feature_cols = [c for c in df.columns if c not in raw_cols]

    df = df.replace([np.inf, -np.inf], np.nan)
    bad_cols = [c for c in feature_cols if df[c].isna().sum() > len(df) * 0.05]
    feature_cols = [c for c in feature_cols if c not in bad_cols]

    df["target"] = df["close"].pct_change(HORIZON).shift(-HORIZON)
    df = df.dropna(subset=feature_cols + ["target"])

    return df, feature_cols


def train_and_save(df: pd.DataFrame, feature_cols: list[str]) -> tuple[Pipeline, dict]:
    """Train model on train split, save to data/, return model + metrics."""
    X = df[feature_cols].values
    y_raw = df["target"].values
    y = (y_raw > 0).astype(int)

    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingClassifier(
            max_iter=3000, max_depth=6, learning_rate=0.01,
            min_samples_leaf=30, l2_regularization=0.2,
            max_bins=128, random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]
    train_acc = float((pipe.predict(X_train) == y_train).mean())
    test_acc = float((preds == y_test).mean())

    model_path = DATA_DIR / "model_v1.joblib"
    joblib.dump(pipe, model_path)
    print(f"Saved model: {model_path} ({model_path.stat().st_size / 1024:.0f} KB)")

    meta: dict = {
        "feature_cols": feature_cols,
        "horizon_bars": HORIZON,
        "train_acc": round(train_acc, 4),
        "test_acc": round(test_acc, 4),
        "edge": round(test_acc - 0.5, 4),
        "split_idx": split_idx,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    from sklearn.inspection import permutation_importance
    perm = permutation_importance(pipe, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    importances = perm.importances_mean
    top_idx = np.argsort(importances)[::-1][:20]
    meta["importance"] = {
        "names": [feature_cols[i] for i in top_idx],
        "values": [round(float(importances[i]), 4) for i in top_idx],
    }

    meta_path = DATA_DIR / "model_v1_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved metadata: {meta_path}")

    return pipe, {
        **meta,
        "preds": preds,
        "probs": probs,
        "y_test": y_test,
        "y_raw_test": y_raw[split_idx:split_idx + len(y_test)],
        "ts_test": df["timestamp"].values[split_idx:split_idx + len(y_test)].tolist(),
        "close_test": df["close"].values[split_idx:split_idx + len(y_test)].tolist(),
    }


def load_cached(df: pd.DataFrame, feature_cols: list[str]) -> tuple[Pipeline, dict] | None:
    """Load cached model and recompute predictions. Returns None on cache miss."""
    model_path = DATA_DIR / "model_v1.joblib"
    meta_path = DATA_DIR / "model_v1_meta.json"
    if not model_path.exists() or not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text())
    if meta.get("feature_cols") != feature_cols or "importance" not in meta:
        return None

    print("Using cached model (delete data/model_v1.joblib to retrain)")
    pipe = joblib.load(model_path)

    X = df[feature_cols].values
    y_raw = df["target"].values
    y = (y_raw > 0).astype(int)
    split_idx = meta["split_idx"]
    y_test = y[split_idx:]

    preds = pipe.predict(X[split_idx:])
    probs = pipe.predict_proba(X[split_idx:])[:, 1]

    return pipe, {
        **meta,
        "preds": preds,
        "probs": probs,
        "y_test": y_test,
        "y_raw_test": y_raw[split_idx:split_idx + len(y_test)],
        "ts_test": df["timestamp"].values[split_idx:split_idx + len(y_test)].tolist(),
        "close_test": df["close"].values[split_idx:split_idx + len(y_test)].tolist(),
    }
