"""
Build HTML visualizations by injecting real data into templates.

Usage:
    uv run python -m viz.build_viz
    uv run python viz/build_viz.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python viz/build_viz.py` from subnet/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viz.prepare import load_and_prepare, load_cached, train_and_save
from viz import build_input_data, build_model_perf


def main() -> None:
    df, feature_cols = load_and_prepare()
    print(f"Loaded {len(df)} rows, {len(feature_cols)} features")

    cached = load_cached(df, feature_cols)
    if cached:
        _, model_info = cached
    else:
        _, model_info = train_and_save(df, feature_cols)
    print(f"Train: {model_info['train_acc']:.1%} | Test: {model_info['test_acc']:.1%} | Edge: {model_info['edge']:+.2%}")
    print()

    build_input_data.build(df, model_info)
    build_model_perf.build(model_info)


if __name__ == "__main__":
    main()
