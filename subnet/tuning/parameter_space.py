"""
Parameter Space Definition

Enumerates every tunable parameter in the Insignia subnet, defines bounds,
constraints, and provides encoding/decoding between flat numeric vectors
(for the optimizer) and structured configuration objects (for the subnet).

The parameter space is organized into groups:
  - L1 scoring weights (must sum to 1.0)
  - L2 scoring weights (must sum to 1.0)
  - Overfitting detector thresholds
  - Cross-layer promotion criteria
  - Cross-layer feedback weights
  - Anti-gaming thresholds
  - L2 trading engine parameters
  - Buyback mechanism parameters
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from insignia.scoring import WeightConfig
from insignia.cross_layer import PromotionConfig


@dataclass
class ParameterBounds:
    name: str
    lower: float
    upper: float
    group: str
    description: str = ""


# All tunable parameters with bounds
PARAMETER_DEFINITIONS: List[ParameterBounds] = [
    # L1 Scoring Weights (6 params, must sum to 1.0)
    # Uses variance-penalized mean - λ·std formulation from GBDT HPO
    ParameterBounds("l1_penalized_f1",       0.05, 0.45, "l1_weights", "Weight for penalized F1 (mean_f1 - λ·std_f1)"),
    ParameterBounds("l1_penalized_sharpe",   0.05, 0.45, "l1_weights", "Weight for penalized Sharpe (mean_sharpe - λ·std_sharpe)"),
    ParameterBounds("l1_max_drawdown",       0.05, 0.30, "l1_weights", "Weight for max drawdown penalty"),
    ParameterBounds("l1_generalization_gap", 0.05, 0.40, "l1_weights", "Weight for |train_f1 - val_f1| overfitting"),
    ParameterBounds("l1_feature_efficiency", 0.01, 0.15, "l1_weights", "Weight for feature efficiency"),
    ParameterBounds("l1_latency",            0.01, 0.20, "l1_weights", "Weight for latency score"),

    # L2 Scoring Weights (6 params, must sum to 1.0)
    ParameterBounds("l2_realized_pnl",     0.05, 0.45, "l2_weights", "Weight for realized P&L"),
    ParameterBounds("l2_omega",            0.05, 0.35, "l2_weights", "Weight for Omega ratio"),
    ParameterBounds("l2_max_drawdown",     0.05, 0.30, "l2_weights", "Weight for max drawdown"),
    ParameterBounds("l2_win_rate",         0.05, 0.25, "l2_weights", "Weight for win rate"),
    ParameterBounds("l2_consistency",      0.05, 0.35, "l2_weights", "Weight for consistency"),
    ParameterBounds("l2_model_attribution",0.01, 0.25, "l2_weights", "Weight for model attribution"),

    # Variance Penalty (λ in mean - λ·std formulation)
    ParameterBounds("variance_penalty",  0.1, 1.5,  "variance_penalty", "λ coefficient: higher = stricter variance punishment"),
    ParameterBounds("n_eval_windows",    3,   10,   "variance_penalty", "K rolling windows for mean/std computation"),

    # Cross-Layer Promotion
    ParameterBounds("promotion_top_n",                 3,   20,   "promotion", "Number of models promoted to L2"),
    ParameterBounds("promotion_min_consecutive_epochs", 1,    5,   "promotion", "Min epochs in top-N before promotion"),
    ParameterBounds("promotion_max_overfitting_score",  0.1,  0.6, "promotion", "Max overfitting score for promotion"),
    ParameterBounds("promotion_max_score_decay_pct",    0.05, 0.4, "promotion", "Max allowed epoch-over-epoch score decay"),
    ParameterBounds("promotion_expiry_epochs",          3,   15,   "promotion", "Epochs without L2 usage before expiry"),

    # Cross-Layer Feedback
    ParameterBounds("feedback_bonus_weight",   0.0,  0.40, "feedback", "L2 success bonus for L1 scores"),
    ParameterBounds("feedback_penalty_weight", 0.0,  0.30, "feedback", "L2 failure penalty for L1 scores"),

    # Anti-Gaming
    ParameterBounds("fingerprint_correlation_threshold", 0.80, 0.99, "anti_gaming", "Prediction correlation threshold for plagiarism"),
    ParameterBounds("copy_trade_time_tolerance",         10,   300,   "anti_gaming", "Time window (sec) for copy-trade matching"),
    ParameterBounds("copy_trade_size_tolerance",         0.01, 0.15,  "anti_gaming", "Size tolerance (%) for copy-trade matching"),
    ParameterBounds("copy_trade_correlation_threshold",  0.75, 0.98,  "anti_gaming", "Correlation threshold for copy-trade detection"),

    # L2 Trading Engine
    ParameterBounds("slippage_base_spread_bps",     0.5,  10.0, "trading", "Base spread in basis points"),
    ParameterBounds("slippage_vol_impact_factor",   0.1,   2.0, "trading", "Volatility impact multiplier"),
    ParameterBounds("slippage_size_impact_factor",  0.01,  0.5, "trading", "Size impact multiplier"),
    ParameterBounds("slippage_fee_bps",             1.0,  15.0, "trading", "Taker fee in basis points"),
    ParameterBounds("trading_max_position_pct",     0.02,  0.20, "trading", "Max position as % of portfolio"),
    ParameterBounds("trading_max_drawdown_pct",     0.10,  0.35, "trading", "Max drawdown before kill switch"),

    # Buyback
    ParameterBounds("buyback_pct",             0.05, 0.50, "buyback", "% of firm P&L used for buybacks"),
    ParameterBounds("buyback_min_profit",      100,  5000, "buyback", "Min P&L before buyback triggers"),
]

N_PARAMS = len(PARAMETER_DEFINITIONS)
PARAM_NAMES = [p.name for p in PARAMETER_DEFINITIONS]


def get_bounds() -> Tuple[np.ndarray, np.ndarray]:
    """Return (lower_bounds, upper_bounds) arrays for the optimizer."""
    lower = np.array([p.lower for p in PARAMETER_DEFINITIONS])
    upper = np.array([p.upper for p in PARAMETER_DEFINITIONS])
    return lower, upper


def get_group_indices() -> Dict[str, List[int]]:
    """Return parameter indices grouped by category."""
    groups: Dict[str, List[int]] = {}
    for i, p in enumerate(PARAMETER_DEFINITIONS):
        groups.setdefault(p.group, []).append(i)
    return groups


def repair_weights(x: np.ndarray) -> np.ndarray:
    """
    Repair operator: normalize L1 and L2 weight groups to sum to 1.0.
    Preserves relative proportions while enforcing the constraint.
    """
    x = x.copy()
    groups = get_group_indices()

    for group_name in ("l1_weights", "l2_weights"):
        indices = groups[group_name]
        vals = x[indices]
        total = vals.sum()
        if total > 0:
            x[indices] = vals / total
        else:
            x[indices] = 1.0 / len(indices)

    return x


def decode(x: np.ndarray) -> Dict[str, Any]:
    """
    Decode a flat parameter vector into structured configuration objects.
    Returns a dict with all config objects needed to run the subnet.
    """
    x = repair_weights(x)
    p = {name: float(val) for name, val in zip(PARAM_NAMES, x)}

    weight_config = WeightConfig(
        l1_penalized_f1=p["l1_penalized_f1"],
        l1_penalized_sharpe=p["l1_penalized_sharpe"],
        l1_max_drawdown=p["l1_max_drawdown"],
        l1_generalization_gap=p["l1_generalization_gap"],
        l1_feature_efficiency=p["l1_feature_efficiency"],
        l1_latency=p["l1_latency"],
        l2_realized_pnl=p["l2_realized_pnl"],
        l2_omega=p["l2_omega"],
        l2_max_drawdown=p["l2_max_drawdown"],
        l2_win_rate=p["l2_win_rate"],
        l2_consistency=p["l2_consistency"],
        l2_model_attribution=p["l2_model_attribution"],
        variance_penalty=p["variance_penalty"],
        n_eval_windows=int(round(p["n_eval_windows"])),
    )

    promotion_config = PromotionConfig(
        top_n=int(round(p["promotion_top_n"])),
        min_consecutive_epochs=int(round(p["promotion_min_consecutive_epochs"])),
        max_overfitting_score=p["promotion_max_overfitting_score"],
        max_score_decay_pct=p["promotion_max_score_decay_pct"],
        expiry_epochs_without_usage=int(round(p["promotion_expiry_epochs"])),
    )

    return {
        "weight_config": weight_config,
        "promotion_config": promotion_config,
        "raw_params": p,
        "variance_penalty_config": {
            "variance_penalty": p["variance_penalty"],
            "n_eval_windows": int(round(p["n_eval_windows"])),
        },
        "feedback": {
            "bonus_weight": p["feedback_bonus_weight"],
            "penalty_weight": p["feedback_penalty_weight"],
        },
        "anti_gaming": {
            "fingerprint_correlation_threshold": p["fingerprint_correlation_threshold"],
            "copy_trade_time_tolerance": p["copy_trade_time_tolerance"],
            "copy_trade_size_tolerance": p["copy_trade_size_tolerance"],
            "copy_trade_correlation_threshold": p["copy_trade_correlation_threshold"],
        },
        "trading": {
            "base_spread_bps": p["slippage_base_spread_bps"],
            "volatility_impact_factor": p["slippage_vol_impact_factor"],
            "size_impact_factor": p["slippage_size_impact_factor"],
            "fee_bps": p["slippage_fee_bps"],
            "max_position_pct": p["trading_max_position_pct"],
            "max_drawdown_pct": p["trading_max_drawdown_pct"],
        },
        "buyback": {
            "buyback_pct": p["buyback_pct"],
            "min_profit_threshold": p["buyback_min_profit"],
        },
    }


def encode_defaults() -> np.ndarray:
    """Encode the default parameter configuration as a flat vector."""
    defaults = {
        "l1_penalized_f1": 0.25, "l1_penalized_sharpe": 0.25, "l1_max_drawdown": 0.15,
        "l1_generalization_gap": 0.20, "l1_feature_efficiency": 0.05,
        "l1_latency": 0.10,
        "l2_realized_pnl": 0.25, "l2_omega": 0.20, "l2_max_drawdown": 0.15,
        "l2_win_rate": 0.10, "l2_consistency": 0.20, "l2_model_attribution": 0.10,
        "variance_penalty": 0.5, "n_eval_windows": 5,
        "promotion_top_n": 10, "promotion_min_consecutive_epochs": 2,
        "promotion_max_overfitting_score": 0.40, "promotion_max_score_decay_pct": 0.20,
        "promotion_expiry_epochs": 5,
        "feedback_bonus_weight": 0.15, "feedback_penalty_weight": 0.10,
        "fingerprint_correlation_threshold": 0.95, "copy_trade_time_tolerance": 60,
        "copy_trade_size_tolerance": 0.05, "copy_trade_correlation_threshold": 0.90,
        "slippage_base_spread_bps": 2.0, "slippage_vol_impact_factor": 0.5,
        "slippage_size_impact_factor": 0.1, "slippage_fee_bps": 5.0,
        "trading_max_position_pct": 0.10, "trading_max_drawdown_pct": 0.20,
        "buyback_pct": 0.20, "buyback_min_profit": 1000.0,
    }
    return np.array([defaults[name] for name in PARAM_NAMES])


def summarize_config(config: Dict[str, Any]) -> str:
    """Human-readable summary of a decoded configuration."""
    lines = []
    p = config["raw_params"]

    lines.append("=== L1 Scoring Weights (mean - λ·std formulation) ===")
    for k in ["l1_penalized_f1", "l1_penalized_sharpe", "l1_max_drawdown",
              "l1_generalization_gap", "l1_feature_efficiency", "l1_latency"]:
        lines.append(f"  {k}: {p[k]:.4f}")

    lines.append("=== L2 Scoring Weights ===")
    for k in ["l2_realized_pnl", "l2_omega", "l2_max_drawdown",
              "l2_win_rate", "l2_consistency", "l2_model_attribution"]:
        lines.append(f"  {k}: {p[k]:.4f}")

    lines.append("=== Variance Penalty (mean - λ·std) ===")
    lines.append(f"  variance_penalty (λ): {p['variance_penalty']:.4f}")
    lines.append(f"  n_eval_windows (K): {int(p['n_eval_windows'])}")

    lines.append("=== Promotion ===")
    lines.append(f"  top_n: {int(p['promotion_top_n'])}")
    lines.append(f"  min_consecutive_epochs: {int(p['promotion_min_consecutive_epochs'])}")
    lines.append(f"  max_overfitting_score: {p['promotion_max_overfitting_score']:.4f}")

    lines.append("=== Feedback ===")
    lines.append(f"  bonus_weight: {p['feedback_bonus_weight']:.4f}")
    lines.append(f"  penalty_weight: {p['feedback_penalty_weight']:.4f}")

    return "\n".join(lines)


if __name__ == "__main__":
    print(f"Parameter space: {N_PARAMS} parameters")
    lower, upper = get_bounds()
    print(f"Bounds shape: lower={lower.shape}, upper={upper.shape}")

    defaults = encode_defaults()
    config = decode(defaults)
    print("\nDefault configuration:")
    print(summarize_config(config))

    print("\nL1 weights sum:", sum(
        config["raw_params"][k] for k in
        ["l1_penalized_f1", "l1_penalized_sharpe", "l1_max_drawdown",
         "l1_generalization_gap", "l1_feature_efficiency", "l1_latency"]
    ))
    print("L2 weights sum:", sum(
        config["raw_params"][k] for k in
        ["l2_realized_pnl", "l2_omega", "l2_max_drawdown",
         "l2_win_rate", "l2_consistency", "l2_model_attribution"]
    ))
