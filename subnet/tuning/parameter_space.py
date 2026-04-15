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
    # L1 Scoring Weights (7 params, must sum to 1.0)
    ParameterBounds("l1_penalized_f1",         0.05, 0.40, "l1_weights", "Weight for Penalized F1"),
    ParameterBounds("l1_penalized_sharpe",     0.05, 0.40, "l1_weights", "Weight for Penalized Sharpe Ratio"),
    ParameterBounds("l1_max_drawdown",         0.05, 0.30, "l1_weights", "Weight for max drawdown penalty"),
    ParameterBounds("l1_variance_score",       0.05, 0.30, "l1_weights", "Weight for Variance Score (cross-regime consistency)"),
    ParameterBounds("l1_overfitting_penalty",  0.05, 0.35, "l1_weights", "Weight for overfitting penalty"),
    ParameterBounds("l1_feature_efficiency",   0.01, 0.15, "l1_weights", "Weight for feature efficiency"),
    ParameterBounds("l1_latency",              0.01, 0.20, "l1_weights", "Weight for latency score"),

    # L2 Scoring Weights (10 params, must sum to 1.0)
    ParameterBounds("l2_realized_pnl",            0.05, 0.40, "l2_weights", "Weight for realized P&L"),
    ParameterBounds("l2_omega",                   0.05, 0.30, "l2_weights", "Weight for Omega ratio"),
    ParameterBounds("l2_max_drawdown",            0.05, 0.30, "l2_weights", "Weight for max drawdown"),
    ParameterBounds("l2_win_rate",                0.02, 0.25, "l2_weights", "Weight for win rate"),
    ParameterBounds("l2_consistency",             0.05, 0.30, "l2_weights", "Weight for consistency"),
    ParameterBounds("l2_model_attribution",       0.01, 0.25, "l2_weights", "Weight for model attribution"),
    ParameterBounds("l2_execution_quality",       0.05, 0.30, "l2_weights", "Weight for execution quality (latency, reliability, slippage)"),
    ParameterBounds("l2_annualized_volatility",   0.02, 0.15, "l2_weights", "Weight for annualized volatility (inverted — lower vol = higher score)"),
    ParameterBounds("l2_sharpe_ratio",            0.02, 0.15, "l2_weights", "Weight for Sharpe ratio (risk-adjusted return per unit total vol)"),
    ParameterBounds("l2_sortino_ratio",           0.02, 0.15, "l2_weights", "Weight for Sortino ratio (risk-adjusted return per unit downside vol)"),

    # Overfitting Detector
    ParameterBounds("overfit_gap_threshold", 0.05, 0.40, "overfitting", "IS/OOS gap before penalty kicks in"),
    ParameterBounds("overfit_decay_rate",    1.0,  15.0, "overfitting", "Exponential decay rate for penalty"),

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

    # Emission Distribution (reverse sigmoid)
    ParameterBounds("emission_sigmoid_midpoint",   0.2,  0.8,  "emissions", "Reverse sigmoid midpoint — controls how many miners get high emissions"),
    ParameterBounds("emission_sigmoid_steepness",  1.0,  20.0, "emissions", "Reverse sigmoid steepness — controls drop-off gradient"),
    ParameterBounds("l1_l2_emission_split",        0.40, 0.80, "emissions", "Fraction of emissions to L1 (remainder to L2)"),

    # Rate Limiting
    ParameterBounds("rate_limit_epoch_seconds",    3600, 172800, "rate_limit", "Minimum seconds between miner submissions"),

    # Cross-Layer Feedback Thresholds
    ParameterBounds("feedback_min_l2_epochs",      1,    10,  "feedback_thresholds", "Min L2 epochs of data before feedback applies"),
    ParameterBounds("feedback_bonus_threshold",    0.4,  0.8, "feedback_thresholds", "L2 score above which L1 bonus kicks in"),
    ParameterBounds("feedback_penalty_threshold",  0.1,  0.5, "feedback_thresholds", "L2 score below which L1 penalty kicks in"),

    # --- New parameters from orchestration research (2026-03-29) ---

    # Validation Timing (defend against validator latency exploitation)
    ParameterBounds("min_prediction_lead_time",            5,    60,   "validation_timing", "Min seconds between data publication and trade submission"),
    ParameterBounds("validator_latency_penalty_weight",    0.0,  0.5,  "validation_timing", "Weight applied to penalize high-latency validator scores"),
    ParameterBounds("high_latency_threshold_ms",           500,  5000, "validation_timing", "Latency (ms) above which validator scores are penalized"),
    ParameterBounds("commit_rate_threshold",                0.50, 0.95, "validation_timing", "Minimum acceptable commit participation rate"),
    ParameterBounds("commitment_violation_weight",          0.001, 0.05, "validation_timing", "Weight for strategic non-commitment detection"),
    ParameterBounds("selective_reveal_warning_streak",      1,    3,    "validation_timing", "Consecutive no-reveals before warning only"),
    ParameterBounds("selective_reveal_penalty_streak",      2,    5,    "validation_timing", "Consecutive no-reveals before halving score"),
    ParameterBounds("selective_reveal_zero_streak",         3,    6,    "validation_timing", "Consecutive no-reveals before zeroing score"),

    # Consensus Integrity (defend against miner-validator collusion)
    ParameterBounds("weight_entropy_minimum",              0.5,  2.0,  "consensus_integrity", "Min entropy of validator weight distribution"),
    ParameterBounds("cross_validator_score_variance_max",  0.1,  0.5,  "consensus_integrity", "Max allowed score variance across validators for one miner"),
    ParameterBounds("validator_rotation_max_consecutive_epochs", 3, 10, "consensus_integrity", "Max epochs same validator scores same miner"),
    ParameterBounds("validator_agreement_threshold",       0.1,  0.4,  "consensus_integrity", "Max scoring deviation from validator consensus"),
    ParameterBounds("collusion_detection_lookback_epochs", 5,    20,   "consensus_integrity", "Epochs of history for collusion pattern detection"),

    # Economic mechanisms (report-backed sybil resistance and ensemble improvements)
    ParameterBounds("identity_bond_threshold",             0.50, 0.90, "economic_mechanisms", "Identity verification / bonding threshold for miner admission"),
    ParameterBounds("stake_weight_consensus",              0.10, 0.60, "economic_mechanisms", "Relative weight assigned to stake-informed consensus checks"),
    ParameterBounds("bayesian_model_weight",               0.40, 0.90, "ensemble_detection", "Weight assigned to Bayesian model averaging in ensemble fusion"),
    ParameterBounds("dominant_pair_warning_ratio",         1.05, 2.50, "market_data", "BTC/ETH activity ratio above which sybil pressure warnings intensify"),

    # L1/L2 Cross-Layer Balance
    ParameterBounds("cross_layer_penalty_strength",        0.0,  1.0,  "cross_layer_balance", "Penalty strength for L1/L2 weight skew"),
    ParameterBounds("cross_layer_latency",                 10,   1000, "cross_layer_timing", "Max allowed latency (ms) for cross-layer sync"),
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
        l1_variance_score=p["l1_variance_score"],
        l1_overfitting_penalty=p["l1_overfitting_penalty"],
        l1_feature_efficiency=p["l1_feature_efficiency"],
        l1_latency=p["l1_latency"],
        l2_realized_pnl=p["l2_realized_pnl"],
        l2_omega=p["l2_omega"],
        l2_max_drawdown=p["l2_max_drawdown"],
        l2_win_rate=p["l2_win_rate"],
        l2_consistency=p["l2_consistency"],
        l2_model_attribution=p["l2_model_attribution"],
        l2_execution_quality=p["l2_execution_quality"],
        l2_annualized_volatility=p["l2_annualized_volatility"],
        l2_sharpe_ratio=p["l2_sharpe_ratio"],
        l2_sortino_ratio=p["l2_sortino_ratio"],
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
        "overfitting": {
            "gap_threshold": p["overfit_gap_threshold"],
            "decay_rate": p["overfit_decay_rate"],
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
        "emissions": {
            "sigmoid_midpoint": p["emission_sigmoid_midpoint"],
            "sigmoid_steepness": p["emission_sigmoid_steepness"],
            "l1_l2_split": p["l1_l2_emission_split"],
        },
        "rate_limit": {
            "epoch_seconds": p["rate_limit_epoch_seconds"],
        },
        "feedback_thresholds": {
            "min_l2_epochs": int(round(p["feedback_min_l2_epochs"])),
            "bonus_threshold": p["feedback_bonus_threshold"],
            "penalty_threshold": p["feedback_penalty_threshold"],
        },
        "validation_timing": {
            "min_prediction_lead_time": p["min_prediction_lead_time"],
            "validator_latency_penalty_weight": p["validator_latency_penalty_weight"],
            "high_latency_threshold_ms": p["high_latency_threshold_ms"],
            "commit_rate_threshold": p["commit_rate_threshold"],
            "commitment_violation_weight": p["commitment_violation_weight"],
            "selective_reveal_warning_streak": int(round(p["selective_reveal_warning_streak"])),
            "selective_reveal_penalty_streak": int(round(p["selective_reveal_penalty_streak"])),
            "selective_reveal_zero_streak": int(round(p["selective_reveal_zero_streak"])),
        },
        "consensus_integrity": {
            "weight_entropy_minimum": p["weight_entropy_minimum"],
            "cross_validator_score_variance_max": p["cross_validator_score_variance_max"],
            "validator_rotation_max_consecutive_epochs": int(round(p["validator_rotation_max_consecutive_epochs"])),
            "validator_agreement_threshold": p["validator_agreement_threshold"],
            "collusion_detection_lookback_epochs": int(round(p["collusion_detection_lookback_epochs"])),
        },
        "economic_mechanisms": {
            "identity_bond_threshold": p["identity_bond_threshold"],
            "stake_weight_consensus": p["stake_weight_consensus"],
        },
        "cross_layer_balance": {
            "penalty_strength": p["cross_layer_penalty_strength"],
        },
        "cross_layer_timing": {
            "max_latency_ms": p["cross_layer_latency"],
        },
        "ensemble_detection": {
            "fusion_strategy": "bayesian_model_averaging",
            "correlation_threshold": 0.80,
            "entropy_threshold_lower": 0.20,
            "symbol_diversity_threshold": 0.33,
            "response_vote_threshold": 3,
            "bayesian_weight": p["bayesian_model_weight"],
        },
        "market_data": {
            "trading_pairs": [
                "BTC-USDT-PERP",
                "ETH-USDT-PERP",
                "SOL-USDT-PERP",
                "AVAX-USDT-PERP",
                "ADA-USDT-PERP",
            ],
            "dominant_pair_warning_ratio": p["dominant_pair_warning_ratio"],
        },
        "research_targets": {
            "seed_lineage": ["EXP-116", "EXP-118", "EXP-140", "EXP-141"],
            "best_experiment": "EXP-140",
            "runner_up_experiment": "EXP-141",
            "target_breach_rate": 5e-6,
        },
    }


def encode_defaults() -> np.ndarray:
    """Encode the default parameter configuration as a flat vector."""
    defaults = {
        "l1_penalized_f1": 0.22, "l1_penalized_sharpe": 0.18, "l1_max_drawdown": 0.14,
        "l1_variance_score": 0.16, "l1_overfitting_penalty": 0.14, "l1_feature_efficiency": 0.06,
        "l1_latency": 0.10,
        "l2_realized_pnl": 0.18, "l2_omega": 0.12, "l2_max_drawdown": 0.12,
        "l2_win_rate": 0.05, "l2_consistency": 0.18, "l2_model_attribution": 0.11,
        "l2_execution_quality": 0.09,
        "l2_annualized_volatility": 0.05, "l2_sharpe_ratio": 0.05, "l2_sortino_ratio": 0.05,
        "overfit_gap_threshold": 0.15, "overfit_decay_rate": 5.0,
        "promotion_top_n": 8, "promotion_min_consecutive_epochs": 3,
        "promotion_max_overfitting_score": 0.35, "promotion_max_score_decay_pct": 0.15,
        "promotion_expiry_epochs": 5,
        "feedback_bonus_weight": 0.12, "feedback_penalty_weight": 0.12,
        "fingerprint_correlation_threshold": 0.95, "copy_trade_time_tolerance": 60,
        "copy_trade_size_tolerance": 0.05, "copy_trade_correlation_threshold": 0.90,
        "slippage_base_spread_bps": 2.5, "slippage_vol_impact_factor": 0.6,
        "slippage_size_impact_factor": 0.12, "slippage_fee_bps": 5.0,
        "trading_max_position_pct": 0.08, "trading_max_drawdown_pct": 0.18,
        "buyback_pct": 0.20, "buyback_min_profit": 1000.0,
        "emission_sigmoid_midpoint": 0.50, "emission_sigmoid_steepness": 8.0,
        "l1_l2_emission_split": 0.65,
        "rate_limit_epoch_seconds": 86400,
        "feedback_min_l2_epochs": 3, "feedback_bonus_threshold": 0.62,
        "feedback_penalty_threshold": 0.28,
        # Validation timing (Phase 5 secure-and-improving run held CR effectiveness at 0.700)
        "min_prediction_lead_time": 35,
        "validator_latency_penalty_weight": 0.28,
        "high_latency_threshold_ms": 1800,
        "commit_rate_threshold": 0.75,
        "commitment_violation_weight": 0.012,
        "selective_reveal_warning_streak": 1,
        "selective_reveal_penalty_streak": 2,
        "selective_reveal_zero_streak": 3,
        # Consensus integrity (Phase 5 secure-and-improving profile)
        "weight_entropy_minimum": 1.45,
        "cross_validator_score_variance_max": 0.18,
        "validator_rotation_max_consecutive_epochs": 4,
        "validator_agreement_threshold": 0.17,
        "collusion_detection_lookback_epochs": 12,
        # Economic mechanisms (strongest signal from EXP-140/141 family)
        "identity_bond_threshold": 0.72,
        "stake_weight_consensus": 0.38,
        "bayesian_model_weight": 0.68,
        "dominant_pair_warning_ratio": 1.35,
        # Cross-layer balance
        "cross_layer_penalty_strength": 0.45,
        "cross_layer_latency": 160,
    }
    return np.array([defaults[name] for name in PARAM_NAMES])


def summarize_config(config: Dict[str, Any]) -> str:
    """Human-readable summary of a decoded configuration."""
    lines = []
    p = config["raw_params"]

    lines.append("=== L1 Scoring Weights ===")
    for k in ["l1_penalized_f1", "l1_penalized_sharpe", "l1_max_drawdown",
              "l1_variance_score", "l1_overfitting_penalty", "l1_feature_efficiency", "l1_latency"]:
        lines.append(f"  {k}: {p[k]:.4f}")

    lines.append("=== L2 Scoring Weights ===")
    for k in ["l2_realized_pnl", "l2_omega", "l2_max_drawdown",
              "l2_win_rate", "l2_consistency", "l2_model_attribution",
              "l2_execution_quality", "l2_annualized_volatility",
              "l2_sharpe_ratio", "l2_sortino_ratio"]:
        lines.append(f"  {k}: {p[k]:.4f}")

    lines.append("=== Overfitting Detector ===")
    lines.append(f"  gap_threshold: {p['overfit_gap_threshold']:.4f}")
    lines.append(f"  decay_rate: {p['overfit_decay_rate']:.2f}")

    lines.append("=== Promotion ===")
    lines.append(f"  top_n: {int(p['promotion_top_n'])}")
    lines.append(f"  min_consecutive_epochs: {int(p['promotion_min_consecutive_epochs'])}")
    lines.append(f"  max_overfitting_score: {p['promotion_max_overfitting_score']:.4f}")

    lines.append("=== Feedback ===")
    lines.append(f"  bonus_weight: {p['feedback_bonus_weight']:.4f}")
    lines.append(f"  penalty_weight: {p['feedback_penalty_weight']:.4f}")

    lines.append("=== Validation Timing ===")
    lines.append(f"  min_prediction_lead_time: {p['min_prediction_lead_time']:.1f}s")
    lines.append(f"  validator_latency_penalty_weight: {p['validator_latency_penalty_weight']:.4f}")
    lines.append(f"  high_latency_threshold_ms: {p['high_latency_threshold_ms']:.0f}")
    lines.append(f"  commit_rate_threshold: {p['commit_rate_threshold']:.2f}")
    lines.append(f"  commitment_violation_weight: {p['commitment_violation_weight']:.4f}")
    lines.append(f"  selective_reveal_warning_streak: {int(p['selective_reveal_warning_streak'])}")
    lines.append(f"  selective_reveal_penalty_streak: {int(p['selective_reveal_penalty_streak'])}")
    lines.append(f"  selective_reveal_zero_streak: {int(p['selective_reveal_zero_streak'])}")

    lines.append("=== Consensus Integrity ===")
    lines.append(f"  weight_entropy_minimum: {p['weight_entropy_minimum']:.4f}")
    lines.append(f"  cross_validator_score_variance_max: {p['cross_validator_score_variance_max']:.4f}")
    lines.append(f"  validator_rotation_max_epochs: {int(p['validator_rotation_max_consecutive_epochs'])}")
    lines.append(f"  validator_agreement_threshold: {p['validator_agreement_threshold']:.4f}")
    lines.append(f"  collusion_lookback_epochs: {int(p['collusion_detection_lookback_epochs'])}")

    lines.append("=== Economic Mechanisms ===")
    lines.append(f"  identity_bond_threshold: {p['identity_bond_threshold']:.4f}")
    lines.append(f"  stake_weight_consensus: {p['stake_weight_consensus']:.4f}")
    lines.append(f"  bayesian_model_weight: {p['bayesian_model_weight']:.4f}")
    lines.append(f"  dominant_pair_warning_ratio: {p['dominant_pair_warning_ratio']:.4f}")

    lines.append("=== Cross-Layer Balance ===")
    lines.append(f"  penalty_strength: {p['cross_layer_penalty_strength']:.4f}")
    lines.append(f"  max_latency_ms: {p['cross_layer_latency']:.0f}")

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
         "l1_variance_score", "l1_overfitting_penalty", "l1_feature_efficiency", "l1_latency"]
    ))
    print("L2 weights sum:", sum(
        config["raw_params"][k] for k in
        ["l2_realized_pnl", "l2_omega", "l2_max_drawdown",
         "l2_win_rate", "l2_consistency", "l2_model_attribution",
         "l2_execution_quality", "l2_annualized_volatility",
         "l2_sharpe_ratio", "l2_sortino_ratio"]
    ))
