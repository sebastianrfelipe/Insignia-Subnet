"""On-chain execution rules for treasury buy-flow (SPEC §0.13, §5).

Non-negotiables encoded here:
- every pool order is limit-bounded (`add-stake-limit`, allow_partial=False);
- large orders are MEV-shielded (`submit_shielded`, ML-KEM-768) — shield AND
  bound, they defend against different attacks;
- orders are TWAP-sliced with randomized timing, each slice inside the
  slippage budget.
"""

from treasury.execution.twap import ExecutionLimits, TwapPlan, TwapSlice, plan_twap

__all__ = ["ExecutionLimits", "TwapPlan", "TwapSlice", "plan_twap"]
