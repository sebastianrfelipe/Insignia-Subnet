"""On-chain execution rules for treasury buy-flow (SPEC §0.13, §5).

Non-negotiables encoded here:
- every pool order is limit-bounded (`add-stake-limit`, allow_partial=False);
- large orders are MEV-shielded (`submit_shielded`, ML-KEM-768) — shield AND
  bound, they defend against different attacks;
- orders are TWAP-sliced with randomized timing, each slice inside the
  slippage budget;
- slash settlement burns are batched one per tempo (add_stake_burn rate
  limit), each batch inside its round-trip slippage budget.
"""

from treasury.execution.burn import (
    BurnBatch,
    BurnClient,
    BurnLimits,
    BurnRateLimited,
    BurnRateLimiter,
    SettlementPlan,
    plan_settlement,
    settlement_round_trip,
)
from treasury.execution.twap import ExecutionLimits, TwapPlan, TwapSlice, plan_twap

__all__ = [
    "BurnBatch", "BurnClient", "BurnLimits", "BurnRateLimited", "BurnRateLimiter",
    "ExecutionLimits", "SettlementPlan", "TwapPlan", "TwapSlice",
    "plan_settlement", "plan_twap", "settlement_round_trip",
]
