"""NAV-band treasury policy, revenue routing, and circuit breakers (SPEC §0.5, §5).

The discipline that makes the flywheel legitimate: the treasury runs a
closed-end-fund band around NAV per alpha, never a target price. Buy-flow is
CONDITIONAL on trading at a discount; at or above NAV the correct buy-flow is
zero and the tranche banks to reserve.
"""

from __future__ import annotations

import enum
import statistics
from dataclasses import dataclass, field


def premium_discount(spot_price: float, nav_per_alpha: float) -> float:
    """The factsheet headline metric: spot / NAV − 1."""
    if nav_per_alpha <= 0:
        raise ValueError("NAV per alpha must be positive")
    return spot_price / nav_per_alpha - 1.0


class BandAction(enum.Enum):
    BUY = "buy"                    # TWAP, limit-bounded, shielded
    HOLD = "hold"                  # absorb miner sell-flow only
    STOP_BUYING = "stop_buying"    # consider OTC issuance; bank revenue as reserve


@dataclass(frozen=True)
class NavBand:
    buy_below: float = 0.90        # spot < 0.9×NAV → accretive buys
    issue_above: float = 1.10      # spot > 1.1×NAV → stop; consider OTC issuance

    def action(self, spot_price: float, nav_per_alpha: float) -> BandAction:
        ratio = spot_price / nav_per_alpha
        if ratio < self.buy_below:
            return BandAction.BUY
        if ratio > self.issue_above:
            return BandAction.STOP_BUYING
        return BandAction.HOLD


@dataclass(frozen=True)
class RoutingPolicy:
    """Nominal revenue split when at a discount; governable (SPEC §5)."""

    buy_flow: float = 0.50
    otc_inventory: float = 0.20
    reserve: float = 0.25
    ops: float = 0.05

    def __post_init__(self):
        total = self.buy_flow + self.otc_inventory + self.reserve + self.ops
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"routing split must sum to 1, got {total}")

    def route(self, revenue: float, action: BandAction) -> dict[str, float]:
        """When not buying, the buy-flow tranche accrues to reserve — revenue is
        banked, never spent above NAV."""
        buying = action is BandAction.BUY
        return {
            "buy_flow": revenue * self.buy_flow if buying else 0.0,
            "otc_inventory": revenue * self.otc_inventory,
            "reserve": revenue * (self.reserve + (0.0 if buying else self.buy_flow)),
            "ops": revenue * self.ops,
        }


RESERVE_TARGET_MONTHS = 6.0   # reflexivity brake: ≥ 6 months of trailing median buy-flow


@dataclass(frozen=True)
class CircuitBreakers:
    """Halt buy-flow when tripped (SPEC §5). Thresholds are governable."""

    price_over_ma30_max: float = 1.25       # (a) don't chase: spot > x·30d MA
    reserve_floor_months: float = 3.0       # (b) reserve below 3 months of buy-flow
    share_drop_wow_max: float = 0.15        # (c) emission share −15% WoW despite flow


@dataclass
class MarketState:
    spot_price: float
    ma30_price: float
    reserve_tao: float
    trailing_monthly_buy_flow: list[float] = field(default_factory=list)
    emission_share_wow_change: float = 0.0  # e.g. −0.2 = down 20% week-over-week


def reserve_months(state: MarketState) -> float:
    flows = [f for f in state.trailing_monthly_buy_flow if f > 0]
    if not flows:
        return float("inf")
    return state.reserve_tao / statistics.median(flows)


def tripped_breakers(state: MarketState,
                     breakers: CircuitBreakers = CircuitBreakers()) -> list[str]:
    tripped = []
    if state.ma30_price > 0 and state.spot_price > breakers.price_over_ma30_max * state.ma30_price:
        tripped.append(f"price {state.spot_price / state.ma30_price:.2f}× the 30d MA — do not chase")
    months = reserve_months(state)
    if months < breakers.reserve_floor_months:
        tripped.append(f"reserve covers only {months:.1f} months of median buy-flow")
    if state.emission_share_wow_change < -breakers.share_drop_wow_max:
        tripped.append(
            f"emission share fell {-state.emission_share_wow_change:.0%} WoW despite flow — "
            "possible parameter regime change, investigate before resuming")
    return tripped


def buy_flow_allowed(action: BandAction, state: MarketState,
                     breakers: CircuitBreakers = CircuitBreakers()) -> bool:
    return action is BandAction.BUY and not tripped_breakers(state, breakers)
