"""Treasury engine: one revenue cycle end-to-end (SPEC §5).

attested revenue → NAV computation → band decision → routing → TWAP plan.
Chain submission is behind the `Executor` protocol so the whole cycle is
testable; production wires an executor built on the bittensor SDK following
treasury/execution rules.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass
from typing import Protocol

from chainio import ParamsProvider
from treasury import pool_math
from treasury.accounting import ConversionLot, NavReport, TreasuryBook, compute_nav
from treasury.execution import TwapPlan, plan_twap
from treasury.policy import (
    BandAction,
    CircuitBreakers,
    MarketState,
    NavBand,
    RoutingPolicy,
    buy_flow_allowed,
    tripped_breakers,
)


class Executor(Protocol):
    """Submits a TWAP plan; returns (tao_spent, alpha_acquired, realised_slippage_bps)."""

    def execute(self, plan: TwapPlan) -> tuple[float, float, float]: ...


@dataclass
class CycleResult:
    nav: NavReport
    action: BandAction
    routed: dict[str, float]
    breakers: list[str]
    plan: TwapPlan | None
    lot: ConversionLot | None


class TreasuryEngine:
    def __init__(self, provider: ParamsProvider, book: TreasuryBook, netuid: int,
                 band: NavBand = NavBand(), routing: RoutingPolicy = RoutingPolicy(),
                 breakers: CircuitBreakers = CircuitBreakers(),
                 accumulation_only: bool = False, twap_window_minutes: float = 24 * 60.0):
        self.provider = provider
        self.book = book
        self.netuid = netuid
        self.band = band
        self.routing = routing
        self.breakers = breakers
        # Phase 1: small scheduled buys only; OTC/issuance paths disabled.
        self.accumulation_only = accumulation_only
        self.twap_window_minutes = twap_window_minutes

    def run_cycle(self, revenue_tao: float, trading_aum_tao: float,
                  circulating_alpha: float, market: MarketState,
                  executor: Executor | None = None,
                  now: dt.datetime | None = None) -> CycleResult:
        now = now or dt.datetime.now(dt.timezone.utc)
        pool = self.provider.pool(self.netuid)
        nav = compute_nav(now, self.book, pool, trading_aum_tao, circulating_alpha)

        action = self.band.action(pool.spot_price, nav.nav_per_alpha)
        tripped = tripped_breakers(market, self.breakers)
        routed = self.routing.route(revenue_tao, action if not tripped else BandAction.HOLD)
        if self.accumulation_only:
            routed["reserve"] += routed.pop("otc_inventory")

        plan, lot = None, None
        if routed.get("buy_flow", 0.0) > 0 and buy_flow_allowed(action, market, self.breakers):
            plan = plan_twap(pool, routed["buy_flow"], self.twap_window_minutes)
            if executor is not None:
                tao_spent, alpha_got, slippage = executor.execute(plan)
                lot = ConversionLot(
                    lot_id=str(uuid.uuid4()), executed_at=now,
                    revenue_usd=0.0,  # revenue arrives pre-converted to TAO here
                    tao_bought=tao_spent, tao_price_usd=0.0,
                    alpha_bought=alpha_got, venue="pool-add-stake-limit",
                    slippage_bps=slippage,
                )
                self.book.record(lot)
        self.book.tao_reserve_balance += routed.get("reserve", 0.0)
        self.book.otc_inventory_alpha += self._inventory_alpha(routed, pool)

        return CycleResult(nav=nav, action=action, routed=routed,
                           breakers=tripped, plan=plan, lot=lot)

    def _inventory_alpha(self, routed: dict[str, float], pool) -> float:
        tao = routed.get("otc_inventory", 0.0)
        return pool_math.quote_add_stake(pool, tao) if tao > 0 else 0.0
