"""OTC quote engine: pool-referenced pricing ± lock-commitment schedule (SPEC §6).

Discounts scale with the committed perpetual-lock duration; unlocked delivery
carries a premium and is discouraged. Quotes are depth-aware (referenced to what
the size would actually cost on the pool, not to spot) and NAV-band-aware:
issuing below NAV transfers value from existing holders and is flagged.
"""

from __future__ import annotations

import bisect
import datetime as dt
from dataclasses import dataclass, field

from chainio import PoolSnapshot
from otc.compliance import KycRegistry, require_legal_signoff
from treasury import pool_math
from treasury.policy import NavBand, premium_discount


@dataclass(frozen=True)
class DiscountSchedule:
    """Illustrative defaults; the desk sets the live schedule per LP agreement.
    Keys are committed perpetual-lock months, values are discounts to the
    pool-referenced price (negative = premium for unlocked delivery)."""

    points: tuple[tuple[int, float], ...] = (
        (0, -0.02),   # unlocked delivery: 2% premium, discouraged
        (6, 0.03),
        (12, 0.08),   # the standard term — spans the EMA + root-proportion ramps
        (24, 0.12),
    )

    def discount(self, lock_months: int) -> float:
        months = sorted(m for m, _ in self.points)
        table = dict(self.points)
        idx = bisect.bisect_right(months, lock_months) - 1
        if idx < 0:
            return table[months[0]]
        return table[months[idx]]


@dataclass(frozen=True)
class OtcQuote:
    counterparty_id: str
    alpha_amount: float
    lock_months: int
    pool_reference_price: float    # depth-aware TAO per alpha for this size
    discount: float
    quote_price: float             # TAO per alpha after discount
    total_tao: float
    nav_per_alpha: float | None
    below_nav: bool                # issuance below NAV dilutes existing holders
    expires_at: dt.datetime


@dataclass
class OtcDesk:
    kyc: KycRegistry
    schedule: DiscountSchedule = field(default_factory=DiscountSchedule)
    band: NavBand = field(default_factory=NavBand)
    quote_ttl: dt.timedelta = dt.timedelta(minutes=30)

    def quote(self, counterparty_id: str, pool: PoolSnapshot, alpha_amount: float,
              lock_months: int, nav_per_alpha: float | None = None,
              now: dt.datetime | None = None) -> OtcQuote:
        require_legal_signoff()
        self.kyc.require(counterparty_id)
        if alpha_amount <= 0:
            raise ValueError("alpha_amount must be positive")

        # Depth-aware reference: what buying this size on the pool would cost
        # per alpha, so OTC never quotes tighter than the LP's real alternative.
        tao_cost_on_pool = _pool_cost_for_alpha(pool, alpha_amount)
        reference_price = tao_cost_on_pool / alpha_amount
        discount = self.schedule.discount(lock_months)
        quote_price = reference_price * (1.0 - discount)

        below_nav = False
        if nav_per_alpha is not None:
            below_nav = premium_discount(quote_price, nav_per_alpha) < 0.0

        now = now or dt.datetime.now(dt.timezone.utc)
        return OtcQuote(
            counterparty_id=counterparty_id, alpha_amount=alpha_amount,
            lock_months=lock_months, pool_reference_price=reference_price,
            discount=discount, quote_price=quote_price,
            total_tao=quote_price * alpha_amount, nav_per_alpha=nav_per_alpha,
            below_nav=below_nav, expires_at=now + self.quote_ttl,
        )


def _pool_cost_for_alpha(pool: PoolSnapshot, alpha_target: float) -> float:
    """TAO required to obtain `alpha_target` from the pool (inverse of
    quote_add_stake, by bisection)."""
    if alpha_target >= pool.alpha_reserve:
        raise ValueError("size exceeds pool alpha reserve; quote in tranches")
    lo, hi = 0.0, pool_math.INSUFFICIENT_LIQUIDITY_MULT * pool.tao_reserve
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if pool_math.quote_add_stake(pool, mid) < alpha_target:
            lo = mid
        else:
            hi = mid
    return hi
