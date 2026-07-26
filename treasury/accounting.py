"""Lot-level accounting, depth-adjusted NAV, proof-of-reserves (SPEC §0.14, §5).

Hard requirement: NAV is QUOTED, not marked. Treasury alpha is valued at what
`quote-unstake` would realise against live reserves, never at spot × amount —
on the reference pool a 500k-alpha spot mark overstates realisable value by 17%.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from chainio import PoolSnapshot
from treasury import pool_math
from treasury.policy import premium_discount


@dataclass(frozen=True)
class ConversionLot:
    """One revenue → TAO → alpha conversion, recorded at execution time."""

    lot_id: str
    executed_at: dt.datetime
    revenue_usd: float
    tao_bought: float
    tao_price_usd: float
    alpha_bought: float          # 0 for lots parked as TAO reserve
    venue: str                   # "cex-twap", "pool-add-stake-limit", ...
    slippage_bps: float = 0.0


@dataclass
class TreasuryBook:
    lots: list[ConversionLot] = field(default_factory=list)
    tao_reserve_balance: float = 0.0       # unstaked TAO buffer (reflexivity brake)
    alpha_holdings: float = 0.0            # treasury-held alpha (staked)
    otc_inventory_alpha: float = 0.0       # alpha earmarked for OTC delivery

    def record(self, lot: ConversionLot) -> None:
        self.lots.append(lot)
        self.alpha_holdings += lot.alpha_bought
        if lot.alpha_bought == 0:
            self.tao_reserve_balance += lot.tao_bought

    @property
    def total_alpha(self) -> float:
        return self.alpha_holdings + self.otc_inventory_alpha


def depth_adjusted_alpha_value(pool: PoolSnapshot, alpha_amount: float) -> float:
    """TAO realisable for `alpha_amount` via quote-unstake against live reserves."""
    if alpha_amount <= 0:
        return 0.0
    return pool_math.quote_unstake(pool, alpha_amount)


@dataclass(frozen=True)
class NavReport:
    as_of: dt.datetime
    trading_aum_tao: float             # desk AUM attested, converted at TAO/USD
    treasury_tao: float
    treasury_alpha: float
    treasury_alpha_value_tao: float    # depth-adjusted
    treasury_alpha_spot_mark_tao: float  # reported ONLY to show the overstatement
    circulating_alpha: float
    spot_price: float

    @property
    def nav_total_tao(self) -> float:
        return self.trading_aum_tao + self.treasury_tao + self.treasury_alpha_value_tao

    @property
    def nav_per_alpha(self) -> float:
        return self.nav_total_tao / self.circulating_alpha

    @property
    def premium_discount(self) -> float:
        return premium_discount(self.spot_price, self.nav_per_alpha)

    @property
    def depth_haircut(self) -> float:
        """How much spot marking would overstate treasury alpha (negative)."""
        if self.treasury_alpha_spot_mark_tao <= 0:
            return 0.0
        return self.treasury_alpha_value_tao / self.treasury_alpha_spot_mark_tao - 1.0


def compute_nav(as_of: dt.datetime, book: TreasuryBook, pool: PoolSnapshot,
                trading_aum_tao: float, circulating_alpha: float) -> NavReport:
    if circulating_alpha <= 0:
        raise ValueError("circulating alpha must be positive")
    return NavReport(
        as_of=as_of,
        trading_aum_tao=trading_aum_tao,
        treasury_tao=book.tao_reserve_balance,
        treasury_alpha=book.total_alpha,
        treasury_alpha_value_tao=depth_adjusted_alpha_value(pool, book.total_alpha),
        treasury_alpha_spot_mark_tao=pool_math.spot_value(pool, book.total_alpha),
        circulating_alpha=circulating_alpha,
        spot_price=pool.spot_price,
    )


def proof_of_reserves(report: NavReport, book: TreasuryBook) -> dict:
    """Monthly publication payload (SPEC §5). On-chain balances are
    independently verifiable; trading AUM is the attested desk figure — the
    standing NAV-oracle conflict is disclosed in RISK_REGISTER.md."""
    return {
        "as_of": report.as_of.isoformat(),
        "tao_reserve": report.treasury_tao,
        "alpha_holdings": book.alpha_holdings,
        "otc_inventory_alpha": book.otc_inventory_alpha,
        "alpha_depth_adjusted_tao": report.treasury_alpha_value_tao,
        "alpha_spot_mark_tao": report.treasury_alpha_spot_mark_tao,
        "depth_haircut": report.depth_haircut,
        "trading_aum_tao_attested": report.trading_aum_tao,
        "nav_per_alpha": report.nav_per_alpha,
        "premium_discount": report.premium_discount,
        "conversion_lots": len(book.lots),
    }
