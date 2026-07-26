"""Depth-adjusted NAV accounting and the treasury cycle (SPEC §0.14, §5)."""

import datetime as dt

import pytest

from chainio import StaticProvider, reference_pool
from treasury import pool_math
from treasury.accounting import TreasuryBook, compute_nav, proof_of_reserves
from treasury.engine import TreasuryEngine
from treasury.policy import BandAction, MarketState

NOW = dt.datetime(2026, 7, 26, tzinfo=dt.timezone.utc)


def _book(alpha: float = 500_000.0, tao: float = 10_000.0) -> TreasuryBook:
    book = TreasuryBook()
    book.alpha_holdings = alpha
    book.tao_reserve_balance = tao
    return book


def test_nav_is_quoted_not_marked():
    pool = reference_pool()
    report = compute_nav(NOW, _book(), pool, trading_aum_tao=100_000.0,
                         circulating_alpha=6e6)
    # treasury alpha valued at quote-unstake, not spot × amount
    assert report.treasury_alpha_value_tao == pytest.approx(
        pool_math.quote_unstake(pool, 500_000.0))
    assert report.treasury_alpha_value_tao < report.treasury_alpha_spot_mark_tao
    # the 500k anchor: spot marking overstates by ~17.1% (SPEC §0.14)
    assert report.depth_haircut == pytest.approx(-0.171, abs=0.003)


def test_proof_of_reserves_payload():
    pool = reference_pool()
    book = _book()
    report = compute_nav(NOW, book, pool, 100_000.0, 6e6)
    payload = proof_of_reserves(report, book)
    assert payload["nav_per_alpha"] == pytest.approx(report.nav_per_alpha)
    assert payload["depth_haircut"] < 0
    assert payload["alpha_spot_mark_tao"] > payload["alpha_depth_adjusted_tao"]


def _healthy_market(pool) -> MarketState:
    return MarketState(spot_price=pool.spot_price, ma30_price=pool.spot_price,
                       reserve_tao=60_000.0, trailing_monthly_buy_flow=[2_000.0] * 6,
                       emission_share_wow_change=0.0)


def test_cycle_buys_at_discount():
    provider = StaticProvider()
    pool = provider.pool()
    engine = TreasuryEngine(provider, TreasuryBook(), netuid=1)
    # deep discount: NAV well above spot
    result = engine.run_cycle(revenue_tao=1_000.0, trading_aum_tao=600_000.0,
                              circulating_alpha=6e6, market=_healthy_market(pool), now=NOW)
    assert result.action is BandAction.BUY
    assert result.plan is not None
    assert result.plan.total_tao == pytest.approx(500.0)  # 50% buy-flow tranche
    assert result.routed["reserve"] == pytest.approx(250.0)


def test_cycle_banks_revenue_above_nav():
    provider = StaticProvider()
    pool = provider.pool()
    engine = TreasuryEngine(provider, TreasuryBook(), netuid=1)
    # spot far above NAV: tiny AUM
    result = engine.run_cycle(revenue_tao=1_000.0, trading_aum_tao=50_000.0,
                              circulating_alpha=6e6, market=_healthy_market(pool), now=NOW)
    assert result.action is BandAction.STOP_BUYING
    assert result.plan is None
    assert result.routed["buy_flow"] == 0.0
    assert result.routed["reserve"] == pytest.approx(750.0)


def test_cycle_halts_on_breaker_even_at_discount():
    provider = StaticProvider()
    pool = provider.pool()
    engine = TreasuryEngine(provider, TreasuryBook(), netuid=1)
    stressed = MarketState(spot_price=pool.spot_price, ma30_price=pool.spot_price,
                           reserve_tao=1_000.0, trailing_monthly_buy_flow=[2_000.0] * 6,
                           emission_share_wow_change=0.0)  # reserve < 3 months
    result = engine.run_cycle(revenue_tao=1_000.0, trading_aum_tao=600_000.0,
                              circulating_alpha=6e6, market=stressed, now=NOW)
    assert result.breakers
    assert result.plan is None
    assert result.routed["buy_flow"] == 0.0
