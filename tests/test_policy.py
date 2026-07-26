"""NAV band, routing, circuit breakers (SPEC §0.5, §5)."""

import pytest

from treasury.policy import (
    BandAction,
    CircuitBreakers,
    MarketState,
    NavBand,
    RoutingPolicy,
    buy_flow_allowed,
    premium_discount,
    reserve_months,
    tripped_breakers,
)


def test_band_actions():
    band = NavBand()
    assert band.action(0.85, 1.0) is BandAction.BUY
    assert band.action(1.0, 1.0) is BandAction.HOLD
    assert band.action(1.15, 1.0) is BandAction.STOP_BUYING


def test_premium_discount():
    assert premium_discount(1.1, 1.0) == pytest.approx(0.10)
    with pytest.raises(ValueError):
        premium_discount(1.0, 0.0)


def test_routing_split_sums_and_banks_above_nav():
    routing = RoutingPolicy()
    at_discount = routing.route(100.0, BandAction.BUY)
    assert sum(at_discount.values()) == pytest.approx(100.0)
    assert at_discount["buy_flow"] == pytest.approx(50.0)

    at_premium = routing.route(100.0, BandAction.STOP_BUYING)
    assert at_premium["buy_flow"] == 0.0
    # the buy-flow tranche accrues to reserve — revenue is never spent above NAV
    assert at_premium["reserve"] == pytest.approx(75.0)
    assert sum(at_premium.values()) == pytest.approx(100.0)


def test_invalid_routing_split_rejected():
    with pytest.raises(ValueError):
        RoutingPolicy(buy_flow=0.9, otc_inventory=0.2, reserve=0.25, ops=0.05)


def _market(**overrides) -> MarketState:
    base = dict(spot_price=1.0, ma30_price=1.0, reserve_tao=6_000.0,
                trailing_monthly_buy_flow=[1_000.0] * 6, emission_share_wow_change=0.0)
    base.update(overrides)
    return MarketState(**base)


def test_circuit_breakers():
    assert tripped_breakers(_market()) == []
    assert len(tripped_breakers(_market(spot_price=1.3))) == 1          # chasing
    assert len(tripped_breakers(_market(reserve_tao=2_000.0))) == 1     # reserve < 3 months
    assert len(tripped_breakers(_market(emission_share_wow_change=-0.2))) == 1  # regime change


def test_buy_flow_requires_band_and_breakers():
    assert buy_flow_allowed(BandAction.BUY, _market())
    assert not buy_flow_allowed(BandAction.HOLD, _market())
    assert not buy_flow_allowed(BandAction.BUY, _market(reserve_tao=1_000.0))


def test_reserve_months_uses_trailing_median():
    state = _market(reserve_tao=5_000.0, trailing_monthly_buy_flow=[500.0, 1_000.0, 2_000.0])
    assert reserve_months(state) == pytest.approx(5.0)
    assert reserve_months(_market(trailing_monthly_buy_flow=[])) == float("inf")
