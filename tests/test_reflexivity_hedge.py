"""Short-hedge relief valve tests for the reflexivity Monte Carlo (SPEC §7, §10.7).

``hedge_relief_frac`` models chain shorting (once shipped) diverting a fraction
of stressed redemption selling away from the spot pool. Default 0.0 must leave
the quarterly grid byte-identical.
"""

from dataclasses import replace

import pytest

from risk import reflexivity
from risk.reflexivity import RevenueShock, ScenarioConfig


SMALL = ScenarioConfig(months=12, n_paths=60, seed=13)
SHOCK_100_6M = RevenueShock(3, 6, 1.0)


def test_zero_relief_matches_default_exactly():
    default = reflexivity.run(replace(SMALL, revenue_shock=SHOCK_100_6M))
    zero = reflexivity.run(replace(SMALL, revenue_shock=SHOCK_100_6M, hedge_relief_frac=0.0))
    assert default.p_spiral == zero.p_spiral
    assert default.p_spiral_by_month == zero.p_spiral_by_month
    assert default.median_terminal_discount == zero.median_terminal_discount
    assert default.p5_terminal_discount == zero.p5_terminal_discount
    assert default.mean_terminal_share == zero.mean_terminal_share
    assert default.summary() == zero.summary()


def test_hedge_relief_does_not_worsen_spiral_probability():
    baseline = reflexivity.run(replace(SMALL, revenue_shock=SHOCK_100_6M))
    hedged = reflexivity.run(replace(SMALL, revenue_shock=SHOCK_100_6M, hedge_relief_frac=0.4))
    assert hedged.p_spiral <= baseline.p_spiral
    assert hedged.median_terminal_discount >= baseline.median_terminal_discount
    assert "hedge relief" in hedged.summary()


def test_hedge_relief_bounds():
    with pytest.raises(ValueError):
        reflexivity.run(replace(SMALL, hedge_relief_frac=-0.01))
    with pytest.raises(ValueError):
        reflexivity.run(replace(SMALL, hedge_relief_frac=1.01))
