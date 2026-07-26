"""TWAP planning under the slippage budget (SPEC §0.13, §5)."""

import pytest

from chainio import reference_pool
from treasury import pool_math
from treasury.execution import ExecutionLimits, plan_twap
from treasury.execution.twap import max_slice_within_budget


def test_slices_sum_to_total_and_respect_budget():
    pool = reference_pool()
    limits = ExecutionLimits()
    plan = plan_twap(pool, total_tao=2_000.0, window_minutes=24 * 60, limits=limits, seed=1)
    assert sum(s.tao_in for s in plan.slices) == pytest.approx(2_000.0)
    # per-slice effective price stays within the budget against fresh reserves
    slice_tao = plan.slices[0].tao_in
    alpha_out = pool_math.quote_add_stake(pool, slice_tao)
    impact_bps = (slice_tao / alpha_out / pool.spot_price - 1.0) * 1e4
    assert impact_bps <= limits.max_slice_slippage_bps + 0.5


def test_every_slice_is_limit_bounded():
    pool = reference_pool()
    plan = plan_twap(pool, total_tao=500.0, window_minutes=60, seed=2)
    for s in plan.slices:
        assert s.limit_price > pool.spot_price
        assert s.offset_minutes >= 0.0


def test_large_slices_are_shielded():
    pool = reference_pool()
    limits = ExecutionLimits(max_slices=2, shield_above_reserve_frac=0.001)
    plan = plan_twap(pool, total_tao=pool.tao_reserve * 0.01, window_minutes=60,
                     limits=limits, seed=3)
    assert plan.shielded_count == len(plan.slices) > 0


def test_max_slice_bisection_is_tight():
    pool = reference_pool()
    budget = 30.0
    max_slice = max_slice_within_budget(pool, budget)
    alpha_out = pool_math.quote_add_stake(pool, max_slice)
    impact_bps = (max_slice / alpha_out / pool.spot_price - 1.0) * 1e4
    assert impact_bps == pytest.approx(budget, abs=1.0)


def test_empty_plan_for_zero_flow():
    assert plan_twap(reference_pool(), 0.0, 60).slices == ()
