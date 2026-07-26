"""Pool math against the published anchors (SPEC §0.13–0.14, §0.5)."""

import math
from dataclasses import replace

import pytest

from chainio import reference_pool
from treasury import pool_math


def test_exit_slippage_matches_spec_anchors():
    pool = reference_pool()
    # SPEC §0.14: unwinding 200k / 500k / 900k alpha realises −7.6% / −17.1% / −27.1% vs spot
    for size, expected in [(200_000, -0.076), (500_000, -0.171), (900_000, -0.271)]:
        assert pool_math.exit_slippage(pool, size) == pytest.approx(expected, abs=0.003)


def test_staged_exit_beats_cliff():
    pool = reference_pool()
    position = 500_000.0
    half_life = 62.4  # chain-derived, ≈ τ·ln2
    increments = pool_math.decay_schedule_increments(position, half_life, horizon_days=365.0)
    staged = pool_math.staged_redemption(pool, increments, refill=True)
    cliff = pool_math.quote_unstake(pool, position)
    spot = position * pool.spot_price
    # v6 model: staged ≈ −1.6% vs spot, cliff ≈ −17.1%
    assert staged / spot - 1.0 == pytest.approx(-0.016, abs=0.01)
    assert staged > cliff * 1.15


def test_price_move_cost_convexity_anchors():
    pool = reference_pool()
    # SPEC §0.5: +10% costs 4.9% of TAO reserve, +50% costs 22.5%, +100% costs 41.4%
    for ratio, frac in [(1.10, 0.049), (1.50, 0.225), (2.00, 0.414)]:
        assert pool_math.price_move_cost(pool, ratio) / pool.tao_reserve == pytest.approx(
            frac, abs=0.001)
    with pytest.raises(ValueError):
        pool_math.price_move_cost(pool, 0.9)


def test_max_fill_lands_spot_at_limit_price():
    pool = replace(reference_pool(), fee_rate=0.0)
    limit = pool.spot_price * 1.2
    fill = pool_math.max_fill_at_limit(pool, limit)
    alpha_out = pool_math.quote_add_stake(pool, fill)
    after = replace(pool, tao_reserve=pool.tao_reserve + fill,
                    alpha_reserve=pool.alpha_reserve - alpha_out)
    assert after.spot_price == pytest.approx(limit, rel=1e-6)
    assert pool_math.max_fill_at_limit(pool, pool.spot_price * 0.9) == 0.0


def test_insufficient_liquidity_guard():
    pool = reference_pool()
    with pytest.raises(pool_math.InsufficientLiquidity):
        pool_math.quote_add_stake(pool, pool.tao_reserve * 1_001)


def test_move_stake_same_subnet_is_not_a_swap():
    assert pool_math.move_stake_same_subnet(123_456.0) == 123_456.0


def test_decay_increments_sum_to_released_mass():
    increments = pool_math.decay_schedule_increments(1000.0, 62.4, horizon_days=365.0)
    released = 1000.0 * (1 - math.exp(-math.log(2) / 62.4 * 365))
    assert sum(increments) == pytest.approx(released, rel=1e-6)
