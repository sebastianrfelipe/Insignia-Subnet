"""Balancer weighted-pool math for the subnet's TAO/alpha pool (SPEC §0.13–0.14).

Sign conventions: quotes return the OUT-side amount for a given IN-side amount.
The pool fee is charged off the input side. Everything here is pure and takes a
`PoolSnapshot`; live reserves come from chainio providers.
"""

from __future__ import annotations

import math
from typing import Iterable

from chainio import PoolSnapshot

# Chain rejects single swaps above this multiple of the TAO reserve.
INSUFFICIENT_LIQUIDITY_MULT = 1_000.0


class InsufficientLiquidity(ValueError):
    pass


def _swap_out(out_reserve: float, out_weight: float,
              in_reserve: float, in_weight: float,
              amount_in: float, fee_rate: float) -> float:
    """Weighted constant-product out-given-in:
    Δout = out_res · (1 − (in_res / (in_res + Δin·(1−fee)))^(w_in/w_out))."""
    if amount_in <= 0:
        return 0.0
    net_in = amount_in * (1.0 - fee_rate)
    ratio = in_reserve / (in_reserve + net_in)
    return out_reserve * (1.0 - ratio ** (in_weight / out_weight))


def quote_add_stake(pool: PoolSnapshot, tao_in: float) -> float:
    """Alpha received for staking `tao_in` TAO into the pool."""
    if tao_in > INSUFFICIENT_LIQUIDITY_MULT * pool.tao_reserve:
        raise InsufficientLiquidity(
            f"swap of {tao_in:,.0f} TAO exceeds {INSUFFICIENT_LIQUIDITY_MULT:g}× reserve")
    return _swap_out(pool.alpha_reserve, pool.w_alpha, pool.tao_reserve, pool.w_tao,
                     tao_in, pool.fee_rate)


def quote_unstake(pool: PoolSnapshot, alpha_in: float) -> float:
    """TAO realised by unstaking `alpha_in` alpha — the ONLY valid way to value a
    position (SPEC §0.14: NAV must be quoted, not marked)."""
    return _swap_out(pool.tao_reserve, pool.w_tao, pool.alpha_reserve, pool.w_alpha,
                     alpha_in, pool.fee_rate)


def spot_value(pool: PoolSnapshot, alpha_amount: float) -> float:
    return alpha_amount * pool.spot_price


def exit_slippage(pool: PoolSnapshot, alpha_amount: float) -> float:
    """Realised-vs-spot shortfall of a single-transaction exit (negative)."""
    spot = spot_value(pool, alpha_amount)
    return quote_unstake(pool, alpha_amount) / spot - 1.0 if spot > 0 else 0.0


def _after_unstake(pool: PoolSnapshot, alpha_in: float, tao_out: float) -> PoolSnapshot:
    from dataclasses import replace
    return replace(pool, tao_reserve=pool.tao_reserve - tao_out,
                   alpha_reserve=pool.alpha_reserve + alpha_in)


def staged_redemption(pool: PoolSnapshot, increments: Iterable[float],
                      refill: bool = False) -> float:
    """Cumulative TAO from unstaking `increments` sequentially, re-quoting each
    against post-trade reserves.

    With refill=False reserves only deplete, which UNDERSTATES the benefit of
    staging — in reality the pool refills between increments via emission
    injections and buy-flow (the v6 model's conservative assumption).
    """
    total = 0.0
    current = pool
    for alpha in increments:
        tao = quote_unstake(current, alpha)
        total += tao
        if not refill:
            current = _after_unstake(current, alpha, tao)
        # refill=True: quote every increment at the original reserves
    return total


def decay_schedule_increments(position: float, half_life_days: float,
                              horizon_days: float, step_days: float = 1.0) -> list[float]:
    """Redeemable-alpha increments released by the exponential decay schedule —
    the natural staging the lock decay imposes (SPEC §4: cliff exits prohibited)."""
    lam = math.log(2) / half_life_days
    increments = []
    prev_redeemed = 0.0
    t = step_days
    while t <= horizon_days + 1e-9:
        redeemed = position * (1.0 - math.exp(-lam * t))
        increments.append(redeemed - prev_redeemed)
        prev_redeemed = redeemed
        t += step_days
    return increments


def price_move_cost(pool: PoolSnapshot, price_ratio: float) -> float:
    """TAO required to move spot price by `price_ratio` (p′/p):
    Δy = y·((p′/p)^w1 − 1). Convex — +10% costs 4.9% of the TAO reserve,
    +100% costs 41.4% (SPEC §0.5 'optimal rate')."""
    if price_ratio < 1.0:
        raise ValueError("price_move_cost quotes upward moves; ratio must be ≥ 1")
    return pool.tao_reserve * (price_ratio ** pool.w_tao - 1.0)


def max_fill_at_limit(pool: PoolSnapshot, limit_price: float) -> float:
    """Maximum TAO fillable by add-stake-limit before spot reaches `limit_price`.
    Same curve as price_move_cost; 0 when the limit is already breached."""
    ratio = limit_price / pool.spot_price
    if ratio <= 1.0:
        return 0.0
    return price_move_cost(pool, ratio)


def move_stake_same_subnet(alpha_amount: float) -> float:
    """`move-stake` between hotkeys on the SAME subnet is not a swap — no fee,
    no price impact (SPEC §0.13). Use for OTC delivery and owner-hotkey
    migration. Cross-subnet moves run two swaps; quote those explicitly."""
    return alpha_amount
