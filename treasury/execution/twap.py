"""TWAP slicing under a slippage budget, with limit and shield decisions.

Pure planning; submission happens through the bittensor SDK at the call site
(`add_stake_limit(..., allow_partial=False)`, `submit_shielded` for flagged
slices). Cost of moving price is convex (Δy = y·((p′/p)^w1 − 1)), so many small
slices strictly dominate one large one (SPEC §0.5 rule 2).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from chainio import PoolSnapshot
from treasury import pool_math


@dataclass(frozen=True)
class ExecutionLimits:
    max_slice_slippage_bps: float = 30.0   # per-slice effective-price impact budget
    limit_buffer: float = 0.005            # limit_price = spot × (1 + buffer)
    shield_above_reserve_frac: float = 0.001  # shield slices > 0.1% of TAO reserve
    max_slices: int = 96                   # ceiling on plan granularity
    timing_jitter_frac: float = 0.35       # ±35% jitter on slice spacing (MEV-aware)


@dataclass(frozen=True)
class TwapSlice:
    tao_in: float
    limit_price: float
    shielded: bool
    offset_minutes: float


@dataclass(frozen=True)
class TwapPlan:
    slices: tuple[TwapSlice, ...]
    total_tao: float
    window_minutes: float

    @property
    def shielded_count(self) -> int:
        return sum(1 for s in self.slices if s.shielded)


def _slippage_bps(pool: PoolSnapshot, tao_in: float) -> float:
    alpha_out = pool_math.quote_add_stake(pool, tao_in)
    if alpha_out <= 0:
        return 0.0
    effective_price = tao_in / alpha_out
    return (effective_price / pool.spot_price - 1.0) * 1e4


def max_slice_within_budget(pool: PoolSnapshot, budget_bps: float) -> float:
    """Largest single buy whose effective price stays within `budget_bps` of
    spot, found by bisection against the pool quote."""
    lo, hi = 0.0, pool.tao_reserve
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _slippage_bps(pool, mid) <= budget_bps:
            lo = mid
        else:
            hi = mid
    return lo


def plan_twap(pool: PoolSnapshot, total_tao: float, window_minutes: float,
              limits: ExecutionLimits = ExecutionLimits(),
              seed: int | None = None) -> TwapPlan:
    if total_tao <= 0:
        return TwapPlan(slices=(), total_tao=0.0, window_minutes=window_minutes)

    max_slice = max_slice_within_budget(pool, limits.max_slice_slippage_bps)
    if max_slice <= 0:
        raise ValueError("slippage budget unmeetable against current reserves")
    n = min(limits.max_slices, max(1, -(-total_tao // max_slice).__int__()))
    slice_tao = total_tao / n
    limit_price = pool.spot_price * (1.0 + limits.limit_buffer)
    shielded = slice_tao > limits.shield_above_reserve_frac * pool.tao_reserve

    rng = random.Random(seed)
    spacing = window_minutes / n
    slices = []
    for i in range(n):
        jitter = rng.uniform(-limits.timing_jitter_frac, limits.timing_jitter_frac)
        slices.append(TwapSlice(
            tao_in=slice_tao,
            limit_price=limit_price,
            shielded=shielded,
            offset_minutes=max(0.0, (i + 0.5 + jitter) * spacing),
        ))
    return TwapPlan(slices=tuple(slices), total_tao=total_tao, window_minutes=window_minutes)
