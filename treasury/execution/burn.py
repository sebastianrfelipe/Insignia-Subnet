"""Slash-settlement burn pipeline (SPEC §5; INCENTIVE_MECHANISM §Deployment
Collateral).

Two legs per settlement batch, both limit-bounded:
  1. slash leg  — unstake slashed escrow alpha into the pool → TAO proceeds;
  2. burn leg   — subnet owner calls `add_stake_burn` with those proceeds:
                  TAO enters the pool reserve, the AMM-equivalent alpha is
                  removed from the alpha reserve and burned in the same
                  transaction.

Net effect: circulating alpha falls by ≈ the slashed amount less fees on both
legs, pool TAO round-trips. `add_stake_burn` is rate-limited to ONE call per
tempo per subnet (`AddStakeBurnRateLimitExceeded` on violation) — settlement is
therefore batched per tempo, and oversized batches split across tempos to stay
inside the slippage budget.

M2 gate: as with lockmgr.locks, extrinsic names/arguments must be verified
end-to-end on testnet before mainnet use.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from chainio import PoolSnapshot
from treasury import pool_math


class BurnCallError(RuntimeError):
    pass


class BurnRateLimited(BurnCallError):
    """Raised locally before submission would hit AddStakeBurnRateLimitExceeded."""


@dataclass(frozen=True)
class BurnLimits:
    """The batch bound targets the SLASH LEG's price impact, not the supply
    cost: the round-trip supply shortfall is ≈ 2× the input-side fee regardless
    of size (the legs' slippage cancels — see RoundTrip), but between the legs
    the pool sits displaced and front-runnable, and that transient displacement
    IS monotone in batch size."""

    max_batch_slippage_bps: float = 30.0   # slash-leg realized-vs-spot budget
    burn_limit_buffer: float = 0.01        # min alpha out = quote × (1 − buffer)
    max_tempos_pending: int = 8            # alert threshold: queue older than this


@dataclass(frozen=True)
class RoundTrip:
    """Pure math for one settlement batch against a pool snapshot."""

    alpha_slashed: float
    tao_proceeds: float          # slash leg: quote_unstake(alpha_slashed)
    alpha_burned: float          # burn leg: quote_add_stake(post-unstake pool, proceeds)
    pool_after: PoolSnapshot

    @property
    def supply_reduction_shortfall(self) -> float:
        """(slashed − burned) / slashed. Near-size-invariant at ≈ 2× the
        input-side fee: the burn leg re-buys on the very pool the slash leg
        displaced, so the legs' slippage cancels and only fees remain. Batch
        sizing is therefore bounded on the slash leg's transient price impact
        (BurnLimits), not on this."""
        return 1.0 - self.alpha_burned / self.alpha_slashed if self.alpha_slashed > 0 else 0.0


def settlement_round_trip(pool: PoolSnapshot, alpha_slashed: float) -> RoundTrip:
    if alpha_slashed <= 0:
        return RoundTrip(0.0, 0.0, 0.0, pool)
    tao = pool_math.quote_unstake(pool, alpha_slashed)
    after_unstake = replace(pool, tao_reserve=pool.tao_reserve - tao,
                            alpha_reserve=pool.alpha_reserve + alpha_slashed)
    burned = pool_math.quote_add_stake(after_unstake, tao)
    pool_after = replace(after_unstake, tao_reserve=after_unstake.tao_reserve + tao,
                         alpha_reserve=after_unstake.alpha_reserve - burned)
    return RoundTrip(alpha_slashed, tao, burned, pool_after)


@dataclass(frozen=True)
class BurnBatch:
    alpha_to_unstake: float
    expected_tao: float
    min_alpha_burned: float      # limit for the burn leg (slippage tolerance)


@dataclass(frozen=True)
class SettlementPlan:
    """One batch per tempo, sized to the slippage budget. `batches[0]` executes
    this tempo; the remainder is the queue carried forward."""

    batches: tuple[BurnBatch, ...]
    total_alpha: float


def plan_settlement(pool: PoolSnapshot, pending_alpha: float,
                    limits: BurnLimits = BurnLimits()) -> SettlementPlan:
    """Split `pending_alpha` into per-tempo batches whose SLASH-LEG price
    impact (realized vs spot, pool_math.exit_slippage) stays inside
    `max_batch_slippage_bps`. Halves the batch until it fits — exit slippage is
    monotone in size, so this terminates unless the budget sits below the
    input-side fee floor. Batches re-quote against post-batch reserves; in
    production each executes in its own tempo and SHOULD be re-planned against
    live reserves first (emissions and buy-flow refill between tempos)."""
    batches: list[BurnBatch] = []
    remaining = pending_alpha
    current = pool
    floor = pending_alpha * 1e-6
    while remaining > 1e-9:
        size = remaining
        while -pool_math.exit_slippage(current, size) * 10_000 > limits.max_batch_slippage_bps:
            size /= 2.0
            if size < floor:
                raise ValueError(
                    "slippage budget is below the input-side fee floor "
                    f"(~{pool.fee_rate * 10_000:.0f} bps); raise max_batch_slippage_bps")
        trip = settlement_round_trip(current, size)
        batches.append(BurnBatch(
            alpha_to_unstake=size,
            expected_tao=trip.tao_proceeds,
            min_alpha_burned=trip.alpha_burned * (1.0 - limits.burn_limit_buffer),
        ))
        remaining -= size
        current = trip.pool_after
    return SettlementPlan(tuple(batches), pending_alpha)


@dataclass
class BurnRateLimiter:
    """One add_stake_burn per tempo per subnet. Tracks the last successful burn
    block locally so the pipeline never submits into a guaranteed
    AddStakeBurnRateLimitExceeded."""

    tempo_blocks: int
    last_burn_block: int | None = None

    def can_burn(self, current_block: int) -> bool:
        if self.last_burn_block is None:
            return True
        return current_block // self.tempo_blocks > self.last_burn_block // self.tempo_blocks

    def record(self, block: int) -> None:
        self.last_burn_block = block

    def blocks_until_allowed(self, current_block: int) -> int:
        if self.can_burn(current_block):
            return 0
        next_tempo_start = (current_block // self.tempo_blocks + 1) * self.tempo_blocks
        return next_tempo_start - current_block


class BurnClient:
    """Extrinsic wrapper for the settlement legs, modeled on lockmgr.locks.
    Signs with the escrow coldkey for the slash leg and the OWNER coldkey for
    add_stake_burn (owner-only extrinsic) — inject the matching wallet."""

    def __init__(self, subtensor: Any, escrow_wallet: Any, owner_wallet: Any, netuid: int):
        self._st = subtensor
        self._escrow = escrow_wallet
        self._owner = owner_wallet
        self._netuid = netuid

    def _compose(self, wallet: Any, call_function: str, call_params: dict) -> Any:
        substrate = self._st.substrate
        call = substrate.compose_call(
            call_module="SubtensorModule",
            call_function=call_function,
            call_params=call_params,
        )
        extrinsic = substrate.create_signed_extrinsic(call=call, keypair=wallet.coldkey)
        receipt = substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
        if not receipt.is_success:
            message = str(receipt.error_message)
            if "AddStakeBurnRateLimitExceeded" in message:
                raise BurnRateLimited(message)
            raise BurnCallError(f"{call_function} failed: {message}")
        return receipt

    def escrow_return(self, hotkey: str, dest_coldkey: str, amount_alpha: float) -> Any:
        """Bond release back to a miner coldkey — same-subnet transfer_stake,
        no swap, no price impact (SPEC §0.13)."""
        return self._compose(self._escrow, "transfer_stake", {
            "destination_coldkey": dest_coldkey,
            "hotkey": hotkey,
            "origin_netuid": self._netuid,
            "destination_netuid": self._netuid,
            "alpha_amount": _to_rao(amount_alpha),
        })

    def slash_leg(self, hotkey: str, batch: BurnBatch) -> Any:
        """Unstake slashed escrow alpha, limit-bounded (remove_stake_limit,
        allow_partial=False — all-or-nothing inside the planned batch)."""
        limit_price = batch.expected_tao / batch.alpha_to_unstake
        return self._compose(self._escrow, "remove_stake_limit", {
            "hotkey": hotkey,
            "netuid": self._netuid,
            "amount_unstaked": _to_rao(batch.alpha_to_unstake),
            "limit_price": _to_rao(limit_price * (1.0 - 0.01)),
            "allow_partial": False,
        })

    def burn_leg(self, hotkey: str, batch: BurnBatch, limiter: BurnRateLimiter,
                 current_block: int) -> Any:
        if not limiter.can_burn(current_block):
            raise BurnRateLimited(
                f"add_stake_burn already used this tempo; retry in "
                f"{limiter.blocks_until_allowed(current_block)} blocks")
        receipt = self._compose(self._owner, "add_stake_burn", {
            "hotkey": hotkey,
            "netuid": self._netuid,
            "amount_staked": _to_rao(batch.expected_tao),
            "limit_price": _to_rao(batch.expected_tao / batch.min_alpha_burned),
            "allow_partial": False,
        })
        limiter.record(current_block)
        return receipt


def _to_rao(value: float) -> int:
    return int(round(value * 1e9))
