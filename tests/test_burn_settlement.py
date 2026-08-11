"""Slash-settlement burn pipeline: round-trip math, per-tempo batching, rate
limit (SPEC §5; INCENTIVE_MECHANISM §Deployment Collateral)."""

import pytest

from chainio import ChainParams, reference_pool
from treasury import pool_math
from treasury.execution.burn import (
    BurnClient,
    BurnLimits,
    BurnRateLimited,
    BurnRateLimiter,
    plan_settlement,
    settlement_round_trip,
)


def test_round_trip_burns_slashed_amount_less_fees():
    pool = reference_pool()
    trip = settlement_round_trip(pool, 1_000.0)
    # Small batch: shortfall ≈ two input-side fees (2 × ~5 bps), well under 0.1%
    assert 0.0 < trip.supply_reduction_shortfall < 2.5 * pool.fee_rate
    assert trip.alpha_burned == pytest.approx(1_000.0, rel=0.005)
    # Pool TAO round-trips exactly (both legs move the same TAO amount)
    assert trip.pool_after.tao_reserve == pytest.approx(pool.tao_reserve)
    # Pool alpha keeps only the unburned residue of the unstaked amount
    assert trip.pool_after.alpha_reserve - pool.alpha_reserve == pytest.approx(
        1_000.0 - trip.alpha_burned)


def test_round_trip_shortfall_is_fees_not_slippage():
    # The burn leg re-buys on the pool the slash leg displaced, so slippage
    # cancels between the legs: supply shortfall ≈ 2× input-side fee at ANY
    # size. Batch bounds exist for the transient inter-leg displacement.
    pool = reference_pool()
    for size in (1_000.0, 50_000.0, 500_000.0):
        trip = settlement_round_trip(pool, size)
        assert trip.supply_reduction_shortfall == pytest.approx(
            2 * pool.fee_rate, rel=0.25)


def test_plan_single_batch_when_inside_budget():
    plan = plan_settlement(reference_pool(), 5_000.0)
    assert len(plan.batches) == 1
    assert plan.batches[0].alpha_to_unstake == pytest.approx(5_000.0)
    assert plan.batches[0].min_alpha_burned < 5_000.0


def test_plan_splits_oversized_queue_across_tempos():
    pool = reference_pool()
    plan = plan_settlement(pool, 1_000_000.0, BurnLimits(max_batch_slippage_bps=20.0))
    assert len(plan.batches) > 1
    assert sum(b.alpha_to_unstake for b in plan.batches) == pytest.approx(1_000_000.0)
    # every batch's slash leg respects the price-impact budget against its own
    # (post-prior-batch) pool
    current = pool
    for batch in plan.batches:
        assert -pool_math.exit_slippage(current, batch.alpha_to_unstake) * 10_000 <= 20.0 + 1e-6
        current = settlement_round_trip(current, batch.alpha_to_unstake).pool_after


def test_plan_rejects_budget_below_fee_floor():
    with pytest.raises(ValueError, match="fee floor"):
        plan_settlement(reference_pool(), 10_000.0, BurnLimits(max_batch_slippage_bps=0.5))


def test_rate_limiter_one_burn_per_tempo():
    tempo = ChainParams().tempo_blocks
    limiter = BurnRateLimiter(tempo_blocks=tempo)
    assert limiter.can_burn(1_000)
    limiter.record(1_000)
    assert not limiter.can_burn(1_000 + tempo - (1_000 % tempo) - 1)  # same tempo
    assert limiter.blocks_until_allowed(1_010) == (1_000 // tempo + 1) * tempo - 1_010
    assert limiter.can_burn((1_000 // tempo + 1) * tempo)  # next tempo boundary


class FakeReceipt:
    def __init__(self, ok=True, error=""):
        self.is_success = ok
        self.error_message = error


class FakeSubstrate:
    def __init__(self, receipt: FakeReceipt):
        self.receipt = receipt
        self.calls = []

    def compose_call(self, call_module, call_function, call_params):
        self.calls.append((call_function, call_params))
        return (call_function, call_params)

    def create_signed_extrinsic(self, call, keypair):
        return call

    def submit_extrinsic(self, extrinsic, wait_for_inclusion):
        return self.receipt


class FakeChain:
    def __init__(self, receipt=None):
        self.substrate = FakeSubstrate(receipt or FakeReceipt())


class FakeWallet:
    coldkey = "kp"


def make_client(chain: FakeChain) -> BurnClient:
    return BurnClient(chain, FakeWallet(), FakeWallet(), netuid=4)


def test_burn_leg_submits_and_records_tempo():
    chain = FakeChain()
    limiter = BurnRateLimiter(tempo_blocks=360)
    plan = plan_settlement(reference_pool(), 5_000.0)
    make_client(chain).burn_leg("owner_hk", plan.batches[0], limiter, current_block=720)
    assert limiter.last_burn_block == 720
    (fn, params), = chain.substrate.calls
    assert fn == "add_stake_burn"
    assert params["allow_partial"] is False
    assert params["limit_price"] > 0


def test_burn_leg_refuses_second_call_same_tempo():
    chain = FakeChain()
    limiter = BurnRateLimiter(tempo_blocks=360, last_burn_block=700)
    plan = plan_settlement(reference_pool(), 5_000.0)
    with pytest.raises(BurnRateLimited):
        make_client(chain).burn_leg("owner_hk", plan.batches[0], limiter, current_block=719)
    assert chain.substrate.calls == []  # never submitted


def test_chain_rate_limit_error_raises_typed():
    chain = FakeChain(FakeReceipt(ok=False, error="AddStakeBurnRateLimitExceeded"))
    limiter = BurnRateLimiter(tempo_blocks=360)
    plan = plan_settlement(reference_pool(), 5_000.0)
    with pytest.raises(BurnRateLimited):
        make_client(chain).burn_leg("owner_hk", plan.batches[0], limiter, current_block=100)
    assert limiter.last_burn_block is None  # failed burn is not recorded


def test_slash_leg_is_limit_bounded():
    chain = FakeChain()
    plan = plan_settlement(reference_pool(), 5_000.0)
    make_client(chain).slash_leg("owner_hk", plan.batches[0])
    (fn, params), = chain.substrate.calls
    assert fn == "remove_stake_limit"
    assert params["allow_partial"] is False
