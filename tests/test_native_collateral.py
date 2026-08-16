"""Native Subtensor registration collateral: pallet settle math, unlock
horizon, validator gate, retention lever (docs/COLLATERAL.md)."""

import math

import pytest

from chainio import (
    CollateralPolicy,
    MinerCollateralPosition,
    StaticProvider,
    apply_collateral_gate,
    insignia_default_policy,
    native_locked_fraction,
    registration_split,
    settle_miner_collateral,
    total_native_locked,
    unlock_horizon_days,
)
from chainio.collateral import (
    LOCK_SHARE_CHAIN_CAP,
    deployment_bond_headroom,
    lock_share_from_u16,
)
from lockmgr import monitor
from risk import alerts
from treasury import emissions


def _pos(**kw) -> MinerCollateralPosition:
    defaults = dict(hotkey="hk-1", coldkey="ck-1", uid="1", locked=100.0,
                    min_locked=0.0, earned=0.0, drain_ratio=1.0, stake=100.0)
    defaults.update(kw)
    return MinerCollateralPosition(**defaults)


def test_registration_split_and_disabled_policy():
    burn, lock = registration_split(10.0, 0.5)
    assert burn == pytest.approx(5.0)
    assert lock == pytest.approx(5.0)
    assert registration_split(10.0, 0.0) == (10.0, 0.0)
    with pytest.raises(ValueError):
        registration_split(10.0, 0.99)          # above pallet 95% cap
    with pytest.raises(ValueError):
        registration_split(-1.0, 0.5)
    assert not CollateralPolicy(netuid=1).enabled
    assert insignia_default_policy(1).enabled


def test_lock_share_u16_roundtrip():
    assert lock_share_from_u16(0) == 0.0
    assert lock_share_from_u16(65_535) == pytest.approx(LOCK_SHARE_CHAIN_CAP)
    policy = insignia_default_policy(1)
    # 0.5 × 65535 ≈ 32768; decoding that u16 is ~0.5
    assert lock_share_from_u16(policy.lock_share_u16()) == pytest.approx(0.5, abs=0.001)


def test_unlock_horizon_is_the_statistical_significance_bond():
    # 1,000 α locked, k=1, 10 α/day earned → 100 days to recover. k=0.5 doubles it.
    assert unlock_horizon_days(1_000.0, 1.0, 10.0) == pytest.approx(100.0)
    assert unlock_horizon_days(1_000.0, 0.5, 10.0) == pytest.approx(200.0)
    # floor stops the drain
    assert unlock_horizon_days(1_000.0, 1.0, 10.0, min_locked=1_000.0) == 0.0
    # stop earning → freeze (Const: remainder stays locked indefinitely)
    assert unlock_horizon_days(1_000.0, 1.0, 0.0) == math.inf
    assert unlock_horizon_days(1_000.0, 0.0, 10.0) == math.inf


def test_settle_drains_above_floor_and_captures_below():
    # Above floor, k=1: 50 emission releases 50 locked, captures nothing.
    above = settle_miner_collateral(_pos(locked=100.0, min_locked=20.0, stake=100.0),
                                    emission=50.0, capturable=50.0)
    assert above.captured == 0.0
    assert above.released == pytest.approx(50.0)
    assert above.position.locked == pytest.approx(50.0)
    assert above.position.earned == pytest.approx(50.0)

    # Below floor: capturable fills the shortfall, no release.
    below = settle_miner_collateral(_pos(locked=10.0, min_locked=40.0, stake=10.0),
                                    emission=100.0, capturable=100.0)
    assert below.released == 0.0
    assert below.captured == pytest.approx(30.0)
    assert below.position.locked == pytest.approx(40.0)
    assert below.position.stake == pytest.approx(40.0)

    # Fully drained with no floor → row removed.
    gone = settle_miner_collateral(_pos(locked=10.0), emission=10.0, capturable=10.0)
    assert gone.position is None
    assert gone.released == pytest.approx(10.0)

    # Zero emission is a no-op (does not touch earned either — pallet returns
    # before mutating when emission.is_zero()).
    noop = settle_miner_collateral(_pos(locked=10.0, earned=5.0), 0.0, 10.0)
    assert noop.position.earned == 5.0
    assert noop.released == 0.0


def test_nominator_capturable_must_not_exceed_owner_slice():
    # Document the pallet invariant: capturable is the owner's slice. Passing
    # a smaller capturable than the shortfall only fills what belongs to them.
    result = settle_miner_collateral(
        _pos(locked=0.0, min_locked=100.0, stake=0.0),
        emission=50.0, capturable=20.0)
    assert result.captured == pytest.approx(20.0)
    assert result.position.locked == pytest.approx(20.0)


def test_deployment_bond_cannot_consume_native_lock():
    pos = _pos(locked=80.0, stake=100.0)
    assert pos.free_alpha == pytest.approx(20.0)
    assert deployment_bond_headroom(pos, 20.0) == pytest.approx(0.0)
    assert deployment_bond_headroom(pos, 50.0) < 0   # starved


def test_gate_zeros_short_and_frozen_miners():
    weights = {"a": 0.5, "b": 0.3, "c": 0.2}
    positions = {
        "a": _pos(hotkey="a", uid="a", locked=100.0),
        "b": _pos(hotkey="b", uid="b", locked=10.0),
    }
    gated, ids = apply_collateral_gate(weights, positions, required_min=50.0,
                                       freeze_uids={"c"})
    assert gated["a"] == pytest.approx(0.5)
    assert gated["b"] == 0.0
    assert gated["c"] == 0.0
    assert set(ids) == {"b", "c"}
    # disabled floor, no freeze → identity
    same, none = apply_collateral_gate(weights, positions, required_min=0.0)
    assert same == weights and none == []


def test_native_locked_fraction_is_disjoint_from_deployment_bonds():
    positions = [_pos(locked=200.0), _pos(hotkey="hk-2", locked=300.0)]
    assert total_native_locked(positions) == pytest.approx(500.0)
    assert native_locked_fraction(positions, 2_000.0) == pytest.approx(0.25)
    assert native_locked_fraction(positions, 0.0) == 0.0
    # σ after both levers: 50% base, 40% deployed-bonded, 25% native-locked
    # unsellable = 0.65 → σ = 0.5 × 0.35 = 0.175
    assert emissions.effective_sell_through(0.5, 0.4, 0.25) == pytest.approx(0.175)
    # existing call sites (bonded only) still work
    assert emissions.effective_sell_through(0.5, 0.4) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        emissions.effective_sell_through(0.5, 0.4, 1.5)


def test_static_provider_serves_collateral():
    policy = insignia_default_policy(7)
    pos = [_pos()]
    provider = StaticProvider(collateral_policy=policy, miner_collateral=pos)
    assert provider.collateral_policy(7).lock_share == pytest.approx(0.5)
    assert provider.miner_collateral(7) == pos
    # with_params preserves collateral state
    bumped = provider.with_params(root_tao=6e6)
    assert bumped.collateral_policy(7).lock_share == pytest.approx(0.5)
    assert bumped.miner_collateral(7) == pos
    # default provider is disabled policy, empty rows
    empty = StaticProvider()
    assert not empty.collateral_policy(1).enabled
    assert empty.miner_collateral(1) == []


def test_native_collateral_monitor_visibility_and_starvation():
    enabled = CollateralPolicy(netuid=1, lock_share=0.5, drain_ratio=1.0,
                               required_min_alpha=50.0)
    watch = monitor.NativeCollateralWatch(netuid=1, policy=enabled, positions=[])
    kinds = [f.kind for f in monitor.native_collateral_findings(watch)]
    assert kinds == ["native_collateral_no_visibility"]

    short = _pos(locked=10.0, stake=10.0)
    starved = _pos(hotkey="hk-desk", locked=90.0, stake=100.0)
    watch = monitor.NativeCollateralWatch(
        netuid=1, policy=enabled, positions=[short, starved],
        previous_policy=CollateralPolicy(netuid=1, lock_share=0.8, drain_ratio=1.0),
        deployment_bond_by_hotkey={"hk-desk": 50.0})
    findings = monitor.native_collateral_findings(watch)
    kinds = {f.kind for f in findings}
    assert "native_collateral_policy" in kinds
    assert "native_collateral_floor" in kinds
    assert "native_collateral_starves_bond" in kinds
    assert "native_collateral_level" in kinds
    # lock_share 0.8 → 0.5 is a 37.5% relative drop → page
    policy_f = [f for f in findings if f.kind == "native_collateral_policy"][0]
    assert policy_f.severity == "page"

    alerts_out = alerts.from_native_collateral(findings)
    assert any(a.source.startswith("monitor.native_collateral") for a in alerts_out)
