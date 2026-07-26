"""Vesting/conviction math and the lock state machine (SPEC §4, §10.1)."""

import pytest

from chainio import ChainParams
from lockmgr import schedules
from lockmgr.schedules import LockState, LpLock, ScheduleViolation


PARAMS = ChainParams()


def test_half_life_is_tau_ln2():
    # τ = 648,000 blocks ≈ 90 days → half-life ≈ 62.4 days (docs' "~60 days")
    assert PARAMS.unlock_tau_days == pytest.approx(90.0)
    assert PARAMS.lock_half_life_days == pytest.approx(62.38, abs=0.05)
    assert schedules.locked_mass(1.0, PARAMS.lock_half_life_days, PARAMS) == pytest.approx(0.5)


def test_redeemable_checkpoints_match_spec():
    # SPEC §4: ~50% redeemable at +2 months, ~88% at +6, ~98.5% at +12
    assert schedules.redeemable(1.0, 60, PARAMS) == pytest.approx(0.50, abs=0.02)
    assert schedules.redeemable(1.0, 180, PARAMS) == pytest.approx(0.88, abs=0.02)
    assert schedules.redeemable(1.0, 365, PARAMS) == pytest.approx(0.985, abs=0.005)


def test_owner_conviction_is_instant_nonowner_ramps():
    assert schedules.conviction_owner(750.0) == 750.0
    # non-owner reaches 50% at τ·ln2 and ~98.3% at 1 year
    hl = PARAMS.maturity_tau_days * 0.6931471805599453
    assert schedules.conviction_nonowner(1.0, hl, PARAMS) == pytest.approx(0.5, rel=1e-6)
    assert schedules.conviction_nonowner(1.0, 365, PARAMS) == pytest.approx(0.983, abs=0.003)
    assert schedules.conviction_nonowner(1.0, 0.0, PARAMS) == 0.0


def test_roll_forward_equal_rates_matches_limit():
    m, c = 1000.0, 200.0
    dt = 7200.0 * 30  # 30 days in blocks
    mass_eq, conv_eq = schedules.roll_forward(m, c, dt, 648_000.0, 648_000.0)
    mass_near, conv_near = schedules.roll_forward(m, c, dt, 648_000.0, 648_000.0 * (1 + 1e-7))
    assert mass_eq == pytest.approx(mass_near, rel=1e-6)
    assert conv_eq == pytest.approx(conv_near, rel=1e-4)


def test_roll_forward_is_exact_solution():
    # composing two half-steps must equal one full step
    m, c = 1000.0, 0.0
    dt = 7200.0 * 60
    m1, c1 = schedules.roll_forward(m, c, dt / 2, 648_000.0, 400_000.0)
    m2, c2 = schedules.roll_forward(m1, c1, dt / 2, 648_000.0, 400_000.0)
    m_full, c_full = schedules.roll_forward(m, c, dt, 648_000.0, 400_000.0)
    assert m2 == pytest.approx(m_full, rel=1e-9)
    assert c2 == pytest.approx(c_full, rel=1e-9)


def _lock(lp_id: str, m0: float, outer_bound: float, cohort: str = "a") -> LpLock:
    return LpLock(lp_id=lp_id, coldkey=f"ck-{lp_id}", hotkey="owner-hk", netuid=1,
                  m0=m0, lock_day=0.0, outer_bound_day=outer_bound, cohort=cohort)


def test_state_machine_rejects_illegal_transitions():
    lock = _lock("lp1", 1e6, 365)
    lock.transition(LockState.LOCKED, 0.0)
    lock.transition(LockState.PERPETUAL, 0.0)
    with pytest.raises(ScheduleViolation):
        lock.transition(LockState.CLOSED, 1.0)  # cannot close a perpetual lock


def test_outer_bound_can_accelerate_never_extend():
    lock = _lock("lp1", 1e6, 365)
    lock.schedule_toggle(300)          # accelerate: fine
    assert lock.outer_bound_day == 300
    with pytest.raises(ScheduleViolation):
        lock.schedule_toggle(400)      # extend past the (new) bound: refused


def test_mass_decays_from_effective_toggle_even_without_desk_action():
    # The outer bound auto-flips: schedules assume decay begins there regardless.
    lock = _lock("lp1", 1e6, 365)
    assert lock.locked_mass_at(300, PARAMS) == 1e6
    assert lock.locked_mass_at(365 + PARAMS.lock_half_life_days, PARAMS) == pytest.approx(5e5, rel=1e-6)
    assert lock.redeemable_at(365 + PARAMS.lock_half_life_days, PARAMS) == pytest.approx(5e5, rel=1e-6)


def test_cohort_window_cap():
    staggered = [_lock(f"lp{i}", 1e6, 365 + 100 * i) for i in range(4)]
    schedules.validate_cohort_windows(staggered)  # 25% per window: at the cap, passes

    clustered = [_lock("lp0", 1e6, 365), _lock("lp1", 1e6, 375),
                 _lock("lp2", 1e6, 390), _lock("lp3", 1e6, 700)]
    share, _, offenders = schedules.redemption_exposure(clustered)
    assert share == pytest.approx(0.75)
    assert len(offenders) == 3
    with pytest.raises(ScheduleViolation):
        schedules.validate_cohort_windows(clustered)


def test_redeemable_supply_curve_monotone():
    locks = [_lock(f"lp{i}", 1e6, 365 + 100 * i) for i in range(3)]
    curve = schedules.redeemable_supply_curve(locks, PARAMS, horizon_days=900, step_days=30)
    values = [v for _, v in curve]
    assert values == sorted(values)
    assert values[-1] <= 3e6
