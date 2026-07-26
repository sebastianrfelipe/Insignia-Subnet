"""Per-LP vesting schedules, conviction math, cohort windows (SPEC §4, §10.1).

All curves are parameterized by on-chain rates via `ChainParams` — τ values are
root-mutable and must never be hardcoded downstream. Rates are exponential time
constants (`exp(-dt/τ)`), matching the chain's roll-forward convention; the
"~60-day half-life" in docs is `τ · ln 2` ≈ 62.4 days at current defaults.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

from chainio import ChainParams


# --------------------------------------------------------------------------- curves

def decay_factor(dt_days: float, tau_days: float) -> float:
    return math.exp(-dt_days / tau_days)


def locked_mass(m0: float, days_since_toggle: float, params: ChainParams) -> float:
    """Mass remaining in a DECAYING lock `days_since_toggle` after perpetual→decay."""
    if days_since_toggle <= 0:
        return m0
    return m0 * decay_factor(days_since_toggle, params.unlock_tau_days)


def redeemable(m0: float, days_since_toggle: float, params: ChainParams) -> float:
    return m0 - locked_mass(m0, days_since_toggle, params)


def conviction_owner(mass: float) -> float:
    """Locks to the subnet owner's hotkey earn conviction = locked mass instantly."""
    return mass


def conviction_nonowner(m0: float, days_locked: float, params: ChainParams) -> float:
    """Perpetual lock to a non-owner hotkey matures as 1 − e^(−t/τ)."""
    if days_locked <= 0:
        return 0.0
    return m0 * (1.0 - decay_factor(days_locked, params.maturity_tau_days))


def roll_forward(
    mass: float,
    conviction: float,
    dt_blocks: float,
    unlock_rate_blocks: float,
    maturity_rate_blocks: float,
) -> tuple[float, float]:
    """General conviction/mass roll-forward from the chain code (SPEC §4).

    decay_x = exp(−dt/UnlockRate); decay_z = exp(−dt/MaturityRate)
    γ = UnlockRate·(decay_x − decay_z)/(UnlockRate − MaturityRate)
        → (dt/UnlockRate)·decay_x when the rates are equal
    C_new = decay_z·C_old + γ·M_old
    """
    decay_x = math.exp(-dt_blocks / unlock_rate_blocks)
    decay_z = math.exp(-dt_blocks / maturity_rate_blocks)
    if math.isclose(unlock_rate_blocks, maturity_rate_blocks, rel_tol=1e-9):
        gamma = (dt_blocks / unlock_rate_blocks) * decay_x
    else:
        gamma = unlock_rate_blocks * (decay_x - decay_z) / (unlock_rate_blocks - maturity_rate_blocks)
    return mass * decay_x, decay_z * conviction + gamma * mass


# --------------------------------------------------------------------------- state machine

class LockState(enum.Enum):
    DELIVERED = "delivered"              # OTC settlement complete, staked, not yet locked
    LOCKED = "locked"                    # lock_stake executed, decaying by default
    PERPETUAL = "perpetual"              # set_perpetual_lock(true); conviction = mass (owner hotkey)
    VESTING_COMPLETE = "vesting_complete"  # outer-bound month reached; toggle eligible
    DECAYING = "decaying"                # set_perpetual_lock(false); mass halves every ~60d
    CLOSED = "closed"                    # residual dust unstaked


_TRANSITIONS = {
    LockState.DELIVERED: {LockState.LOCKED},
    LockState.LOCKED: {LockState.PERPETUAL, LockState.DECAYING},
    LockState.PERPETUAL: {LockState.VESTING_COMPLETE, LockState.DECAYING},
    LockState.VESTING_COMPLETE: {LockState.DECAYING},
    LockState.DECAYING: {LockState.CLOSED},
    LockState.CLOSED: set(),
}

DUST_FRACTION = 0.015  # ≈ residual mass after 1 year of decay; below this, CLOSED is allowed


class ScheduleViolation(RuntimeError):
    pass


@dataclass
class LpLock:
    """One LP's lock (one lock per coldkey per subnet; single hotkey target)."""

    lp_id: str
    coldkey: str
    hotkey: str                 # MUST be the subnet owner hotkey (SPEC §4 invariant)
    netuid: int
    m0: float                   # alpha locked at conversion
    lock_day: float             # fund-clock day the lock was placed
    outer_bound_day: float      # hard bound: toggle auto-flips here; desk may accelerate, never extend
    cohort: str = "default"
    state: LockState = LockState.DELIVERED
    toggle_day: float | None = None   # actual perpetual→decay day, once flipped
    history: list[tuple[float, LockState]] = field(default_factory=list)

    def transition(self, new_state: LockState, day: float) -> None:
        if new_state not in _TRANSITIONS[self.state]:
            raise ScheduleViolation(f"{self.lp_id}: illegal transition {self.state} → {new_state}")
        if new_state is LockState.DECAYING:
            self.toggle_day = day
        self.state = new_state
        self.history.append((day, new_state))

    def schedule_toggle(self, day: float) -> None:
        """Desk may accelerate the toggle inside the outer bound, never extend it.

        This is the code-level enforcement of SPEC §10.1: the outer-bound
        transaction is pre-scheduled and the desk cannot cancel it.
        """
        if day > self.outer_bound_day:
            raise ScheduleViolation(
                f"{self.lp_id}: toggle day {day} exceeds hard outer bound {self.outer_bound_day}"
            )
        self.outer_bound_day = day

    def effective_toggle_day(self) -> float:
        return self.toggle_day if self.toggle_day is not None else self.outer_bound_day

    def locked_mass_at(self, day: float, params: ChainParams) -> float:
        if self.state is LockState.CLOSED:
            return 0.0
        toggle = self.effective_toggle_day()
        if day < toggle:
            return self.m0
        return locked_mass(self.m0, day - toggle, params)

    def redeemable_at(self, day: float, params: ChainParams) -> float:
        if self.state is LockState.CLOSED:
            return 0.0
        return self.m0 - self.locked_mass_at(day, params)

    def conviction_at(self, day: float, params: ChainParams, owner_hotkey: str) -> float:
        """Owner-hotkey locks: conviction tracks locked mass instantly.
        Non-owner locks (never intentional for Insignia) mature on the τ ramp."""
        mass = self.locked_mass_at(day, params)
        if self.hotkey == owner_hotkey:
            return conviction_owner(mass)
        return conviction_nonowner(mass, day - self.lock_day, params)


# --------------------------------------------------------------------------- cohort windows

def redemption_exposure(
    locks: list[LpLock], window_days: float = 60.0
) -> tuple[float, float, list[LpLock]]:
    """Worst sliding redemption window across all locks.

    Returns (max_share, window_start_day, offending_locks): the largest fraction
    of total locked supply whose decay begins inside any `window_days` window.
    M6 acceptance: max_share ≤ 0.25 (SPEC §9).
    """
    active = [l for l in locks if l.state is not LockState.CLOSED]
    total = sum(l.m0 for l in active)
    if total == 0:
        return 0.0, 0.0, []
    by_toggle = sorted(active, key=lambda l: l.effective_toggle_day())
    best_share, best_start, best_locks = 0.0, 0.0, []
    running = 0.0
    window: list[LpLock] = []
    for lock in by_toggle:
        running += lock.m0
        window.append(lock)
        start = lock.effective_toggle_day() - window_days
        while window and window[0].effective_toggle_day() < start:
            running -= window[0].m0
            window.pop(0)
        share = running / total
        if share > best_share:
            best_share, best_start, best_locks = share, start, list(window)
    return best_share, best_start, best_locks


def validate_cohort_windows(
    locks: list[LpLock], cap: float = 0.25, window_days: float = 60.0
) -> None:
    share, start, offenders = redemption_exposure(locks, window_days)
    if share > cap:
        cohorts = sorted({l.cohort for l in offenders})
        raise ScheduleViolation(
            f"{share:.1%} of locked supply shares the {window_days:.0f}-day redemption window "
            f"starting day {start:.0f} (cap {cap:.0%}); cohorts: {', '.join(cohorts)}"
        )


def redeemable_supply_curve(
    locks: list[LpLock], params: ChainParams, horizon_days: float = 730.0, step_days: float = 7.0
) -> list[tuple[float, float]]:
    """(day, aggregate redeemable alpha) — the §8 lock-cohort schedule series."""
    curve = []
    day = 0.0
    while day <= horizon_days:
        curve.append((day, sum(l.redeemable_at(day, params) for l in locks)))
        day += step_days
    return curve
