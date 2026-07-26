"""Chain-state pollers enforcing the SPEC §4 invariants.

Run once per epoch (tempo). Emits `MonitorFinding`s consumed by risk/alerts.
Covers: parameter drift (recompute schedules on UnlockRate/MaturityRate change),
owner-hotkey change, unstaked LP alpha, and subnet-king early warning — tracked
even while the king transfer is disabled (SPEC §0.6).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from chainio import ChainParams
from lockmgr.locks import LockClient, OnChainLock
from lockmgr.schedules import LpLock, LockState

# Defensive posture if subnet-king is enabled (SPEC §10.3): owner aggregate
# ≥ KING_DEFENSE_RATIO × largest external hotkey AND external conviction below
# KING_CONVICTION_THRESHOLD × SubnetAlphaOut.
KING_DEFENSE_RATIO = 2.0
KING_CONVICTION_THRESHOLD = 0.10


@dataclass(frozen=True)
class MonitorFinding:
    severity: str        # "page" | "warn" | "info"
    kind: str
    detail: str


def param_drift(previous: ChainParams, current: ChainParams) -> list[MonitorFinding]:
    """Any root-driven parameter change invalidates cached schedules — page on
    the lock rates, warn on everything else."""
    findings = []
    page_fields = {"unlock_rate_blocks", "conviction_maturity_blocks"}
    for f in dataclasses.fields(ChainParams):
        old, new = getattr(previous, f.name), getattr(current, f.name)
        if old != new:
            findings.append(MonitorFinding(
                severity="page" if f.name in page_fields else "warn",
                kind="param_change",
                detail=f"{f.name}: {old} → {new}; recompute all vesting schedules",
            ))
    return findings


def verify_lock_invariants(
    expected: LpLock, on_chain: OnChainLock | None, owner_hotkey: str, params: ChainParams,
    day: float, mass_tolerance: float = 0.01,
) -> list[MonitorFinding]:
    findings = []
    if on_chain is None:
        if expected.state in (LockState.LOCKED, LockState.PERPETUAL, LockState.VESTING_COMPLETE,
                              LockState.DECAYING):
            findings.append(MonitorFinding("page", "lock_missing",
                            f"{expected.lp_id}: no on-chain lock for coldkey {expected.coldkey}"))
        return findings
    if on_chain.hotkey != owner_hotkey:
        findings.append(MonitorFinding("page", "wrong_hotkey",
                        f"{expected.lp_id}: locked to {on_chain.hotkey}, not owner hotkey — "
                        "forfeits instant conviction and weakens king defense"))
    should_be_perpetual = expected.state in (LockState.PERPETUAL, LockState.VESTING_COMPLETE)
    if should_be_perpetual and not on_chain.perpetual:
        findings.append(MonitorFinding("page", "decay_leak",
                        f"{expected.lp_id}: perpetual flag off before toggle day "
                        f"{expected.effective_toggle_day():.0f} — mass is decaying early"))
    expected_mass = expected.locked_mass_at(day, params)
    if expected_mass > 0 and abs(on_chain.locked_mass - expected_mass) / expected_mass > mass_tolerance:
        findings.append(MonitorFinding("warn", "mass_mismatch",
                        f"{expected.lp_id}: on-chain mass {on_chain.locked_mass:,.0f} vs "
                        f"scheduled {expected_mass:,.0f}"))
    return findings


def unstaked_lp_positions(staked_by_coldkey: dict[str, float],
                          locks: list[LpLock]) -> list[MonitorFinding]:
    """Locked LP alpha must remain staked at all times — unstaked positions eat
    the full dilution hurdle instead of recapturing issuance (SPEC §0.5)."""
    findings = []
    for lock in locks:
        if lock.state is LockState.CLOSED:
            continue
        staked = staked_by_coldkey.get(lock.coldkey, 0.0)
        if staked <= 0:
            findings.append(MonitorFinding("page", "unstaked_lp",
                            f"{lock.lp_id}: coldkey {lock.coldkey} shows no staked alpha"))
    return findings


@dataclass(frozen=True)
class KingWatch:
    owner_aggregate: float
    largest_external: float
    largest_external_hotkey: str | None
    subnet_alpha_out: float

    @property
    def defense_ratio(self) -> float:
        return self.owner_aggregate / self.largest_external if self.largest_external > 0 else float("inf")

    @property
    def external_share_of_alpha_out(self) -> float:
        return self.largest_external / self.subnet_alpha_out if self.subnet_alpha_out > 0 else 0.0


def king_early_warning(watch: KingWatch) -> list[MonitorFinding]:
    findings = []
    if watch.defense_ratio < KING_DEFENSE_RATIO:
        findings.append(MonitorFinding("warn", "king_defense_ratio",
                        f"owner conviction only {watch.defense_ratio:.2f}× largest external "
                        f"({watch.largest_external_hotkey}); target ≥ {KING_DEFENSE_RATIO}×"))
    if watch.external_share_of_alpha_out >= KING_CONVICTION_THRESHOLD:
        findings.append(MonitorFinding("page", "king_threshold",
                        f"external hotkey {watch.largest_external_hotkey} holds "
                        f"{watch.external_share_of_alpha_out:.1%} of SubnetAlphaOut rolled conviction — "
                        "above the king-transfer eligibility threshold"))
    return findings


def poll(client: LockClient, locks: list[LpLock], owner_hotkey: str,
         previous_params: ChainParams, current_params: ChainParams, day: float,
         staked_by_coldkey: dict[str, float], watch: KingWatch) -> list[MonitorFinding]:
    """One epoch's full sweep; feed the result to risk.alerts.dispatch."""
    findings = param_drift(previous_params, current_params)
    for lock in locks:
        findings += verify_lock_invariants(
            lock, client.get_coldkey_lock(lock.coldkey), owner_hotkey, current_params, day)
    findings += unstaked_lp_positions(staked_by_coldkey, locks)
    findings += king_early_warning(watch)
    return findings
