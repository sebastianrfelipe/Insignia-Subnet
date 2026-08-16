"""Chain-state pollers enforcing the SPEC §4 invariants.

Run once per epoch (tempo). Emits `MonitorFinding`s consumed by risk/alerts.
Covers: parameter drift (recompute schedules on UnlockRate/MaturityRate change),
owner-hotkey change, unstaked LP alpha, subnet-king early warning — tracked
even while the king transfer is disabled (SPEC §0.6) — and, post-Root-Reborn,
beta-basket escrow stake per root validator (SPEC §0.16; R15/R16). Native
registration collateral (SPEC §0.17, docs/COLLATERAL.md): policy drift, missing
visibility, floor shortfall, and deployment-bond starvation.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from chainio import ChainParams, ValidatorBasket, total_escrow_alpha
from chainio.collateral import CollateralPolicy, MinerCollateralPosition, total_native_locked
from lockmgr.locks import LockClient, OnChainLock
from lockmgr.schedules import LpLock, LockState

# Defensive posture if subnet-king is enabled (SPEC §10.3): owner aggregate
# ≥ KING_DEFENSE_RATIO × largest external hotkey AND external conviction below
# KING_CONVICTION_THRESHOLD × SubnetAlphaOut. SubnetAlphaOut INCLUDES beta-basket
# escrow (real stake, conviction-inert) — the denominator both sides fight over.
KING_DEFENSE_RATIO = 2.0
KING_CONVICTION_THRESHOLD = 0.10

# Root Reborn escrow thresholds (SPEC §0.16; R15/R16)
ESCROW_CONSENSUS_WARN_SHARE = 0.05   # one validator's escrow ≥5% of AlphaOut →
                                     # material stake-weight in subnet consensus
ROTATION_WARN_REL_DROP = 0.25        # w_ins down ≥25% epoch-over-epoch → R15


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


@dataclass(frozen=True)
class EscrowWatch:
    """One epoch's view of beta-basket escrow on our netuid (SPEC §0.16).

    `subnet_alpha_out` must be the chain value, which already includes escrow —
    basket positions are real stake entries. `previous_weights` is last epoch's
    per-validator weight toward our netuid, for rotation early-warning (R15).
    """

    netuid: int
    baskets: list[ValidatorBasket]
    subnet_alpha_out: float
    previous_weights: dict[str, float] = field(default_factory=dict)
    trailing_claim_alpha_30d: float = 0.0

    @property
    def total_escrow(self) -> float:
        return total_escrow_alpha(self.baskets)

    @property
    def escrow_share_of_alpha_out(self) -> float:
        if self.subnet_alpha_out <= 0:
            return 0.0
        return self.total_escrow / self.subnet_alpha_out


def escrow_findings(watch: EscrowWatch) -> list[MonitorFinding]:
    """Escrow stake, rotation, and overhang checks (R15/R16).

    Escrow is conviction-inert (the keyless escrow can never sign lock_stake),
    so it never appears in king numerators — but it inflates SubnetAlphaOut for
    attacker and defender alike, and it is the claim-flow overhang.
    """
    findings: list[MonitorFinding] = []
    if not watch.baskets:
        findings.append(MonitorFinding("warn", "escrow_no_visibility",
                        "no beta-basket state readable — treat as missing data, "
                        "not as zero escrow (verify BetaBasketApi on this runtime)"))
        return findings

    for b in watch.baskets:
        if watch.subnet_alpha_out > 0 and \
                b.escrow_alpha / watch.subnet_alpha_out >= ESCROW_CONSENSUS_WARN_SHARE:
            findings.append(MonitorFinding("warn", "escrow_consensus_weight",
                            f"validator {b.hotkey} escrow holds "
                            f"{b.escrow_alpha / watch.subnet_alpha_out:.1%} of SubnetAlphaOut — "
                            "material stake-weight in subnet consensus"))
        prev = watch.previous_weights.get(b.hotkey)
        now = b.weight_to(watch.netuid)
        if prev is not None and prev > 0 and (prev - now) / prev >= ROTATION_WARN_REL_DROP:
            findings.append(MonitorFinding("warn", "basket_rotation",
                            f"validator {b.hotkey} weight to netuid {watch.netuid} "
                            f"fell {prev:.1%} → {now:.1%} — R15 rotation early-warning; "
                            "expect the dividend bid to shrink next epoch"))

    if watch.trailing_claim_alpha_30d > 0 and watch.total_escrow > 0:
        annualized = watch.trailing_claim_alpha_30d * 12.0 / watch.total_escrow
        if annualized > 2.0:
            findings.append(MonitorFinding("page", "claim_cluster",
                            f"escrow claims running {annualized:.1f}×/yr of the basket — "
                            "R16 claim clustering; check reserve coverage vs overhang"))
    findings.append(MonitorFinding("info", "escrow_level",
                    f"beta-basket escrow {watch.total_escrow:,.0f} α "
                    f"({watch.escrow_share_of_alpha_out:.1%} of SubnetAlphaOut) "
                    f"across {len(watch.baskets)} validators"))
    return findings


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


# Native registration collateral (docs/COLLATERAL.md; R11 / R17)
COLLATERAL_HEADROOM_WARN = 0.0          # free alpha below a posted deployment bond
NATIVE_LOCK_SHARE_PAGE_DROP = 0.25      # lock_share cut ≥25% relative → policy shock


@dataclass(frozen=True)
class NativeCollateralWatch:
    """One epoch's view of native miner-registration collateral.

    Empty `positions` with an *enabled* policy is missing visibility (warn),
    never 'no collateral'. `deployment_bond_by_hotkey` is the alpha each
    miner still needs to `transfer_stake` for a desk bond — native locks
    cannot fund that transfer.
    """

    netuid: int
    policy: CollateralPolicy
    positions: list[MinerCollateralPosition]
    previous_policy: CollateralPolicy | None = None
    deployment_bond_by_hotkey: dict[str, float] = field(default_factory=dict)


def native_collateral_findings(watch: NativeCollateralWatch) -> list[MonitorFinding]:
    """Policy drift, floor shortfalls, and deployment-bond starvation."""
    findings: list[MonitorFinding] = []
    if watch.previous_policy is not None:
        prev, cur = watch.previous_policy, watch.policy
        if prev.lock_share != cur.lock_share or prev.drain_ratio != cur.drain_ratio:
            rel_drop = 0.0
            if prev.lock_share > 0:
                rel_drop = (prev.lock_share - cur.lock_share) / prev.lock_share
            severity = "page" if rel_drop >= NATIVE_LOCK_SHARE_PAGE_DROP else "warn"
            findings.append(MonitorFinding(
                severity, "native_collateral_policy",
                f"lock_share {prev.lock_share:.2%} → {cur.lock_share:.2%}, "
                f"drain_ratio {prev.drain_ratio} → {cur.drain_ratio}; "
                "existing miners keep their snapshotted drain until re-registration"))

    if watch.policy.enabled and not watch.positions:
        findings.append(MonitorFinding(
            "warn", "native_collateral_no_visibility",
            "native collateral policy is enabled but no per-UID rows are "
            "readable — treat as missing data, not as zero locks"))
        return findings

    required = watch.policy.required_min_alpha
    short = [p for p in watch.positions if not p.covers(required)]
    if short:
        findings.append(MonitorFinding(
            "warn", "native_collateral_floor",
            f"{len(short)} miner(s) below the published floor "
            f"({required:,.0f} α) — validators must zero their weights"))

    for p in watch.positions:
        bond = watch.deployment_bond_by_hotkey.get(p.hotkey, 0.0)
        if bond > 0 and p.free_alpha + 1e-12 < bond:
            findings.append(MonitorFinding(
                "warn", "native_collateral_starves_bond",
                f"hotkey {p.hotkey}: free {p.free_alpha:,.0f} α < deployment "
                f"bond {bond:,.0f} α — registration lock is blocking desk escrow"))

    locked = total_native_locked(watch.positions)
    findings.append(MonitorFinding(
        "info", "native_collateral_level",
        f"native registration collateral {locked:,.0f} α across "
        f"{len(watch.positions)} positions "
        f"(lock_share={watch.policy.lock_share:.0%}, "
        f"drain_ratio={watch.policy.drain_ratio})"))
    return findings


def poll(client: LockClient, locks: list[LpLock], owner_hotkey: str,
         previous_params: ChainParams, current_params: ChainParams, day: float,
         staked_by_coldkey: dict[str, float], watch: KingWatch,
         escrow: EscrowWatch | None = None,
         native_collateral: NativeCollateralWatch | None = None) -> list[MonitorFinding]:
    """One epoch's full sweep; feed the result to risk.alerts.dispatch."""
    findings = param_drift(previous_params, current_params)
    for lock in locks:
        findings += verify_lock_invariants(
            lock, client.get_coldkey_lock(lock.coldkey), owner_hotkey, current_params, day)
    findings += unstaked_lp_positions(staked_by_coldkey, locks)
    findings += king_early_warning(watch)
    if escrow is not None:
        findings += escrow_findings(escrow)
    if native_collateral is not None:
        findings += native_collateral_findings(native_collateral)
    return findings
