"""Native Subtensor miner-registration collateral (pallet `subnets/collateral.rs`).

A distinct primitive from conviction `lock_stake` AND from Insignia deployment
bonds (`treasury/collateral.py`). Three layers, do not conflate them:

1. **Conviction lock** — governance weight; LP / owner-hotkey path (SPEC §4).
2. **Native registration collateral (this module)** — a time-bond recovered
   only by earning emission. `lock_share` splits the registration price into a
   burned share and a locked share; `drain_ratio` releases locked alpha per
   alpha of hotkey emission earned. Stopping earning freezes the remainder
   indefinitely; deregistration does not unlock it. See docs/COLLATERAL.md.
3. **Deployment collateral** — loss-linked P&L bond on the desk-deployed
   `(researcher, trader)` pair; slashed alpha is burned (`add_stake_burn`).

This module mirrors the pallet's settle / coverage math in floats for the fund
layer. Validators enforce a published floor by zeroing weights (they cannot
write another miner's `min_locked` on-chain — that extrinsic is miner-signed).

Engineering defaults below are NOT chain commitments. The owner sets
`CollateralLockShare` / `CollateralDrainRatio` on-chain; every reader goes
through a live `ParamsProvider` (SPEC §0.15).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

# Pallet: lock_share is u16 with u16::MAX = 100%; admin rejects ≥ 95%.
U16_MAX = 65_535
LOCK_SHARE_CHAIN_CAP = 0.95

# Engineering defaults for Insignia policy — labelled as such, never hardcoded
# into production paths. lock_share = 0.5: half the registration price is a
# recoverable bond, half still burns. drain_ratio = 1.0: one locked alpha
# releases per one alpha of emission earned (k < 1 stretches the horizon).
DEFAULT_LOCK_SHARE = 0.50
DEFAULT_DRAIN_RATIO = 1.0
DEFAULT_REQUIRED_MIN_ALPHA = 0.0
DEFAULT_DRAWDOWN_FREEZE = 0.20          # same hard ceiling as trading scoring


@dataclass(frozen=True)
class CollateralPolicy:
    """Per-subnet native-collateral configuration.

    `lock_share` / `drain_ratio` are chain state. `required_min_alpha` is the
    validator-published floor Const described: miners `set_min_collateral` to
    track it; validators zero weights if `locked` (or the miner-set floor)
    is short. The pallet will not let a validator write another miner's floor.
    """

    netuid: int
    lock_share: float = 0.0              # 0 disables native collateral
    drain_ratio: float = DEFAULT_DRAIN_RATIO
    required_min_alpha: float = DEFAULT_REQUIRED_MIN_ALPHA

    @property
    def enabled(self) -> bool:
        return self.lock_share > 0.0

    def lock_share_u16(self) -> int:
        """On-chain encoding: u16::MAX = 100%."""
        return int(round(self.lock_share * U16_MAX))


@dataclass(frozen=True)
class MinerCollateralPosition:
    """One `(netuid, hotkey, coldkey)` standing collateral row.

    Pallet storage: `MinerCollateral`. Nominators on the same hotkey are not
    frozen by the owner's bond — keyed by the triple, not the hotkey alone.
    `stake` is the position's total alpha (locked + free); used to size the
    leftover that can still `transfer_stake` a deployment bond.
    """

    hotkey: str
    coldkey: str = ""
    uid: str | int | None = None
    locked: float = 0.0
    min_locked: float = 0.0
    earned: float = 0.0
    drain_ratio: float = DEFAULT_DRAIN_RATIO
    stake: float = 0.0

    @property
    def free_alpha(self) -> float:
        """Alpha that can leave this position without uncovering collateral.
        Pallet: `available_to_unstake_from_hotkey` (collateral leg only)."""
        return max(0.0, self.stake - self.locked)

    @property
    def at_or_above_floor(self) -> bool:
        return self.locked + 1e-12 >= self.min_locked

    def covers(self, required_min: float) -> bool:
        """Validator-side floor check: standing lock AND miner-set floor."""
        if required_min <= 0:
            return True
        return self.locked + 1e-12 >= required_min


@dataclass(frozen=True)
class SettleResult:
    """One tempo of `settle_miner_collateral`.

    `captured` is emission diverted INTO the lock when below the miner-set
    floor — the caller must credit only the remainder of the capturable slice
    to the owner. Release does not capture; it just drops `locked`.
    `position` is None when the row fully drained with no floor.
    """

    position: MinerCollateralPosition | None
    captured: float
    released: float


def lock_share_from_u16(raw: int) -> float:
    if raw <= 0:
        return 0.0
    return min(LOCK_SHARE_CHAIN_CAP, raw / U16_MAX)


def registration_split(registration_cost_tao: float, lock_share: float
                       ) -> tuple[float, float]:
    """(burned_tao, collateral_tao) — the pallet's `pay_registration` split.

    `collateral_tao = p × registration_cost`; the rest burns. lock_share = 0
    is classic burned registration.
    """
    if registration_cost_tao < 0:
        raise ValueError("registration cost must be non-negative")
    if not 0.0 <= lock_share <= LOCK_SHARE_CHAIN_CAP:
        raise ValueError(f"lock_share must be in [0, {LOCK_SHARE_CHAIN_CAP}]")
    collateral = registration_cost_tao * lock_share
    return registration_cost_tao - collateral, collateral


def unlock_horizon_days(locked: float, drain_ratio: float,
                        daily_emission: float, min_locked: float = 0.0) -> float:
    """Days of earning needed to drain `locked` down to `min_locked`.

    Pallet: release = min(k × emission, locked − min_locked) per settle.
    k ≤ 0 or zero emission → inf (the freeze Const described: a miner who
    stops earning keeps the remainder indefinitely).
    """
    excess = max(0.0, locked - min_locked)
    if excess <= 0:
        return 0.0
    if drain_ratio <= 0 or daily_emission <= 0:
        return float("inf")
    return excess / (drain_ratio * daily_emission)


def settle_miner_collateral(position: MinerCollateralPosition,
                            emission: float,
                            capturable: float) -> SettleResult:
    """Mirror of pallet `settle_miner_collateral` in float alpha.

    Two directions around the miner-set floor (`min_locked`):
    - Below: up to `min(capturable, shortfall)` is captured into the lock
      (staked to the hotkey itself).
    - Above: `min(drain_ratio × emission, locked − min_locked)` is released
      back to withdrawable stake.

    `emission` drives lifetime earned and the release rate. `capturable` must
    be value that already belongs to the owner (full miner incentive, or only
    the validator's take) — nominator / root-claimable shares must never be
    passed as capturable.
    """
    if emission <= 0:
        return SettleResult(position, 0.0, 0.0)

    earned = position.earned + emission
    locked = position.locked
    shortfall = max(0.0, position.min_locked - locked)

    if shortfall > 0:
        captured = min(max(0.0, capturable), shortfall)
        if captured <= 0:
            updated = replace(position, earned=earned)
            return SettleResult(updated, 0.0, 0.0)
        updated = replace(position, locked=locked + captured, earned=earned,
                          stake=position.stake + captured)
        return SettleResult(updated, captured, 0.0)

    release = drain_ratio_release(emission, position.drain_ratio)
    releasable = max(0.0, locked - position.min_locked)
    released = min(releasable, release)
    new_locked = locked - released
    if new_locked <= 1e-15 and position.min_locked <= 1e-15:
        return SettleResult(None, 0.0, released)
    updated = replace(position, locked=new_locked, earned=earned)
    return SettleResult(updated, 0.0, released)


def drain_ratio_release(emission: float, drain_ratio: float) -> float:
    return max(0.0, emission) * max(0.0, drain_ratio)


def total_native_locked(positions: list[MinerCollateralPosition]) -> float:
    """Aggregate standing lock — the unsellable native-collateral stock (R11)."""
    return sum(p.locked for p in positions)


def native_locked_fraction(positions: list[MinerCollateralPosition],
                           miner_held_alpha: float) -> float:
    """Share of miner-held alpha sitting in native locks. Capped at 1.

    Disjoint from deployment-bond escrow: native collateral cannot be
    `transfer_stake`'d, so the two stocks do not overlap.
    """
    if miner_held_alpha <= 0:
        return 0.0
    return min(1.0, total_native_locked(positions) / miner_held_alpha)


def deployment_bond_headroom(position: MinerCollateralPosition,
                             bond_alpha: float) -> float:
    """How much free alpha remains after posting `bond_alpha` as a deployment
    bond. Negative → the registration lock is starving the desk-tier escrow
    (docs/COLLATERAL.md §Interaction)."""
    return position.free_alpha - bond_alpha


def apply_collateral_gate(
    weights: dict[str, float],
    positions: dict[str, MinerCollateralPosition],
    required_min: float = 0.0,
    freeze_uids: set[str] | None = None,
) -> tuple[dict[str, float], list[str]]:
    """Zero weights for miners who fail the published floor or are frozen.

    This is the native primitive's teeth on a trading subnet: a martingale
    blow-up (or an exploit) that zeros weights stops emission, which freezes
    remaining collateral until the miner re-registers and earns it back.
    Keyed by the same strings as `weights` (uid or hotkey).

    Returns (gated_weights, gated_ids). Gated miners stay in the dict at 0
    so Yuma still sees them; they earn nothing and cannot drain.
    """
    frozen = freeze_uids or set()
    gated: list[str] = []
    out = dict(weights)
    for uid, weight in weights.items():
        pos = positions.get(uid)
        short = required_min > 0 and (pos is None or not pos.covers(required_min))
        if uid in frozen or short:
            if weight != 0.0:
                out[uid] = 0.0
            gated.append(uid)
    return out, gated


def insignia_default_policy(netuid: int) -> CollateralPolicy:
    """Documented Insignia starting policy. Owner must still set it on-chain."""
    return CollateralPolicy(
        netuid=netuid,
        lock_share=DEFAULT_LOCK_SHARE,
        drain_ratio=DEFAULT_DRAIN_RATIO,
        required_min_alpha=DEFAULT_REQUIRED_MIN_ALPHA,
    )
