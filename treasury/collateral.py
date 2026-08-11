"""Deployment collateral: staked-alpha bonds slashed against realized live P&L
(SPEC §0.5, subnet/docs/INCENTIVE_MECHANISM.md §Deployment Collateral).

Burn, don't just withhold: scoring penalties forfeit upside a gamed pair never
had; the bond creates real downside for a deployed pair that loses the desk
money. Slashed alpha is BURNED, never redistributed — redistribution is a
bounty for inducing other pairs' losses and is recyclable by sybil clusters.

This module is pure bookkeeping and slash math. The chain legs (escrow via
transfer_stake, per-tempo unstake + add_stake_burn settlement) live in
treasury.execution.burn. Attribution of realized P&L to a pair follows the
same split the buyback mechanism already uses.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

# Bond sizing and slash caps are contractual terms (deployment agreement,
# Phase-0 counsel review) — these are engineering defaults, not commitments.
DEFAULT_BOND_RATIO = 0.10        # bond = 10% of allocated deployment capital (TAO terms)
DEFAULT_WINDOW_SLASH_CAP = 0.25  # ≤ 25% of the ORIGINAL bond per settlement window


class CollateralError(RuntimeError):
    pass


class BondState(enum.Enum):
    PENDING = "pending"        # deployment offered; escrow transfer not yet verified
    ACTIVE = "active"          # escrow verified on-chain; pair is deployable
    RELEASING = "releasing"    # undeployed; awaiting escrow return (less slashes)
    CLOSED = "closed"          # escrow returned or fully slashed


def required_bond_alpha(deployed_capital_tao: float, alpha_spot_price: float,
                        bond_ratio: float = DEFAULT_BOND_RATIO) -> float:
    """Bond sized against allocated deployment capital, converted at spot.
    Sizing uses spot (not quote_unstake): the bond is posted small relative to
    pool depth, and overstating it slightly is the conservative direction."""
    if alpha_spot_price <= 0:
        raise ValueError("alpha spot price must be positive")
    return deployed_capital_tao * bond_ratio / alpha_spot_price


@dataclass
class Bond:
    """One deployed pair's collateral. `contributions` maps each miner coldkey
    (researcher and trader) to the alpha it escrowed; slashes apply pro-rata."""

    pair_id: str
    contributions: dict[str, float]
    deployed_capital_tao: float
    state: BondState = BondState.PENDING
    slashed_alpha: float = 0.0            # lifetime total taken from this bond

    @property
    def original_alpha(self) -> float:
        return sum(self.contributions.values())

    @property
    def remaining_alpha(self) -> float:
        return self.original_alpha - self.slashed_alpha


@dataclass(frozen=True)
class SlashResult:
    pair_id: str
    window_id: str
    realized_loss_tao: float
    slash_alpha: float                    # total, queued for burn settlement
    per_coldkey: dict[str, float]         # pro-rata by contribution
    capped: bool                          # per-window cap bound before loss did


def slash_for_window(bond: Bond, realized_loss_tao: float, window_id: str,
                     window_cap: float = DEFAULT_WINDOW_SLASH_CAP) -> SlashResult:
    """Slash for one settlement window's NET realized loss.

    slash = original_bond × (loss / deployed_capital), capped at
    `window_cap` × original bond per window and at the remaining bond. Windows
    net internally, but a profitable window never restores prior slashes —
    the bond only ratchets down (gains pay through the standard reward split).
    """
    if bond.state is not BondState.ACTIVE:
        raise CollateralError(f"{bond.pair_id}: cannot slash bond in state {bond.state.value}")
    if realized_loss_tao <= 0:
        return SlashResult(bond.pair_id, window_id, realized_loss_tao, 0.0, {}, False)

    loss_ratio = realized_loss_tao / bond.deployed_capital_tao
    uncapped = bond.original_alpha * loss_ratio
    cap = bond.original_alpha * window_cap
    slash = min(uncapped, cap, bond.remaining_alpha)

    original = bond.original_alpha
    per_coldkey = {ck: slash * amt / original for ck, amt in bond.contributions.items()}
    bond.slashed_alpha += slash
    return SlashResult(bond.pair_id, window_id, realized_loss_tao, slash,
                       per_coldkey, capped=uncapped > slash + 1e-12)


@dataclass
class BondRegistry:
    """Fund-side ledger of all bonds plus the slash queue awaiting burn
    settlement. `pending_burn_alpha` drains one batch per tempo through
    treasury.execution.burn (add_stake_burn is rate-limited to one call per
    tempo per subnet)."""

    bonds: dict[str, Bond] = field(default_factory=dict)
    pending_burn_alpha: float = 0.0
    settled_burn_alpha: float = 0.0       # lifetime alpha actually burned
    slash_log: list[SlashResult] = field(default_factory=list)

    def post(self, bond: Bond) -> None:
        if bond.pair_id in self.bonds:
            raise CollateralError(f"bond already exists for pair {bond.pair_id}")
        self.bonds[bond.pair_id] = bond

    def activate(self, pair_id: str, escrow_staked_alpha: float,
                 tolerance: float = 0.001) -> None:
        """Flip PENDING → ACTIVE once the on-chain escrow balance covers the
        bond. Verification input comes from a chain read, never self-reported."""
        bond = self._get(pair_id)
        if bond.state is not BondState.PENDING:
            raise CollateralError(f"{pair_id}: activate from {bond.state.value}")
        if escrow_staked_alpha < bond.original_alpha * (1.0 - tolerance):
            raise CollateralError(
                f"{pair_id}: escrow holds {escrow_staked_alpha:,.2f} alpha, "
                f"bond requires {bond.original_alpha:,.2f}")
        bond.state = BondState.ACTIVE

    def slash(self, pair_id: str, realized_loss_tao: float, window_id: str,
              window_cap: float = DEFAULT_WINDOW_SLASH_CAP) -> SlashResult:
        result = slash_for_window(self._get(pair_id), realized_loss_tao, window_id, window_cap)
        if result.slash_alpha > 0:
            self.pending_burn_alpha += result.slash_alpha
            self.slash_log.append(result)
        return result

    def mark_settled(self, alpha_burned_from_queue: float) -> None:
        """Called by the settlement pipeline after a successful burn batch."""
        if alpha_burned_from_queue > self.pending_burn_alpha + 1e-9:
            raise CollateralError("settled more than the pending burn queue")
        self.pending_burn_alpha -= alpha_burned_from_queue
        self.settled_burn_alpha += alpha_burned_from_queue

    def release(self, pair_id: str) -> dict[str, float]:
        """Undeploy: returns per-coldkey escrow to give back (original minus
        pro-rata slashes). Accrued staking emissions on the escrow follow the
        same pro-rata split at the execution layer."""
        bond = self._get(pair_id)
        if bond.state is not BondState.ACTIVE:
            raise CollateralError(f"{pair_id}: release from {bond.state.value}")
        bond.state = BondState.RELEASING
        original = bond.original_alpha
        scale = bond.remaining_alpha / original if original > 0 else 0.0
        return {ck: amt * scale for ck, amt in bond.contributions.items()}

    def close(self, pair_id: str) -> None:
        bond = self._get(pair_id)
        if bond.state is not BondState.RELEASING and bond.remaining_alpha > 1e-9:
            raise CollateralError(f"{pair_id}: close from {bond.state.value} with bond remaining")
        bond.state = BondState.CLOSED

    def _get(self, pair_id: str) -> Bond:
        if pair_id not in self.bonds:
            raise CollateralError(f"no bond for pair {pair_id}")
        return self.bonds[pair_id]

    # --- reporting / invariants (SPEC §8 factsheet; risk.alerts inputs) ---

    @property
    def total_bonded_alpha(self) -> float:
        """Alpha that cannot be sold while its pair is deployed — the R11
        retention lever, reported monthly."""
        return sum(b.remaining_alpha for b in self.bonds.values()
                   if b.state is BondState.ACTIVE)

    def escrow_shortfall(self, escrow_staked_alpha: float) -> float:
        """Positive when the on-chain escrow coldkey holds less than active
        bonds + unsettled slashes — page-severity custody breach."""
        expected = self.total_bonded_alpha + self.pending_burn_alpha + sum(
            b.remaining_alpha for b in self.bonds.values() if b.state is BondState.RELEASING)
        return max(0.0, expected - escrow_staked_alpha)
