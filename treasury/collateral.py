"""Deployment collateral: staked-alpha bonds slashed against realized live P&L
(SPEC §0.5, subnet/docs/INCENTIVE_MECHANISM.md §Deployment Collateral).

Burn, don't just withhold: scoring penalties forfeit upside a gamed pair never
had; the bond creates real downside for a deployed pair that loses the desk
money. Slashed alpha is BURNED, never redistributed — redistribution is a
bounty for inducing other pairs' losses and is recyclable by sybil clusters.

This module is pure bookkeeping and slash math. The chain legs (escrow via
transfer_stake, per-tempo unstake + add_stake_burn settlement) live in
treasury.execution.burn. Native Subtensor registration collateral is a
different primitive (time-bond recovered by earning emission) — see
chainio.collateral and docs/COLLATERAL.md. The two stocks are disjoint:
native locks cannot be transfer_stake'd into this escrow.

Researcher and trader are SEPARATE miners who do not choose each other —
PAIRING_MECHANISM.md §2.3 assigns pairs deterministically from chain block hash
and hides partner identity until evaluation. Splitting a slash by bond size
alone would therefore punish one miner for the other's error with no screening
or monitoring channel to justify it, and would reintroduce exactly the partner
noise the emission side removes via the K-partner floor and the
variance-penalized credit formula (pairing.py::MarginalContributionCredit).
Slashes are therefore split by ATTRIBUTION (§blame_split): what the per-role
diagnostics explain lands on that role, and only the genuinely joint residual
is shared — at reduced exposure, because an unexplained loss is a weaker
justification for punishment than an explained one.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

# Bond sizing and slash caps are contractual terms (deployment agreement,
# Phase-0 counsel review) — these are engineering defaults, not commitments.
DEFAULT_BOND_RATIO = 0.10        # bond = 10% of allocated deployment capital (TAO terms)
DEFAULT_WINDOW_SLASH_CAP = 0.25  # ≤ 25% of the ORIGINAL bond per settlement window
DEFAULT_AMBIGUOUS_EXPOSURE = 0.5  # fraction of an UNEXPLAINED loss that is slashed at all

# Normalized per-role diagnostic keys (higher = better) from the subnet's
# PairScore.model_breakdown / trading_breakdown. Kept as plain strings: the
# fund layer must not import the subnet package (separate distributions).
RESEARCHER_DIAGNOSTICS = ("overfitting_penalty", "penalized_f1", "variance")
TRADER_DIAGNOSTICS = ("execution_quality", "consistency", "penalized_sharpe")


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


def degradation(validated: dict[str, float], live: dict[str, float],
                keys: tuple[str, ...]) -> float:
    """How much a role's own diagnostics deteriorated from validation to live,
    in [0, 1]. Mean relative drop across `keys`, which are normalized
    higher-is-better scores from the subnet's per-role breakdowns.

    0 means the role performed as validated (its diagnostics do not explain the
    loss); 1 means total breakdown. Keys absent from either dict are skipped —
    an unavailable diagnostic must read as "explains nothing", never as fault.
    """
    drops = []
    for key in keys:
        before, after = validated.get(key), live.get(key)
        if before is None or after is None or before <= 0:
            continue
        drops.append(min(1.0, max(0.0, (before - after) / before)))
    return sum(drops) / len(drops) if drops else 0.0


@dataclass(frozen=True)
class LossAttribution:
    """Per-role degradation for one settlement window, keyed by miner coldkey.

    Build with `degradation()` against the pair's validation-time and live
    diagnostic breakdowns. Omit entirely (pass None to the slash call) when
    diagnostics are unavailable — the loss is then treated as fully
    unexplained, which reduces the slash rather than defaulting to blame.
    """

    researcher_coldkey: str
    trader_coldkey: str
    researcher_degradation: float
    trader_degradation: float

    def __post_init__(self):
        for value in (self.researcher_degradation, self.trader_degradation):
            if not 0.0 <= value <= 1.0:
                raise ValueError("degradation must be in [0, 1]")


@dataclass(frozen=True)
class BlameSplit:
    """Decomposition of one slash into explained and unexplained parts."""

    explained_fraction: float             # of the loss, what diagnostics account for
    attributed: dict[str, float]          # coldkey → share of the EXPLAINED part
    ambiguous_fraction: float             # 1 − explained_fraction
    slashable_fraction: float             # explained + ambiguous×exposure ≤ 1


def blame_split(bond: Bond, attribution: LossAttribution | None,
                ambiguous_exposure: float = DEFAULT_AMBIGUOUS_EXPOSURE) -> BlameSplit:
    """Split responsibility for a loss between two independently-operating miners.

    Explained portion: `min(1, d_researcher + d_trader)` of the loss, divided
    between the roles in proportion to their own degradation. Unexplained
    residual: a genuine joint-mismatch cost (a sound model and a sound strategy
    can still be a bad pairing), so it is shared pro-rata by bond — but only
    `ambiguous_exposure` of it is slashed at all. Punishment scales with the
    strength of the justification; the unslashed remainder is simply forgiven,
    since the bond is an incentive device, not a loss-recovery claim.
    """
    if attribution is None:
        return BlameSplit(0.0, {}, 1.0, ambiguous_exposure)
    d_r = attribution.researcher_degradation
    d_t = attribution.trader_degradation
    total_d = d_r + d_t
    explained = min(1.0, total_d)
    attributed = {} if total_d <= 0 else {
        attribution.researcher_coldkey: explained * d_r / total_d,
        attribution.trader_coldkey: explained * d_t / total_d,
    }
    ambiguous = 1.0 - explained
    return BlameSplit(explained, attributed, ambiguous,
                      explained + ambiguous * ambiguous_exposure)


@dataclass(frozen=True)
class SlashResult:
    pair_id: str
    window_id: str
    realized_loss_tao: float
    slash_alpha: float                    # total, queued for burn settlement
    per_coldkey: dict[str, float]         # by attribution, not by bond size alone
    capped: bool                          # per-window cap bound before loss did
    blame: BlameSplit


def slash_for_window(bond: Bond, realized_loss_tao: float, window_id: str,
                     attribution: LossAttribution | None = None,
                     window_cap: float = DEFAULT_WINDOW_SLASH_CAP,
                     ambiguous_exposure: float = DEFAULT_AMBIGUOUS_EXPOSURE) -> SlashResult:
    """Slash for one settlement window's NET realized loss.

    Size: `original_bond × (loss / deployed_capital)`, reduced by the share of
    the loss no diagnostic explains, then capped at `window_cap` × the original
    bond and at the remaining bond. Windows net internally, but a profitable
    window never restores prior slashes — the bond only ratchets down (gains
    pay through the standard reward split).

    Split: by `blame_split` — the explained part follows the degraded role, the
    unexplained remainder is shared pro-rata. See the module docstring for why
    a pure pro-rata split is not acceptable across unaffiliated miners.
    """
    if bond.state is not BondState.ACTIVE:
        raise CollateralError(f"{bond.pair_id}: cannot slash bond in state {bond.state.value}")
    blame = blame_split(bond, attribution, ambiguous_exposure)
    if realized_loss_tao <= 0:
        return SlashResult(bond.pair_id, window_id, realized_loss_tao, 0.0, {}, False, blame)

    loss_ratio = realized_loss_tao / bond.deployed_capital_tao
    uncapped = bond.original_alpha * loss_ratio * blame.slashable_fraction
    cap = bond.original_alpha * window_cap
    slash = min(uncapped, cap, bond.remaining_alpha)

    # `blame.attributed` values already sum to explained_fraction, and the
    # slashed ambiguous weight is the rest of slashable_fraction; dividing by
    # slashable_fraction renormalizes both onto the actual (possibly capped)
    # slash so the per-coldkey shares always sum to it exactly.
    original = bond.original_alpha
    ambiguous_weight = blame.slashable_fraction - blame.explained_fraction

    per_coldkey = {ck: 0.0 for ck in bond.contributions}
    for coldkey, share in blame.attributed.items():
        if coldkey not in per_coldkey:
            raise CollateralError(
                f"{bond.pair_id}: attribution names {coldkey}, not a bond contributor")
        per_coldkey[coldkey] += slash * share / blame.slashable_fraction
    for coldkey, amount in bond.contributions.items():
        per_coldkey[coldkey] += slash * (amount / original) * ambiguous_weight / blame.slashable_fraction

    bond.slashed_alpha += slash
    return SlashResult(bond.pair_id, window_id, realized_loss_tao, slash,
                       per_coldkey, capped=uncapped > slash + 1e-12, blame=blame)


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
              attribution: LossAttribution | None = None,
              window_cap: float = DEFAULT_WINDOW_SLASH_CAP,
              ambiguous_exposure: float = DEFAULT_AMBIGUOUS_EXPOSURE) -> SlashResult:
        result = slash_for_window(self._get(pair_id), realized_loss_tao, window_id,
                                  attribution, window_cap, ambiguous_exposure)
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
