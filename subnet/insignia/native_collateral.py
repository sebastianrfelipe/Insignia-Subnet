"""Native Subtensor registration collateral — validator-side gate.

Fund-layer pallet math lives in `chainio/collateral.py` (the subnet package
must not import the fund distribution). This module is the live path the
paired validator uses to:

1. Zero Yuma weights when a miner is short of the published collateral floor.
2. Freeze a miner after a martingale / max-drawdown blow-up so they cannot
   immediately farm emission (and drain the lock) on the next lucky window.

Const's premise (SN8 Sharpe/Sortino gaming): a miner who levers up until they
blow up looks statistically brilliant on a short window. Scoring already
withholds that epoch's upside. The freeze keeps weights at zero across
subsequent epochs so remaining native collateral stays locked until they
re-register and earn it back. See docs/COLLATERAL.md and
docs/INCENTIVE_MECHANISM.md §Native Registration Collateral.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Same hard ceiling as TradingValidator.max_drawdown_limit / scoring docs.
DEFAULT_DRAWDOWN_FREEZE = 0.20


@dataclass(frozen=True)
class CollateralPosition:
    """Minimal per-miner view the gate needs. `uid` matches the weight dict."""

    uid: str
    locked: float = 0.0
    min_locked: float = 0.0
    stake: float = 0.0

    def covers(self, required_min: float) -> bool:
        if required_min <= 0:
            return True
        return self.locked + 1e-12 >= required_min


@dataclass(frozen=True)
class FreezeRecord:
    uid: str
    reason: str
    since_generation: int
    min_generations: int = 0   # 0 = until the UID leaves the metagraph


@dataclass
class FreezeLedger:
    """Martingale / exploit freeze. Weights stay zero → emission stops →
    native collateral cannot drain. Dropped when the UID is gone (pruned or
    re-registered) or `min_generations` have elapsed, whichever is later."""

    records: dict[str, FreezeRecord] = field(default_factory=dict)

    def freeze(self, uid: str, reason: str, generation: int,
               min_generations: int = 0) -> FreezeRecord:
        existing = self.records.get(uid)
        if existing is not None:
            return existing
        rec = FreezeRecord(uid, reason, generation, min_generations)
        self.records[uid] = rec
        return rec

    def sweep(self, present_uids: set[str], generation: int) -> list[str]:
        """Drop records whose UID left the metagraph and whose minimum
        freeze length has elapsed. Returns the released uids."""
        released = []
        for uid, rec in list(self.records.items()):
            gone = uid not in present_uids
            aged = generation - rec.since_generation >= rec.min_generations
            if gone and aged:
                del self.records[uid]
                released.append(uid)
        return released

    def active_uids(self, present_uids: set[str] | None = None) -> set[str]:
        if present_uids is None:
            return set(self.records)
        return set(self.records) & present_uids


def should_freeze_drawdown(max_drawdown: float,
                           limit: float = DEFAULT_DRAWDOWN_FREEZE) -> bool:
    """True when the trading book has breached the hard drawdown ceiling."""
    return max_drawdown >= limit


def apply_collateral_gate(
    weights: dict[str, float],
    positions: dict[str, CollateralPosition],
    required_min: float = 0.0,
    freeze_uids: set[str] | None = None,
) -> tuple[dict[str, float], list[str]]:
    """Zero weights for miners short of the published floor or frozen.

    Gated miners stay in the dict at 0 so Yuma still sees them; they earn
    nothing and cannot drain native collateral. Same contract as
    `chainio.collateral.apply_collateral_gate`.
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
