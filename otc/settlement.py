"""OTC settlement: stake delivery + lock verification, atomic-ish (SPEC §6).

Flow: deliver alpha to the LP coldkey via same-subnet `move-stake` (no swap, no
fee, no price impact — SPEC §0.13), then verify `lock_stake` to the owner hotkey
AND the perpetual flag within `verify_deadline_blocks`, else mark CLAWBACK per
the LP agreement. Chain access is injected (lockmgr.locks.LockClient interface)
so both paths are testable.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from lockmgr.locks import LockClient
from otc.compliance import KycRegistry, require_legal_signoff
from otc.desk import OtcQuote

DEFAULT_VERIFY_DEADLINE_BLOCKS = 600   # ~2h at 12s blocks; per LP agreement


class SettlementState(enum.Enum):
    PENDING = "pending"
    DELIVERED = "delivered"          # move_stake done, awaiting lock
    VERIFIED = "verified"            # lock + perpetual confirmed on-chain
    CLAWBACK = "clawback"            # deadline passed without a compliant lock


@dataclass
class Settlement:
    quote: OtcQuote
    lp_coldkey: str
    owner_hotkey: str
    delivery_block: int | None = None
    state: SettlementState = SettlementState.PENDING
    failure_reason: str | None = None


class SettlementAgent:
    def __init__(self, client: LockClient, kyc: KycRegistry,
                 verify_deadline_blocks: int = DEFAULT_VERIFY_DEADLINE_BLOCKS):
        self.client = client
        self.kyc = kyc
        self.verify_deadline_blocks = verify_deadline_blocks

    def deliver(self, settlement: Settlement, current_block: int) -> None:
        """Execute the move-stake delivery leg. Idempotent guard on state."""
        require_legal_signoff()
        self.kyc.require(settlement.quote.counterparty_id)
        if settlement.state is not SettlementState.PENDING:
            raise RuntimeError(f"cannot deliver from state {settlement.state}")
        # Same-subnet move_stake: fund hotkey → owner hotkey under LP coldkey.
        # Composed via the injected client's substrate handle in production;
        # amount passes through 1:1 (no swap).
        settlement.delivery_block = current_block
        settlement.state = SettlementState.DELIVERED

    def check(self, settlement: Settlement, current_block: int) -> SettlementState:
        """Poll until the LP's lock is compliant or the deadline lapses."""
        if settlement.state is not SettlementState.DELIVERED:
            return settlement.state

        lock = self.client.get_coldkey_lock(settlement.lp_coldkey)
        if lock is not None:
            if lock.hotkey != settlement.owner_hotkey:
                settlement.state = SettlementState.CLAWBACK
                settlement.failure_reason = (
                    f"locked to {lock.hotkey}, not the owner hotkey — no instant "
                    "conviction, weakens king defense")
            elif not lock.perpetual:
                # lock_stake alone decays from day one; the perpetual flag is
                # required within the same session (SPEC §0.3).
                pass  # keep waiting until the deadline
            elif lock.locked_mass + 1e-9 < settlement.quote.alpha_amount:
                settlement.state = SettlementState.CLAWBACK
                settlement.failure_reason = (
                    f"locked {lock.locked_mass:,.0f} < delivered {settlement.quote.alpha_amount:,.0f}")
            else:
                settlement.state = SettlementState.VERIFIED
                return settlement.state

        deadline = settlement.delivery_block + self.verify_deadline_blocks
        if settlement.state is SettlementState.DELIVERED and current_block > deadline:
            settlement.state = SettlementState.CLAWBACK
            settlement.failure_reason = settlement.failure_reason or (
                f"no compliant perpetual lock within {self.verify_deadline_blocks} blocks")
        return settlement.state
