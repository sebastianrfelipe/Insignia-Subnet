"""Phase-0 compliance gate (SPEC §2 Phase 0).

Investor-facing functionality is BLOCKED until securities counsel signs off by
placing `LEGAL_SIGNOFF.md` at the repository root. This module is the code-level
enforcement: OTC quoting, settlement, and factsheet publication all call
`require_legal_signoff()` first. Do not special-case around it; the gate being
annoying is the point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGAL_SIGNOFF_FILE = REPO_ROOT / "LEGAL_SIGNOFF.md"


class ComplianceGateError(RuntimeError):
    pass


def legal_signoff_present(signoff_file: Path = LEGAL_SIGNOFF_FILE) -> bool:
    return signoff_file.is_file()


def require_legal_signoff(signoff_file: Path = LEGAL_SIGNOFF_FILE) -> None:
    if not legal_signoff_present(signoff_file):
        raise ComplianceGateError(
            "Phase-0 gate: LEGAL_SIGNOFF.md not found at repo root. Investor-facing "
            "functionality (OTC quotes, settlement, factsheet publication) is blocked "
            "until securities counsel signs off — see docs/SPEC.md §2 Phase 0."
        )


@dataclass
class KycRegistry:
    """Counterparties cleared by the Phase-0 KYC/AML process. Every OTC
    counterparty must be registered before quoting (SPEC §6)."""

    approved: set[str] = field(default_factory=set)

    def approve(self, counterparty_id: str) -> None:
        self.approved.add(counterparty_id)

    def require(self, counterparty_id: str) -> None:
        if counterparty_id not in self.approved:
            raise ComplianceGateError(
                f"counterparty {counterparty_id!r} has not passed the Phase-0 KYC gate"
            )
