"""Compliance gate, OTC quoting, and settlement verification (SPEC §2 Phase 0, §6)."""

import pytest

from chainio import reference_pool
from lockmgr.locks import OnChainLock
from otc import compliance, desk as desk_mod
from otc.compliance import ComplianceGateError, KycRegistry
from otc.desk import DiscountSchedule, OtcDesk
from otc.settlement import Settlement, SettlementAgent, SettlementState
from otc.desk import OtcQuote
import datetime as dt


def test_legal_gate_blocks_without_signoff(tmp_path):
    missing = tmp_path / "LEGAL_SIGNOFF.md"
    assert not compliance.legal_signoff_present(missing)
    with pytest.raises(ComplianceGateError):
        compliance.require_legal_signoff(missing)
    missing.write_text("signed", encoding="utf-8")
    compliance.require_legal_signoff(missing)  # no raise once counsel signs off


def test_repo_gate_is_closed_and_blocks_quotes():
    # There is deliberately no LEGAL_SIGNOFF.md in this repo yet (Phase 0 open).
    assert not compliance.legal_signoff_present()
    kyc = KycRegistry(approved={"lp-1"})
    with pytest.raises(ComplianceGateError):
        OtcDesk(kyc=kyc).quote("lp-1", reference_pool(), 100_000.0, 12)


@pytest.fixture
def open_gate(monkeypatch):
    monkeypatch.setattr(desk_mod, "require_legal_signoff", lambda: None)
    import otc.settlement as settlement_mod

    monkeypatch.setattr(settlement_mod, "require_legal_signoff", lambda: None)


def test_kyc_required(open_gate):
    with pytest.raises(ComplianceGateError):
        OtcDesk(kyc=KycRegistry()).quote("stranger", reference_pool(), 100_000.0, 12)


def test_discount_scales_with_lock_commitment(open_gate):
    schedule = DiscountSchedule()
    assert schedule.discount(0) == -0.02      # unlocked delivery: premium
    assert schedule.discount(11) == 0.03      # falls back to the 6-month tier
    assert schedule.discount(12) == 0.08
    assert schedule.discount(36) == 0.12

    otc = OtcDesk(kyc=KycRegistry(approved={"lp-1"}))
    pool = reference_pool()
    q_locked = otc.quote("lp-1", pool, 200_000.0, 12, nav_per_alpha=0.06)
    q_unlocked = otc.quote("lp-1", pool, 200_000.0, 0, nav_per_alpha=0.06)
    assert q_locked.quote_price < q_unlocked.quote_price
    # depth-aware reference is above spot: buying 200k on the pool costs more per alpha
    assert q_locked.pool_reference_price > pool.spot_price
    assert q_locked.below_nav  # quoting below NAV must be surfaced, never silent


def _quote(amount: float) -> OtcQuote:
    return OtcQuote(counterparty_id="lp-1", alpha_amount=amount, lock_months=12,
                    pool_reference_price=0.056, discount=0.08, quote_price=0.0515,
                    total_tao=0.0515 * amount, nav_per_alpha=None, below_nav=False,
                    expires_at=dt.datetime(2026, 8, 1, tzinfo=dt.timezone.utc))


class FakeLockClient:
    def __init__(self, lock: OnChainLock | None):
        self.lock = lock

    def get_coldkey_lock(self, coldkey: str):
        return self.lock


def _agent(lock, deadline=600) -> SettlementAgent:
    return SettlementAgent(FakeLockClient(lock), KycRegistry(approved={"lp-1"}),
                           verify_deadline_blocks=deadline)


def _delivered(agent) -> Settlement:
    s = Settlement(quote=_quote(100_000.0), lp_coldkey="ck-lp1", owner_hotkey="owner-hk")
    s.delivery_block = 1_000
    s.state = SettlementState.DELIVERED
    return s


def test_settlement_verifies_compliant_lock(open_gate):
    lock = OnChainLock("ck-lp1", "owner-hk", 1, 100_000.0, 100_000.0, perpetual=True)
    agent = _agent(lock)
    s = _delivered(agent)
    assert agent.check(s, current_block=1_100) is SettlementState.VERIFIED


def test_settlement_clawback_on_wrong_hotkey(open_gate):
    lock = OnChainLock("ck-lp1", "rogue-hk", 1, 100_000.0, 0.0, perpetual=True)
    agent = _agent(lock)
    s = _delivered(agent)
    assert agent.check(s, current_block=1_100) is SettlementState.CLAWBACK
    assert "owner hotkey" in s.failure_reason


def test_settlement_waits_for_perpetual_then_claws_back_at_deadline(open_gate):
    lock = OnChainLock("ck-lp1", "owner-hk", 1, 100_000.0, 0.0, perpetual=False)
    agent = _agent(lock)
    s = _delivered(agent)
    # perpetual flag not yet set: keep waiting inside the deadline
    assert agent.check(s, current_block=1_100) is SettlementState.DELIVERED
    # deadline passes without the flag: clawback per agreement
    assert agent.check(s, current_block=1_700) is SettlementState.CLAWBACK
