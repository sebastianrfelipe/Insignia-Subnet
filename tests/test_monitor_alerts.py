"""Monitor invariants and alert rules (SPEC §4, §7)."""

from dataclasses import replace

from chainio import ChainParams
from lockmgr import monitor
from lockmgr.locks import OnChainLock
from lockmgr.schedules import LockState, LpLock
from risk import alerts

PARAMS = ChainParams()


def _lock() -> LpLock:
    lock = LpLock(lp_id="lp1", coldkey="ck-1", hotkey="owner-hk", netuid=1,
                  m0=1e6, lock_day=0.0, outer_bound_day=365.0)
    lock.state = LockState.PERPETUAL
    return lock


def test_param_drift_pages_on_lock_rates():
    current = replace(PARAMS, unlock_rate_blocks=500_000.0, tao_weight=0.20)
    findings = monitor.param_drift(PARAMS, current)
    kinds = {(f.severity, f.detail.split(":")[0]) for f in findings}
    assert ("page", "unlock_rate_blocks") in kinds
    assert ("warn", "tao_weight") in kinds


def test_wrong_hotkey_and_early_decay_page():
    on_chain = OnChainLock("ck-1", "rogue-hk", 1, 1e6, 0.0, perpetual=False)
    findings = monitor.verify_lock_invariants(_lock(), on_chain, "owner-hk", PARAMS, day=100.0)
    kinds = {f.kind for f in findings}
    assert "wrong_hotkey" in kinds
    assert "decay_leak" in kinds


def test_missing_lock_pages():
    findings = monitor.verify_lock_invariants(_lock(), None, "owner-hk", PARAMS, day=100.0)
    assert findings[0].kind == "lock_missing"


def test_unstaked_lp_pages():
    findings = monitor.unstaked_lp_positions({"ck-1": 0.0}, [_lock()])
    assert findings and findings[0].kind == "unstaked_lp"
    assert monitor.unstaked_lp_positions({"ck-1": 1e6}, [_lock()]) == []


def test_king_early_warning_thresholds():
    safe = monitor.KingWatch(owner_aggregate=10e6, largest_external=1e6,
                             largest_external_hotkey="hk-x", subnet_alpha_out=20e6)
    assert monitor.king_early_warning(safe) == []

    exposed = monitor.KingWatch(owner_aggregate=3e6, largest_external=2.5e6,
                                largest_external_hotkey="hk-x", subnet_alpha_out=20e6)
    kinds = {f.kind for f in monitor.king_early_warning(exposed)}
    assert kinds == {"king_defense_ratio", "king_threshold"}


def test_dispatcher_dedupes():
    sent_log = []
    dispatcher = alerts.Dispatcher(sinks=[sent_log.append])
    batch = [alerts.Alert("warn", "x", "same"), alerts.Alert("warn", "x", "same")]
    assert len(dispatcher.dispatch(batch)) == 1
    assert len(dispatcher.dispatch(batch)) == 0
    assert len(sent_log) == 1


def test_cohort_alert_levels():
    def lock(day: float) -> LpLock:
        return LpLock(lp_id=f"lp{day}", coldkey=f"ck{day}", hotkey="owner-hk", netuid=1,
                      m0=1e6, lock_day=0.0, outer_bound_day=day)

    clustered = [lock(365), lock(370), lock(380), lock(700)]
    result = alerts.from_cohorts(clustered)
    assert result and result[0].severity == "page"

    staggered = [lock(365 + 100 * i) for i in range(4)]
    assert [a.severity for a in alerts.from_cohorts(staggered)] in ([], ["warn"])
