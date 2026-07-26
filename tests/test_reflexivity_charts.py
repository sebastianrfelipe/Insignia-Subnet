"""Reflexivity Monte Carlo smoke tests and chart regeneration (SPEC §7, §8)."""

from dataclasses import replace

import pytest

from risk import reflexivity
from risk.reflexivity import RevenueShock, ScenarioConfig


SMALL = ScenarioConfig(months=12, n_paths=40, seed=11)


def test_run_produces_valid_report():
    report = reflexivity.run(SMALL)
    assert 0.0 <= report.p_spiral <= 1.0
    assert len(report.p_spiral_by_month) == SMALL.months
    assert all(0.0 <= p <= 1.0 for p in report.p_spiral_by_month)
    assert "P(spiral)" in report.summary()


def test_severe_shock_is_not_safer_than_baseline():
    baseline = reflexivity.run(replace(SMALL, revenue_shock=None))
    severe = reflexivity.run(replace(SMALL, revenue_shock=RevenueShock(3, 9, 1.0)))
    assert severe.p_spiral >= baseline.p_spiral - 0.05
    assert severe.median_terminal_discount <= baseline.median_terminal_discount + 0.10


def test_charts_regenerate(tmp_path):
    from dashboards import charts

    paths = charts.regenerate_all(tmp_path)
    assert len(paths) == 4
    for p in paths:
        assert p.exists() and p.stat().st_size > 10_000

    summary = charts.summary(charts.ChainParams(), charts.reference_pool())
    assert "Staged vs cliff" in summary


def test_factsheet_builds_and_publication_is_gated(tmp_path):
    import datetime as dt

    from chainio import ChainParams, reference_pool
    from dashboards.investor_api import build_factsheet, render_markdown
    from dashboards.investor_api import factsheet as fs_mod
    from lockmgr.schedules import LpLock
    from otc.compliance import ComplianceGateError
    from treasury.accounting import TreasuryBook, compute_nav

    pool = reference_pool()
    nav = compute_nav(dt.datetime(2026, 7, 26, tzinfo=dt.timezone.utc), TreasuryBook(),
                      pool, trading_aum_tao=400_000.0, circulating_alpha=6e6)
    locks = [LpLock(lp_id=f"lp{i}", coldkey=f"ck{i}", hotkey="owner-hk", netuid=1,
                    m0=1e6, lock_day=0.0, outer_bound_day=365 + 100 * i) for i in range(4)]
    fs = build_factsheet(
        period="2026-07", params=ChainParams(), nav=nav, subnet_age_days=365,
        miner_sell_through=0.3, staker_apy_alpha=0.30, staker_apy_usd=0.12,
        emission_share=0.004, emission_share_trend_wow=0.001, net_tao_flow=1_500.0,
        conviction_owner=4e6, conviction_top_external=0.5e6, locks=locks,
        reserve_coverage_months=6.5, buy_flow_executed_tao=1_800.0,
        revenue_attested_tao=4_000.0,
    )
    md = render_markdown(fs)
    assert "Premium/discount to NAV" in md
    assert "do not blend" in md

    with pytest.raises(ComplianceGateError):
        fs_mod.publish(fs, tmp_path)  # no LEGAL_SIGNOFF.md in this repo: gate closed
