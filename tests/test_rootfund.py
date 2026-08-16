"""Root Reborn overlay against SPEC §0.16 — protocol flow, not the LP product.

LP return is the wrapper identity in test_emissions.py / SYSTEM_EQUATIONS §9.
This module covers basket bid, escrow, claim monitoring, and R15/R16 shocks
on top of locked-alpha LPs.
"""

from dataclasses import replace

import pytest

from chainio import (ChainParams, StaticProvider, ValidatorBasket,
                     stake_weighted_insignia_weight, total_escrow_alpha)
from lockmgr import monitor
from risk import reflexivity
from treasury import emissions

PARAMS = ChainParams()
NETUID = 77


# --- emissions: SPEC §0.16 overlay ------------------------------------------


def test_root_base_yield_is_the_coupon_not_lp_yield():
    # 983 τ/day × 365 / 5,374,582 ≈ 6.7%/yr — root TAO coupon, not wrapper return
    assert emissions.root_base_yield(PARAMS) == pytest.approx(0.0668, abs=0.001)
    y_alpha = emissions.lp_annual_yield(PARAMS, 365, 12e6)
    assert y_alpha == pytest.approx(0.076, abs=0.002)
    # a 100% alpha move doubles the LP; it barely moves the root coupon
    assert emissions.lp_return(y_alpha, 1.0) == pytest.approx(2.0 * (1 + y_alpha) - 1)


def test_dividend_bid_scales_with_root_stake_not_lp_notional():
    # w=0.9 — 16.5 / 41.1 / 82.3 / 164.6 τ/day at 100k/250k/500k/1M τ of *root* stake
    for stake, expected in [(1e5, 16.5), (2.5e5, 41.1), (5e5, 82.3), (1e6, 164.6)]:
        assert emissions.dividend_bid(PARAMS, stake, 0.9) == pytest.approx(expected, abs=0.2)


def test_breakeven_sell_through_from_standing_bid():
    bid = emissions.dividend_bid(PARAMS, 5e5, 0.9)
    assert emissions.breakeven_sell_through(PARAMS, bid, 0.02) == pytest.approx(0.57, abs=0.01)


def test_escrow_steady_state():
    # F ≈ 82.3 τ/day, p = 0.02, c = 0.5/yr → E* ≈ 3.0M α
    bid = emissions.dividend_bid(PARAMS, 5e5, 0.9)
    assert emissions.escrow_steady_state(bid, 0.02, 0.5) == pytest.approx(3.0e6, rel=0.01)


def test_staker_yield_keeps_lp_alpha_in_the_base():
    # SYSTEM_EQUATIONS §5: 7.6% at S=12M, rp≈0.155. Escrow dilutes per-unit yield.
    assert emissions.staker_yield_with_escrow(PARAMS, 365, 12e6, 0.0) == pytest.approx(
        0.076, abs=0.002)
    with_escrow = emissions.staker_yield_with_escrow(PARAMS, 365, 12e6, 1.8e6)
    assert with_escrow < emissions.staker_yield_with_escrow(PARAMS, 365, 12e6, 0.0)


def test_deferred_root_slice_is_leaked_not_retained():
    # retention already excludes the root slice; the deferred slice is what
    # accrues to escrow instead of auto-selling (SPEC §0.16)
    age = 365
    slice_ = emissions.deferred_root_slice(PARAMS, age)
    assert slice_ == pytest.approx(PARAMS.validator_cut - emissions.alpha_staker_share(PARAMS, age),
                                   abs=1e-9)
    assert emissions.issuance_retention(PARAMS, age, 0.0) + slice_ == pytest.approx(
        1.0, abs=1e-9)


def test_maintenance_and_external_bid():
    assert emissions.maintenance_flow(PARAMS, 0.65, 0.02) == pytest.approx(93.6)
    # default (curation unset) follower weight ≈ emission share: 1% → 9.83 τ/day
    assert emissions.external_basket_bid(PARAMS, 0.01) == pytest.approx(9.83)


# --- chainio: basket read layer ----------------------------------------------


def _baskets() -> list[ValidatorBasket]:
    return [
        ValidatorBasket("hk-insignia", 5e5, {NETUID: 0.9, 0: 0.03}, escrow_alpha=1.2e6),
        ValidatorBasket("hk-follower", 1.5e6, {NETUID: 0.05, 0: 0.5}, escrow_alpha=3.0e5),
        ValidatorBasket("hk-passive", 3e6, {}, escrow_alpha=0.0),
    ]


def test_stake_weighted_weight_and_escrow_totals():
    baskets = _baskets()
    w = stake_weighted_insignia_weight(baskets, NETUID)
    assert w == pytest.approx((0.9 * 5e5 + 0.05 * 1.5e6) / 5e6)
    assert total_escrow_alpha(baskets) == pytest.approx(1.5e6)
    assert stake_weighted_insignia_weight([], NETUID) == 0.0


def test_static_provider_serves_baskets():
    provider = StaticProvider(baskets=_baskets())
    assert len(provider.root_baskets(NETUID)) == 3
    assert provider.with_params(root_tao=6e6).chain_params().root_tao == 6e6


# --- lockmgr: escrow monitoring (R15/R16) ------------------------------------


def test_escrow_no_visibility_is_a_warning_not_zero():
    watch = monitor.EscrowWatch(netuid=NETUID, baskets=[], subnet_alpha_out=5e6)
    kinds = [f.kind for f in monitor.escrow_findings(watch)]
    assert kinds == ["escrow_no_visibility"]


def test_escrow_consensus_weight_and_rotation_warnings():
    watch = monitor.EscrowWatch(
        netuid=NETUID, baskets=_baskets(), subnet_alpha_out=10e6,
        previous_weights={"hk-follower": 0.10})  # 0.10 → 0.05: −50% rotation
    findings = monitor.escrow_findings(watch)
    kinds = {f.kind for f in findings}
    assert "escrow_consensus_weight" in kinds     # 1.2M / 10M = 12% ≥ 5%
    assert "basket_rotation" in kinds             # R15 early-warning
    assert "escrow_level" in kinds
    assert "claim_cluster" not in kinds


def test_claim_cluster_pages():
    watch = monitor.EscrowWatch(
        netuid=NETUID, baskets=_baskets(), subnet_alpha_out=10e6,
        trailing_claim_alpha_30d=0.3e6)  # 3.6M/yr on a 1.5M basket = 2.4×
    findings = monitor.escrow_findings(watch)
    cluster = [f for f in findings if f.kind == "claim_cluster"]
    assert cluster and cluster[0].severity == "page"


# --- reflexivity: wrapper baseline + overlay shocks --------------------------


FAST = dict(n_paths=60, months=18)


def test_default_scenario_is_locked_alpha_lps():
    config = reflexivity.ScenarioConfig(**FAST)
    assert config.locked_alpha > 0
    assert config.validator_root_tao == 0.0
    report = reflexivity.run(config)
    assert 0.0 <= report.p_spiral <= 1.0
    assert "root basket" not in report.summary()


def test_neutral_basket_defaults_match_wrapper_only():
    wrapper = reflexivity.run(reflexivity.ScenarioConfig(**FAST))
    explicit = reflexivity.run(reflexivity.ScenarioConfig(
        **FAST, validator_root_tao=0.0, w_ins=0.0, w_ext=0.0))
    assert wrapper.p_spiral == explicit.p_spiral
    assert wrapper.median_terminal_discount == explicit.median_terminal_discount


def test_rotation_shock_weakens_the_bid():
    base = reflexivity.ScenarioConfig(
        **FAST, locked_alpha=3e6, validator_root_tao=5e5, w_ins=0.9, w_ext=0.05,
        revenue_shock=None)
    calm = reflexivity.run(base)
    rotated = reflexivity.run(replace(
        base, basket_rotation=reflexivity.BasketRotationShock(
            3, 6, validator_outflow_monthly=0.5)))
    assert rotated.mean_terminal_share <= calm.mean_terminal_share + 1e-9
    assert rotated.p_spiral >= calm.p_spiral - 1e-9


def test_claim_clustering_is_priced():
    base = reflexivity.ScenarioConfig(
        **FAST, locked_alpha=3e6, escrow_alpha0=1.5e6, w_ext=0.05,
        revenue_shock=reflexivity.RevenueShock(3, 6, 1.0))
    mild = reflexivity.run(replace(base, claim_stress_mult=1.0))
    clustered = reflexivity.run(replace(base, claim_stress_mult=8.0))
    assert clustered.p_spiral >= mild.p_spiral - 1e-9


def test_quarterly_report_covers_the_wrapper_grid():
    text = reflexivity.quarterly_report(reflexivity.ScenarioConfig(n_paths=20, months=12))
    for name in reflexivity.STANDARD_SCENARIOS:
        assert f"[{name}]" in text
