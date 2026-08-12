"""Root-basket (v3) model against every number the ROOTFUND spec publishes,
plus the R15/R16 reflexivity scenarios and escrow monitoring."""

import pytest

from chainio import (ChainParams, StaticProvider, ValidatorBasket,
                     stake_weighted_insignia_weight, total_escrow_alpha)
from lockmgr import monitor
from risk import reflexivity
from treasury import emissions

PARAMS = ChainParams()
NETUID = 77


# --- emissions: ROOTFUND spec §2–§6 -----------------------------------------


def test_root_base_yield_matches_release_notes():
    # 983 τ/day × 365 / 5,374,582 ≈ 6.7%/yr — "the starting point, not the story"
    assert emissions.root_base_yield(PARAMS) == pytest.approx(0.0668, abs=0.001)


def test_sleeve_yield_table_matches_spec():
    # ROOTFUND §2: w=0.9, t=0.18 — 4.0% at g=−30%, 5.5% flat, 10.4% at +100%
    for g, expected in [(-0.30, 0.040), (0.0, 0.055), (0.50, 0.079),
                        (1.00, 0.104), (1.50, 0.129)]:
        assert emissions.sleeve_yield(PARAMS, 0.9, g) == pytest.approx(expected, abs=0.001)


def test_dividend_bid_table_matches_spec():
    # ROOTFUND §2: w=0.9 — 16.5 / 41.1 / 82.3 / 164.6 τ/day at 100k/250k/500k/1M τ
    for sleeve, expected in [(1e5, 16.5), (2.5e5, 41.1), (5e5, 82.3), (1e6, 164.6)]:
        assert emissions.dividend_bid(PARAMS, sleeve, 0.9) == pytest.approx(expected, abs=0.2)


def test_breakeven_sell_through_matches_spec():
    # ROOTFUND §6: a 500k τ sleeve alone absorbs σ ≈ 0.57 at p = 0.02
    bid = emissions.dividend_bid(PARAMS, 5e5, 0.9)
    assert emissions.breakeven_sell_through(PARAMS, bid, 0.02) == pytest.approx(0.57, abs=0.01)


def test_escrow_steady_state_matches_spec():
    # ROOTFUND §4: F ≈ 82.3 τ/day, p = 0.02, c = 0.5/yr → E* ≈ 3.0M α
    bid = emissions.dividend_bid(PARAMS, 5e5, 0.9)
    assert emissions.escrow_steady_state(bid, 0.02, 0.5) == pytest.approx(3.0e6, rel=0.01)


def test_staker_yield_with_escrow_matches_spec():
    # ROOTFUND §4: S=12M, L=4M — 10.9% at E*=0.36M, 9.3% at E*=1.8M (rp=0.155)
    for escrow, expected in [(0.36e6, 0.109), (1.8e6, 0.093)]:
        y = emissions.staker_yield_with_escrow(PARAMS, 365, 12e6, 4e6, escrow)
        assert y == pytest.approx(expected, abs=0.002)
    # and the v1 baseline it improves on: 7.6% with LP alpha in the base
    assert emissions.staker_yield_with_escrow(PARAMS, 365, 12e6, 0.0, 0.0) == pytest.approx(
        0.076, abs=0.002)


def test_staked_lp_drag_matches_spec():
    # ROOTFUND §0.5: S=12M, rp=0.155 — 1.4 / 4.1 / 7.2 / 10.4 pp at σ = 0/30/65/100%
    for sigma, expected_pp in [(0.0, 1.4), (0.30, 4.1), (0.65, 7.2), (1.0, 10.4)]:
        drag = emissions.staked_lp_drag(PARAMS, 12e6, 365, sigma)
        assert drag * 100 == pytest.approx(expected_pp, abs=0.15)
    # staking + wrapper reinvestment vs the unstaked hurdle (21.9% at 12M)
    assert emissions.staked_lp_drag(PARAMS, 12e6, 365, 0.65) < \
        emissions.dilution_hurdle(PARAMS, 12e6) / 2


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


# --- reflexivity: v3 scenarios ------------------------------------------------


FAST = dict(n_paths=60, months=18)


def test_neutral_defaults_reproduce_v1_model():
    # sleeve/weights default to zero — pre-v441 paths must be bit-identical
    v1 = reflexivity.run(reflexivity.ScenarioConfig(**FAST))
    v1_again = reflexivity.run(reflexivity.ScenarioConfig(**FAST, sleeve_tao=0.0,
                                                          w_ins=0.0, w_ext=0.0))
    assert v1.p_spiral == v1_again.p_spiral
    assert v1.median_terminal_discount == v1_again.median_terminal_discount


def test_v3_baseline_runs_and_reports():
    config = reflexivity.v3_baseline(reflexivity.ScenarioConfig(**FAST))
    assert config.locked_alpha == 0.0 and config.sleeve_tao == 5e5
    report = reflexivity.run(config)
    assert 0.0 <= report.p_spiral <= 1.0
    assert "root basket" in report.summary()


def test_rotation_shock_weakens_the_bid():
    from dataclasses import replace
    base = reflexivity.v3_baseline(reflexivity.ScenarioConfig(**FAST))
    calm = reflexivity.run(base)
    rotated = reflexivity.run(replace(
        base, basket_rotation=reflexivity.BasketRotationShock(3, 6, sleeve_outflow_monthly=0.5)))
    # losing the dividend bid cannot make outcomes better
    assert rotated.mean_terminal_share <= calm.mean_terminal_share + 1e-9
    assert rotated.p_spiral >= calm.p_spiral - 1e-9


def test_claim_clustering_is_priced():
    from dataclasses import replace
    base = reflexivity.v3_baseline(reflexivity.ScenarioConfig(**FAST))
    shocked = replace(base, revenue_shock=reflexivity.RevenueShock(3, 6, 1.0))
    mild = reflexivity.run(replace(shocked, claim_stress_mult=1.0))
    clustered = reflexivity.run(replace(shocked, claim_stress_mult=8.0))
    assert clustered.p_spiral >= mild.p_spiral - 1e-9


def test_v3_quarterly_report_covers_the_grid():
    text = reflexivity.quarterly_report_v3(reflexivity.ScenarioConfig(n_paths=20, months=12))
    for name in reflexivity.V3_SCENARIOS:
        assert f"[{name}]" in text
