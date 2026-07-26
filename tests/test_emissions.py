"""Emission math against every number the spec publishes (SPEC §0.7–0.12, §0.5)."""

import pytest

from chainio import ChainParams
from treasury import emissions

PARAMS = ChainParams()


def test_alpha_staker_share_ramp_matches_spec():
    # SPEC §0.10: 12.7% at 1 month, 23.5% at 3, 29.9% at 6, 34.6% at 12 — vs the 41% headline
    for age_days, expected in [(30, 0.127), (90, 0.235), (180, 0.299), (365, 0.346)]:
        assert emissions.alpha_staker_share(PARAMS, age_days) == pytest.approx(expected, abs=0.002)


def test_issuance_retention_matches_spec():
    # SPEC §0.5: at a 1-year-old subnet — 93.6% retained at 0% miner sell-through,
    # 81.3% at 30%, 69.0% at 60%, 52.6% at 100%
    for sell, expected in [(0.0, 0.936), (0.3, 0.813), (0.6, 0.690), (1.0, 0.526)]:
        assert emissions.issuance_retention(PARAMS, 365, sell) == pytest.approx(expected, abs=0.002)


def test_dilution_hurdle_matches_spec():
    # SPEC §0.5: 263% at 1M alpha, 52.6% at 5M, 26.3% at 10M, 13.1% at 20M
    for supply, expected in [(1e6, 2.63), (5e6, 0.526), (10e6, 0.263), (20e6, 0.131)]:
        assert emissions.dilution_hurdle(PARAMS, supply) == pytest.approx(expected, abs=0.005)


def test_ema_responsiveness_ramp_matches_spec():
    # SPEC §0.8: 20% at day 7, 50% at day 28, 76% at day 90, 93% at day 365
    bpd = PARAMS.blocks_per_day
    for day, expected in [(7, 0.20), (28, 0.50), (90, 0.76), (365, 0.93)]:
        assert emissions.ema_responsiveness(PARAMS, day * bpd) == pytest.approx(expected, abs=0.005)


def test_emission_share_caps_price_and_taxes_burn():
    # p is capped at 1.0
    assert emissions.emission_share(5.0, 0.0, [(1.0, 0.0)]) == pytest.approx(0.5)
    # b_i taxes share one-for-one (SPEC §0.11): a 51% owner-burn roughly halves share
    clean = emissions.emission_share(1.0, 0.0, [(1.0, 0.0)] * 9)
    burned = emissions.emission_share(1.0, 0.51, [(1.0, 0.0)] * 9)
    assert burned / clean == pytest.approx(0.49, abs=0.03)


def test_leakage_drag_matches_v6_model():
    # v6 model printout at 12M supply: 0.7 / 3.4 / 6.1 / 9.7 pp for 0/30/60/100% sell
    for sell, expected_pp in [(0.0, 0.7), (0.3, 3.4), (0.6, 6.1), (1.0, 9.7)]:
        assert emissions.leakage_drag(PARAMS, 12e6, sell) * 100 == pytest.approx(
            expected_pp, abs=0.15)


def test_lp_daily_yield_is_recapture_scale():
    # 3M staked on a 1-year subnet: daily yield ≈ 7,200 × 34.6% / 3M ≈ 0.083%/day
    daily = emissions.lp_daily_yield(PARAMS, 365, 3e6)
    assert daily == pytest.approx(7200 * 0.346 / 3e6, rel=0.02)
