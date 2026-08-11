"""Deployment collateral: bond lifecycle, pro-rata slashing, retention lever
(SPEC §0.5, §5; INCENTIVE_MECHANISM §Deployment Collateral)."""

import pytest

from chainio import ChainParams
from risk import alerts
from treasury import emissions
from treasury.collateral import (
    Bond,
    BondRegistry,
    BondState,
    CollateralError,
    required_bond_alpha,
    slash_for_window,
)


def make_bond(researcher=60_000.0, trader=40_000.0, capital=10_000.0,
              state=BondState.ACTIVE) -> Bond:
    return Bond(pair_id="pair-1",
                contributions={"ck_researcher": researcher, "ck_trader": trader},
                deployed_capital_tao=capital, state=state)


def test_bond_sizing_at_spot():
    # 10k TAO deployed at 10% ratio, alpha at 0.05 TAO → 20k alpha bond
    assert required_bond_alpha(10_000.0, 0.05) == pytest.approx(20_000.0)
    with pytest.raises(ValueError):
        required_bond_alpha(10_000.0, 0.0)


def test_slash_is_pro_rata_and_loss_proportional():
    bond = make_bond()
    # 5% realized loss of deployed capital → 5% of the original bond
    result = slash_for_window(bond, realized_loss_tao=500.0, window_id="W1")
    assert result.slash_alpha == pytest.approx(0.05 * 100_000.0)
    assert result.per_coldkey["ck_researcher"] == pytest.approx(3_000.0)
    assert result.per_coldkey["ck_trader"] == pytest.approx(2_000.0)
    assert not result.capped
    assert bond.remaining_alpha == pytest.approx(95_000.0)


def test_window_cap_binds_on_large_loss():
    bond = make_bond()
    # 60% loss would take 60% of the bond; the 25% per-window cap binds
    result = slash_for_window(bond, realized_loss_tao=6_000.0, window_id="W1")
    assert result.slash_alpha == pytest.approx(25_000.0)
    assert result.capped


def test_gains_never_slash_or_restore():
    bond = make_bond()
    slash_for_window(bond, 500.0, "W1")
    before = bond.remaining_alpha
    result = slash_for_window(bond, realized_loss_tao=-2_000.0, window_id="W2")
    assert result.slash_alpha == 0.0
    assert bond.remaining_alpha == before  # the bond only ratchets down


def test_slash_exhausts_at_remaining():
    bond = make_bond()
    for window in range(5):
        slash_for_window(bond, 6_000.0, f"W{window}", window_cap=0.25)
    assert bond.remaining_alpha == pytest.approx(0.0)
    assert slash_for_window(bond, 6_000.0, "W9").slash_alpha == 0.0


def test_slash_requires_active_state():
    with pytest.raises(CollateralError):
        slash_for_window(make_bond(state=BondState.PENDING), 500.0, "W1")


def test_registry_lifecycle_and_burn_queue():
    reg = BondRegistry()
    bond = make_bond(state=BondState.PENDING)
    reg.post(bond)
    with pytest.raises(CollateralError):
        reg.activate("pair-1", escrow_staked_alpha=90_000.0)  # under-escrowed
    reg.activate("pair-1", escrow_staked_alpha=100_000.0)
    assert bond.state is BondState.ACTIVE
    assert reg.total_bonded_alpha == pytest.approx(100_000.0)

    reg.slash("pair-1", 500.0, "W1")
    assert reg.pending_burn_alpha == pytest.approx(5_000.0)
    assert reg.total_bonded_alpha == pytest.approx(95_000.0)

    reg.mark_settled(5_000.0)
    assert reg.pending_burn_alpha == 0.0
    assert reg.settled_burn_alpha == pytest.approx(5_000.0)
    with pytest.raises(CollateralError):
        reg.mark_settled(1.0)  # nothing pending

    returned = reg.release("pair-1")
    # release returns original minus pro-rata slashes: 95% of each contribution
    assert returned["ck_researcher"] == pytest.approx(57_000.0)
    assert returned["ck_trader"] == pytest.approx(38_000.0)
    reg.close("pair-1")
    assert bond.state is BondState.CLOSED


def test_escrow_shortfall_covers_bonds_and_unsettled_slashes():
    reg = BondRegistry()
    reg.post(make_bond(state=BondState.PENDING))
    reg.activate("pair-1", 100_000.0)
    reg.slash("pair-1", 500.0, "W1")
    # escrow must hold remaining bond (95k) + unburned slash (5k)
    assert reg.escrow_shortfall(100_000.0) == 0.0
    assert reg.escrow_shortfall(98_000.0) == pytest.approx(2_000.0)


def test_effective_sell_through_is_the_r11_lever():
    # bonding the cohort earning 40% of miner emissions cuts sigma 40%
    assert emissions.effective_sell_through(0.5, 0.4) == pytest.approx(0.3)
    params = ChainParams()
    with_bonds = emissions.retention_with_bonds(params, 365.0, 0.5, 0.4)
    without = emissions.issuance_retention(params, 365.0, 0.5)
    assert with_bonds > without
    with pytest.raises(ValueError):
        emissions.effective_sell_through(0.5, 1.5)


def test_post_burn_supply_lowers_dilution_hurdle():
    params = ChainParams()
    base = emissions.dilution_hurdle(params, 10_000_000.0)
    burned = emissions.dilution_hurdle(
        params, emissions.post_burn_supply(10_000_000.0, 500_000.0))
    assert burned > base  # smaller supply → higher hurdle per remaining alpha
    with pytest.raises(ValueError):
        emissions.post_burn_supply(1_000.0, 2_000.0)


def test_collateral_alerts():
    reg = BondRegistry()
    reg.post(make_bond(state=BondState.PENDING))
    reg.activate("pair-1", 100_000.0)
    reg.slash("pair-1", 500.0, "W1")

    assert alerts.from_collateral(reg, escrow_staked_alpha=100_000.0) == []
    short = alerts.from_collateral(reg, escrow_staked_alpha=90_000.0)
    assert [a.severity for a in short] == ["page"]
    stuck = alerts.from_collateral(reg, 100_000.0, tempos_oldest_pending=12)
    assert [a.source for a in stuck] == ["collateral.settlement"]
