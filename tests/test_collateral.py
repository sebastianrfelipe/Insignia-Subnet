"""Deployment collateral: bond lifecycle, pro-rata slashing, retention lever
(SPEC §0.5, §5; INCENTIVE_MECHANISM §Deployment Collateral)."""

import pytest

from chainio import ChainParams
from risk import alerts
from treasury import emissions
from treasury.collateral import (
    RESEARCHER_DIAGNOSTICS,
    Bond,
    BondRegistry,
    BondState,
    CollateralError,
    LossAttribution,
    blame_split,
    degradation,
    required_bond_alpha,
    slash_for_window,
)


def make_bond(researcher=60_000.0, trader=40_000.0, capital=10_000.0,
              state=BondState.ACTIVE) -> Bond:
    return Bond(pair_id="pair-1",
                contributions={"ck_researcher": researcher, "ck_trader": trader},
                deployed_capital_tao=capital, state=state)


def attribution(researcher: float, trader: float) -> LossAttribution:
    return LossAttribution("ck_researcher", "ck_trader", researcher, trader)


def test_bond_sizing_at_spot():
    # 10k TAO deployed at 10% ratio, alpha at 0.05 TAO → 20k alpha bond
    assert required_bond_alpha(10_000.0, 0.05) == pytest.approx(20_000.0)
    with pytest.raises(ValueError):
        required_bond_alpha(10_000.0, 0.0)


def test_fully_attributed_loss_lands_only_on_the_degraded_role():
    bond = make_bond()
    # The trader's diagnostics fully explain the loss; the researcher performed
    # as validated and must not pay for a partner it never chose.
    result = slash_for_window(bond, realized_loss_tao=500.0, window_id="W1",
                              attribution=attribution(researcher=0.0, trader=1.0))
    assert result.slash_alpha == pytest.approx(5_000.0)   # fully explained: no discount
    assert result.per_coldkey["ck_trader"] == pytest.approx(5_000.0)
    assert result.per_coldkey["ck_researcher"] == pytest.approx(0.0)


def test_partial_degradation_splits_by_blame_not_bond_size():
    bond = make_bond()   # researcher posted 60%, trader 40%
    result = slash_for_window(bond, 500.0, "W1",
                              attribution=attribution(researcher=0.2, trader=0.6))
    # explained = 0.8 of the loss, split 1:3 by degradation; 0.2 ambiguous at
    # 50% exposure = 0.1 shared pro-rata. Total slashable = 0.9 of 5,000.
    assert result.blame.explained_fraction == pytest.approx(0.8)
    assert result.slash_alpha == pytest.approx(4_500.0)
    # explained 4,000 splits 1,000/3,000; ambiguous 500 splits 300/200 by bond
    assert result.per_coldkey["ck_researcher"] == pytest.approx(1_300.0)
    assert result.per_coldkey["ck_trader"] == pytest.approx(3_200.0)
    # trader carries the larger share despite posting the smaller bond
    assert sum(result.per_coldkey.values()) == pytest.approx(result.slash_alpha)


def test_unexplained_loss_is_shared_but_discounted():
    bond = make_bond()
    result = slash_for_window(bond, 500.0, "W1",
                              attribution=attribution(researcher=0.0, trader=0.0))
    # Nothing is explained, so only the ambiguous exposure (50%) is slashed at
    # all — punishment scales with the strength of the justification.
    assert result.blame.explained_fraction == 0.0
    assert result.slash_alpha == pytest.approx(2_500.0)
    assert result.per_coldkey["ck_researcher"] == pytest.approx(1_500.0)
    assert result.per_coldkey["ck_trader"] == pytest.approx(1_000.0)


def test_missing_diagnostics_reduce_the_slash_rather_than_assigning_blame():
    bond = make_bond()
    result = slash_for_window(bond, 500.0, "W1", attribution=None)
    assert result.slash_alpha == pytest.approx(2_500.0)   # same as fully unexplained
    assert result.blame.ambiguous_fraction == 1.0


def test_attribution_must_name_bond_contributors():
    bond = make_bond()
    stranger = LossAttribution("ck_someone_else", "ck_trader", 0.5, 0.5)
    with pytest.raises(CollateralError, match="not a bond contributor"):
        slash_for_window(bond, 500.0, "W1", attribution=stranger)


def test_degradation_from_breakdowns():
    validated = {"overfitting_penalty": 0.8, "penalized_f1": 0.6, "variance": 0.5}
    live = {"overfitting_penalty": 0.4, "penalized_f1": 0.3, "variance": 0.5}
    # relative drops of 50%, 50%, 0% → mean 1/3
    assert degradation(validated, live, RESEARCHER_DIAGNOSTICS) == pytest.approx(1 / 3)
    # an absent diagnostic explains nothing; it must never read as fault
    assert degradation({}, {}, RESEARCHER_DIAGNOSTICS) == 0.0
    # improvement is not negative degradation
    assert degradation({"variance": 0.4}, {"variance": 0.9}, ("variance",)) == 0.0


@pytest.mark.parametrize("d_r,d_t", [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.2, 0.6),
                                     (1.0, 1.0), (0.5, 0.5), (0.05, 0.0)])
@pytest.mark.parametrize("loss", [1.0, 500.0, 6_000.0])
def test_per_coldkey_always_sums_to_the_slash(d_r, d_t, loss):
    """The split must never create or destroy slashed alpha — the burn queue is
    fed from slash_alpha while miners are debited from per_coldkey."""
    result = slash_for_window(make_bond(), loss, "W1", attribution(d_r, d_t))
    assert sum(result.per_coldkey.values()) == pytest.approx(result.slash_alpha)
    assert all(v >= 0 for v in result.per_coldkey.values())


def test_blame_split_never_exceeds_the_whole_loss():
    bond = make_bond()
    split = blame_split(bond, attribution(researcher=1.0, trader=1.0))
    assert split.explained_fraction == 1.0      # capped, not 2.0
    assert split.slashable_fraction == 1.0
    assert sum(split.attributed.values()) == pytest.approx(1.0)


def test_window_cap_binds_on_large_loss():
    bond = make_bond()
    # 60% loss fully explained would take 60% of the bond; the 25% cap binds
    result = slash_for_window(bond, realized_loss_tao=6_000.0, window_id="W1",
                              attribution=attribution(0.5, 0.5))
    assert result.slash_alpha == pytest.approx(25_000.0)
    assert result.capped


def test_gains_never_slash_or_restore():
    bond = make_bond()
    slash_for_window(bond, 500.0, "W1", attribution(0.5, 0.5))
    before = bond.remaining_alpha
    result = slash_for_window(bond, realized_loss_tao=-2_000.0, window_id="W2",
                              attribution=attribution(0.5, 0.5))
    assert result.slash_alpha == 0.0
    assert bond.remaining_alpha == before  # the bond only ratchets down


def test_slash_exhausts_at_remaining():
    bond = make_bond()
    for window in range(5):
        slash_for_window(bond, 6_000.0, f"W{window}", attribution(0.5, 0.5))
    assert bond.remaining_alpha == pytest.approx(0.0)
    assert slash_for_window(bond, 6_000.0, "W9", attribution(0.5, 0.5)).slash_alpha == 0.0


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

    reg.slash("pair-1", 500.0, "W1", attribution(researcher=0.5, trader=0.5))
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
    reg.slash("pair-1", 500.0, "W1", attribution(researcher=0.5, trader=0.5))
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
    reg.slash("pair-1", 500.0, "W1", attribution(researcher=0.5, trader=0.5))

    assert alerts.from_collateral(reg, escrow_staked_alpha=100_000.0) == []
    short = alerts.from_collateral(reg, escrow_staked_alpha=90_000.0)
    assert [a.severity for a in short] == ["page"]
    stuck = alerts.from_collateral(reg, 100_000.0, tempos_oldest_pending=12)
    assert [a.source for a in stuck] == ["collateral.settlement"]
