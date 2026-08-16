"""Emission-share, root-proportion, retention, and dilution math (SPEC §0.7–0.12).

The June-2026 regime: emission share follows the EMA of spot price (NOT flow),
young subnets pay alpha stakers far below the headline 41%, and issuance
retention inside the wrapper is the first-order NAV lever.
"""

from __future__ import annotations

from chainio import ChainParams


def root_proportion(params: ChainParams, cumulative_alpha_issuance: float) -> float:
    """Slice of the validator half reserved for root TAO stakers:
    root_tao·w / (root_tao·w + alpha_issuance). Large while the subnet is young —
    the year-1 headwind (SPEC §0.10)."""
    weighted_root = params.root_tao * params.tao_weight
    return weighted_root / (weighted_root + cumulative_alpha_issuance)


def cumulative_issuance(params: ChainParams, subnet_age_days: float) -> float:
    """Total alpha issuance (alpha_out + alpha_in) driving the root-proportion
    ramp. Ignores halvings and the 21M cap — fine below ~4 years of age."""
    return subnet_age_days * params.alpha_issuance_per_day


def alpha_staker_share(params: ChainParams, subnet_age_days: float) -> float:
    """Fraction of alpha_out actually paid to alpha stakers at a given age:
    validator_cut × (1 − root_proportion). ≈12.7% at 1 month → 34.6% at 1 year,
    against the 41% headline. Paid only while Σ subnet EMA prices > 1.0."""
    rp = root_proportion(params, cumulative_issuance(params, subnet_age_days))
    return params.validator_cut * (1.0 - rp)


def ema_responsiveness(params: ChainParams, blocks_since_start: float) -> float:
    """SubnetMovingPrice EMA responsiveness ramp (SPEC §0.8):
    base_alpha × b/(b + 201,600). 20% at day 7, 50% at day 28, 76% at day 90,
    93% at day 365 — sustained buy-flow cannot buy emission share quickly."""
    return blocks_since_start / (blocks_since_start + params.ema_horizon_blocks)


def emission_share(moving_price: float, owner_burn: float,
                   competitors: list[tuple[float, float]]) -> float:
    """share_i = p_i(1−b_i) / Σ_j p_j(1−b_j), with p capped at 1.0 (SPEC §0.7).

    `owner_burn` (b_i) is the share of last tempo's miner incentive directed to
    owner hotkeys — it taxes emission share one-for-one, which is why miner
    incentive must NEVER route to owner hotkeys (SPEC §0.11).
    """
    own = min(moving_price, 1.0) * (1.0 - owner_burn)
    total = own + sum(min(p, 1.0) * (1.0 - b) for p, b in competitors)
    return own / total if total > 0 else 0.0


def dilution_hurdle(params: ChainParams, circulating_supply: float) -> float:
    """Annual trading return required to hold NAV per alpha flat against
    issuance: 7,200×365/supply. 263% at 1M alpha, 26.3% at 10M (SPEC §0.5)."""
    return params.alpha_out_per_day * 365.0 / circulating_supply


def issuance_retention(params: ChainParams, subnet_age_days: float,
                       miner_sell_through: float) -> float:
    """Share of alpha_out retained inside the wrapper:
    owner cut (back to fund) + staker share (to LPs) + unsold miner alpha.
    93.6% at 0% sell-through, 52.6% at 100% (1-year-old subnet)."""
    staker = alpha_staker_share(params, subnet_age_days)
    return params.owner_cut + staker + params.miner_cut * (1.0 - miner_sell_through)


def leakage_drag(params: ChainParams, circulating_supply: float,
                 miner_sell_through: float) -> float:
    """Annual drag on LP return from issuance leaking OUT of the wrapper —
    the v6 replacement for the retired revenue-yield model (fig 3):
    drag = dilution × [validator_cut × root_proportion + miner_cut × sell_through].
    The owner cut accrues to NAV; staked LPs recapture their own share."""
    dilution = params.alpha_out_per_day * 365.0 / circulating_supply
    rp = root_proportion(params, circulating_supply)
    leak = params.validator_cut * rp + params.miner_cut * miner_sell_through
    return dilution * leak


def effective_sell_through(base_sell_through: float, bonded_fraction: float,
                           native_locked_fraction: float = 0.0) -> float:
    """Miner sell-through σ after both collateral levers (R11):

    - `bonded_fraction` — share of miner-cut emissions earned by pairs with
      ACTIVE deployment bonds (cannot sell while deployed).
    - `native_locked_fraction` — share of miner-held alpha sitting in native
      registration collateral (cannot unstake; recovered only by earning).
      Disjoint from deployment escrow: native locks cannot be transfer_stake'd.

    Combined unsellable share is additive and capped at 1. SPEC §0.5;
    docs/COLLATERAL.md.
    """
    if not 0.0 <= bonded_fraction <= 1.0:
        raise ValueError("bonded_fraction must be in [0, 1]")
    if not 0.0 <= native_locked_fraction <= 1.0:
        raise ValueError("native_locked_fraction must be in [0, 1]")
    unsellable = min(1.0, bonded_fraction + native_locked_fraction)
    return base_sell_through * (1.0 - unsellable)


def retention_with_bonds(params: ChainParams, subnet_age_days: float,
                         base_sell_through: float, bonded_fraction: float,
                         native_locked_fraction: float = 0.0) -> float:
    """issuance_retention with both collateral levers applied — the number the
    monthly factsheet reports next to raw retention (SPEC §8)."""
    return issuance_retention(
        params, subnet_age_days,
        effective_sell_through(base_sell_through, bonded_fraction,
                               native_locked_fraction))


def post_burn_supply(circulating_supply: float, cumulative_burned: float) -> float:
    """Circulating supply net of slash-settlement burns. Feed this into
    dilution_hurdle / leakage_drag — burns permanently lower the issuance
    hurdle's denominator base (SPEC §5 slash-settlement pipeline)."""
    if cumulative_burned < 0 or cumulative_burned > circulating_supply:
        raise ValueError("cumulative_burned must be in [0, circulating_supply]")
    return circulating_supply - cumulative_burned


def lp_daily_yield(params: ChainParams, subnet_age_days: float,
                   total_staked_alpha: float) -> float:
    """Daily staking yield per unit of staked LP alpha (alpha terms). This is
    DILUTION RECAPTURE, not profit — it offsets the 7,200/day issuance."""
    share = alpha_staker_share(params, subnet_age_days)
    return params.alpha_out_per_day * share / total_staked_alpha


# --- Root Reborn / root-basket structure (ROOTFUND spec §2–§6, SPEC §0.16) ---


def deferred_root_slice(params: ChainParams, subnet_age_days: float) -> float:
    """The root slice of alpha_out — validator_cut × root_proportion — which
    post-Root-Reborn accrues as staked alpha in beta-basket escrows instead of
    auto-selling. Still LEAKED for retention accounting (economically owned by
    root stakers); realization timing is claim flow, not a per-block drain.
    Reported separately in the factsheet (SPEC §8)."""
    rp = root_proportion(params, cumulative_issuance(params, subnet_age_days))
    return params.validator_cut * rp


def root_base_yield(params: ChainParams) -> float:
    """Annual root staking yield before take: 983 τ/day × 365 / root_tao ≈ 6.7%.
    'The starting point, not the story' — v441 release notes."""
    return params.root_dividend_per_day * 365.0 / params.root_tao


def sleeve_yield(params: ChainParams, w_ins: float, alpha_price_return: float,
                 take: float | None = None) -> float:
    """Effective annual yield to a root-sleeve staker on the Insignia validator:
    (1 − t) × y_root × (1 + w_ins × g_α). The transmission of desk performance
    into LP yield — principal stays plain TAO (ROOTFUND spec §2)."""
    t = params.root_validator_take if take is None else take
    return (1.0 - t) * root_base_yield(params) * (1.0 + w_ins * alpha_price_return)


def dividend_bid(params: ChainParams, sleeve_tao: float, w_ins: float) -> float:
    """τ/day of structural Insignia buy-flow from the fund's own validator:
    w_ins × 983 × K_v / τ_root. Dividends are pro-rata to delegated stake, so
    the bid scales with product demand — the v3 answer to miner sell that the
    v2 buyback could not give (ROOTFUND spec §2, §4)."""
    return w_ins * params.root_dividend_per_day * sleeve_tao / params.root_tao


def external_basket_bid(params: ChainParams, stake_weighted_w_ins: float) -> float:
    """τ/day of follower flow from all OTHER validators' baskets, using the
    stake-weighted mean weight toward Insignia (chainio.stake_weighted_insignia_weight).
    Default (curation disabled / unset) ≈ the subnet's emission share."""
    return stake_weighted_w_ins * params.root_dividend_per_day


def maintenance_flow(params: ChainParams, miner_sell_through: float,
                     alpha_price: float) -> float:
    """τ/day of buy-flow needed to hold price flat against emission sell
    pressure: 7,200 × σ × p (SYSTEM_EQUATIONS §8)."""
    return params.alpha_out_per_day * miner_sell_through * alpha_price


def breakeven_sell_through(params: ChainParams, total_bid_tao_day: float,
                           alpha_price: float) -> float:
    """σ* — the miner sell-through fully absorbed by a given τ/day of standing
    bid: σ* = B / (7,200 × p). At a 500k τ sleeve, w=0.9, p=0.02: ≈ 0.57."""
    return total_bid_tao_day / (params.alpha_out_per_day * alpha_price)


def escrow_steady_state(bid_tao_day: float, alpha_price: float,
                        annual_claim_rate: float) -> float:
    """Steady-state beta-basket escrow alpha on our subnet, from
    dE/dt = F/p − (c/365)·E with F in τ/day and c annualized:
    E* = F × 365 / (p × c) (ROOTFUND spec §5). At F=82.3 τ/day, p=0.02, c=0.5:
    ≈ 3.0M α — the claim-flow overhang (R16) and the conviction-inert share of
    SubnetAlphaOut."""
    if annual_claim_rate <= 0:
        raise ValueError("annual_claim_rate must be positive")
    if alpha_price <= 0:
        return 0.0
    return bid_tao_day * 365.0 / (alpha_price * annual_claim_rate)


def staker_yield_with_escrow(params: ChainParams, subnet_age_days: float,
                             staked_alpha: float, lp_alpha_diverted: float,
                             escrow_alpha: float) -> float:
    """Annual per-unit staking yield with the v3 staked base: LP alpha out of
    the denominator, basket escrow in (it is real stake earning every epoch).
    The miner's mechanical gain from separation (ROOTFUND spec §4)."""
    base = staked_alpha - lp_alpha_diverted + escrow_alpha
    if base <= 0:
        raise ValueError("staked base must be positive after diversion")
    share = alpha_staker_share(params, subnet_age_days)
    return params.alpha_out_per_day * share * 365.0 / base


def staked_lp_drag(params: ChainParams, total_staked_alpha: float,
                   subnet_age_days: float, miner_sell_through: float) -> float:
    """Annual drag on a STAKED LP position in the v1 wrapper — the corrected
    dilution number (ROOTFUND spec §0.5): the owner cut returns via treasury
    NAV backing and the staker tranche is recaptured, so only the root slice
    and miner sell-through leak:
    drag = (7,200 × 365 / S) × validator_cut × (rp + σ).
    1.4–10.4 pp/yr at S=12M vs the 21.9 pp unstaked hurdle — the rebuttal to
    'LPs get diluted'."""
    dilution = params.alpha_out_per_day * 365.0 / total_staked_alpha
    rp = root_proportion(params, cumulative_issuance(params, subnet_age_days))
    return dilution * params.validator_cut * (rp + miner_sell_through)
