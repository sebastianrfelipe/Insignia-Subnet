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


def effective_sell_through(base_sell_through: float, bonded_fraction: float) -> float:
    """Miner sell-through σ after the deployment-collateral lever (R11):
    bonded alpha cannot be sold while its pair is deployed, so the fraction of
    miner emissions accruing to bonded deployed pairs is removed from the
    sellable base. `bonded_fraction` = share of miner-cut emissions earned by
    pairs with ACTIVE bonds (SPEC §0.5; INCENTIVE_MECHANISM §Deployment
    Collateral)."""
    if not 0.0 <= bonded_fraction <= 1.0:
        raise ValueError("bonded_fraction must be in [0, 1]")
    return base_sell_through * (1.0 - bonded_fraction)


def retention_with_bonds(params: ChainParams, subnet_age_days: float,
                         base_sell_through: float, bonded_fraction: float) -> float:
    """issuance_retention with the collateral lever applied — the number the
    monthly factsheet reports next to raw retention (SPEC §8)."""
    return issuance_retention(
        params, subnet_age_days,
        effective_sell_through(base_sell_through, bonded_fraction))


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
