"""Monte Carlo of the flywheel under stress (SPEC §7) — published quarterly to LPs.

The loop being simulated is the known failure mode: revenue stalls → buy-flow
stops → EMA price decays → emission share falls → yield falls → LPs toggle to
decay → releasing supply into thin liquidity depresses price further. A path is
in a SPIRAL state when emission share is below threshold AND the redeemable
supply overhang exceeds what the pool can absorb inside the slippage tolerance.

This is a scenario model, not a forecast: monthly steps, stylized price
formation (NAV anchor + flow impact via real pool math), conservative refill
assumptions. Parameters are explicit so the quarterly report can table them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from chainio import ChainParams, PoolSnapshot, reference_pool
from treasury import emissions, pool_math
from treasury.policy import BandAction, NavBand, RoutingPolicy


@dataclass(frozen=True)
class RevenueShock:
    start_month: int
    duration_months: int
    severity: float          # 0.5 = −50% revenue, 1.0 = −100%


@dataclass(frozen=True)
class ScenarioConfig:
    months: int = 24
    n_paths: int = 2_000
    seed: int = 7

    # fund + wrapper state at t0
    params: ChainParams = field(default_factory=ChainParams)
    pool: PoolSnapshot = field(default_factory=reference_pool)
    circulating_alpha: float = 6.0e6
    locked_alpha: float = 3.0e6
    subnet_age_days: float = 365.0
    trading_aum_tao: float = 250_000.0
    monthly_revenue_tao: float = 4_000.0
    monthly_desk_return_mu: float = 0.015     # on AUM, lognormal-ish
    monthly_desk_return_sigma: float = 0.05

    # shocks (SPEC §7 scenario axes)
    revenue_shock: RevenueShock | None = RevenueShock(6, 6, 1.0)
    tao_drawdown_corr: float = 0.6            # extra spot pressure during the shock
    competitor_growth_monthly: float = 0.02   # share erosion from rival subnets

    # LP behaviour
    toggle_prob_base: float = 0.02            # monthly, per unit of still-perpetual mass
    toggle_prob_stressed: float = 0.30        # when discount is deep or yield collapsed
    stress_discount: float = -0.30            # premium_discount below this = stressed
    redemption_sell_frac: float = 0.5         # share of newly redeemable alpha sold monthly
    hedge_relief_frac: float = 0.0            # share of stressed selling diverted to
                                              # short-side hedging (chain shorting), never
                                              # hitting the spot pool; 0 until it ships

    # spiral definition
    spiral_share_threshold: float = 0.004     # emission share floor
    absorbable_slippage: float = 0.25         # pool "depth" = size exiting within −25%

    policy_band: NavBand = field(default_factory=NavBand)
    routing: RoutingPolicy = field(default_factory=RoutingPolicy)


@dataclass
class SpiralReport:
    config: ScenarioConfig
    p_spiral: float
    p_spiral_by_month: list[float]
    median_terminal_discount: float
    p5_terminal_discount: float
    mean_terminal_share: float

    def summary(self) -> str:
        shock = self.config.revenue_shock
        shock_txt = (f"revenue −{shock.severity:.0%} months {shock.start_month}–"
                     f"{shock.start_month + shock.duration_months}" if shock else "no shock")
        text = (
            f"reflexivity MC ({self.config.n_paths} paths, {self.config.months}m, {shock_txt}):\n"
            f"  P(spiral)              {self.p_spiral:6.1%}\n"
            f"  median terminal disc.  {self.median_terminal_discount:+6.1%}\n"
            f"  p5 terminal discount   {self.p5_terminal_discount:+6.1%}\n"
            f"  mean terminal share    {self.mean_terminal_share:6.2%}"
        )
        if self.config.hedge_relief_frac > 0:
            text += f"\n  hedge relief           {self.config.hedge_relief_frac:6.1%} of stressed selling diverted"
        return text


def _absorbable_alpha(pool: PoolSnapshot, max_slippage: float) -> float:
    """Largest exit whose realised-vs-spot shortfall stays inside tolerance."""
    lo, hi = 0.0, pool.alpha_reserve * 10.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if -pool_math.exit_slippage(pool, mid) <= max_slippage:
            lo = mid
        else:
            hi = mid
    return lo


def run(config: ScenarioConfig = ScenarioConfig()) -> SpiralReport:
    if not 0.0 <= config.hedge_relief_frac <= 1.0:
        raise ValueError(f"hedge_relief_frac must be in [0, 1], got {config.hedge_relief_frac}")
    rng = np.random.default_rng(config.seed)
    p = config.params
    spiral_hits = np.zeros(config.months, dtype=float)
    spiraled = np.zeros(config.n_paths, dtype=bool)
    terminal_disc = np.zeros(config.n_paths)
    terminal_share = np.zeros(config.n_paths)

    for path in range(config.n_paths):
        pool = config.pool
        aum = config.trading_aum_tao
        perpetual = config.locked_alpha
        redeemable_overhang = 0.0
        competitor_index = 1.0
        ema_price = pool.spot_price
        age_days = config.subnet_age_days

        for m in range(config.months):
            age_days += 30.0
            shocked = (config.revenue_shock is not None
                       and config.revenue_shock.start_month <= m
                       < config.revenue_shock.start_month + config.revenue_shock.duration_months)
            rev_mult = (1.0 - config.revenue_shock.severity) if shocked else 1.0
            revenue = config.monthly_revenue_tao * rev_mult * math.exp(rng.normal(0.0, 0.25))

            desk_ret = rng.normal(config.monthly_desk_return_mu * rev_mult,
                                  config.monthly_desk_return_sigma)
            aum = max(aum * (1.0 + desk_ret) + revenue, 0.0)

            treasury_alpha_value = pool_math.quote_unstake(pool, 0.0)  # treasury alpha omitted
            nav = (aum + treasury_alpha_value) / config.circulating_alpha
            disc = pool.spot_price / nav - 1.0 if nav > 0 else 0.0

            # policy: buy the discount only
            action = config.policy_band.action(pool.spot_price, nav)
            buy_tao = config.routing.route(revenue, action)["buy_flow"]
            if buy_tao > 0:
                bought = pool_math.quote_add_stake(pool, buy_tao)
                pool = replace(pool, tao_reserve=pool.tao_reserve + buy_tao,
                               alpha_reserve=pool.alpha_reserve - bought)

            # LP toggles: stressed when the discount is deep or yield collapsed
            yield_ann = emissions.lp_daily_yield(p, age_days, config.locked_alpha) * 365.0
            stressed = disc < config.stress_discount or yield_ann < emissions.dilution_hurdle(
                p, config.circulating_alpha) * 0.25
            toggle_p = config.toggle_prob_stressed if stressed else config.toggle_prob_base
            toggled = perpetual * toggle_p * rng.uniform(0.5, 1.5)
            perpetual = max(perpetual - toggled, 0.0)

            # decay releases ~1 − e^(−30/τ) of toggled mass per month into the overhang
            monthly_release = 1.0 - math.exp(-30.0 / p.unlock_tau_days)
            redeemable_overhang += toggled * monthly_release
            sold = redeemable_overhang * config.redemption_sell_frac
            spot_sold = sold * (1.0 - config.hedge_relief_frac)
            redeemable_overhang -= sold
            if spot_sold > 0:
                tao_out = pool_math.quote_unstake(pool, spot_sold)
                extra = 0.10 * config.tao_drawdown_corr if shocked else 0.0
                pool = replace(pool,
                               tao_reserve=(pool.tao_reserve - tao_out) * (1.0 - extra),
                               alpha_reserve=pool.alpha_reserve + spot_sold)

            # EMA responds to time-at-price on the anti-manipulation ramp
            resp = emissions.ema_responsiveness(p, age_days * p.blocks_per_day)
            ema_price += resp * 0.5 * (pool.spot_price - ema_price)
            competitor_index *= 1.0 + config.competitor_growth_monthly
            share = emissions.emission_share(ema_price, 0.0, [(competitor_index, 0.0)])

            depth = _absorbable_alpha(pool, config.absorbable_slippage)
            if share < config.spiral_share_threshold and redeemable_overhang > depth:
                if not spiraled[path]:
                    spiraled[path] = True
                spiral_hits[m] += 1

        terminal_disc[path] = disc
        terminal_share[path] = share

    return SpiralReport(
        config=config,
        p_spiral=float(spiraled.mean()),
        p_spiral_by_month=list(spiral_hits / config.n_paths),
        median_terminal_discount=float(np.median(terminal_disc)),
        p5_terminal_discount=float(np.percentile(terminal_disc, 5)),
        mean_terminal_share=float(terminal_share.mean()),
    )


STANDARD_SCENARIOS: dict[str, RevenueShock | None] = {
    "baseline": None,
    "rev_-50_3m": RevenueShock(6, 3, 0.5),
    "rev_-50_6m": RevenueShock(6, 6, 0.5),
    "rev_-100_3m": RevenueShock(6, 3, 1.0),
    "rev_-100_6m": RevenueShock(6, 6, 1.0),
    "rev_-100_12m": RevenueShock(6, 12, 1.0),
}


def quarterly_report(base: ScenarioConfig = ScenarioConfig()) -> str:
    """The scenario grid published to LPs each quarter (SPEC §7)."""
    lines = []
    for name, shock in STANDARD_SCENARIOS.items():
        report = run(replace(base, revenue_shock=shock))
        lines.append(f"[{name}]\n{report.summary()}\n")
    return "\n".join(lines)


if __name__ == "__main__":
    print(quarterly_report())
