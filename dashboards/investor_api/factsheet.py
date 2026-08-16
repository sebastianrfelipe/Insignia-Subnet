"""Monthly factsheet assembly (SPEC §8).

Headline metric: premium/discount to NAV per alpha. Everything else supports
it: issuance retention, realized staker APY vs the dilution hurdle, emission
share trend, conviction table, redeemable-supply schedule, reserve coverage,
buy-flow executed vs revenue attested. `publish()` is investor-facing and sits
behind the Phase-0 gate; `build_factsheet()`/`render_markdown()` are internal.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from chainio import ChainParams
from lockmgr.schedules import LpLock, redeemable_supply_curve, redemption_exposure
from otc.compliance import require_legal_signoff
from treasury import emissions
from treasury.accounting import NavReport
from treasury.policy import RESERVE_TARGET_MONTHS


@dataclass(frozen=True)
class Factsheet:
    period: str                          # "2026-07"
    nav: NavReport
    premium_discount: float              # headline
    issuance_retention: float
    miner_sell_through: float
    staker_apy_alpha: float              # realized, alpha terms
    staker_apy_usd: float                # realized, USD terms (separate, never blended)
    dilution_hurdle: float
    emission_share: float
    emission_share_trend_wow: float
    net_tao_flow: float
    conviction_owner: float
    conviction_top_external: float
    redeemable_curve: list[tuple[float, float]]   # next 24 months
    worst_redemption_window_share: float
    reserve_coverage_months: float
    buy_flow_executed_tao: float
    revenue_attested_tao: float
    toggle_delays_disclosed: list[str]   # §10.1(c): any desk delay + reason
    native_collateral_locked: float = 0.0
    native_lock_share: float = 0.0
    deployment_bonded_alpha: float = 0.0
    cumulative_burned_alpha: float = 0.0


def build_factsheet(period: str, params: ChainParams, nav: NavReport,
                    subnet_age_days: float, miner_sell_through: float,
                    staker_apy_alpha: float, staker_apy_usd: float,
                    emission_share: float, emission_share_trend_wow: float,
                    net_tao_flow: float, conviction_owner: float,
                    conviction_top_external: float, locks: list[LpLock],
                    reserve_coverage_months: float, buy_flow_executed_tao: float,
                    revenue_attested_tao: float,
                    toggle_delays_disclosed: list[str] | None = None,
                    native_collateral_locked: float = 0.0,
                    native_lock_share: float = 0.0,
                    deployment_bonded_alpha: float = 0.0,
                    cumulative_burned_alpha: float = 0.0) -> Factsheet:
    worst_window, _, _ = redemption_exposure(locks)
    return Factsheet(
        period=period,
        nav=nav,
        premium_discount=nav.premium_discount,
        issuance_retention=emissions.issuance_retention(params, subnet_age_days, miner_sell_through),
        miner_sell_through=miner_sell_through,
        staker_apy_alpha=staker_apy_alpha,
        staker_apy_usd=staker_apy_usd,
        dilution_hurdle=emissions.dilution_hurdle(params, nav.circulating_alpha),
        emission_share=emission_share,
        emission_share_trend_wow=emission_share_trend_wow,
        net_tao_flow=net_tao_flow,
        conviction_owner=conviction_owner,
        conviction_top_external=conviction_top_external,
        redeemable_curve=redeemable_supply_curve(locks, params, horizon_days=730.0, step_days=30.0),
        worst_redemption_window_share=worst_window,
        reserve_coverage_months=reserve_coverage_months,
        buy_flow_executed_tao=buy_flow_executed_tao,
        revenue_attested_tao=revenue_attested_tao,
        toggle_delays_disclosed=toggle_delays_disclosed or [],
        native_collateral_locked=native_collateral_locked,
        native_lock_share=native_lock_share,
        deployment_bonded_alpha=deployment_bonded_alpha,
        cumulative_burned_alpha=cumulative_burned_alpha,
    )


def render_markdown(fs: Factsheet) -> str:
    defense = (fs.conviction_owner / fs.conviction_top_external
               if fs.conviction_top_external > 0 else float("inf"))
    lines = [
        f"# Insignia factsheet — {fs.period}",
        "",
        f"## Premium/discount to NAV: {fs.premium_discount:+.1%}",
        "",
        f"NAV per alpha (depth-adjusted): {fs.nav.nav_per_alpha:.6f} TAO · "
        f"spot: {fs.nav.spot_price:.6f} TAO · "
        f"spot-mark overstatement of treasury alpha: {-fs.nav.depth_haircut:.1%}",
        "",
        "| Metric | Value | Note |",
        "|---|---|---|",
        f"| Issuance retention | {fs.issuance_retention:.1%} | miner sell-through {fs.miner_sell_through:.0%} |",
        f"| Staker APY (alpha terms) | {fs.staker_apy_alpha:.1%} | dilution recapture, not profit |",
        f"| Staker APY (USD terms) | {fs.staker_apy_usd:.1%} | separate basis — do not blend |",
        f"| Dilution hurdle | {fs.dilution_hurdle:.1%}/yr | NAV-flat trading return required |",
        f"| Emission share | {fs.emission_share:.2%} | {fs.emission_share_trend_wow:+.1%} WoW |",
        f"| Net TAO flow | {fs.net_tao_flow:+,.0f} τ | |",
        f"| Conviction defense | {defense:.1f}× | owner vs top external hotkey |",
        f"| Worst 60-day redemption window | {fs.worst_redemption_window_share:.1%} | cap 25% |",
        f"| Reserve coverage | {fs.reserve_coverage_months:.1f} months | target ≥ {RESERVE_TARGET_MONTHS:.0f} |",
        f"| Buy-flow vs revenue | {fs.buy_flow_executed_tao:,.0f} / {fs.revenue_attested_tao:,.0f} τ | executed / attested |",
        f"| Native registration collateral | {fs.native_collateral_locked:,.0f} α | lock_share {fs.native_lock_share:.0%} |",
        f"| Deployment bonds / burned | {fs.deployment_bonded_alpha:,.0f} / {fs.cumulative_burned_alpha:,.0f} α | active escrow / slash-settlement burns |",
        "",
    ]
    if fs.toggle_delays_disclosed:
        lines += ["## Toggle delays exercised this period", ""]
        lines += [f"- {reason}" for reason in fs.toggle_delays_disclosed]
        lines.append("")
    lines += ["## Redeemable supply, next 24 months", "",
              "| Day | Redeemable alpha |", "|---|---|"]
    lines += [f"| {int(day)} | {amount:,.0f} |" for day, amount in fs.redeemable_curve]
    return "\n".join(lines)


def publish(fs: Factsheet, out_dir: Path) -> Path:
    """Investor-facing publication — Phase-0 gated (SPEC §2)."""
    require_legal_signoff()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"factsheet_{fs.period}.md"
    out.write_text(render_markdown(fs), encoding="utf-8")
    return out
