"""Regenerates docs/investor/*.png — port of the v6 quantitative model (SPEC §8).

All curves are computed by the same modules the fund actually runs
(lockmgr.schedules, treasury.pool_math, treasury.emissions), so the investor
charts can never drift from the mechanism code. Chart-only presentation lives
here; no math is defined in this file.

Figures:
  v6_plot1_vesting_corrected.png  LP vesting lifecycle (conviction instant via
                                  owner hotkey; staking-yield overlay)
  v6_plot2_staged_exit.png        decay schedule as slippage control (staged vs cliff)
  v6_plot3_leakage_drag.png       LP return drag from issuance leakage
                                  (replaces the retired revenue-yield model)
  conviction_mechanics.png        owner vs non-owner conviction; decaying default
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chainio import ChainParams, PoolSnapshot, reference_pool
from lockmgr import schedules
from treasury import emissions, pool_math

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "investor"

# Validated categorical palette (dataviz reference instance, light mode).
BLUE = "#2a78d6"      # slot 1 — conviction / primary series
ORANGE = "#eb6834"    # slot 2 — locked mass
AQUA = "#1baf7a"      # slot 3 — redeemable / staged exit
VIOLET = "#4a3aa7"    # slot 7 — staking-yield overlay
RED = "#e34948"       # slot 8 — cliff-exit comparator
ORDINAL_BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]  # ordered supply levels
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#eceae6"
SURFACE = "#fcfcfb"
NEUTRAL = "#9c9a94"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": SURFACE, "axes.edgecolor": "#d8d6d1",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 11,
    "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})


def fig_vesting_lifecycle(params: ChainParams, out_dir: Path = OUT_DIR,
                          toggle_day: float = 365.0, subnet_age_at_lock: float = 180.0,
                          total_staked: float = 3.0e6, horizon_days: float = 730.0) -> Path:
    days = np.linspace(0, horizon_days, 900)
    mass = np.array([
        schedules.locked_mass(1.0, d - toggle_day, params) if d >= toggle_day else 1.0
        for d in days
    ])
    conviction = mass.copy()          # owner hotkey: instant, tracks locked mass
    redeemable = 1.0 - mass

    daily_yield = np.array([
        emissions.lp_daily_yield(params, subnet_age_at_lock + d, total_staked) for d in days
    ])
    cum_yield = np.concatenate([[0], np.cumsum(np.diff(days) * daily_yield[:-1])])

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.fill_between(days, 0, 100 * mass, color=ORANGE, alpha=0.13)
    ax.plot(days, 100 * mass, color=ORANGE, lw=2.6, label="Locked alpha (staked throughout)")
    ax.plot(days, 100 * conviction, color=BLUE, lw=3.4, ls=(0, (1, 1.6)),
            label="Conviction — instant via owner hotkey (tracks locked mass)")
    ax.plot(days, 100 * redeemable, color=AQUA, lw=2.6, ls="--", label="Redeemable by LP")
    ax.plot(days, 100 * cum_yield, color=VIOLET, lw=2.6,
            label="Cumulative staking yield earned (alpha terms)")
    ax.axvline(toggle_day, color=RED, lw=1.6, ls=":")
    ax.annotate("Month 12: set_perpetual_lock(false)", (toggle_day, 104), fontsize=9,
                color=RED, ha="center")
    ax.annotate("Locking to the OWNER hotkey grants\nconviction = locked mass immediately —\n"
                "no 60-day maturity ramp.",
                xy=(0.04, 0.62), xycoords="axes fraction", fontsize=9, color=BLUE,
                bbox=dict(boxstyle="round,pad=0.5", fc="#eef4fc", ec=BLUE, lw=1.2))
    ax.annotate("The 12-month term is not arbitrary:\nit spans the EMA ramp (20%→93%) and the\n"
                "root-proportion ramp (13%→35% to stakers).",
                xy=(0.42, 0.13), xycoords="axes fraction", fontsize=9, color=INK,
                bbox=dict(boxstyle="round,pad=0.5", fc="#fdf3ee", ec=ORANGE, lw=1.1))
    ax.set_xlabel("Days since LP lock")
    ax.set_ylabel("% of LP position")
    ax.set_title("LP vesting lifecycle — conviction instant, yield overlaid")
    ax.set_xlim(0, horizon_days)
    ax.set_ylim(0, 112)
    ax.legend(fontsize=8.5, loc="center left")
    fig.text(0.01, 0.012,
             f"Lock to subnet-owner hotkey; decay τ = {params.unlock_rate_blocks:,.0f} blocks "
             f"(half-life ≈ {params.lock_half_life_days:.0f} d). Yield assumes the subnet is "
             f"{subnet_age_at_lock:.0f} d old at lock, {total_staked / 1e6:.0f}M alpha staked, "
             "root_proportion ramping with issuance.\nYield is DILUTION RECAPTURE, not profit — "
             "it offsets the 7,200 alpha/day issuance rather than adding to it.",
             fontsize=7, color=NEUTRAL)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = out_dir / "v6_plot1_vesting_corrected.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_staged_exit(pool: PoolSnapshot, params: ChainParams, out_dir: Path = OUT_DIR,
                    position: float = 500_000.0, horizon_days: float = 365.0) -> Path:
    t = np.linspace(0, horizon_days, 400)
    lam = math.log(2) / params.lock_half_life_days
    redeem_alpha = position * (1 - np.exp(-lam * t))
    spot_val = redeem_alpha * pool.spot_price

    increments = np.diff(np.concatenate([[0], redeem_alpha]))
    staged = np.cumsum([pool_math.quote_unstake(pool, a) for a in increments])
    cliff = pool_math.quote_unstake(pool, position)
    spot_full = position * pool.spot_price

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.plot(t, spot_val, color=NEUTRAL, lw=2.0, ls="--", label="Spot mark of redeemed alpha")
    ax.plot(t, staged, color=AQUA, lw=2.8, label="Realised via staged exponential redemption")
    ax.axhline(cliff, color=RED, lw=2.2, ls=":",
               label=f"Realised via a single cliff exit ({100 * (cliff / spot_full - 1):+.1f}% vs spot)")
    ax.annotate("The exponential decay is not just a lockup —\n"
                "it is a slippage-control mechanism. Staged exit\n"
                "recovers materially more than a cliff.",
                xy=(0.30, 0.30), xycoords="axes fraction", fontsize=9.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.5", fc="#eefaf4", ec=AQUA, lw=1.2))
    ax.set_xlabel("Days after the perpetual toggle is released")
    ax.set_ylabel("TAO realised")
    ax.set_title("A benefit of the decay schedule — it paces the exit")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_xlim(0, horizon_days)
    fig.text(0.01, 0.012,
             f"{position:,.0f} alpha position against the finney SN4 pool snapshot "
             f"({pool.tao_reserve:,.0f} tau / {pool.alpha_reserve:,.2f} alpha). Staged path re-quotes each "
             "increment; pool reserves held static, so this UNDERSTATES\nthe benefit (in reality the pool "
             "refills between increments via emission and buy-flow). Cliff exits are contractually prohibited.",
             fontsize=7, color=NEUTRAL)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = out_dir / "v6_plot2_staged_exit.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_leakage_drag(params: ChainParams, out_dir: Path = OUT_DIR,
                     supplies: tuple[float, ...] = (3e6, 6e6, 12e6, 24e6)) -> Path:
    sell = np.linspace(0, 1, 300)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for supply, color in zip(supplies, ORDINAL_BLUES):
        drag = np.array([emissions.leakage_drag(params, supply, s) for s in sell])
        dilution = params.alpha_out_per_day * 365 / supply
        ax.plot(100 * sell, 100 * drag, lw=2.5, color=color,
                label=f"{supply / 1e6:.0f}M alpha supply (dilution {100 * dilution:.0f}%/yr)")
        ax.annotate(f"{supply / 1e6:.0f}M", (100.5, 100 * drag[-1]), fontsize=8.5,
                    color=color, va="center", annotation_clip=False)
    ax.set_xlabel("Miner sell-through (%)")
    ax.set_ylabel("Annual drag on LP return (percentage points)")
    ax.set_title("LP return = trading return − leakage drag (replaces the old yield model)")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.annotate("Revenue creates NAV, not emissions. The question is\n"
                "how much of issuance LEAKS out of the wrapper —\n"
                "a fully-staked LP recaptures the rest.",
                xy=(0.38, 0.14), xycoords="axes fraction", fontsize=9.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.5", fc="#fdf3ee", ec=ORANGE, lw=1.2))
    fig.text(0.01, 0.012,
             "drag = (7,200 × 365 / supply) × [0.41 × root_proportion + 0.41 × miner_sell_through]. "
             "Owner cut (18%) returns to the fund and accrues to NAV;\nalpha stakers recapture their share. "
             "LP return in USD ≈ desk trading return − this drag ± any move in the premium/discount to NAV.",
             fontsize=7, color=NEUTRAL)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = out_dir / "v6_plot3_leakage_drag.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_conviction_mechanics(params: ChainParams, out_dir: Path = OUT_DIR,
                             horizon_days: float = 365.0) -> Path:
    t = np.linspace(0, horizon_days, 500)
    owner = np.ones_like(t)
    nonowner = np.array([schedules.conviction_nonowner(1.0, d, params) for d in t])
    decaying = np.array([schedules.locked_mass(1.0, d, params) for d in t])

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(t, 100 * owner, color=BLUE, lw=2.8,
            label="Perpetual lock to OWNER hotkey — conviction instant (Insignia's path)")
    ax.plot(t, 100 * nonowner, color=ORANGE, lw=2.6,
            label="Perpetual lock to non-owner hotkey — 1 − e^(−t/τ) maturity")
    ax.plot(t, 100 * decaying, color=AQUA, lw=2.6, ls="--",
            label="Default decaying lock — mass without set_perpetual_lock(true)")
    half_life = params.lock_half_life_days
    ax.axvline(half_life, color=NEUTRAL, lw=1.2, ls=":")
    ax.annotate(f"half-life ≈ {half_life:.0f} d", (half_life + 6, 8), fontsize=8.5, color=INK_2)
    ax.set_xlabel("Days since lock")
    ax.set_ylabel("% of locked mass / conviction")
    ax.set_title("Conviction mechanics — why LP locks target the owner hotkey")
    ax.set_xlim(0, horizon_days)
    ax.set_ylim(0, 108)
    ax.legend(fontsize=8.5, loc="center right")
    fig.text(0.01, 0.012,
             f"τ = {params.unlock_rate_blocks:,.0f} blocks (UnlockRate) for decay; "
             f"{params.conviction_maturity_blocks:,.0f} blocks (ConvictionMaturityRate) for maturity. "
             "Both root-mutable — curves recompute from chain each epoch.",
             fontsize=7, color=NEUTRAL)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = out_dir / "conviction_mechanics.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def summary(params: ChainParams, pool: PoolSnapshot,
            subnet_age_at_lock: float = 180.0, total_staked: float = 3.0e6,
            position: float = 500_000.0) -> str:
    lines = ["Cumulative LP staking yield (alpha terms), subnet "
             f"{subnet_age_at_lock:.0f}d old at lock:"]
    for d in [90, 180, 365, 547, 730]:
        days = np.linspace(0, d, max(d, 2))
        y = np.array([emissions.lp_daily_yield(params, subnet_age_at_lock + x, total_staked)
                      for x in days])
        cum = float(np.trapezoid(y, days))
        lines.append(f"  day {d:>4}: {100 * cum:6.1f}%")

    lam = math.log(2) / params.lock_half_life_days
    t = np.linspace(0, 365, 400)
    increments = np.diff(np.concatenate([[0], position * (1 - np.exp(-lam * t))]))
    staged = sum(pool_math.quote_unstake(pool, a) for a in increments)
    cliff = pool_math.quote_unstake(pool, position)
    spot = position * pool.spot_price
    lines += [
        f"\nStaged vs cliff exit of {position:,.0f} alpha:",
        f"  spot mark      {spot:9,.0f} tau",
        f"  cliff exit     {cliff:9,.0f} tau  ({100 * (cliff / spot - 1):+.1f}%)",
        f"  staged (1yr)   {staged:9,.0f} tau  ({100 * (staged / spot - 1):+.1f}%)",
        "\nLP drag (pp/yr) at 12M supply:",
    ]
    for s in [0.0, 0.3, 0.6, 1.0]:
        lines.append(f"  miner sell {100 * s:3.0f}% -> "
                     f"{100 * emissions.leakage_drag(params, 12e6, s):5.1f} pp")
    return "\n".join(lines)


def regenerate_all(out_dir: Path = OUT_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    params = ChainParams()
    pool = reference_pool()
    return [
        fig_vesting_lifecycle(params, out_dir),
        fig_staged_exit(pool, params, out_dir),
        fig_leakage_drag(params, out_dir),
        fig_conviction_mechanics(params, out_dir),
    ]


if __name__ == "__main__":
    for path in regenerate_all():
        print(f"wrote {path}")
    print()
    print(summary(ChainParams(), reference_pool()))
