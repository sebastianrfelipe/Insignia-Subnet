"""Live alert rules tied to chain data (SPEC §7) — the pager layer.

Inputs come from lockmgr.monitor sweeps (RPC: get_coldkey_lock,
get_hotkey_conviction, get_most_convicted_hotkey_on_subnet) and treasury
market state (taostats API). Sinks are pluggable; default is logging, with a
webhook stub for the pager integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable

from lockmgr.monitor import MonitorFinding
from lockmgr.schedules import LpLock, redemption_exposure
from treasury.collateral import BondRegistry
from treasury.policy import (
    CircuitBreakers,
    MarketState,
    NavBand,
    premium_discount,
    tripped_breakers,
)

log = logging.getLogger("insignia.risk.alerts")


@dataclass(frozen=True)
class Alert:
    severity: str        # "page" | "warn" | "info"
    source: str
    message: str


def from_monitor(findings: Iterable[MonitorFinding]) -> list[Alert]:
    return [Alert(f.severity, f"monitor.{f.kind}", f.detail) for f in findings]


def from_market(state: MarketState, nav_per_alpha: float,
                band: NavBand = NavBand(),
                breakers: CircuitBreakers = CircuitBreakers()) -> list[Alert]:
    alerts = [Alert("page", "treasury.breaker", msg) for msg in tripped_breakers(state, breakers)]
    disc = premium_discount(state.spot_price, nav_per_alpha)
    if disc > band.issue_above - 1.0:
        alerts.append(Alert("warn", "treasury.band",
                            f"spot at {disc:+.1%} to NAV — above band; buying halted, "
                            "consider OTC issuance"))
    elif disc < band.buy_below - 1.0:
        alerts.append(Alert("info", "treasury.band",
                            f"spot at {disc:+.1%} to NAV — accretive buy zone"))
    return alerts


def from_cohorts(locks: list[LpLock], cap: float = 0.25,
                 window_days: float = 60.0) -> list[Alert]:
    share, start, _ = redemption_exposure(locks, window_days)
    if share > cap:
        return [Alert("page", "lockmgr.cohorts",
                      f"{share:.1%} of locked supply shares the {window_days:.0f}-day "
                      f"redemption window from day {start:.0f} (cap {cap:.0%}) — M6 breach")]
    if share > cap * 0.8:
        return [Alert("warn", "lockmgr.cohorts",
                      f"redemption window at {share:.1%}, approaching the {cap:.0%} cap")]
    return []


def from_collateral(registry: BondRegistry, escrow_staked_alpha: float,
                    tempos_oldest_pending: int = 0,
                    max_tempos_pending: int = 8) -> list[Alert]:
    """Deployment-collateral invariants (SPEC §5; INCENTIVE_MECHANISM
    §Deployment Collateral): escrow must cover bonds + unsettled slashes, and
    the burn queue must drain — one add_stake_burn per tempo means a queue
    aging past a few tempos indicates a stuck pipeline, not a big slash."""
    alerts = []
    shortfall = registry.escrow_shortfall(escrow_staked_alpha)
    if shortfall > 0:
        alerts.append(Alert("page", "collateral.escrow",
                            f"escrow coldkey short {shortfall:,.2f} alpha vs bond ledger — "
                            "custody breach, halt deployments"))
    if registry.pending_burn_alpha > 0 and tempos_oldest_pending > max_tempos_pending:
        alerts.append(Alert("warn", "collateral.settlement",
                            f"{registry.pending_burn_alpha:,.2f} slashed alpha unburned for "
                            f"{tempos_oldest_pending} tempos (max {max_tempos_pending}) — "
                            "settlement pipeline stuck or slippage-budget-bound"))
    return alerts


def from_native_collateral(findings: Iterable[MonitorFinding]) -> list[Alert]:
    """Native registration-collateral findings (docs/COLLATERAL.md)."""
    return [Alert(f.severity, f"monitor.{f.kind}", f.detail) for f in findings]


Sink = Callable[[Alert], None]


def _log_sink(alert: Alert) -> None:
    level = {"page": logging.CRITICAL, "warn": logging.WARNING}.get(alert.severity, logging.INFO)
    log.log(level, "[%s] %s", alert.source, alert.message)


@dataclass
class Dispatcher:
    sinks: list[Sink] = field(default_factory=lambda: [_log_sink])
    _seen: set[tuple[str, str]] = field(default_factory=set)

    def dispatch(self, alerts: Iterable[Alert]) -> list[Alert]:
        """De-duplicated fan-out to all sinks; returns what was actually sent."""
        sent = []
        for alert in alerts:
            key = (alert.source, alert.message)
            if key in self._seen:
                continue
            self._seen.add(key)
            for sink in self.sinks:
                sink(alert)
            sent.append(alert)
        return sent


def webhook_sink(url: str) -> Sink:
    """Pager webhook stub — wire to the ops pager before M5 game-day."""

    def _send(alert: Alert) -> None:
        import json
        import urllib.request

        payload = json.dumps(
            {"severity": alert.severity, "source": alert.source, "message": alert.message}
        ).encode()
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)

    return _send
