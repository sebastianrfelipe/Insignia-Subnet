"""Chain parameters, pool snapshots, and the providers that fetch them live.

Defaults reflect mainnet as of 2026-07 (SPEC §0.15) and exist for simulation and
tests only. Production paths MUST go through a live provider each epoch — every
one of these values is mutable by root (SPEC §0.16).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Protocol


@dataclass(frozen=True)
class ChainParams:
    """Protocol parameters the fund layer depends on.

    Rates named *_blocks are exponential time constants in blocks, matching the
    chain's roll-forward `exp(-dt/Rate)` convention — NOT half-lives. The
    ~60-day figure quoted in docs is `unlock_tau_days * ln(2)` ≈ 62.4 days.
    """

    unlock_rate_blocks: float = 648_000.0          # lock decay τ (UnlockRate)
    conviction_maturity_blocks: float = 648_000.0  # non-owner conviction τ (ConvictionMaturityRate)
    block_time_s: float = 12.0
    tempo_blocks: int = 360

    owner_cut: float = 0.18
    miner_cut: float = 0.41
    validator_cut: float = 0.41

    tao_weight: float = 0.18                       # root TAO weight in root_proportion
    alpha_out_per_day: float = 7_200.0             # participant emissions (1 alpha/block pre-halving)
    alpha_issuance_per_day: float = 14_400.0       # alpha_out + alpha_in, drives root_proportion ramp
    ema_horizon_blocks: float = 201_600.0          # SubnetMovingPrice EMA anti-manipulation horizon
    fee_rate: float = 33 / 65_535                  # per-subnet pool FeeRate, input side

    root_tao: float = 5_374_582.0                  # TAO staked on root (live value; refresh)

    # Root Reborn (runtime v441, SPEC §0.16 / ROOTFUND spec §2)
    root_dividend_per_day: float = 983.0           # τ/day root dividend, network-wide
    root_validator_take: float = 0.18              # default take on root staker yields

    @property
    def blocks_per_day(self) -> float:
        return 86_400.0 / self.block_time_s

    @property
    def unlock_tau_days(self) -> float:
        return self.unlock_rate_blocks / self.blocks_per_day

    @property
    def maturity_tau_days(self) -> float:
        return self.conviction_maturity_blocks / self.blocks_per_day

    @property
    def lock_half_life_days(self) -> float:
        return self.unlock_tau_days * math.log(2)


@dataclass(frozen=True)
class PoolSnapshot:
    """One reading of a subnet's balancer weighted pool."""

    tao_reserve: float
    alpha_reserve: float
    w_tao: float = 0.5      # w1; weights bounded [0.01, 0.99], shifted by emission injections
    w_alpha: float = 0.5    # w2
    fee_rate: float = 33 / 65_535
    block: int | None = None

    @property
    def spot_price(self) -> float:
        """TAO per alpha: p = (w1 · TAO_res) / (w2 · alpha_res)."""
        return (self.w_tao * self.tao_reserve) / (self.w_alpha * self.alpha_reserve)


def reference_pool() -> PoolSnapshot:
    """Finney SN4 snapshot used across the v6 model, SPEC §0.14, and tests."""
    return PoolSnapshot(tao_reserve=131_662.0, alpha_reserve=2_431_632.87)


@dataclass(frozen=True)
class ValidatorBasket:
    """One root validator's beta-basket state as seen from our subnet
    (Root Reborn, runtime v441; SPEC §0.16, ROOTFUND spec §2–§5).

    `weights` is the normalized `Weights[ROOT]` vector keyed by netuid (uid 0 =
    the held-TAO stability slot). `escrow_alpha` is the alpha this validator's
    fund holds on OUR netuid via the keyless pallet escrow — real stake, counted
    in SubnetAlphaOut, conviction-inert.
    """

    hotkey: str
    root_stake_tao: float                  # K_v — delegated root stake
    weights: dict[int, float]              # netuid → normalized basket weight
    escrow_alpha: float = 0.0              # basket alpha held on our netuid
    nav_tao: float | None = None           # realizable fund NAV, if queryable

    def weight_to(self, netuid: int) -> float:
        return self.weights.get(netuid, 0.0)


def stake_weighted_insignia_weight(baskets: list[ValidatorBasket], netuid: int) -> float:
    """w̄_ins — stake-weighted mean basket weight toward our subnet, the
    multiplier on the network dividend that becomes structural bid
    (ROOTFUND spec §2: F_ins = w̄ · 983 τ/day scaled by stake share)."""
    total = sum(b.root_stake_tao for b in baskets)
    if total <= 0:
        return 0.0
    return sum(b.weight_to(netuid) * b.root_stake_tao for b in baskets) / total


def total_escrow_alpha(baskets: list[ValidatorBasket]) -> float:
    """Aggregate beta-basket escrow alpha on our netuid — the claim-flow
    overhang (R16) and the conviction-inert share of SubnetAlphaOut."""
    return sum(b.escrow_alpha for b in baskets)


class ParamsProvider(Protocol):
    """Live read layer. Implementations must re-read every epoch and never cache
    across a tempo boundary — parameter changes must surface within one epoch
    (lockmgr.monitor alerts on deltas)."""

    def chain_params(self) -> ChainParams: ...

    def pool(self, netuid: int) -> PoolSnapshot: ...

    def root_baskets(self, netuid: int) -> list[ValidatorBasket]: ...


class StaticProvider:
    """Fixed values for tests and simulation."""

    def __init__(self, params: ChainParams | None = None, pool: PoolSnapshot | None = None,
                 baskets: list[ValidatorBasket] | None = None):
        self._params = params or ChainParams()
        self._pool = pool or reference_pool()
        self._baskets = baskets or []

    def chain_params(self) -> ChainParams:
        return self._params

    def pool(self, netuid: int = 0) -> PoolSnapshot:
        return self._pool

    def root_baskets(self, netuid: int = 0) -> list[ValidatorBasket]:
        return self._baskets

    def with_params(self, **changes) -> "StaticProvider":
        return StaticProvider(replace(self._params, **changes), self._pool, self._baskets)


class SubtensorProvider:
    """Direct-RPC provider via the bittensor SDK (lazy import).

    M2 note: conviction v2 storage (UnlockRate, ConvictionMaturityRate, lock
    maps) may not yet be surfaced by the SDK — fall back to raw substrate
    storage queries via `subtensor.substrate.query` and verify names on testnet
    against subtensor PRs #2658/#2687/#2696 before trusting this path.
    """

    def __init__(self, network: str = "finney"):
        import bittensor  # deferred: not needed for math/tests

        self._st = bittensor.subtensor(network=network)

    def chain_params(self) -> ChainParams:
        base = ChainParams()
        substrate = self._st.substrate
        overrides: dict[str, float] = {}
        for field, pallet_item in [
            ("unlock_rate_blocks", "UnlockRate"),
            ("conviction_maturity_blocks", "ConvictionMaturityRate"),
        ]:
            try:
                value = substrate.query("SubtensorModule", pallet_item)
                if value is not None:
                    overrides[field] = float(value.value)
            except Exception:
                # Storage item absent (conviction v2 not deployed on this
                # network) — keep the documented default and let monitor flag it.
                continue
        return replace(base, **overrides) if overrides else base

    def pool(self, netuid: int) -> PoolSnapshot:
        info = self._st.subnet(netuid)
        return PoolSnapshot(
            tao_reserve=float(info.tao_in.tao),
            alpha_reserve=float(info.alpha_in.tao),
            block=self._st.get_current_block(),
        )

    def root_baskets(self, netuid: int) -> list[ValidatorBasket]:
        """Beta-basket state per root validator (Root Reborn, v441).

        Uses the betaBasket runtime API where exposed by the SDK; falls back to
        raw runtime calls. Names must be verified on testnet against subtensor
        PR #2968 before trusting this path — a missing API returns [], which
        lockmgr.monitor treats as "no escrow visibility" (warn), never as
        "no escrow".
        """
        substrate = self._st.substrate
        baskets: list[ValidatorBasket] = []
        try:
            neurons = self._st.neurons_lite(netuid=0)
        except Exception:
            return baskets
        for n in neurons:
            hotkey = n.hotkey
            try:
                weights_raw = substrate.runtime_call(
                    "BetaBasketApi", "get_validator_weights", [hotkey]).value
                weights = {int(u): float(w) for u, w in (weights_raw or [])}
                total = sum(weights.values())
                if total > 0:
                    weights = {u: w / total for u, w in weights.items()}
                basket_raw = substrate.runtime_call(
                    "BetaBasketApi", "get_validator_basket", [hotkey]).value or []
                escrow = sum(float(a) for u, a in basket_raw if int(u) == netuid)
                nav = substrate.runtime_call(
                    "BetaBasketApi", "get_validator_nav", [hotkey]).value
                baskets.append(ValidatorBasket(
                    hotkey=hotkey,
                    root_stake_tao=float(n.stake),
                    weights=weights,
                    escrow_alpha=escrow / 1e9,  # rao → alpha; verify scaling on testnet
                    nav_tao=float(nav) / 1e9 if nav is not None else None,
                ))
            except Exception:
                # API absent on this runtime — surface partial state; monitor
                # flags missing visibility rather than assuming zero escrow.
                continue
        return baskets


class TaostatsProvider:
    """Pool/price reads via the taostats API (https://taostats.io/).

    Used by risk/alerts and dashboards where an RPC connection is unavailable.
    Requires an API key; endpoint shapes must be verified against current
    taostats API docs before production use.
    """

    BASE_URL = "https://api.taostats.io/api"

    def __init__(self, api_key: str):
        import requests  # deferred

        self._session = requests.Session()
        self._session.headers["Authorization"] = api_key

    def chain_params(self) -> ChainParams:
        # taostats does not expose conviction v2 params; use RPC for those.
        return ChainParams()

    def root_baskets(self, netuid: int) -> list[ValidatorBasket]:
        # taostats does not expose beta-basket state; use the RPC provider.
        return []

    def pool(self, netuid: int) -> PoolSnapshot:
        resp = self._session.get(f"{self.BASE_URL}/dtao/pool/latest/v1", params={"netuid": netuid}, timeout=30)
        resp.raise_for_status()
        data = resp.json()["data"][0]
        return PoolSnapshot(
            tao_reserve=float(data["total_tao"]),
            alpha_reserve=float(data["total_alpha"]),
            block=int(data["block_number"]),
        )
