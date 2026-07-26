"""Wrappers for the conviction v2 lock extrinsics (SPEC §4).

M2 gate: every call here must be exercised end-to-end on testnet before mainnet
use — extrinsic and storage names come from subtensor PRs #2658/#2687/#2696 and
may drift before mainnet deployment. The SDK may not expose them yet, in which
case calls are composed raw via the substrate interface.

Key architecture (SPEC §10.1): prefer LP-held coldkeys granting the desk a
limited proxy over `lock_stake` / `set_perpetual_lock` ONLY — no transfer or
unstake authority. Verify on testnet which proxy type actually gates the lock
extrinsics; if none does, fall back to fund-custodied cohort coldkeys under
multisig and flag the custody change to Phase-0 counsel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class LockCallError(RuntimeError):
    pass


@dataclass(frozen=True)
class OnChainLock:
    """Decoded result of a get_coldkey_lock query."""

    coldkey: str
    hotkey: str
    netuid: int
    locked_mass: float
    conviction: float
    perpetual: bool


class LockClient:
    """Thin extrinsic wrapper. Chain access is injected so the lifecycle logic
    (otc.settlement, lockmgr.monitor) stays testable against a fake."""

    def __init__(self, subtensor: Any, wallet: Any, netuid: int):
        self._st = subtensor
        self._wallet = wallet
        self._netuid = netuid

    def _compose(self, call_function: str, call_params: dict) -> Any:
        substrate = self._st.substrate
        call = substrate.compose_call(
            call_module="SubtensorModule",
            call_function=call_function,
            call_params=call_params,
        )
        extrinsic = substrate.create_signed_extrinsic(call=call, keypair=self._wallet.coldkey)
        receipt = substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
        if not receipt.is_success:
            raise LockCallError(f"{call_function} failed: {receipt.error_message}")
        return receipt

    def lock_stake(self, hotkey: str, amount_alpha: float) -> Any:
        """Creates a DECAYING lock by default — must be followed by
        set_perpetual_lock(True) in the same session (SPEC §0.3)."""
        return self._compose(
            "lock_stake",
            {"hotkey": hotkey, "netuid": self._netuid, "amount_locked": _to_rao(amount_alpha)},
        )

    def set_perpetual_lock(self, perpetual: bool) -> Any:
        return self._compose(
            "set_perpetual_lock", {"netuid": self._netuid, "perpetual": perpetual}
        )

    def get_coldkey_lock(self, coldkey: str) -> OnChainLock | None:
        substrate = self._st.substrate
        raw = substrate.query("SubtensorModule", "ColdkeyLock", [coldkey, self._netuid])
        if raw is None or raw.value is None:
            return None
        v = raw.value
        return OnChainLock(
            coldkey=coldkey,
            hotkey=v["hotkey"],
            netuid=self._netuid,
            locked_mass=_from_rao(v["locked_amount"]),
            conviction=_from_rao(v.get("conviction", 0)),
            perpetual=bool(v.get("perpetual", False)),
        )

    def get_hotkey_conviction(self, hotkey: str) -> float:
        raw = self._st.substrate.query(
            "SubtensorModule", "HotkeyConviction", [hotkey, self._netuid]
        )
        return _from_rao(raw.value) if raw is not None and raw.value is not None else 0.0

    def get_most_convicted_hotkey(self) -> tuple[str, float] | None:
        raw = self._st.substrate.query(
            "SubtensorModule", "MostConvictedHotkey", [self._netuid]
        )
        if raw is None or raw.value is None:
            return None
        return raw.value["hotkey"], _from_rao(raw.value["conviction"])


def _to_rao(alpha: float) -> int:
    return int(round(alpha * 1e9))


def _from_rao(rao: int | float) -> float:
    return float(rao) / 1e9
