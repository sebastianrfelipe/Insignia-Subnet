"""Python SDK backend for SubnetManager / WalletManager.

When `btcli` is not in PATH (e.g. in constrained agent environments), the
managers fall back to this module, which drives the same operations through
the `bittensor` Python SDK directly (`bt.subtensor`, `bt.wallet`).

The backend is a drop-in companion to the btcli code paths in
`subnet_manager.py` and `wallet_manager.py` — it mirrors the operations
those managers need and returns the same shapes (`Optional[int]` netuids,
`bool` success flags, `Dict[str, float]` balances). Each call wraps
exceptions so a chain-side failure degrades to a logged warning + `None` /
`False`, matching the existing btcli fallback behaviour.

The bittensor SDK API surface drifts between versions; calls here target
the 10.x line (the version pinned in the agent environment per the
2026-07-02 deployer report: `bittensor 10.5.0`). Where a method name
differs across versions we try the modern name first and fall back to the
legacy alias.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .config import EmulatorConfig, NetworkTarget

logger = logging.getLogger("subtensor_sdk")


class PySdkBackend:
    """Drive subnet + wallet operations through `bittensor` directly.

    Instantiate lazily on first use (the managers construct this only when
    `btcli` is not available) so the import cost is paid only by code paths
    that actually need it.
    """

    def __init__(self, config: EmulatorConfig):
        self.config = config
        self._subtensor = None
        self._wallets: Dict[str, Any] = {}
        self._connect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Connect to the configured subtensor endpoint."""
        try:
            import bittensor as bt  # type: ignore
        except ImportError as e:
            logger.warning(
                "bittensor SDK not installed (%s); SDK backend unavailable. "
                "Install with: pip install bittensor",
                e,
            )
            return

        try:
            if self.config.network == NetworkTarget.TESTNET:
                self._subtensor = bt.subtensor(network="test")
            else:
                self._subtensor = bt.subtensor(chain_endpoint=self.config.endpoint)
            logger.info(
                "SDK backend connected to %s at block %s",
                self.config.endpoint,
                getattr(self._subtensor, "block", "?"),
            )
        except Exception as e:
            logger.warning("SDK backend failed to connect to %s: %s",
                           self.config.endpoint, e)
            self._subtensor = None

    @property
    def is_online(self) -> bool:
        return self._subtensor is not None

    def _wallet(self, name: str, hotkey: str = "default"):
        """Return a (cached) `bt.wallet` handle."""
        key = f"{name}:{hotkey}"
        if key not in self._wallets:
            try:
                import bittensor as bt  # type: ignore
                self._wallets[key] = bt.wallet(name=name, hotkey=hotkey)
            except Exception as e:
                logger.warning("bt.wallet(%s/%s) failed: %s", name, hotkey, e)
                self._wallets[key] = None
        return self._wallets[key]

    # ------------------------------------------------------------------
    # Subnet lifecycle
    # ------------------------------------------------------------------

    def create_subnet(self) -> Optional[int]:
        """Register a new subnet. Returns the netuid or None."""
        if not self.is_online:
            return None
        wallet = self._wallet(
            self.config.wallets.owner_coldkey,
            self.config.wallets.owner_hotkey,
        )
        if wallet is None:
            return None
        try:
            # bittensor >= 9.x: sub.subnet_create(wallet=..., ...)
            # Some versions return a netuid, others return a success bool
            # and require a separate query for the netuid.
            try:
                result = self._subtensor.subnet_create(wallet=wallet)
            except AttributeError:
                result = self._subtensor.create_subnet(wallet=wallet)

            netuid = self._extract_netuid(result)
            if netuid is not None:
                logger.info("SDK: subnet created, netuid=%d", netuid)
                return netuid
            # Fallback: scan for an owned subnet if create returned a bool.
            return self.find_owned_subnet()
        except Exception as e:
            logger.warning("SDK: subnet_create failed: %s", e)
            return None

    def register_neuron(self, wallet_name: str, hotkey: str = "default") -> bool:
        """Register a hotkey on the configured subnet (burns TAO on testnet)."""
        netuid = self.config.netuid
        if not self.is_online or netuid is None:
            return False
        wallet = self._wallet(wallet_name, hotkey)
        if wallet is None:
            return False
        try:
            # burned_register is the canonical API; some versions expose
            # `register` as an alias.
            try:
                ok = self._subtensor.burned_register(wallet=wallet, netuid=netuid)
            except AttributeError:
                ok = self._subtensor.register(wallet=wallet, netuid=netuid)
            logger.info("SDK: register %s/%s on netuid=%d -> %s",
                        wallet_name, hotkey, netuid, ok)
            return bool(ok)
        except Exception as e:
            logger.warning("SDK: register %s/%s failed: %s", wallet_name, hotkey, e)
            return False

    def set_hyperparameter(self, name: str, value: str) -> bool:
        """Set a single subnet hyperparameter on-chain."""
        netuid = self.config.netuid
        if not self.is_online or netuid is None:
            return False
        wallet = self._wallet(
            self.config.wallets.owner_coldkey,
            self.config.wallets.owner_hotkey,
        )
        if wallet is None:
            return False
        try:
            # Coerce numeric strings back to int for the SDK call.
            try:
                coerced: Any = int(value)
            except ValueError:
                coerced = value
            ok = self._subtensor.set_hyperparameter(
                netuid=netuid,
                parameter=name,
                value=coerced,
                wallet=wallet,
            )
            return bool(ok)
        except Exception as e:
            logger.warning("SDK: set_hyperparameter %s=%s failed: %s", name, value, e)
            return False

    def get_subnet_info(self) -> Optional[Dict[str, Any]]:
        netuid = self.config.netuid
        if not self.is_online or netuid is None:
            return None
        try:
            mg = self._subtensor.metagraph(netuid=netuid)
            return {
                "netuid": netuid,
                "n": int(getattr(mg, "n", 0)),
                "block": int(getattr(mg, "block", 0)),
                "total_stake": float(getattr(mg, "total_stake", 0.0)),
            }
        except Exception as e:
            logger.warning("SDK: get_subnet_info failed: %s", e)
            return None

    def get_metagraph(self) -> Optional[Any]:
        netuid = self.config.netuid
        if not self.is_online or netuid is None:
            return None
        try:
            return self._subtensor.metagraph(netuid=netuid)
        except Exception as e:
            logger.warning("SDK: get_metagraph failed: %s", e)
            return None

    def find_owned_subnet(self) -> Optional[int]:
        """Scan the subnet list for one owned by our owner wallet."""
        if not self.is_online:
            return None
        owner_ss58 = self.get_address(
            self.config.wallets.owner_coldkey,
            self.config.wallets.owner_hotkey,
        )
        if owner_ss58 is None:
            return None
        try:
            # get_all_subnets() returns a list of SubnetInfo-like objects;
            # the attribute name for the owner key varies by version.
            try:
                subnets = self._subtensor.get_all_subnets()
            except AttributeError:
                subnets = self._subtensor.subnets()
            for sn in subnets:
                owner_attr = getattr(sn, "owner", None) or getattr(
                    sn, "owner_coldkey", None
                )
                if owner_attr is None:
                    continue
                owner_str = str(owner_attr)
                if owner_ss58 in owner_str or owner_str in owner_ss58:
                    netuid = int(getattr(sn, "netuid", -1))
                    if netuid >= 0:
                        return netuid
        except Exception as e:
            logger.warning("SDK: find_owned_subnet failed: %s", e)
        return None

    def start_emissions(self) -> bool:
        """Emissions on public testnet are managed by the network; local only."""
        if self.config.network != NetworkTarget.LOCAL:
            return True
        # Local-chain emission start is btcli-only in the current code; the
        # SDK path would call subtensor.serve_axon or similar, which is out
        # of scope for this backend. Return False so callers know to use
        # btcli for local emission starts.
        logger.warning(
            "SDK: start_emissions not supported on local chain via SDK; "
            "use btcli for local emission starts."
        )
        return False

    def stake_validator(
        self, wallet_name: str, hotkey: str = "default", amount: float = 1000.0
    ) -> bool:
        if not self.is_online:
            return False
        wallet = self._wallet(wallet_name, hotkey)
        if wallet is None:
            return False
        try:
            ok = self._subtensor.add_stake(wallet=wallet, amount_staked=amount)
            return bool(ok)
        except Exception as e:
            logger.warning("SDK: add_stake failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Wallet operations
    # ------------------------------------------------------------------

    def create_wallet(
        self, coldkey: str, hotkey: str, role: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Create a wallet if it doesn't exist; return its info dict."""
        wallet = self._wallet(coldkey, hotkey)
        if wallet is None:
            return None
        try:
            wallet.create_if_non_existent()
        except Exception as e:
            logger.warning("SDK: wallet create %s/%s failed: %s", coldkey, hotkey, e)
            return None
        ss58 = self.get_address(coldkey, hotkey)
        return {
            "coldkey_name": coldkey,
            "hotkey_name": hotkey,
            "ss58_address": ss58,
            "balance": 0.0,
            "role": role,
        }

    def get_address(self, coldkey: str, hotkey: str) -> Optional[str]:
        wallet = self._wallet(coldkey, hotkey)
        if wallet is None:
            return None
        try:
            hk = getattr(wallet, "hotkey", None)
            if hk is not None and getattr(hk, "ss58_address", None):
                return str(hk.ss58_address)
            ck = getattr(wallet, "coldkeypub", None)
            if ck is not None and getattr(ck, "ss58_address", None):
                return str(ck.ss58_address)
        except Exception as e:
            logger.warning("SDK: get_address %s/%s failed: %s", coldkey, hotkey, e)
        return None

    def get_balance(self, coldkey: str) -> float:
        wallet = self._wallet(coldkey, "default")
        if wallet is None or not self.is_online:
            return 0.0
        try:
            ck = getattr(wallet, "coldkeypub", None)
            if ck is None:
                return 0.0
            bal = self._subtensor.get_balance(ck)
            return float(bal) if bal is not None else 0.0
        except Exception as e:
            logger.warning("SDK: get_balance %s failed: %s", coldkey, e)
            return 0.0

    def transfer(
        self, from_wallet: str, to_address: str, amount: float
    ) -> bool:
        if not self.is_online:
            return False
        wallet = self._wallet(from_wallet, "default")
        if wallet is None:
            return False
        try:
            ok = self._subtensor.transfer(
                wallet=wallet,
                dest=to_address,
                amount=amount,
            )
            return bool(ok)
        except Exception as e:
            logger.warning("SDK: transfer %s->%s failed: %s", from_wallet, to_address, e)
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_netuid(result: Any) -> Optional[int]:
        """Best-effort netuid extraction from a subnet_create return value."""
        if result is None:
            return None
        # Some versions return a tuple (success, netuid).
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            try:
                return int(result[1])
            except (TypeError, ValueError):
                return None
        if isinstance(result, bool):
            return None
        if isinstance(result, int):
            return result
        netuid = getattr(result, "netuid", None)
        if netuid is not None:
            try:
                return int(netuid)
            except (TypeError, ValueError):
                pass
        return None
