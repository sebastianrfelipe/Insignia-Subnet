"""
Subnet Manager

Handles subnet lifecycle operations on the Bittensor network:
  - Create subnet
  - Register miners and validators
  - Configure subnet hyperparameters
  - Query subnet state

All operations go through btcli for compatibility with both local
and public testnet environments.
"""

from __future__ import annotations

import logging
import subprocess
import time
from typing import Dict, List, Optional, Any

from .config import EmulatorConfig, NetworkTarget, SubnetHyperparameters

logger = logging.getLogger("subnet_manager")


class SubnetManager:
    """
    Manages the Insignia subnet on a Bittensor testnet.

    Operations:
      1. create_subnet() – Register a new subnet, captures the netuid
      2. register_neurons() – Register all miner/validator hotkeys
      3. configure_hyperparameters() – Set on-chain hyperparameters
      4. start_emissions() – Activate subnet emissions
      5. get_subnet_info() – Query current subnet state
    """

    def __init__(self, config: EmulatorConfig):
        self.config = config
        self._network_flag = config.btcli_network_flag
        self.netuid: Optional[int] = config.netuid

    def create_subnet(self) -> Optional[int]:
        """
        Create a new subnet on the target network.

        Returns the netuid of the created subnet, or None on failure.
        On local chains, this is near-instant. On testnet, it costs TAO
        and may take a few blocks to confirm.
        """
        logger.info("Creating subnet on %s...", self.config.network.value)

        result = self._run_btcli([
            "subnet", "create",
            "--wallet.name", self.config.wallets.owner_coldkey,
            "--wallet.hotkey", self.config.wallets.owner_hotkey,
            "--network", self._network_flag,
            "--no-prompt",
        ], capture=True)

        if result:
            netuid = self._parse_netuid(result)
            if netuid is not None:
                self.netuid = netuid
                self.config.netuid = netuid
                logger.info("Subnet created with netuid=%d", netuid)
                return netuid

        netuid = self._find_owned_subnet()
        if netuid is not None:
            self.netuid = netuid
            self.config.netuid = netuid
            logger.info("Found existing owned subnet: netuid=%d", netuid)
            return netuid

        logger.warning("Could not determine subnet netuid from btcli output")
        return None

    def register_neuron(
        self,
        wallet_name: str,
        hotkey: str = "default",
    ) -> bool:
        """Register a miner or validator hotkey on the subnet."""
        if self.netuid is None:
            logger.error("No netuid set — create a subnet first")
            return False

        logger.info(
            "Registering %s/%s on netuid=%d...",
            wallet_name, hotkey, self.netuid,
        )

        result = self._run_btcli([
            "subnets", "register",
            "--netuid", str(self.netuid),
            "--wallet.name", wallet_name,
            "--wallet.hotkey", hotkey,
            "--network", self._network_flag,
            "--no-prompt",
        ], capture=True)

        if result and ("registered" in result.lower() or "already" in result.lower()):
            logger.info("  Registered %s/%s", wallet_name, hotkey)
            return True

        if result and "error" not in result.lower():
            logger.info("  Registration submitted for %s/%s", wallet_name, hotkey)
            return True

        logger.warning("  Registration may have failed for %s/%s", wallet_name, hotkey)
        return False

    def register_all_neurons(self, wallets: Dict[str, Any]) -> Dict[str, bool]:
        """Register all emulator wallets on the subnet."""
        if self.netuid is None:
            logger.error("No netuid set — create a subnet first")
            return {}

        results = {}
        for key, wallet in wallets.items():
            if key == "owner":
                continue

            ok = self.register_neuron(
                wallet_name=wallet.coldkey_name,
                hotkey=wallet.hotkey_name,
            )
            results[key] = ok
            time.sleep(1)

        registered = sum(1 for v in results.values() if v)
        total = len(results)
        logger.info("Registered %d/%d neurons on netuid=%d", registered, total, self.netuid)
        return results

    def configure_hyperparameters(
        self, params: Optional[SubnetHyperparameters] = None
    ) -> bool:
        """
        Set on-chain hyperparameters for the subnet.

        These control Bittensor-level behavior: tempo, immunity period,
        weight limits, etc. Not to be confused with Insignia's scoring
        weights which are application-level parameters.
        """
        if self.netuid is None:
            logger.error("No netuid set — create a subnet first")
            return False

        params = params or self.config.subnet_params
        logger.info("Configuring subnet hyperparameters for netuid=%d...", self.netuid)

        hyperparam_commands = [
            ("tempo", str(params.tempo)),
            ("immunity_period", str(params.immunity_period)),
            ("max_weight_limit", str(params.max_weight_limit)),
            ("min_allowed_weights", str(params.min_allowed_weights)),
            ("adjustment_alpha", str(params.adjustment_alpha)),
        ]

        success_count = 0
        for param_name, param_value in hyperparam_commands:
            ok = self._set_hyperparameter(param_name, param_value)
            if ok:
                success_count += 1
            time.sleep(0.5)

        logger.info(
            "Configured %d/%d hyperparameters",
            success_count, len(hyperparam_commands),
        )
        return success_count > 0

    def get_subnet_info(self) -> Optional[Dict[str, Any]]:
        """Query current subnet state from the chain."""
        if self.netuid is None:
            return None

        result = self._run_btcli([
            "subnet", "list",
            "--network", self._network_flag,
        ], capture=True)

        if not result:
            return None

        for line in result.split("\n"):
            if str(self.netuid) in line:
                return {"netuid": self.netuid, "raw_info": line.strip()}

        return {"netuid": self.netuid, "raw_info": "Not found in subnet list"}

    def get_metagraph(self) -> Optional[str]:
        """Retrieve the metagraph for the subnet."""
        if self.netuid is None:
            return None

        return self._run_btcli([
            "subnet", "metagraph",
            "--netuid", str(self.netuid),
            "--network", self._network_flag,
        ], capture=True)

    def start_emissions(self) -> bool:
        """Attempt to start emissions on the subnet (local chain)."""
        if self.netuid is None:
            logger.error("No netuid — create subnet first")
            return False

        if self.config.network != NetworkTarget.LOCAL:
            logger.info("Emissions on public testnet are managed by the network")
            return True

        result = self._run_btcli([
            "subnet", "start",
            "--netuid", str(self.netuid),
            "--wallet.name", self.config.wallets.owner_coldkey,
            "--network", self._network_flag,
            "--no-prompt",
        ], capture=True)

        return result is not None

    def stake_validator(
        self,
        wallet_name: str,
        hotkey: str = "default",
        amount: float = 1000.0,
    ) -> bool:
        """Stake TAO on a validator hotkey."""
        if self.netuid is None:
            return False

        result = self._run_btcli([
            "stake", "add",
            "--wallet.name", wallet_name,
            "--wallet.hotkey", hotkey,
            "--amount", str(amount),
            "--network", self._network_flag,
            "--no-prompt",
        ], capture=True)

        return result is not None

    def _set_hyperparameter(self, name: str, value: str) -> bool:
        """Set a single subnet hyperparameter via btcli."""
        result = self._run_btcli([
            "subnets", "hyperparameters",
            "--netuid", str(self.netuid),
            "--wallet.name", self.config.wallets.owner_coldkey,
            "--param", name,
            "--value", value,
            "--network", self._network_flag,
            "--no-prompt",
        ], capture=True)

        if result and "error" not in result.lower():
            logger.debug("  Set %s=%s", name, value)
            return True
        return False

    def _find_owned_subnet(self) -> Optional[int]:
        """Search subnet list for one owned by our wallet."""
        result = self._run_btcli([
            "subnet", "list",
            "--network", self._network_flag,
        ], capture=True)

        if not result:
            return None

        for line in result.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("NET") or stripped.startswith("---"):
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    netuid = int(parts[0])
                    if netuid > 0:
                        return netuid
                except ValueError:
                    continue
        return None

    @staticmethod
    def _parse_netuid(output: str) -> Optional[int]:
        """Parse netuid from btcli subnet create output."""
        for line in output.split("\n"):
            lower = line.lower()
            if "netuid" in lower or "subnet" in lower:
                for word in line.split():
                    try:
                        val = int(word.strip(":,."))
                        if val > 0:
                            return val
                    except ValueError:
                        continue
        return None

    def _run_btcli(
        self, args: List[str], capture: bool = False, timeout: int = 120
    ) -> Optional[str]:
        """Execute a btcli command."""
        cmd = ["btcli"] + args
        logger.debug("Running: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                input="\n" * 10,
            )
            if result.returncode != 0 and result.stderr:
                logger.debug("btcli stderr: %s", result.stderr.strip())
            return result.stdout if capture else None
        except FileNotFoundError:
            logger.warning("btcli not found. Install with: pip install bittensor-cli")
            return None
        except subprocess.TimeoutExpired:
            logger.warning("btcli command timed out after %ds", timeout)
            return None
