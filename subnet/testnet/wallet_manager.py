"""
Wallet Manager

Automates Bittensor wallet creation and funding for testnet emulation.
Wraps btcli commands and provides a Python API for the emulator.

On a local subtensor chain, the pre-funded "alice" account is used to
seed all emulator wallets with test TAO. On the public testnet, wallets
must be funded manually via the Bittensor Discord faucet.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import EmulatorConfig, NetworkTarget

logger = logging.getLogger("wallet_manager")


@dataclass
class WalletInfo:
    coldkey_name: str
    hotkey_name: str
    ss58_address: Optional[str] = None
    balance: float = 0.0
    role: str = ""


class WalletManager:
    """
    Creates and manages Bittensor wallets for testnet emulation.

    Wallet hierarchy for the emulator:
      - 1 owner wallet (creates and configures the subnet)
      - N validator wallets (run L1/L2 validation and set weights)
      - M miner wallets (simulated honest + adversarial agents)
    """

    def __init__(self, config: EmulatorConfig):
        self.config = config
        self.wallets: Dict[str, WalletInfo] = {}
        self._network_flag = config.btcli_network_flag

    def setup_all_wallets(self) -> Dict[str, WalletInfo]:
        """Create all wallets needed for the emulator."""
        logger.info("Setting up wallets for %s network...", self.config.network.value)

        owner = self._create_wallet(
            self.config.wallets.owner_coldkey,
            self.config.wallets.owner_hotkey,
            role="owner",
        )
        self.wallets["owner"] = owner

        for i in range(self.config.wallets.n_validators):
            name = self.config.wallets.validator_coldkey(i)
            w = self._create_wallet(
                name,
                self.config.wallets.validator_hotkey,
                role="validator",
            )
            self.wallets[f"validator_{i}"] = w

        for i in range(self.config.wallets.n_miners):
            name = self.config.wallets.miner_coldkey(i)
            w = self._create_wallet(
                name,
                self.config.wallets.miner_hotkey,
                role="miner",
            )
            self.wallets[f"miner_{i}"] = w

        logger.info(
            "Created %d wallets: 1 owner, %d validators, %d miners",
            len(self.wallets),
            self.config.wallets.n_validators,
            self.config.wallets.n_miners,
        )
        return self.wallets

    def fund_from_alice(self, amount_per_wallet: float = 10000.0) -> bool:
        """
        Fund all wallets from the pre-funded alice account (local chain only).

        The local subtensor Docker image includes an "alice" account with
        a large balance. This method transfers test TAO to each emulator wallet.
        """
        if self.config.network != NetworkTarget.LOCAL:
            logger.warning(
                "Auto-funding only available on local chain. "
                "For testnet, fund wallets via the Bittensor Discord faucet."
            )
            return False

        logger.info("Funding wallets from alice (%.0f TAO each)...", amount_per_wallet)

        for key, wallet in self.wallets.items():
            if not wallet.ss58_address:
                wallet.ss58_address = self._get_address(
                    wallet.coldkey_name, wallet.hotkey_name
                )

            if wallet.ss58_address:
                ok = self._transfer(
                    from_wallet="alice",
                    to_address=wallet.ss58_address,
                    amount=amount_per_wallet,
                )
                if ok:
                    wallet.balance = amount_per_wallet
                    logger.info("  Funded %s (%s): %.0f TAO", key, wallet.coldkey_name, amount_per_wallet)
                else:
                    logger.warning("  Failed to fund %s", key)

        return True

    def check_balances(self) -> Dict[str, float]:
        """Query on-chain balances for all wallets."""
        balances = {}
        for key, wallet in self.wallets.items():
            balance = self._get_balance(wallet.coldkey_name)
            wallet.balance = balance
            balances[key] = balance
        return balances

    def get_wallet_summary(self) -> List[Dict]:
        """Return a summary of all wallets for logging/reporting."""
        return [
            {
                "key": key,
                "coldkey": w.coldkey_name,
                "hotkey": w.hotkey_name,
                "role": w.role,
                "ss58": w.ss58_address or "unknown",
                "balance": w.balance,
            }
            for key, w in self.wallets.items()
        ]

    def _create_wallet(self, coldkey: str, hotkey: str, role: str = "") -> WalletInfo:
        """Create a wallet via btcli if it doesn't exist."""
        wallet_path = Path.home() / ".bittensor" / "wallets" / coldkey
        if wallet_path.exists():
            logger.info("  Wallet '%s' already exists, skipping creation", coldkey)
        else:
            self._run_btcli([
                "wallet", "create",
                "--wallet.name", coldkey,
                "--wallet.hotkey", hotkey,
                "--no-password",
            ])

        ss58 = self._get_address(coldkey, hotkey)

        return WalletInfo(
            coldkey_name=coldkey,
            hotkey_name=hotkey,
            ss58_address=ss58,
            role=role,
        )

    def _get_address(self, coldkey: str, hotkey: str) -> Optional[str]:
        """Retrieve the SS58 address for a wallet's hotkey."""
        try:
            result = self._run_btcli([
                "wallet", "overview",
                "--wallet.name", coldkey,
                "--wallet.hotkey", hotkey,
            ], capture=True)
            if result:
                for line in result.split("\n"):
                    if "5" in line and len(line.strip()) >= 48:
                        parts = line.strip().split()
                        for part in parts:
                            if part.startswith("5") and len(part) >= 46:
                                return part
        except Exception:
            pass
        return None

    def _get_balance(self, coldkey: str) -> float:
        """Query the on-chain balance for a coldkey."""
        try:
            result = self._run_btcli([
                "wallet", "balance",
                "--wallet.name", coldkey,
                "--network", self._network_flag,
            ], capture=True)
            if result:
                for line in result.split("\n"):
                    if "τ" in line or "TAO" in line:
                        parts = line.replace("τ", "").replace("TAO", "").strip().split()
                        for part in parts:
                            try:
                                return float(part.replace(",", ""))
                            except ValueError:
                                continue
        except Exception as e:
            logger.debug("Balance query failed for %s: %s", coldkey, e)
        return 0.0

    def _transfer(
        self, from_wallet: str, to_address: str, amount: float
    ) -> bool:
        """Transfer TAO from one wallet to another address."""
        try:
            self._run_btcli([
                "wallet", "transfer",
                "--wallet.name", from_wallet,
                "--destination", to_address,
                "--amount", str(amount),
                "--network", self._network_flag,
                "--no-prompt",
            ])
            return True
        except Exception as e:
            logger.warning("Transfer failed: %s", e)
            return False

    def _run_btcli(
        self, args: List[str], capture: bool = False, timeout: int = 60
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
            logger.warning(
                "btcli not found. Install with: pip install bittensor-cli"
            )
            return None
        except subprocess.TimeoutExpired:
            logger.warning("btcli command timed out after %ds", timeout)
            return None
