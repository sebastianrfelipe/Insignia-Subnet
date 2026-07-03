"""Offline smoke tests for the Python SDK backend and manager fallback wiring.

These tests do NOT hit a live chain — they verify:
  1. `_subtensor_sdk.PySdkBackend` imports cleanly and degrades gracefully
     when `bittensor` is not installed (returns `is_online == False`).
  2. `SubnetManager` and `WalletManager` correctly detect whether `btcli`
     is in PATH and wire to the SDK backend when it isn't.
  3. The SDK backend's no-op fallbacks return the documented shapes
     (`None` / `False` / `0.0`) when offline, so callers can rely on the
     same contracts the btcli path provides.

Live chain validation (subnet create, register, transfer) requires a
funded wallet on the public testnet and is out of scope for CI.
"""

import importlib
import unittest
from unittest import mock

from testnet._subtensor_sdk import PySdkBackend
from testnet.config import EmulatorConfig, NetworkTarget
from testnet.subnet_manager import SubnetManager
from testnet.wallet_manager import WalletManager


def _fresh_sdk_module():
    """Re-import the SDK module so `bittensor` import side-effects reset
    between tests (PySdkBackend caches the subtensor connection at construct
    time)."""
    import testnet._subtensor_sdk as mod
    importlib.reload(mod)
    return mod


class PySdkBackendOfflineTests(unittest.TestCase):
    def _config(self) -> EmulatorConfig:
        return EmulatorConfig(network=NetworkTarget.TESTNET)

    def test_imports_cleanly(self):
        """The SDK module imports without bittensor installed."""
        mod = _fresh_sdk_module()
        self.assertTrue(hasattr(mod, "PySdkBackend"))

    def test_offline_backend_reports_offline(self):
        """When the subtensor connection fails, is_online is False and
        operations return their documented offline shapes."""
        backend = PySdkBackend(self._config())
        # Whether or not bittensor is installed in CI, a connection to the
        # live testnet is not assumed. Either path must leave the backend
        # in a known state.
        if backend.is_online:
            # If we happen to be online (e.g. dev machine with bittensor
            # and network access), skip the offline-shape assertions.
            self.skipTest("Live testnet connection available; offline path not exercised.")
        self.assertIsNone(backend.create_subnet())
        self.assertFalse(backend.register_neuron("insignia-owner", "default"))
        self.assertFalse(backend.set_hyperparameter("tempo", "360"))
        self.assertIsNone(backend.get_subnet_info())
        self.assertIsNone(backend.get_metagraph())
        self.assertIsNone(backend.find_owned_subnet())
        self.assertEqual(backend.get_balance("insignia-owner"), 0.0)
        self.assertFalse(backend.transfer("alice", "5Gxh...", 100.0))

    def test_extract_netuid_handles_shapes(self):
        """_extract_netuid tolerates the version-varying return shapes."""
        extract = PySdkBackend._extract_netuid
        self.assertIsNone(extract(None))
        self.assertIsNone(extract(True))
        self.assertIsNone(extract(False))
        self.assertEqual(extract(13), 13)
        self.assertEqual(extract((True, 13)), 13)
        self.assertEqual(extract([False, 7]), 7)

        class _FakeSN:
            netuid = 42

        self.assertEqual(extract(_FakeSN()), 42)


class ManagerBtcliDetectionTests(unittest.TestCase):
    """Managers must pick the SDK backend when btcli is not in PATH."""

    def test_subnet_manager_uses_sdk_when_btcli_missing(self):
        cfg = EmulatorConfig(network=NetworkTarget.TESTNET)
        with mock.patch("testnet.subnet_manager.shutil.which", return_value=None):
            mgr = SubnetManager(cfg)
        self.assertFalse(mgr._btcli_available)
        # `sdk` property must yield a PySdkBackend without raising.
        # NOTE: this constructs a real PySdkBackend which may attempt a
        # network connection; we only assert the type, not the connection.
        self.assertIsInstance(mgr.sdk, PySdkBackend)

    def test_subnet_manager_uses_btcli_when_available(self):
        cfg = EmulatorConfig(network=NetworkTarget.TESTNET)
        with mock.patch(
            "testnet.subnet_manager.shutil.which", return_value="/usr/bin/btcli"
        ):
            mgr = SubnetManager(cfg)
        self.assertTrue(mgr._btcli_available)

    def test_wallet_manager_uses_sdk_when_btcli_missing(self):
        cfg = EmulatorConfig(network=NetworkTarget.TESTNET)
        with mock.patch("testnet.wallet_manager.shutil.which", return_value=None):
            mgr = WalletManager(cfg)
        self.assertFalse(mgr._btcli_available)
        self.assertIsInstance(mgr.sdk, PySdkBackend)

    def test_wallet_manager_uses_btcli_when_available(self):
        cfg = EmulatorConfig(network=NetworkTarget.TESTNET)
        with mock.patch(
            "testnet.wallet_manager.shutil.which", return_value="/usr/bin/btcli"
        ):
            mgr = WalletManager(cfg)
        self.assertTrue(mgr._btcli_available)


class ManagerSdkFallbackDispatchTests(unittest.TestCase):
    """When btcli is absent, public methods must dispatch to the SDK backend
    rather than shelling out to a missing binary."""

    def _make_subnet_mgr_with_mock_sdk(self):
        cfg = EmulatorConfig(network=NetworkTarget.TESTNET, netuid=13)
        with mock.patch("testnet.subnet_manager.shutil.which", return_value=None):
            mgr = SubnetManager(cfg)
        # Replace the lazy SDK property with a mock so no live connection
        # is attempted.
        mgr._sdk = mock.MagicMock(spec=PySdkBackend)
        return mgr

    def _make_wallet_mgr_with_mock_sdk(self):
        cfg = EmulatorConfig(network=NetworkTarget.TESTNET)
        with mock.patch("testnet.wallet_manager.shutil.which", return_value=None):
            mgr = WalletManager(cfg)
        mgr._sdk = mock.MagicMock(spec=PySdkBackend)
        return mgr

    def test_subnet_manager_create_subnet_dispatches_to_sdk(self):
        mgr = self._make_subnet_mgr_with_mock_sdk()
        mgr._sdk.create_subnet.return_value = 13
        self.assertEqual(mgr.create_subnet(), 13)
        mgr._sdk.create_subnet.assert_called_once()

    def test_subnet_manager_register_neuron_dispatches_to_sdk(self):
        mgr = self._make_subnet_mgr_with_mock_sdk()
        mgr._sdk.register_neuron.return_value = True
        self.assertTrue(mgr.register_neuron("insignia-miner-0", "default"))
        mgr._sdk.register_neuron.assert_called_once_with("insignia-miner-0", "default")

    def test_subnet_manager_set_hyperparameter_dispatches_to_sdk(self):
        mgr = self._make_subnet_mgr_with_mock_sdk()
        mgr._sdk.set_hyperparameter.return_value = True
        self.assertTrue(mgr._set_hyperparameter("tempo", "360"))
        mgr._sdk.set_hyperparameter.assert_called_once_with("tempo", "360")

    def test_subnet_manager_get_subnet_info_dispatches_to_sdk(self):
        mgr = self._make_subnet_mgr_with_mock_sdk()
        mgr._sdk.get_subnet_info.return_value = {"netuid": 13, "n": 0}
        self.assertEqual(mgr.get_subnet_info(), {"netuid": 13, "n": 0})

    def test_subnet_manager_find_owned_subnet_dispatches_to_sdk(self):
        mgr = self._make_subnet_mgr_with_mock_sdk()
        mgr._sdk.find_owned_subnet.return_value = 13
        self.assertEqual(mgr._find_owned_subnet(), 13)

    def test_wallet_manager_create_wallet_dispatches_to_sdk(self):
        mgr = self._make_wallet_mgr_with_mock_sdk()
        mgr._sdk.create_wallet.return_value = {
            "coldkey_name": "insignia-owner",
            "hotkey_name": "default",
            "ss58_address": "5Gxh...",
            "balance": 0.0,
            "role": "owner",
        }
        info = mgr._create_wallet("insignia-owner", "default", role="owner")
        self.assertEqual(info.coldkey_name, "insignia-owner")
        self.assertEqual(info.ss58_address, "5Gxh...")
        mgr._sdk.create_wallet.assert_called_once()

    def test_wallet_manager_get_balance_dispatches_to_sdk(self):
        mgr = self._make_wallet_mgr_with_mock_sdk()
        mgr._sdk.get_balance.return_value = 1.0
        self.assertEqual(mgr._get_balance("insignia-owner"), 1.0)

    def test_wallet_manager_transfer_dispatches_to_sdk(self):
        mgr = self._make_wallet_mgr_with_mock_sdk()
        mgr._sdk.transfer.return_value = True
        self.assertTrue(mgr._transfer("alice", "5Gxh...", 100.0))


if __name__ == "__main__":
    unittest.main()
