"""Native registration-collateral gate and martingale freeze ledger.

Run from the subnet package root:

    cd subnet && python -m pytest tests/test_native_collateral.py
"""

import unittest

import numpy as np

from insignia.native_collateral import (
    CollateralPosition,
    FreezeLedger,
    apply_collateral_gate,
    should_freeze_drawdown,
)
from insignia.scoring import ScoreVector
from neurons.validator import PairedValidator
from insignia.pairing import PairingConfig


class TestCollateralGate(unittest.TestCase):
    def test_zeros_shortfall_and_freeze(self):
        weights = {"a": 0.5, "b": 0.5}
        positions = {"a": CollateralPosition("a", locked=100.0)}
        gated, ids = apply_collateral_gate(
            weights, positions, required_min=50.0, freeze_uids={"b"})
        self.assertEqual(gated["a"], 0.5)
        self.assertEqual(gated["b"], 0.0)
        self.assertEqual(set(ids), {"b"})

    def test_disabled_floor_is_identity(self):
        weights = {"a": 1.0}
        gated, ids = apply_collateral_gate(weights, {}, required_min=0.0)
        self.assertEqual(gated, weights)
        self.assertEqual(ids, [])


class TestFreezeLedger(unittest.TestCase):
    def test_drawdown_threshold_matches_trading_validator(self):
        self.assertFalse(should_freeze_drawdown(0.19))
        self.assertTrue(should_freeze_drawdown(0.20))
        self.assertTrue(should_freeze_drawdown(0.50))

    def test_freeze_is_idempotent_and_survives_until_uid_gone(self):
        ledger = FreezeLedger()
        first = ledger.freeze("trader_0", "max_drawdown", generation=1)
        second = ledger.freeze("trader_0", "again", generation=2)
        self.assertIs(first, second)
        self.assertEqual(ledger.active_uids({"trader_0", "trader_1"}), {"trader_0"})
        # still present → not released
        self.assertEqual(ledger.sweep({"trader_0"}, generation=9), [])
        # gone from metagraph → released
        self.assertEqual(ledger.sweep(set(), generation=9), ["trader_0"])
        self.assertEqual(ledger.records, {})


class TestPairedValidatorGate(unittest.TestCase):
    def test_blowup_freezes_trader_weight(self):
        validator = PairedValidator(
            pairing_config=PairingConfig(partners_per_miner=1))
        genomes = validator.assign_pairs(
            ["researcher_0"], ["trader_0"], block_seed="t")
        model = ScoreVector(composite=0.8, normalized={}, raw={})
        # 25% drawdown breaches the 20% freeze ceiling
        trading = ScoreVector(
            composite=0.9,
            normalized={"consistency": 0.9},
            raw={"max_drawdown": 0.25},
        )
        validator.score_pair(genomes[0], model, trading)
        summary = validator.finalize_generation()
        self.assertIn("trader_0", summary["collateral_gated"])
        self.assertEqual(summary["weights"].get("trader_0", 0.0), 0.0)
        self.assertGreater(summary["ungated_weights"].get("trader_0", 0.0), 0.0)

    def test_floor_gates_without_drawdown(self):
        validator = PairedValidator(
            pairing_config=PairingConfig(partners_per_miner=1),
            collateral_required_min=50.0,
        )
        genomes = validator.assign_pairs(
            ["researcher_0"], ["trader_0"], block_seed="t")
        rng = np.random.RandomState(0)
        model = ScoreVector(composite=0.7, normalized={}, raw={})
        trading = ScoreVector(
            composite=0.7,
            normalized={"consistency": 0.7},
            raw={"max_drawdown": 0.05},
        )
        validator.score_pair(genomes[0], model, trading)
        short = {"trader_0": CollateralPosition("trader_0", locked=10.0)}
        summary = validator.finalize_generation(collateral_positions=short)
        self.assertIn("trader_0", summary["collateral_gated"])
        self.assertEqual(summary["weights"]["trader_0"], 0.0)
        _ = rng  # silence unused if pairing doesn't need it


if __name__ == "__main__":
    unittest.main()
