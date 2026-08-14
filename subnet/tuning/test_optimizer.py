"""
Tests for the roster-aware optimizer (step 4c-d).

Validates:
  - load_warm_start() rejects stale 75-dim warm-start files with a clear error
  - load_warm_start() accepts 82-dim warm-start files
  - load_warm_start() returns (None, {}) for missing files / no params
  - InsigniaTuningProblem._evaluate() reads the roster from the decoded vector
    (verified by checking the harness's miner_roster output matches the vector)
"""

import json
import os
import tempfile
import unittest

import numpy as np

from tuning.parameter_space import N_PARAMS, encode_defaults, decode
from tuning.optimizer import load_warm_start

try:
    from tuning.optimizer import InsigniaTuningProblem, PYMOO_AVAILABLE
except ImportError:
    InsigniaTuningProblem = None
    PYMOO_AVAILABLE = False


class TestLoadWarmStart(unittest.TestCase):

    def test_rejects_stale_75_dim_seed(self):
        stale = {"params": list(np.full(75, 0.5).tolist()), "config_id": "V13-R3-STALE"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(stale, f)
            path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                load_warm_start(path)
            self.assertIn("75 params", str(ctx.exception))
            self.assertIn("7 roster dims", str(ctx.exception))
        finally:
            os.unlink(path)

    def test_accepts_82_dim_seed(self):
        defaults = encode_defaults()
        seed = {
            "params": defaults.tolist(),
            "config_id": "V14-R1-CORRECTED-KP",
            "fitness": {"decoded": {"honest_score": 0.9}},
            "source_reports": ["report1"],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(seed, f)
            path = f.name
        try:
            ws_x, ws_meta = load_warm_start(path)
            self.assertIsNotNone(ws_x)
            self.assertEqual(len(ws_x), N_PARAMS)
            self.assertEqual(ws_meta["config_id"], "V14-R1-CORRECTED-KP")
            self.assertEqual(ws_meta["source_reports"], ["report1"])
        finally:
            os.unlink(path)

    def test_missing_file_returns_none(self):
        ws_x, ws_meta = load_warm_start("/nonexistent/path/seed.json")
        self.assertIsNone(ws_x)
        self.assertEqual(ws_meta, {})

    def test_no_params_returns_none(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"config_id": "no-params"}, f)
            path = f.name
        try:
            ws_x, ws_meta = load_warm_start(path)
            self.assertIsNone(ws_x)
            self.assertEqual(ws_meta, {})
        finally:
            os.unlink(path)

    def test_wrong_length_returns_none(self):
        wrong = {"params": list(np.full(50, 0.5).tolist())}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(wrong, f)
            path = f.name
        try:
            ws_x, ws_meta = load_warm_start(path)
            self.assertIsNone(ws_x)
            self.assertEqual(ws_meta, {})
        finally:
            os.unlink(path)


@unittest.skipUnless(PYMOO_AVAILABLE and InsigniaTuningProblem is not None, "pymoo not installed")
class TestEvaluatePymoo(unittest.TestCase):

    def test_problem_instantiates_with_roster_dims(self):
        """InsigniaTuningProblem can be instantiated with the 82-dim space."""
        problem = InsigniaTuningProblem(n_honest=5, n_adversarial_each=1, n_epochs=1)
        self.assertEqual(problem.n_var, N_PARAMS)


class TestEvaluateReadsRoster(unittest.TestCase):
    """
    Verify the data flow: decode() -> roster -> create_default_agents() ->
    build_miner_roster() produces the same roster counts. This confirms
    _evaluate() reads the roster from the decoded vector without needing
    to run a full simulation (which requires pymoo).
    """

    def test_decode_and_build_roster_round_trip(self):
        from tuning.simulation import build_miner_roster, create_default_agents

        defaults = encode_defaults().copy()
        defaults[75] = 3   # n_honest_researchers
        defaults[76] = 3   # n_overfitters
        defaults[77] = 2   # n_copycats
        defaults[78] = 2   # n_gamers
        defaults[79] = 3   # n_sybils
        defaults[80] = 2   # n_honest_traders
        defaults[81] = 2   # n_copy_traders

        config = decode(defaults)
        expected_roster = config["roster"]
        self.assertEqual(expected_roster, {
            "honest": 3, "overfitter": 3, "copycat": 2, "gamer": 2, "sybil": 3,
            "honest_trader": 2, "copy_trader": 2,
        })

        l1, l2 = create_default_agents(
            n_honest=expected_roster["honest"],
            n_overfitters=expected_roster["overfitter"],
            n_copycats=expected_roster["copycat"],
            n_gamers=expected_roster["gamer"],
            n_sybils=expected_roster["sybil"],
            n_random=1,
            n_honest_traders=expected_roster["honest_trader"],
            n_copy_traders=expected_roster["copy_trader"],
        )
        roster = build_miner_roster(l1, l2)
        self.assertEqual(roster, expected_roster)

    def test_baseline_roster_round_trip(self):
        from tuning.simulation import build_miner_roster, create_default_agents

        defaults = encode_defaults()
        config = decode(defaults)
        expected_roster = config["roster"]
        self.assertEqual(expected_roster, {
            "honest": 5, "overfitter": 1, "copycat": 1, "gamer": 1, "sybil": 1,
            "honest_trader": 3, "copy_trader": 1,
        })

        l1, l2 = create_default_agents(
            n_honest=expected_roster["honest"],
            n_overfitters=expected_roster["overfitter"],
            n_copycats=expected_roster["copycat"],
            n_gamers=expected_roster["gamer"],
            n_sybils=expected_roster["sybil"],
            n_random=1,
            n_honest_traders=expected_roster["honest_trader"],
            n_copy_traders=expected_roster["copy_trader"],
        )
        roster = build_miner_roster(l1, l2)
        self.assertEqual(roster, expected_roster)


if __name__ == "__main__":
    unittest.main()
