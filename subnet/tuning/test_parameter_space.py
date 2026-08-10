"""
Tests for the roster-aware parameter space (step 4a-b).

Validates:
  - N_PARAMS == 82 (75 base + 7 roster dims)
  - decode() returns a `roster` dict with the 7 canonical keys
  - encode_defaults() round-trips through decode() with the V14-R1-CORRECTED-KP
    baseline roster (5/1/1/1/1 + 3/1)
  - get_group_indices() includes a `roster` group at indices 75-81
"""

import unittest

import numpy as np

from tuning.parameter_space import (
    N_PARAMS,
    PARAMETER_DEFINITIONS,
    PARAM_NAMES,
    get_bounds,
    get_group_indices,
    encode_defaults,
    decode,
)


class TestParameterSpaceRoster(unittest.TestCase):

    def test_n_params_is_82(self):
        self.assertEqual(N_PARAMS, 82)

    def test_roster_group_has_7_params(self):
        roster_params = [p for p in PARAMETER_DEFINITIONS if p.group == "roster"]
        self.assertEqual(len(roster_params), 7)
        names = [p.name for p in roster_params]
        self.assertEqual(names, [
            "n_honest_researchers",
            "n_overfitters",
            "n_copycats",
            "n_gamers",
            "n_sybils",
            "n_honest_traders",
            "n_copy_traders",
        ])

    def test_roster_indices_are_75_through_81(self):
        groups = get_group_indices()
        self.assertEqual(groups["roster"], [75, 76, 77, 78, 79, 80, 81])

    def test_bounds_shape_matches_n_params(self):
        lower, upper = get_bounds()
        self.assertEqual(lower.shape, (N_PARAMS,))
        self.assertEqual(upper.shape, (N_PARAMS,))

    def test_decode_returns_roster(self):
        defaults = encode_defaults()
        config = decode(defaults)
        self.assertIn("roster", config)
        roster = config["roster"]
        for key in ("honest", "overfitter", "copycat", "gamer", "sybil", "honest_trader", "copy_trader"):
            self.assertIn(key, roster)

    def test_default_roster_matches_v14_r1_corrected_kp(self):
        defaults = encode_defaults()
        config = decode(defaults)
        roster = config["roster"]
        self.assertEqual(roster, {
            "honest": 5,
            "overfitter": 1,
            "copycat": 1,
            "gamer": 1,
            "sybil": 1,
            "honest_trader": 3,
            "copy_trader": 1,
        })

    def test_encode_defaults_length_matches_n_params(self):
        defaults = encode_defaults()
        self.assertEqual(len(defaults), N_PARAMS)

    def test_encode_defaults_round_trips(self):
        defaults = encode_defaults()
        config = decode(defaults)
        raw = config["raw_params"]
        for name in PARAM_NAMES:
            self.assertIn(name, raw)

    def test_roster_bounds(self):
        roster_by_name = {p.name: p for p in PARAMETER_DEFINITIONS if p.group == "roster"}
        self.assertEqual(roster_by_name["n_honest_researchers"].lower, 3)
        self.assertEqual(roster_by_name["n_honest_researchers"].upper, 10)
        self.assertEqual(roster_by_name["n_overfitters"].lower, 0)
        self.assertEqual(roster_by_name["n_overfitters"].upper, 3)
        self.assertEqual(roster_by_name["n_sybils"].lower, 0)
        self.assertEqual(roster_by_name["n_sybils"].upper, 4)
        self.assertEqual(roster_by_name["n_honest_traders"].lower, 2)
        self.assertEqual(roster_by_name["n_honest_traders"].upper, 5)
        self.assertEqual(roster_by_name["n_copy_traders"].lower, 0)
        self.assertEqual(roster_by_name["n_copy_traders"].upper, 3)

    def test_decode_adversarial_heavy_roster(self):
        """A vector encoding the adversarial_heavy roster decodes to the right counts."""
        defaults = encode_defaults().copy()
        defaults[75] = 3   # n_honest_researchers
        defaults[76] = 3   # n_overfitters
        defaults[77] = 2   # n_copycats
        defaults[78] = 2   # n_gamers
        defaults[79] = 3   # n_sybils
        defaults[80] = 2   # n_honest_traders
        defaults[81] = 2   # n_copy_traders
        config = decode(defaults)
        self.assertEqual(config["roster"], {
            "honest": 3,
            "overfitter": 3,
            "copycat": 2,
            "gamer": 2,
            "sybil": 3,
            "honest_trader": 2,
            "copy_trader": 2,
        })


if __name__ == "__main__":
    unittest.main()
