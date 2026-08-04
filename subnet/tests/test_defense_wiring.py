"""Defense wiring test — no defense is assumed unless provably exercised.

Encodes the V13-R3 copycat root-cause lesson (surrogate assumed defenses the
scoring path never implemented) as three permanent checks against
`tuning/defense_registry.py`:

1. Import — every registered implementation symbol resolves.
2. Forward — every declared evidence identifier appears in `tuning/simulation.py`
   source the required number of times (multiplier: definition + application;
   telemetry key: emission). Catches "assumed but unimplemented".
3. Reverse — every penalty-multiplier constant and ensemble detector key the
   simulator defines is claimed by a registry entry. Catches "defined but never
   used".

Run from the subnet package root, same convention as the other subnet tests:

    cd subnet && python -m pytest tests/test_defense_wiring.py
"""

import importlib
import unittest
from pathlib import Path

from tuning.defense_registry import (
    DEFENSE_REGISTRY,
    claimed_evidence,
    discover_simulator_controls,
)


SIM_PATH = Path(__file__).resolve().parents[1] / "tuning" / "simulation.py"


class TestDefenseWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim_source = SIM_PATH.read_text(encoding="utf-8")

    def test_implementations_resolve(self):
        for entry in DEFENSE_REGISTRY.values():
            module_name, _, symbol = entry.implementation.partition(":")
            with self.subTest(control=entry.control_id, implementation=entry.implementation):
                self.assertTrue(symbol, f"{entry.implementation}: missing ':Symbol'")
                module = importlib.import_module(module_name)
                self.assertTrue(
                    hasattr(module, symbol),
                    f"{entry.implementation} does not resolve",
                )

    def test_forward_evidence_is_exercised(self):
        for entry in DEFENSE_REGISTRY.values():
            for identifier, min_occurrences in entry.sim_evidence:
                with self.subTest(control=entry.control_id, evidence=identifier):
                    count = self.sim_source.count(identifier)
                    self.assertGreaterEqual(
                        count,
                        min_occurrences,
                        f"{entry.control_id}: '{identifier}' appears {count}x in "
                        f"simulation.py, needs >= {min_occurrences}x — defense "
                        "assumed but not exercised",
                    )

    def test_reverse_no_unclaimed_controls(self):
        discovered = discover_simulator_controls(self.sim_source)
        claimed = claimed_evidence()
        for family, identifiers in discovered.items():
            unclaimed = identifiers - claimed
            self.assertEqual(
                unclaimed,
                set(),
                f"simulator {family} with no registry entry: {sorted(unclaimed)} — "
                "register them in tuning/defense_registry.py or delete the dead control",
            )

    def test_live_path_only_entries_have_rationale(self):
        for entry in DEFENSE_REGISTRY.values():
            if not entry.sim_evidence:
                with self.subTest(control=entry.control_id):
                    self.assertTrue(entry.live_path_only_rationale)


if __name__ == "__main__":
    unittest.main()
