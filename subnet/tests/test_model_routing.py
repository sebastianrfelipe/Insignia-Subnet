import unittest
from importlib.util import find_spec

if find_spec("pandas") is not None and find_spec("sklearn") is not None:
    from tuning.simulation import create_default_agents
else:
    create_default_agents = None


@unittest.skipIf(create_default_agents is None, "simulation dependencies unavailable")
class ModelRoutingTests(unittest.TestCase):
    def test_route_assignment_is_stable_for_run_seed(self):
        routing = {
            "enabled": True,
            "route_names": ["route-a", "route-b", "route-c"],
            "assignment_seed": 1234,
            "stable_per_run": True,
        }

        l1_agents_a, l2_agents_a = create_default_agents(
            n_honest=2,
            n_overfitters=1,
            n_copycats=1,
            n_gamers=0,
            n_sybils=1,
            n_random=0,
            n_honest_traders=2,
            n_copy_traders=1,
            model_routing=routing,
        )
        l1_agents_b, l2_agents_b = create_default_agents(
            n_honest=2,
            n_overfitters=1,
            n_copycats=1,
            n_gamers=0,
            n_sybils=1,
            n_random=0,
            n_honest_traders=2,
            n_copy_traders=1,
            model_routing=routing,
        )

        manifest_a = {
            agent.uid: (agent.assigned_route, agent.assigned_model_profile)
            for agent in [*l1_agents_a, *l2_agents_a]
        }
        manifest_b = {
            agent.uid: (agent.assigned_route, agent.assigned_model_profile)
            for agent in [*l1_agents_b, *l2_agents_b]
        }

        self.assertEqual(manifest_a, manifest_b)
        self.assertTrue(all(route in routing["route_names"] for route, _ in manifest_a.values()))

    def test_route_profiles_affect_l1_and_l2_behavior_space(self):
        routing = {
            "enabled": True,
            "route_names": ["route-a", "route-b"],
            "assignment_seed": 99,
            "stable_per_run": True,
        }
        l1_agents, l2_agents = create_default_agents(
            n_honest=1,
            n_overfitters=1,
            n_copycats=0,
            n_gamers=0,
            n_sybils=0,
            n_random=0,
            n_honest_traders=1,
            n_copy_traders=0,
            model_routing=routing,
        )

        for agent in l1_agents:
            self.assertIn("n_estimators", agent.assigned_model_profile)
            self.assertIn("learning_rate", agent.assigned_model_profile)

        for agent in l2_agents:
            self.assertIn("max_position_pct", agent.assigned_model_profile)
            self.assertIn("model_load_count", agent.assigned_model_profile)


if __name__ == "__main__":
    unittest.main()
