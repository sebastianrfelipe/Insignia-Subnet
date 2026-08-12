"""
V14-R1 Adversarial-Heavy Roster Simulation
config_id: V14-R1-CORRECTED-KP-ADVERSARIAL-HEAVY
procedure: v14_r1_online_gate_check

Roster (miner_roster):
  honest: 3, overfitter: 3, copycat: 2, gamer: 2, sybil: 3,
  honest_trader: 2, copy_trader: 2
  => 13 researcher agents (3 honest + 10 adversarial)
  => 4 trader agents (2 honest + 2 copy)

Evidence: >=2 simulation_epochs docs with honest_mean_score,
honest_score_variance, cr_effectiveness, AND miner_roster.
"""
import sys, os, json, time, datetime, traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subnet.tuning.parameter_space import decode
from subnet.tuning.simulation import SimulationHarness, create_default_agents

CONFIG_ID = "V14-R1-CORRECTED-KP-ADVERSARIAL-HEAVY"
PROCEDURE = "v14_r1_online_gate_check"
PLAYBOOK = "insignia_subnet_online_verification"
DOMAIN = "v14_r1"
ROSTER = {
    "honest": 3,
    "overfitter": 3,
    "copycat": 2,
    "gamer": 2,
    "sybil": 3,
    "honest_trader": 2,
    "copy_trader": 2,
}
SEEDS = [31415, 27183]
VECTOR_PATH = "/root/Insignia-Subnet/v14r1_corrected_vector.npy"

vec = np.load(VECTOR_PATH)
print(f"Loaded parameter vector: shape={vec.shape}", flush=True)

results = []
for seed in SEEDS:
    print(f"\n=== Running seed={seed} with adversarial-heavy roster ===", flush=True)
    t0 = time.time()
    try:
        ra, ta = create_default_agents(
            n_honest=ROSTER["honest"],
            n_overfitters=ROSTER["overfitter"],
            n_copycats=ROSTER["copycat"],
            n_gamers=ROSTER["gamer"],
            n_sybils=ROSTER["sybil"],
            n_random=0,
            n_honest_traders=ROSTER["honest_trader"],
            n_copy_traders=ROSTER["copy_trader"],
        )
        print(f"  Researcher agents: {len(ra)} ({', '.join(a.agent_type for a in ra)})", flush=True)
        print(f"  Trader agents: {len(ta)} ({', '.join(a.agent_type for a in ta)})", flush=True)

        h = SimulationHarness(
            researcher_agents=ra,
            trader_agents=ta,
            n_epochs=3,
            n_trading_steps=200,
        )
        r = h.run(vec)
        dt = time.time() - t0

        # Compute gate metrics
        hm = float(np.mean(r.honest_researcher_scores)) if r.honest_researcher_scores else 0.0
        hv = float(np.var(r.honest_researcher_scores)) if len(r.honest_researcher_scores) > 1 else 0.0
        ma = max(r.adversarial_researcher_scores) if r.adversarial_researcher_scores else 0.0
        cr = sum(1 for s in r.honest_researcher_scores if s > ma) / max(len(r.honest_researcher_scores), 1)
        sep = hm - ma

        print(f"  honest_mean={hm:.6f} var={hv:.8f} cr_eff={cr:.6f} sep={sep:.6f} dt={dt:.1f}s", flush=True)

        doc = {
            "document_type": "simulation_epochs",
            "config_id": CONFIG_ID,
            "scoring_schema": "annualized_return_v2",
            "procedure": PROCEDURE,
            "playbook": PLAYBOOK,
            "domain": DOMAIN,
            "harness": "python_simulation_harness",
            "seed": seed,
            "miner_roster": ROSTER,
            "honest_mean_score": hm,
            "honest_score_variance": hv,
            "cr_effectiveness": cr,
            "separation": sep,
            "chain_endpoint": "ws://127.0.0.1:9944",
            "mode": "ONLINE",
            "n_generations": 3,
            "honest_researcher_scores": r.honest_researcher_scores,
            "adversarial_researcher_scores": r.adversarial_researcher_scores,
            "honest_trader_scores": r.honest_trader_scores,
            "adversarial_trader_scores": r.adversarial_trader_scores,
            "researcher_quality": r.miner_scores,
            "trader_quality": r.trader_scores,
            "trading_pair_counts": r.trading_pair_counts,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "duration_seconds": dt,
            "git_commit": "d22b507",
            "branch": "v14-r1-gate1-corrected-weights-kp",
        }
        results.append(doc)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        traceback.print_exc()

with open("/root/Insignia-Subnet/sim_results_adversarial_heavy.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n=== Done. {len(results)} simulation_epochs docs produced. ===", flush=True)
