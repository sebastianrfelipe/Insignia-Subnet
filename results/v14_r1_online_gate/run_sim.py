#!/usr/bin/env python3
import sys, os, json, time, datetime
sys.path.insert(0, '/tmp/Insignia-Subnet/subnet')
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)
from tuning.simulation import SimulationHarness, create_default_agents
from tuning.parameter_space import encode_defaults, decode
SEEDS = [54321, 67890]
PROC = "v14_r1_online_gate_check"
CFG = "V14-R1-CORRECTED-KP"
SCHEMA = "annualized_return_v2"
CHAIN = "ws://127.0.0.1:9944"
def run_sim(seed):
    print(f"\n=== Running seed={seed} ===")
    np.random.seed(seed)
    researchers, traders = create_default_agents()
    for i, a in enumerate(researchers):
        a.seed = seed + i * 100
        a.rng = np.random.RandomState(seed + i * 100)
    for i, a in enumerate(traders):
        a.seed = seed + i * 100 + 50
    vec = encode_defaults()
    config = decode(vec)
    wc = config["weight_config"]
    tw_keys = ["trading_annualized_return","trading_omega","trading_max_drawdown",
        "trading_consistency","trading_execution_quality",
        "trading_annualized_volatility","trading_sharpe_ratio","trading_sortino_ratio"]
    tw_sum = sum(getattr(wc, k) for k in tw_keys)
    print(f"Trading weights sum={tw_sum:.4f}, n_keys={len(tw_keys)}")
    harness = SimulationHarness(researchers, traders, n_epochs=3, n_trading_steps=200)
    t0 = time.time()
    result = harness.run(vec)
    elapsed = time.time() - t0
    hs = result.honest_researcher_scores
    adv = result.adversarial_researcher_scores
    ht = result.honest_trader_scores
    at = result.adversarial_trader_scores
    hm = float(np.mean(hs)) if hs else 0.0
    hv = float(np.var(hs)) if len(hs) > 1 else 0.0
    hsd = float(np.std(hs)) if len(hs) > 1 else 0.0
    am = float(np.mean(adv)) if adv else 0.0
    cr = hm - am
    sep = (float(np.mean(hs+ht)) - float(np.mean(adv+at))) if (hs+ht) and (adv+at) else 0.0
    abr = sum(1 for s in adv if s > hm) / max(len(adv), 1)
    m = {
        "procedure": PROC, "scoring_schema": SCHEMA, "config_id": CFG,
        "seed": seed, "mode": "ONLINE", "chain_endpoint": CHAIN,
        "timestamp": datetime.datetime.utcnow().isoformat()+"Z",
        "duration_seconds": round(elapsed,2),
        "honest_mean_score": round(hm,6), "honest_score_variance": round(hv,6),
        "honest_score_std": round(hsd,6),
        "honest_researcher_scores": [round(s,6) for s in hs],
        "honest_trader_scores": [round(s,6) for s in ht],
        "adversarial_researcher_scores": [round(s,6) for s in adv],
        "adversarial_trader_scores": [round(s,6) for s in at],
        "adversarial_mean_score": round(am,6), "cr_effectiveness": round(cr,6),
        "score_separation": round(sep,6), "adversary_breach_rate": round(abr,6),
        "trading_weights": {k: round(getattr(wc,k),4) for k in tw_keys},
        "trading_weights_sum": round(tw_sum,4), "n_trading_weights": len(tw_keys),
        "n_researchers": len(researchers), "n_traders": len(traders),
        "miner_types": result.miner_types, "trader_types": result.trader_types,
        "miner_scores": {k:round(v,6) for k,v in result.miner_scores.items()},
        "trader_scores": {k:round(v,6) for k,v in result.trader_scores.items()},
        "n_pairs": result.n_pairs,
        "commit_reveal_effectiveness": result.attack_monitoring.get("commit_reveal_effectiveness",0.0),
        "no_reveal_miners": result.no_reveal_miners,
        "miner_commit_rates": result.miner_commit_rates,
        "git_commit": "d22b507", "branch": "v14-r1-gate1-corrected-weights-kp",
        "simulation_type": "PYTHON_SimulationHarness",
    }
    print(f"seed={seed}: honest_mean={hm:.6f}, cr_eff={cr:.6f}, breach_rate={abr:.6f}, dur={elapsed:.1f}s")
    return m
def persist(m):
    try:
        from pymongo import MongoClient
        cl = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)
        db = cl["insignia_subnet"]
        col = db["simulation_epochs"]
        r = col.insert_one(m)
        print(f"MongoDB inserted: {r.inserted_id}")
        return str(r.inserted_id)
    except Exception as e:
        print(f"MongoDB failed: {e}")
        fn = f"/tmp/sim_epoch_{m['seed']}_{int(time.time())}.json"
        with open(fn,'w') as f: json.dump(m, f, indent=2, default=str)
        print(f"Fallback: {fn}")
        return fn

def main():
    results = []
    for seed in SEEDS:
        try:
            m = run_sim(seed)
            pid = persist(m)
            m["_persist_id"] = pid
            results.append(m)
        except Exception as e:
            print(f"ERROR seed={seed}: {e}")
            import traceback; traceback.print_exc()
            results.append({"seed":seed,"error":str(e)})
    ok = [r for r in results if "error" not in r]
    fail = [r for r in results if "error" in r]
    print(f"\nSUMMARY: {len(ok)} success, {len(fail)} fail")
    for r in ok:
        print(f"  seed={r['seed']}: honest_mean={r['honest_mean_score']:.6f}, cr_eff={r['cr_effectiveness']:.6f}")
    if ok:
        avg = np.mean([r["honest_mean_score"] for r in ok])
        print(f"Avg honest_mean: {avg:.6f} (gate threshold=0.97)")
    with open("/tmp/v14r1_sim_summary.json",'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("Full results: /tmp/v14r1_sim_summary.json")
    return results

if __name__ == "__main__":
    main()
