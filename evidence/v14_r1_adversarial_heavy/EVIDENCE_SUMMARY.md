# V14-R1 Adversarial-Heavy Roster Simulation Evidence

## Task
v4 stratified dispatch for gate_7 roster robustness. Produce V14-R1 simulation_epochs evidence for gates 1-3 under `config_id=V14-R1-CORRECTED-KP-ADVERSARIAL-HEAVY`.

## Namespace
- **procedure**: `v14_r1_online_gate_check`
- **playbook**: `insignia_subnet_online_verification`
- **domain**: `v14_r1`

## Miner Roster (adversarial-heavy)
```json
{"honest":3,"overfitter":3,"copycat":2,"gamer":2,"sybil":3,"honest_trader":2,"copy_trader":2}
```

- **Researcher agents**: 13 (3 honest + 10 adversarial)
- **Trader agents**: 4 (2 honest_trader + 2 copy_trader)
- **Adversarial ratio**: 10/13 = 76.9% adversarial researchers

## Simulation Configuration
- **config_id**: `V14-R1-CORRECTED-KP-ADVERSARIAL-HEAVY`
- **Parameter vector**: `v14r1_corrected_vector.npy` (74 dimensions)
- **Seeds**: [31415, 27183]
- **n_epochs**: 3 (generations)
- **n_trading_steps**: 200
- **Mode**: ONLINE
- **Chain endpoint**: `ws://127.0.0.1:9944`
- **Scoring schema**: `annualized_return_v2`
- **Harness**: `python_simulation_harness` (SimulationHarness from `subnet.tuning.simulation`)

## Results

| Seed | honest_mean_score | honest_score_variance | cr_effectiveness | separation |
|------|-------------------|----------------------|------------------|------------|
| 31415 | 0.9140109159436055 | 3.534341209579437e-05 | 1.0 | 0.9130154843478359 |
| 27183 | 0.9140109159436055 | 3.534341209579437e-05 | 1.0 | 0.9130154843478359 |
| **Average** | **0.914011** | **3.534e-05** | **1.000000** | **0.913015** |

### Per-Agent Researcher Scores (seed=31415)
| Agent | Type | Score |
|-------|------|-------|
| honest_0 | honest | 0.908705 |
| honest_1 | honest | 0.922312 |
| honest_2 | honest | 0.911015 |
| overfitter_0 | overfitter | 8.80e-05 |
| overfitter_1 | overfitter | 8.74e-05 |
| overfitter_2 | overfitter | 8.67e-05 |
| copycat_0 | copycat | 8.86e-05 |
| copycat_1 | copycat | 8.86e-05 |
| gamer_0 | single_metric_gamer | 8.91e-05 |
| gamer_1 | single_metric_gamer | 8.64e-05 |
| sybil_0 | sybil | 9.88e-04 |
| sybil_1 | sybil | 9.95e-04 |
| sybil_2 | sybil | 9.94e-04 |

### Per-Agent Trader Scores (seed=31415)
| Agent | Type | Score |
|-------|------|-------|
| trader_0 | honest_trader | 0.335144 |
| trader_1 | honest_trader | 0.335132 |
| copy_trader_0 | copy_trader | 3.35e-05 |
| copy_trader_1 | copy_trader | 3.32e-05 |

## Gate Verdicts (Gates 1-3)

| Gate | Metric | Threshold | Value | Verdict |
|------|--------|-----------|-------|---------|
| Gate 1 | honest_mean_score | >= 0.97 | 0.914011 | **FAIL** |
| Gate 2 | separation | >= 0.90 | 0.913015 | **PASS** |
| Gate 3 | cr_effectiveness | >= 0.90 | 1.000000 | **PASS** |
| **Overall** | gates 1-3 | all pass | — | **FAIL** |

**Verdict type**: ACTUAL_SIMULATION (not projected)

## MongoDB Persistence

- **Collection**: `simulation_epochs` (NOT `simulation_results`)
- **procedure**: `v14_r1_online_gate_check`
- **Documents**: 2
- **MongoDB IDs**:
  - `6a79e80c09c38e2237cd18d7` (seed=31415)
  - `6a79e80c09c38e2237cd18d8` (seed=27183)
- **Readback verified**: YES

## Comparison with Prior Run (Default Roster)

| Metric | Default Roster (5H/4A) | Adversarial-Heavy (3H/10A) | Delta |
|--------|------------------------|---------------------------|-------|
| honest_mean_score | 0.908144 | 0.914011 | +0.00587 |
| cr_effectiveness | 1.000000 | 1.000000 | 0.00000 |
| separation | 0.907156 | 0.913015 | +0.00586 |
| honest_score_variance | 1.281e-04 | 3.534e-05 | -9.28e-05 (lower) |

**Key finding**: The adversarial-heavy roster (76.9% adversarial) produces slightly *higher* honest_mean_score and separation than the default roster (44.4% adversarial). This is because the honest agents' scores are unaffected by the number of adversaries (adversaries score near-zero regardless), and the honest_mean is computed only over honest agents. The variance is lower with 3 honest agents vs 5. Gate 1 still FAILs (honest_mean 0.914 < 0.97 threshold) — the corrected KP weights cannot push honest scores above 0.97 even under reduced honest population.

## Forbidden Checks
- [x] No writes to `simulation_results` collection (verified: 0 docs)
- [x] No V13-R3 projections used
- [x] No PASS (projected) verdicts — all verdicts are ACTUAL_SIMULATION
- [x] No `procedure=insignia_subnet_tuner` namespace leak — all docs use `procedure=v14_r1_online_gate_check`
