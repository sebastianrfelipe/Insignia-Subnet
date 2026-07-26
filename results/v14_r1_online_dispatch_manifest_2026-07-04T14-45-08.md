# V14-R1 Online-Mode Gate Verification — Orchestrator Dispatch Manifest

**Generated:** 2026-07-04T14:45:08.292474+00:00
**Config ID:** V14-R1-CORRECTED-KP
**Config source:** orchestrator MongoDB memory key `v14_r1_corrected_config`
**Spec ref:** EMULATOR_SPEC.md §9 acceptance gates (online mode)

## Objective

Verify the 6 online-mode §9 gates against the live V14-R1-CORRECTED-KP config on-chain, across >= 2 reruns with different seeds. If all 6 clear, proceed to the HITL promotion gate (btcli hyperparameter apply + promote converged config as production reference).

## Offline Evidence Package (from `feat/signal-driven-adversary-penalties`)

- **Step 2 (simulator):** `results/v14_r1_empirical_validation_2026-07-04T02-21-09.json`
- **Step 3 (sentinel):** `results/sentinel_coverage_matrix_2026-07-04T14-35-21.json`
- **Step 4 (tuner seed):** `results/tuner_v14_r1_warmstart_seed_2026-07-04T02-39-07.json`
- **Step 5 (gate check):** `results/researcher_v14_r1_gate_check_2026-07-04T14-33-59.json`

**Offline summary:** All 4 harness-mode gates PASS (attack_breach_rate=0.0, separation=0.9004, validator_latency_severity=0.0351, prediction_timing_severity=0.025). 0 adversary leaks across the 19-vector sentinel surface. 2 non-penalty-path breaches remain (random_baseline_discrimination, collusion_temporal_pattern) — both synthetic harness artifacts, not adversary leaks.

## 6 Online-Mode Gates to Verify On-Chain

| # | Gate | Threshold | Evidence source | Note |
|---|---|---|---|---|
| 1 | `honest_mean_score` | 0.97 | live miner composite scores on-chain (mean of honest miners) | The V13-R2 knee claimed 0.9795 (surrogate-predicted, never empirically confirmed). The offline harness produces 0.9007 via its synthetic scorer; the live chain must verify >= 0.97. |
| 2 | `score_variance` | 0.002 | live miner composite score variance across honest miners | Offline harness produces 0.0013 (synthetic); live variance must be <= 0.002. |
| 3 | `commit_reveal_effectiveness` | 0.667 | commit-reveal telemetry on-chain (commit_timestamps, reveal_timestamps, no_reveal_streaks) | Offline harness hardcodes 0.76; live CR effectiveness must be >= 0.667 floor (>= 0.76 target). |
| 4 | `consecutive_clean_validations` | 6 | sentinel breach-free validation streak on-chain | Offline harness hardcodes 6; live streak must be >= 6 consecutive clean validations. |
| 5 | `convergence_contract` | unanimously_met | convergence_metrics MCP (§7 contract: all agents agree + grace period) | Requires orchestrator-side convergence_metrics read. The §7 contract must be unanimously met with the grace period elapsed. |
| 6 | `sentinel_posture` | SECURE_AND_IMPROVING | sentinel_state MCP (security_status field) | Offline harness hardcodes TARGET_ACHIEVED; live sentinel posture must be SECURE_AND_IMPROVING or stronger. |

## Acceptance Criteria

- **Reruns required:** 2 (different seeds for each rerun)
- **All gates must clear:** True
- **If any gate fails:** Do NOT promote V14-R1. File a correction task for the failing gate's root cause and re-run the cycle from the appropriate step.
- **If all gates clear:** Proceed to HITL promotion gate: (1) btcli hyperparameter apply with the V14-R1-CORRECTED-KP config, (2) promote the converged config as the production reference in research_targets.

## MCP Dispatch Commands

**⚠️ Cannot be executed from this repo** — the insignia-local MCP server is not available in this environment. Execute these in the orchestrator's agent env (or trigger via the swarm gateway).

_Execute these commands in the orchestrator's agent env via the insignia-local MCP. These cannot be executed from the offline repo — the insignia-local MCP server is not available in this environment._

### step1_file_task

**MCP tool:** `insignia-local.file_task`

```json
{
  "assignee": "orchestrator",
  "priority": 10,
  "description": "V14-R1 online-mode gate verification: run the live V14-R1-CORRECTED-KP config on-chain across >= 2 reruns with different seeds. Verify the 6 online-mode \u00a79 gates (honest_mean_score >= 0.97, score_variance <= 0.002, commit_reveal_effectiveness >= 0.667, consecutive_clean_validations >= 6, convergence_contract unanimously met, sentinel_posture SECURE_AND_IMPROVING+). If all clear, proceed to HITL promotion.",
  "metadata": {
    "cycle_step": "5_to_HITL",
    "config_id": "V14-R1-CORRECTED-KP",
    "offline_evidence_branch": "feat/signal-driven-adversary-penalties",
    "offline_gate_check": "results/researcher_v14_r1_gate_check_2026-07-04T14-33-59.json"
  }
}
```

### step2_write_agent_memory

**MCP tool:** `insignia-local.write_agent_memory`

```json
{
  "key": "v14_r1_online_verification_dispatch",
  "value": {
    "status": "DISPATCHED",
    "timestamp": "2026-07-04T14:45:08.292491+00:00",
    "config_id": "V14-R1-CORRECTED-KP",
    "config_source": "v14_r1_corrected_config (MongoDB)",
    "offline_evidence": {
      "harness_mode_gates_passed": 4,
      "harness_mode_gates_failed": 0,
      "online_mode_gates_pending": 6,
      "adversary_leaks": 0,
      "separation": 0.9004
    },
    "reruns_required": 2,
    "gates_to_verify": [
      "honest_mean_score",
      "score_variance",
      "commit_reveal_effectiveness",
      "consecutive_clean_validations",
      "convergence_contract",
      "sentinel_posture"
    ]
  }
}
```

### step3_invalidate_prior

**MCP tool:** `insignia-local.write_agent_memory`

```json
{
  "key": "v14_r1_online_verification_result",
  "value": null,
  "note": "Clear any prior result so the orchestrator starts fresh."
}
```

## Post-Verification Actions

### On success (all 6 gates clear across >= 2 reruns)

1. Write the verification result to agent_memory key `v14_r1_online_verification_result` with status=ALL_GATES_CLEARED.
1. File a HITL task for `btcli hyperparameter apply` with the V14-R1-CORRECTED-KP config.
1. Update `research_targets` in parameter_space.py: set target_achieved=True, current_candidate_status=promoted_to_production_reference.

### On failure (any gate fails)

1. Write the verification result to agent_memory key `v14_r1_online_verification_result` with status=GATES_FAILED and the failing gate(s).
1. File a correction task for the failing gate's root cause.
1. Do NOT promote V14-R1.

_JSON manifest: `v14_r1_online_dispatch_manifest_2026-07-04T14-45-08.json`_