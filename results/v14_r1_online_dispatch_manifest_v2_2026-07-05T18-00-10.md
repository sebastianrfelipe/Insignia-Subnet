ing # V14-R1 Online-Mode Gate Verification, Orchestrator Dispatch Manifest (v2)

**Generated:** 2026-07-05T18:00:10.830842+00:00
**Config ID:** V14-R1-CORRECTED-KP
**Config source:** orchestrator MongoDB memory key `v14_r1_corrected_config`
**Spec ref:** EMULATOR_SPEC.md §9 acceptance gates (online mode)

## v1 post-mortem (why v2 exists)

- **v1 manifest:** `results/v14_r1_online_dispatch_manifest_2026-07-04T14-45-08.json`
- **v1 result:** 2/6 done, 4 cancelled, 0 failed (dashboard 2026-07-05)

v1 cancelled 4/6 gates due to two MCP guardrail rejections:

- **`NAMESPACE_FILTER_REQUIRED`**, trigger: mongodb_find on convergence_metrics without namespace filter
  - v2 fix: RUN_NAMESPACE declared + namespace_filter on every MCP read
- **`PLAYBOOK_COLLECTION_FORBIDDEN`**, trigger: mongodb_find on simulation_results (belongs to Insignia subnet tuner playbook)
  - v2 fix: FORBIDDEN_COLLECTIONS list + FILESYSTEM_KRET_ARTIFACTS redirect

## v2 fixes

1. **Namespace declared up front:** `{'playbook': 'insignia_subnet_online_verification', 'domain': 'v14_r1', 'procedure': 'v14_r1_online_gate_check'}`, every MCP read must include one of these filter fields.
2. **Forbidden collections:** `['simulation_results']`, orchestrator must NOT `mongodb_find` on these.
3. **Filesystem KRET redirect:** offline evidence read from filesystem (`results/`), not MongoDB.

## Objective

Verify the 6 online-mode §9 gates against the live V14-R1-CORRECTED-KP config on-chain, across >= 2 reruns with different seeds. If all 6 clear, proceed to the HITL promotion gate (btcli hyperparameter apply + promote converged config as production reference).

## Offline Evidence Package (filesystem KRET artifacts, read from disk, NOT MongoDB)

- **Step 2 (simulator):** `results/v14_r1_empirical_validation_2026-07-04T02-21-09.json`
- **Step 3 (sentinel):** `results/sentinel_coverage_matrix_2026-07-04T14-35-21.json`
- **Step 4 (tuner seed):** `results/tuner_v14_r1_warmstart_seed_2026-07-04T02-39-07.json`
- **Step 5 (gate check):** `results/researcher_v14_r1_gate_check_2026-07-04T14-33-59.json`

**Read instruction:** Read these from the FILESYSTEM, not MongoDB. The orchestrator's mongodb_find on `simulation_results` is forbidden (belongs to the Insignia subnet tuner playbook). These JSON files are the KRET artifacts that substitute for that collection.

**Offline summary:** All 4 harness-mode gates PASS (attack_breach_rate=0.0, separation=0.9004, validator_latency_severity=0.0351, prediction_timing_severity=0.025). 0 adversary leaks across the 19-vector sentinel surface. 2 non-penalty-path breaches remain (random_baseline_discrimination, collusion_temporal_pattern), both synthetic harness artifacts, not adversary leaks.

## 6 Online-Mode Gates to Verify On-Chain (with namespaced evidence sources)

| # | Gate | Threshold | Evidence source | Namespace filter |
|---|---|---|---|---|
| 1 | `honest_mean_score` | 0.97 | on_chain: live miner composite scores on-chain (mean of honest miners) (on-chain) | `playbook=insignia_subnet_online_verification, domain=v14_r1, procedure=v14_r1_online_gate_check, agent_type=miner` |
| 2 | `score_variance` | 0.002 | on_chain: live miner composite score variance across honest miners (on-chain) | `playbook=insignia_subnet_online_verification, domain=v14_r1, procedure=v14_r1_online_gate_check, agent_type=miner` |
| 3 | `commit_reveal_effectiveness` | 0.667 | on_chain: commit-reveal telemetry (commit_timestamps, reveal_timestamps, no_reveal_streaks) (on-chain) | `playbook=insignia_subnet_online_verification, domain=v14_r1, procedure=v14_r1_online_gate_check, agent_type=validator` |
| 4 | `consecutive_clean_validations` | 6 | mcp_read: sentinel breach-free validation streak (sentinel_state) | `playbook=insignia_subnet_online_verification, domain=v14_r1, procedure=v14_r1_online_gate_check, agent_type=sentinel` |
| 5 | `convergence_contract` | unanimously_met | mcp_read: §7 contract: all agents agree + grace period elapsed (convergence_metrics) | `playbook=insignia_subnet_online_verification, domain=v14_r1, procedure=v14_r1_online_gate_check, agent_type=convergence_monitor` |
| 6 | `sentinel_posture` | SECURE_AND_IMPROVING | mcp_read: security_status field (sentinel_state) | `playbook=insignia_subnet_online_verification, domain=v14_r1, procedure=v14_r1_online_gate_check, agent_type=sentinel` |

## Acceptance Criteria

- **Reruns required:** 2 (different seeds for each rerun)
- **All gates must clear:** True
- **If any gate fails:** Do NOT promote V14-R1. File a correction task for the failing gate's root cause and re-run the cycle from the appropriate step.
- **If all gates clear:** Proceed to HITL promotion gate: (1) btcli hyperparameter apply with the V14-R1-CORRECTED-KP config, (2) promote the converged config as the production reference in research_targets.

## MCP Dispatch Commands (v2, namespaced + forbidden-collection-aware)

**Cannot be executed from this repo**, the insignia-local MCP server is not available in this environment. Execute in the orchestrator's agent env.

### step1_file_task

**MCP tool:** `insignia-local.file_task`

```json
{
  "assignee": "orchestrator",
  "priority": 10,
  "description": "V14-R1 online-mode gate verification (v2 \u2014 fixes v1 guardrail rejections): run the live V14-R1-CORRECTED-KP config on-chain across >= 2 reruns with different seeds. Verify the 6 online-mode \u00a79 gates. Scope every MCP read with the namespace {'playbook': 'insignia_subnet_online_verification', 'domain': 'v14_r1', 'procedure': 'v14_r1_online_gate_check'}. Do NOT mongodb_find on ['simulation_results'] \u2014 read filesystem KRET artifacts in results/ instead.",
  "metadata": {
    "cycle_step": "5_to_HITL",
    "config_id": "V14-R1-CORRECTED-KP",
    "manifest_version": "v2",
    "namespace": {
      "playbook": "insignia_subnet_online_verification",
      "domain": "v14_r1",
      "procedure": "v14_r1_online_gate_check"
    },
    "forbidden_collections": [
      "simulation_results"
    ],
    "filesystem_kret_artifacts": {
      "simulator_step2": "results/v14_r1_empirical_validation_2026-07-04T02-21-09.json",
      "sentinel_step3": "results/sentinel_coverage_matrix_2026-07-04T14-35-21.json",
      "tuner_step4": "results/tuner_v14_r1_warmstart_seed_2026-07-04T02-39-07.json",
      "researcher_step5": "results/researcher_v14_r1_gate_check_2026-07-04T14-33-59.json"
    },
    "offline_evidence_branch": "feat/signal-driven-adversary-penalties",
    "offline_gate_check": "results/researcher_v14_r1_gate_check_2026-07-04T14-33-59.json"
  }
}
```

### step2_write_agent_memory_dispatch

**MCP tool:** `insignia-local.write_agent_memory`

```json
{
  "key": "v14_r1_online_verification_dispatch",
  "value": {
    "status": "DISPATCHED_V2",
    "manifest_version": "v2",
    "timestamp": "2026-07-05T18:00:10.830870+00:00",
    "config_id": "V14-R1-CORRECTED-KP",
    "config_source": "v14_r1_corrected_config (MongoDB)",
    "namespace": {
      "playbook": "insignia_subnet_online_verification",
      "domain": "v14_r1",
      "procedure": "v14_r1_online_gate_check"
    },
    "forbidden_collections": [
      "simulation_results"
    ],
    "filesystem_kret_artifacts": {
      "simulator_step2": "results/v14_r1_empirical_validation_2026-07-04T02-21-09.json",
      "sentinel_step3": "results/sentinel_coverage_matrix_2026-07-04T14-35-21.json",
      "tuner_step4": "results/tuner_v14_r1_warmstart_seed_2026-07-04T02-39-07.json",
      "researcher_step5": "results/researcher_v14_r1_gate_check_2026-07-04T14-33-59.json"
    },
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
    ],
    "v1_postmortem": "4/6 cancelled by NAMESPACE_FILTER_REQUIRED + PLAYBOOK_COLLECTION_FORBIDDEN; v2 fixes both."
  }
}
```

### step3_invalidate_prior_result

**MCP tool:** `insignia-local.write_agent_memory`

```json
{
  "key": "v14_r1_online_verification_result",
  "value": null,
  "note": "Clear the v1 result (2/6 done, 4 cancelled) so the orchestrator starts v2 fresh."
}
```

## Post-Verification Actions

### On success (all 6 gates clear across >= 2 reruns)

1. Write agent_memory key `v14_r1_online_verification_result` with status=ALL_GATES_CLEARED.
2. File a HITL task for `btcli hyperparameter apply` with the V14-R1-CORRECTED-KP config.
3. Update `research_targets` in parameter_space.py: target_achieved=True, current_candidate_status=promoted_to_production_reference.

### On failure (any gate fails)

1. Write agent_memory key `v14_r1_online_verification_result` with status=GATES_FAILED and the failing gate(s).
2. File a correction task for the failing gate's root cause.
3. Do NOT promote V14-R1.

_JSON manifest: `v14_r1_online_dispatch_manifest_v2_2026-07-05T18-00-10.json`_