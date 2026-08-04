# V14-R1 Online-Mode Gate Verification, Orchestrator Dispatch Manifest (v3)

**Generated:** 2026-07-07T20:53:46.970624+00:00
**Amended:** 2026-08-03 (AMEND-2026-08-03-trading-metric-revamp, subnet/CHANGELOG.md 2026-08-03)
**Config ID:** V14-R1-CORRECTED-KP
**Config source:** orchestrator MongoDB memory key `v14_r1_corrected_config`
**Spec ref:** EMULATOR_SPEC.md §9 acceptance gates (online mode)

## 2026-08-03 amendment, trading metric revamp

Config compatibility + evidence freshness. V14-R1-CORRECTED-KP predates the parameter-space revamp and must be re-based before verification; tuner warm-start vectors under results/ are invalidated; all gate evidence must be scored under the annualized_return_v2 composite.

Design change (subnet/CHANGELOG.md 2026-08-03): annualized_return replaces realized_pnl as the trading profitability headline; win_rate is demoted to a reported-only diagnostic; the tunable trading-weight space drops from 9 weights to 8 with the survivors renormalized pro-rata x1/0.94. V14-R1-CORRECTED-KP was tuned under the old space, so:

- Added pre-flight check config_matches_current_parameter_space (abort status ABORTED_CONFIG_PARAMETER_SPACE_STALE).
- Added hard block no_stale_parameter_space_promotion.
- Added evidence_scoring_schema requirement (scoring_schema='annualized_return_v2').
- Appended the stale-composite entry to every gate's forbidden_evidence.
- Quarantined tuner_step4 warm-start artifact into invalidated_kret_artifacts (must not be decoded with the new parameter space).

Invalidated KRET artifacts (quarantined, retained for provenance, MUST NOT be decoded with the new parameter space):

- `tuner_step4`: `results/tuner_v14_r1_warmstart_seed_2026-07-04T02-39-07.json`, Warm-start vector encoded in the pre-2026-08-03 parameter space (9 trading weights incl. realized_pnl + win_rate). Must not be decoded with the new 8-weight annualized_return parameter space.

## v2 post-mortem (why v3 exists)

- **v2 manifest:** `results/v14_r1_online_dispatch_manifest_v2_2026-07-05T18-00-10.json`
- **v2 report:** `Orchestration Report/Orchestration Report, 2026-07-07T20-41-48.pdf`
- **v2 result:** 6/6 completed, 0 failed (guardrails fixed)
- **v2 actual outcome:** Premature promotion: orchestrator declared PASS on projected V13-R3 baseline + offline KRET; coder persisted criterion_met=true to MongoDB without V14-R1 runs; parameter_space.py update stuck in PENDING_GITHUB_PUSH; reviewer audit said INSUFFICIENT_EVIDENCE on all 6 gates and was overridden by orchestrator summary.

v2 caveats that were the whole verification:
- Local chain UNREACHABLE, 0 on-chain evidence
- 0 V14-R1 documents in MongoDB across 12+ collections
- 0 reruns completed (manifest required >= 2)
- V13-R3 convergence FAILED (criterion_met=false) yet used as baseline projection

Root causes v3 fixes:

- **v2 failure:** Orchestrator fell back to projections when chain was unreachable
  - **v3 fix:** PRE-FLIGHT GATE: abort if chain unreachable, no fallback
- **v2 failure:** Verdicts cited V13-R3 data and offline files as PASS evidence
  - **v3 fix:** EVIDENCE_REQUIREMENTS: each gate must cite a V14-R1 MongoDB document; V13-R3 / offline refs are forbidden_evidence
- **v2 failure:** Coder wrote criterion_met=true before V14-R1 runs existed
  - **v3 fix:** PROMOTION WRITE-BLOCK: criterion_met=true writes forbidden until >= 2 V14-R1 simulation_epochs + V14-R1 sentinel_state persisted
- **v2 failure:** Coder used procedure=insignia_subnet_tuner for V14-R1 writes (namespace leak)
  - **v3 fix:** NAMESPACE LOCK: V14-R1 writes must use procedure=v14_r1_online_gate_check
- **v2 failure:** Orchestrator summary overrode reviewer's INSUFFICIENT_EVIDENCE
  - **v3 fix:** REVIEWER VETO: orchestrator summary must match reviewer verdicts; disagreement triggers correction task, not promotion
- **v2 failure:** PROMOTED_WITH_PENDING_BTCLI status written without git push
  - **v3 fix:** NO PROMOTION WITHOUT GIT PUSH: PENDING_GITHUB_PUSH is a TODO, not a promotion

## v3 hardening summary

1. **PRE-FLIGHT GATE**, abort if local chain unreachable (no projection fallback).
2. **EVIDENCE-BACKED VERDICTS**, each gate must cite a V14-R1 MongoDB document; V13-R3 / offline refs are forbidden evidence.
3. **PROMOTION WRITE-BLOCK**, `criterion_met=true` writes forbidden until ≥2 V14-R1 `simulation_epochs` + V14-R1 `sentinel_state` persisted.
4. **NAMESPACE LOCK**, V14-R1 writes must use `procedure=v14_r1_online_gate_check` (not `insignia_subnet_tuner`).
5. **REVIEWER VETO**, orchestrator summary must match reviewer verdicts; disagreement → correction task, not promotion.
6. **NO PROJECTION VERDICTS**, `PASS (projected)` / `PASS (conditional)` are forbidden; verdicts are PASS / FAIL / INSUFFICIENT_EVIDENCE.
7. **NO PROMOTION WITHOUT GIT PUSH**, `PENDING_GITHUB_PUSH` is a TODO, not a promotion.
8. **CONFIG PARAMETER-SPACE GATE (2026-08-03)**, abort with `ABORTED_CONFIG_PARAMETER_SPACE_STALE` if V14-R1-CORRECTED-KP is still expressed in the pre-revamp trading-weight space; file a re-basing task instead of verifying.
9. **EVIDENCE SCORING SCHEMA (2026-08-03)**, gate evidence must carry `scoring_schema='annualized_return_v2'`; pre-revamp composite runs do not count toward `min_documents`.

## Objective

Verify the 6 online-mode §9 gates against the live V14-R1-CORRECTED-KP config on-chain, across >= 2 reruns with different seeds, with V14-R1 evidence persisted to MongoDB. If all 6 clear against V14-R1 evidence (not projections), proceed to the HITL promotion gate. v3 hard-blocks the premature-promotion failure mode observed in the v2 run. Amended 2026-08-03: the config must first be re-based to the current parameter space (annualized_return_v2; realized_pnl replaced, win_rate diagnostic-only) and all gate evidence must be scored under the post-revamp composite.

## Pre-flight checks (ABORT on any failure)

| # | Check | Pass criteria | On fail |
|---|---|---|---|
| 1 | `local_chain_reachable` | btcli chain-info (or equivalent SDK call) returns a block within the last 60 seconds. | ABORT. Do not fall back to offline/harness projections. §9 requires online mode. |
| 2 | `v14_r1_config_loaded` | read_memory(v14_r1_corrected_config) returns a non-null config dict. | ABORT. The config is the substrate for the entire verification; without it nothing can be verified. |
| 3 | `config_matches_current_parameter_space` | The loaded config dict has exactly 8 trading-weight keys including trading_annualized_return, and contains neither trading_realized_pnl nor trading_win_rate. | ABORT with status=ABORTED_CONFIG_PARAMETER_SPACE_STALE. File a config re-basing task (drop realized_pnl + win_rate, renormalize the surviving weights x1/0.94, re-persist under config_id V14-R1-CORRECTED-KP) and re-dispatch after it completes. Do NOT verify or promote the stale-space config. |
| 4 | `namespace_writable` | A no-op mongodb_insert_one into audit_log with the RUN_NAMESPACE succeeds. | ABORT. If the namespace isn't writable, the verification cannot persist evidence under the correct procedure. |

## Hard blocks (orchestrator/coder MUST NOT do these)

### `no_projection_verdicts`
**Rule:** Gate verdicts MUST be PASS, FAIL, or INSUFFICIENT_EVIDENCE. The strings 'PASS (projected)', 'PASS (conditional)', 'PASS (inferred from baseline)', 'PASS (with caveats)' are FORBIDDEN.
**Enforcement:** Any agent producing a projection verdict must re-issue it as INSUFFICIENT_EVIDENCE with the missing-evidence list.

### `no_criterion_met_writes_without_evidence`
**Rule:** mongodb_insert_one / mongodb_update_one into convergence_metrics, sentinel_state, audit_log with criterion_met=true or status=PROMOTED for V14-R1 is FORBIDDEN until the EVIDENCE_REQUIREMENTS for all 6 gates are satisfied with V14-R1 documents.
**Enforcement:** The coder's HITL task must read simulation_epochs (>= 2 V14-R1 docs), sentinel_state (>= 1 V14-R1 doc), convergence_metrics (NOT pre-existing) BEFORE any promotion write. If any prerequisite is missing, the HITL task returns BLOCKED, not PROMOTED_WITH_PENDING_BTCLI.

### `no_namespace_leak`
**Rule:** V14-R1 verification writes (mongodb_insert_one / mongodb_update_one) MUST use procedure='v14_r1_online_gate_check'. Writes with procedure='insignia_subnet_tuner' are FORBIDDEN for this run.
**Enforcement:** The coder's mongodb_insert_one calls must include the RUN_NAMESPACE fields. Any write missing them, or using the tuner procedure, must be rejected by the orchestrator's MCP layer (treat as a guardrail, same as NAMESPACE_FILTER_REQUIRED).

### `no_offline_fallback_for_online_gates`
**Rule:** If local_chain_reachable pre-flight fails, ABORT the run. Do not produce verdicts from offline/harness data + V13-R3 projections.
**Enforcement:** The orchestrator's first action is the pre-flight. On fail, write agent_memory v14_r1_online_verification_result with status=ABORTED_CHAIN_UNREACHABLE and exit. No gate verdicts, no HITL dispatch.

### `reviewer_veto_honored`
**Rule:** If the reviewer's audit returns INSUFFICIENT_EVIDENCE or FAIL for any gate, the orchestrator's summary MUST reflect that verdict. The orchestrator MUST NOT override the reviewer with 'PASS (projected)'.
**Enforcement:** The orchestrator's summary table must match the reviewer's per-gate verdicts. Disagreement triggers a correction task targeting the discrepant gate, not a promotion.

### `no_promotion_without_git_push`
**Rule:** The coder MUST NOT write agent_memory with status=PROMOTED or status=PROMOTED_WITH_PENDING_BTCLI unless the parameter_space.py change is actually pushed to the git branch.
**Enforcement:** If GitHub push tools are unavailable, the coder writes status=BLOCKED_NO_GIT_PUSH with the diff description, and the orchestrator files a follow-up task for manual push. PENDING_GITHUB_PUSH is a TODO, not a promotion.

### `no_stale_parameter_space_promotion`
**Rule:** The coder MUST NOT push a parameter_space.py promotion diff, set current_candidate_status=promoted_*, or apply btcli hyperparameters for a config expressed in the pre-2026-08-03 trading-weight space (trading_realized_pnl / trading_win_rate present, or trading_annualized_return absent). Promotion applies only to the re-based config.
**Enforcement:** If the config fails the config_matches_current_parameter_space pre-flight, the HITL promotion task returns BLOCKED_CONFIG_STALE and the orchestrator files a config re-basing task instead of promoting. btcli apply with stale-space weights is a chain-mutating write of a mis-scored production config.

## 6 Online-Mode Gates with V14-R1 Evidence Requirements

| # | Gate | Threshold | Required collection | Required field | Min docs | Forbidden evidence |
|---|---|---|---|---|---|---|
| 1 | `honest_mean_score` | 0.97 | `simulation_epochs` | `honest_mean_score` | 2 | V13-R3 simulation_epochs; offline KRET file references; separation-based projections; pre-2026-08-03 evidence scored under the realized_pnl/win_rate composite |
| 2 | `score_variance` | 0.002 | `simulation_epochs` | `honest_score_variance` | 2 | V13-R3 simulation_epochs; projections; pre-2026-08-03 evidence scored under the realized_pnl/win_rate composite |
| 3 | `commit_reveal_effectiveness` | 0.667 | `simulation_epochs` | `cr_effectiveness` | 2 | V13-R3 sentinel_state cr_effectiveness; agent_memory sentinel_state from V13-R3; pre-2026-08-03 evidence scored under the realized_pnl/win_rate composite |
| 4 | `consecutive_clean_validations` | 6 | `sentinel_state` | `consecutive_evals_below_threshold` | 1 | V13-R3 sentinel_state; agent_memory sentinel_state from V13-R3; pre-2026-08-03 evidence scored under the realized_pnl/win_rate composite |
| 5 | `convergence_contract` | unanimously_met | `convergence_metrics` | `criterion_met` | 1 | V13-R3 convergence_metrics (criterion_met=false); offline summary claims; pre-2026-08-03 evidence scored under the realized_pnl/win_rate composite |
| 6 | `sentinel_posture` | SECURE_AND_IMPROVING | `sentinel_state` | `security_status` | 1 | V13-R3 sentinel_state; agent_memory sentinel_state from V13-R3; pre-2026-08-03 evidence scored under the realized_pnl/win_rate composite |

## Evidence scoring schema (2026-08-03)

Every V14-R1 evidence document must carry scoring_schema='annualized_return_v2' (composite scored under the 2026-08-03 metric revamp: annualized_return headline, win_rate diagnostic-only). Documents without the tag, or produced under the pre-revamp realized_pnl/win_rate composite, do NOT count toward any gate's min_documents.

Applies to collections: simulation_epochs, sentinel_state, convergence_metrics. Source: subnet/CHANGELOG.md 2026-08-03, trading metric revamp.

## Acceptance Criteria

- **Reruns required:** 2 (different seeds for each rerun)
- **All gates must clear:** True
- **Evidence must be V14-R1:** True
- **No projection verdicts:** True
- **If any gate fails:** Do NOT promote V14-R1. File a correction task for the failing gate's root cause. Do NOT write criterion_met=true to MongoDB.
- **If all gates clear:** Proceed to HITL promotion gate: (1) push the parameter_space.py change to the git branch (current_candidate_status=promoted_to_production_reference), (2) btcli hyperparameter apply with the V14-R1-CORRECTED-KP config, (3) write the promotion record to audit_log with the RUN_NAMESPACE. All three steps must complete; PENDING_GITHUB_PUSH is not a promotion.

## MCP Dispatch Commands (v3, pre-flight + write-block + namespace lock)

**Cannot be executed from this repo**, the insignia-local MCP server is not available in this environment. Execute in the orchestrator's agent env.

### step1_file_task

**MCP tool:** `insignia-local.file_task`

```json
{
  "assignee": "orchestrator",
  "priority": 10,
  "description": "V14-R1 online-mode gate verification (v3 \u2014 hard-blocks premature promotion): run the live V14-R1-CORRECTED-KP config on-chain across >= 2 reruns with different seeds. PRE-FLIGHT: abort if local chain unreachable; abort with ABORTED_CONFIG_PARAMETER_SPACE_STALE if the config is still in the pre-2026-08-03 trading-weight space (trading_realized_pnl / trading_win_rate), file a re-basing task instead of verifying. EVIDENCE: each gate must cite a V14-R1 MongoDB document tagged scoring_schema='annualized_return_v2' (V13-R3 projections and pre-revamp composite evidence forbidden). PROMOTION WRITE-BLOCK: do not write criterion_met=true until >= 2 V14-R1 simulation_epochs + V14-R1 sentinel_state are persisted. NAMESPACE: every write uses {'playbook': 'insignia_subnet_online_verification', 'domain': 'v14_r1', 'procedure': 'v14_r1_online_gate_check'} (procedure=insignia_subnet_tuner is FORBIDDEN). REVIEWER VETO: orchestrator summary must match reviewer verdicts.",
  "metadata": {
    "cycle_step": "5_to_HITL_v3",
    "config_id": "V14-R1-CORRECTED-KP",
    "manifest_version": "v3",
    "namespace": {
      "playbook": "insignia_subnet_online_verification",
      "domain": "v14_r1",
      "procedure": "v14_r1_online_gate_check"
    },
    "forbidden_collections": [
      "simulation_results"
    ],
    "preflight_checks": [
      "local_chain_reachable",
      "v14_r1_config_loaded",
      "config_matches_current_parameter_space",
      "namespace_writable"
    ],
    "hard_blocks": [
      "no_projection_verdicts",
      "no_criterion_met_writes_without_evidence",
      "no_namespace_leak",
      "no_offline_fallback_for_online_gates",
      "reviewer_veto_honored",
      "no_promotion_without_git_push",
      "no_stale_parameter_space_promotion"
    ],
    "v2_postmortem_doc": "results/v14_r1_v2_run_postmortem_2026-07-07.md"
  }
}
```

### step2_write_agent_memory_dispatch

**MCP tool:** `insignia-local.write_agent_memory`

```json
{
  "key": "v14_r1_online_verification_dispatch",
  "value": {
    "status": "DISPATCHED_V3",
    "manifest_version": "v3",
    "timestamp": "2026-07-07T20:53:46.970652+00:00",
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
    "preflight_checks": [
      "local_chain_reachable",
      "v14_r1_config_loaded",
      "config_matches_current_parameter_space",
      "namespace_writable"
    ],
    "hard_blocks": [
      "no_projection_verdicts",
      "no_criterion_met_writes_without_evidence",
      "no_namespace_leak",
      "no_offline_fallback_for_online_gates",
      "reviewer_veto_honored",
      "no_promotion_without_git_push",
      "no_stale_parameter_space_promotion"
    ],
    "reruns_required": 2,
    "gates_to_verify": [
      "honest_mean_score",
      "score_variance",
      "commit_reveal_effectiveness",
      "consecutive_clean_validations",
      "convergence_contract",
      "sentinel_posture"
    ],
    "v2_postmortem": "Premature promotion in v2; v3 hard-blocks recurrence."
  }
}
```

### step3_invalidate_v2_result

**MCP tool:** `insignia-local.write_agent_memory`

```json
{
  "key": "v14_r1_online_verification_result",
  "value": null,
  "note": "Clear the v2 result (GATES_VERIFIED_WITH_CAVEATS / PROMOTED_WITH_PENDING_BTCLI) so the orchestrator starts v3 fresh."
}
```

### step4_quarantine_v2_premature_writes

**MCP tool:** `insignia-local.write_agent_memory`

```json
{
  "key": "v14_r1_v2_premature_writes_to_quarantine",
  "value": {
    "reason": "v2 persisted criterion_met=true / SECURE_AND_IMPROVING for V14-R1 without V14-R1 runs backing them (procedure=insignia_subnet_tuner, namespace leak).",
    "collections_to_quarantine": [
      "convergence_metrics (V14-R1 documents with procedure=insignia_subnet_tuner)",
      "sentinel_state (V14-R1 documents with procedure=insignia_subnet_tuner)",
      "audit_log (V14-R1 hitl_promotion_btcli_apply_pending events)"
    ],
    "action": "Re-tag with verification_mode=offline_harness_fallback_projected, criterion_met=false, OR delete. Do NOT treat as V14-R1 evidence in v3."
  }
}
```

## Post-Verification Actions

### On success (all 6 gates clear against V14-R1 evidence)

1. All 6 gates PASS against V14-R1 MongoDB evidence tagged scoring_schema='annualized_return_v2' (not projections, not pre-revamp composite runs).
2. >= 2 V14-R1 simulation_epochs documents persisted with distinct seeds.
3. >= 1 V14-R1 sentinel_state document persisted with security_status=SECURE_AND_IMPROVING+ and consecutive_evals_below_threshold >= 6.
4. >= 1 V14-R1 convergence_metrics document persisted with criterion_met=true (this is the evidence, written LAST).
5. parameter_space.py change PUSHED to git branch (not PENDING_GITHUB_PUSH), expressing the re-based V14-R1-CORRECTED-KP values in the current 8 trading-weight (annualized_return) space.
6. btcli hyperparameter apply completed (not pending) with the re-based config.
7. Write agent_memory v14_r1_online_verification_result with status=ALL_GATES_CLEARED_V14_R1_EVIDENCE.

### On failure (any gate fails or INSUFFICIENT_EVIDENCE)

1. Write agent_memory v14_r1_online_verification_result with status=GATES_FAILED or INSUFFICIENT_EVIDENCE and the per-gate verdicts.
2. Do NOT write criterion_met=true to any V14-R1 document.
3. Do NOT mark parameter_space.py as promoted.
4. File a correction task for the failing gate's root cause.

### On pre-flight abort (chain unreachable, etc.)

1. Write agent_memory v14_r1_online_verification_result with status=ABORTED_CHAIN_UNREACHABLE (or ABORTED_<preflight_id>).
2. Do NOT evaluate any gate.
3. Do NOT dispatch the HITL promotion task.
4. File a task to restore the local chain, then re-dispatch v3.
5. On config_matches_current_parameter_space failure: status=ABORTED_CONFIG_PARAMETER_SPACE_STALE; file a config re-basing task (renormalize surviving trading weights x1/0.94 into the annualized_return space, re-persist under config_id V14-R1-CORRECTED-KP), then re-dispatch v3 after it completes.

_JSON manifest: `v14_r1_online_dispatch_manifest_v3_2026-07-07T20-53-46.json`_

_v2 post-mortem: `results/v14_r1_v2_run_postmortem_2026-07-07.md`_