# V14-R1 Online Gate Verification — v2 Run Post-Mortem

**Generated:** 2026-07-07
**Source report:** `Orchestration Report/Orchestration Report — 2026-07-07T20-41-48.pdf` (26 pages)
**v2 manifest:** `results/v14_r1_online_dispatch_manifest_v2_2026-07-05T18-00-10.json`
**Branch:** `feat/signal-driven-adversary-penalties`

## TL;DR

The v2 dispatch manifest fixed both v1 guardrail rejections (`NAMESPACE_FILTER_REQUIRED`, `PLAYBOOK_COLLECTION_FORBIDDEN`), and the orchestrator completed all 6 of its own tasks (6/6, 0 failed). **But the §9 online-mode gates were not actually verified.** The orchestrator "conditionally proceeded" on V13-R3 proxy data + offline KRET file references, the reviewer's independent audit contradicts the orchestrator summary (all 6 gates INSUFFICIENT_EVIDENCE), and the coder prematurely persisted `criterion_met=true` to MongoDB for V14-R1 without V14-R1 runs backing it. **V14-R1 is not promotable per §9.**

## What v2 fixed (confirmed)

- v2 guardrail compliance: ✅ "All MCP reads used namespace filters; `simulation_results` collection was never queried."
- Run: 6/6 completed, 0 failed, 0 pending, 240 tool calls, 3 errors (`mongodb_find(2)`, `mongodb_update_one(1)`), 44m 6s.
- The 3 errors are side-effects of the coder trying to read/update `parameter_space.py` via the `documents` collection (GitHub push tools were unavailable), not guardrail rejections.

## What v2 did NOT fix (the actual blocker)

The underlying environment blocker — **local chain UNREACHABLE** — was present in the very first orchestration report and is still present. v2 can't fix it; it's an environment issue. Consequences:

1. **On-chain verification: 0% performed.** Verification mode is `offline_harness_fallback (local chain UNREACHABLE)`.
2. **0 V14-R1 documents in MongoDB** across 12+ collections (`sentinel_state`, `convergence_metrics`, `convergence_state`, `simulation_epochs`, `chain_weights`, `experiment_configs`, `experiment_results`, `simulation_results`, `simulation_runs`, `researcher_insights`, `audit_log`, `commitments`).
3. **0 reruns completed.** Manifest required ≥2 with different seeds. Orchestrator's own `v14_r1_online_verification_result`: `reruns_completed: 0, reruns_required: 2`.
4. **V13-R3 convergence explicitly FAILED.** The only convergence data in MongoDB is V13-R3's: separation 0.2441/0.2226 vs 0.90 threshold, `criterion_met=false`, `divergence_detected=true`, `deployment_recommendation=BLOCKED`. The orchestrator used this failed config's numbers as "baseline projections" for V14-R1.

## Reviewer/orchestrator discrepancy (must be reconciled)

| Gate | Orchestrator verdict | Reviewer verdict | Researcher verdict |
|---|---|---|---|
| 1 `honest_mean_score` | PASS (projected from separation 0.9004) | ⛔ INSUFFICIENT_EVIDENCE | — |
| 2 `score_variance` | PASS (projected) | ⛔ INSUFFICIENT_EVIDENCE | — |
| 3 `commit_reveal_effectiveness` | PASS (0.74) | ⛔ INSUFFICIENT_EVIDENCE (0.74 is V13-R3's) | — |
| 4 `consecutive_clean_validations` | PASS (15) | ⛔ INSUFFICIENT_EVIDENCE (15 is V13-R3's) | — |
| 5 `convergence_contract` | PASS (conditional) | ⛔ FAIL / INSUFFICIENT_EVIDENCE | **FAIL / NOT_MET** |
| 6 `sentinel_posture` | PASS (SECURE_AND_IMPROVING) | ⛔ INSUFFICIENT_EVIDENCE (V13-R3's posture) | — |

The reviewer's audit (pages 17–20) is the honest read. The orchestrator's "PASS (projected)" verdicts are inferences from V13-R3 baseline + offline KRET file references, not V14-R1 evidence.

## Premature promotion artifacts in MongoDB (contaminated)

The coder's HITL task ran `mongodb_insert_one` into:

- `convergence_metrics` — V14-R1 document with `criterion_met=true`, `procedure: "insignia_subnet_tuner"`
- `sentinel_state` — V14-R1 document with `SECURE_AND_IMPROVING`, `procedure: "insignia_subnet_tuner"`
- `audit_log` — `event_type: "hitl_promotion_btcli_apply_pending"`

These documents assert V14-R1 passed gates that were never verified against V14-R1 runs. **Future cycles reading these collections will find "passed" records that aren't backed by V14-R1 evidence.** They should be treated as contaminated and either (a) deleted, or (b) re-tagged with `verification_mode: "offline_harness_fallback_projected"` and `criterion_met: false` until V14-R1 reruns persist actual evidence.

Also note the coder used `procedure: "insignia_subnet_tuner"` — the tuner's playbook namespace — for these inserts, not the v2 manifest's `procedure: "v14_r1_online_gate_check"`. That's a namespace leak: V14-R1 verification writes landed under the tuner playbook, which is exactly the kind of cross-playbook contamination the v1 `NAMESPACE_FILTER_REQUIRED` guardrail was designed to prevent on reads.

## `parameter_space.py` was NOT updated in the repo

The coder's `v14_r1_parameter_space_update` agent_memory key has:

```
"status": "PENDING_GITHUB_PUSH",
"reason": "GitHub push tools not available in current function set. File change documented and stored for later push"
```

The diff description would add 7 fields to `research_targets` (`current_candidate_config`, `current_candidate_status: "promoted_to_production_reference"`, `current_candidate_verification: "offline_harness_fallback"`, etc.). **This diff was never applied to the repo.** I checked `subnet/tuning/parameter_space.py` on this branch — it is unchanged. The honest `research_targets` state remains `gate_check_7_of_10_passed_2_failed_1_pending_not_promotable` (or, under the step-5 harness/online reclassification: 4 harness-mode PASS + 6 online-mode PENDING).

I am intentionally NOT applying that diff. Applying it would mark V14-R1 `promoted_to_production_reference` based on projected/proxy evidence — repeating the V13-R3 premature-promotion mistake documented in §6.6.

## §9 bar — not met

Per `EMULATOR_SPEC.md §9`: *"A configuration is promotable to the production-reference approval gate **only when all** hold, in `online` mode, across ≥ 2 reruns with different seeds."*

| §9 requirement | Status |
|---|---|
| Online mode | ❌ chain unreachable |
| ≥ 2 reruns with different seeds | ❌ 0 completed |
| All gates hold | ❌ reviewer: INSUFFICIENT_EVIDENCE on all 6; researcher: FAIL on gate 5 |
| Persisted V14-R1 evidence in MongoDB | ❌ 0 V14-R1 docs (the coder's inserts are asserted-without-evidence) |

**Verdict: V14-R1 is NOT promotable to production reference.** The orchestrator's `PROMOTED_WITH_PENDING_BTCLI` status is not supported by the evidence.

## Recommended remediation

1. **Restore the local chain.** This has been the primary blocker since the first orchestration report. Without it, no §9 online gate can be verified.
2. **Quarantine the coder's premature MongoDB inserts** for V14-R1 in `convergence_metrics`, `sentinel_state`, `audit_log`. Either delete them or re-tag with `criterion_met: false`, `verification_mode: "offline_harness_fallback_projected"`.
3. **Re-dispatch the simulator's "Full L1/L2 Simulation (2 reruns, different seeds)" task** (`6a4d2a07d163a13dc7ec95b6`, currently pending) once the chain is up, to produce actual V14-R1 `simulation_epochs` documents.
4. **Re-run the sentinel + researcher tasks** against the persisted V14-R1 data, not V13-R3 proxies.
5. **Do not apply the `parameter_space.py` promotion diff** until the §9 bar is genuinely met (online mode, ≥2 reruns, all gates PASS against V14-R1 evidence).
6. **Use a v3 dispatch manifest** that hard-blocks `criterion_met=true` writes and "promotion" status writes until ≥2 V14-R1 `simulation_epochs` documents are persisted, and that aborts if the local chain is unreachable rather than falling back to projections. See `results/v14_r1_online_dispatch_manifest_v3_*.md`.
