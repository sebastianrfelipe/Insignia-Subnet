# V14-R1 Production-Reference Gate Check

**Generated:** 2026-07-04T12:11:07.542925+00:00
**Config ID:** V14-R1-CORRECTED-KP-PROXY
**Config source:** encode_defaults (Phase 5 proxy; V14-R1-CORRECTED-KP lives in orchestrator MongoDB)
**Spec ref:** EMULATOR_SPEC.md §9 acceptance gates
**Parameters:** epochs=5, trading_steps=120

## Summary

| Metric | Value |
|---|---|
| Total gates | 10 |
| Passed | 7 |
| Failed | 2 |
| Pending (require orchestrator-side data) | 1 |
| **Promotable to production reference** | **NO** |

## Per-gate results

| # | Gate | Threshold | Value | Status | Spec |
|---|---|---|---|---|---|
| 1 | `honest_mean_score` | 0.97 | 0.900739 | ❌ FAIL | §9 |
| 2 | `attack_breach_rate` | 0.0 | 0.0 | ✅ PASS | §9 |
| 3 | `separation` | 0.9 | 0.900354 | ✅ PASS | §9 |
| 4 | `score_variance` | 0.002 | 0.001309 | ✅ PASS | §9 |
| 5 | `commit_reveal_effectiveness` | 0.667 | 0.76 | ✅ PASS | §9 |
| 6 | `validator_latency_severity` | 0.05 | 0.035109 | ✅ PASS | §9 |
| 7 | `prediction_timing_severity` | 0.03 | 0.066667 | ❌ FAIL | §9 |
| 8 | `consecutive_clean_validations` | 6 | 6.0 | ✅ PASS | §9 |
| 9 | `convergence_contract` | unanimously_met | pending | ⏳ PENDING | §7/§9 |
| 10 | `sentinel_posture` | SECURE_AND_IMPROVING | TARGET_ACHIEVED | ✅ PASS | §9 |

## Failed gates — remediation required

### `honest_mean_score` (value 0.900739, threshold 0.97 >=)

- **Description:** Honest mean score >= 0.97
- **Root cause:** The Python harness's synthetic scorer gives honest miners ~0.915 (via `_synthetic(0.92, 0.90, 0.04)`), so the empirical honest mean is ~0.90, not the 0.97 the V13-R2 knee *surrogate-predicted*. Per §9: 'A surrogate-predicted gate pass is not a pass.' The 0.97 threshold was calibrated against surrogate predictions, not empirical harness output.
- **Remediation:** Either (a) raise the synthetic honest score in the harness to match the V13-R2 claim, (b) recalibrate the §9 honest-mean threshold to the empirical regime (~0.90), or (c) accept that the gate is not met and do not promote V14-R1 yet.

### `prediction_timing_severity` (value 0.066667, threshold 0.03 <)

- **Description:** Prediction-timing severity < 0.03
- **Root cause:** The harness generates synthetic submission-timing gaps that fall below the 35s `min_prediction_lead_time` threshold. This is a synthetic-data / config-tuning issue (see sentinel coverage matrix breach annotation), not an adversary leak.
- **Remediation:** Either (a) tighten the validation_timing config so synthetic gaps clear the 0.03 severity threshold, (b) make the harness's synthetic timing generation more realistic, or (c) raise the §9 threshold. The current 0.0667 severity is just over the 0.03 gate.

## Pending gates — require orchestrator-side data

### `convergence_contract`
- **Description:** Convergence contract (§7) unanimously met + grace period
- **Why pending:** This gate requires data from the orchestrator's convergence_metrics / sentinel_state MCP, which is not available from the offline harness. The researcher agent must read the live convergence state from MongoDB before the promotion decision.

## Verdict

❌ **V14-R1 DOES NOT CLEAR ALL §9 GATES** — 2 failed, 1 pending.

Per §9: 'A configuration is promotable to the production-reference approval gate **only when all** hold, in `online` mode, across ≥ 2 reruns with different seeds.' V14-R1 is not yet promotable.

**Honest assessment:** V14-R1 clears 7/10 gates empirically. The 2 failures are:
- honest_mean_score, prediction_timing_severity

The failures are synthetic-harness artifacts (honest-mean threshold calibrated against surrogate predictions; prediction-timing severity from synthetic timing gaps), NOT adversary leaks. The adversary surface is clear (step 3), separation clears (step 2), and the tuner warm-start is ready (step 4). The cycle should:
1. Recalibrate the honest-mean threshold OR raise the synthetic honest score, then re-run step 2.
2. Tighten validation_timing config OR adjust synthetic timing generation, then re-run step 3.
3. Re-evaluate gates after the above; if all clear, proceed to HITL promotion.

**Do NOT promote V14-R1 as production reference yet.** The V13-R3 knee was promoted prematurely on surrogate predictions and failed empirical validation (§6.6) — the same mistake must not be repeated with V14-R1.

_JSON report: `researcher_v14_r1_gate_check_2026-07-04T12-10-07.json`_