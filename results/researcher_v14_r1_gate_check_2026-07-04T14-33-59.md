# V14-R1 Production-Reference Gate Check

**Generated:** 2026-07-04T14:34:48.644500+00:00
**Config ID:** V14-R1-CORRECTED-KP-PROXY
**Config source:** encode_defaults (Phase 5 proxy; V14-R1-CORRECTED-KP lives in orchestrator MongoDB)
**Spec ref:** EMULATOR_SPEC.md §9 acceptance gates
**Parameters:** epochs=5, trading_steps=120

## Summary

| Metric | Value |
|---|---|
| Total gates | 10 |
| Harness-mode gates (offline-checkable) | 4 |
| Online-mode gates (require live chain per §9) | 6 |
| Passed (harness-mode) | 4 |
| Failed (harness-mode) | 0 |
| Pending (online-mode, require live chain) | 6 |
| **Promotable to production reference** | **NO, requires online verification** |

## Per-gate results

| # | Gate | Mode | Threshold | Value | Status | Spec |
|---|---|---|---|---|---|---|
| 1 | `honest_mean_score` | online | 0.97 | 0.900739 | ⏳ PENDING | §9 |
| 2 | `attack_breach_rate` | harness | 0.0 | 0.0 | ✅ PASS | §9 |
| 3 | `separation` | harness | 0.9 | 0.900354 | ✅ PASS | §9 |
| 4 | `score_variance` | online | 0.002 | 0.001309 | ⏳ PENDING | §9 |
| 5 | `commit_reveal_effectiveness` | online | 0.667 | 0.76 | ⏳ PENDING | §9 |
| 6 | `validator_latency_severity` | harness | 0.05 | 0.035109 | ✅ PASS | §9 |
| 7 | `prediction_timing_severity` | harness | 0.03 | 0.025 | ✅ PASS | §9 |
| 8 | `consecutive_clean_validations` | online | 6 | 6.0 | ⏳ PENDING | §9 |
| 9 | `convergence_contract` | online | unanimously_met | pending | ⏳ PENDING | §7/§9 |
| 10 | `sentinel_posture` | online | SECURE_AND_IMPROVING | TARGET_ACHIEVED | ⏳ PENDING | §9 |

## Pending online-mode gates, require live chain verification per §9

Per §9: "A configuration is promotable to the production-reference approval gate **only when all** hold, in `online` mode, across ≥ 2 reruns with different seeds." The offline harness cannot verify these gates, they require live chain data.

### `honest_mean_score` (synthetic value 0.900739, threshold 0.97)
- **Description:** Honest mean score >= 0.97 (online mode)
- **Note:** Online-mode gate per §9, requires live chain data, not offline harness.

### `score_variance` (synthetic value 0.001309, threshold 0.002)
- **Description:** Honest score variance <= 0.002 (online mode)
- **Note:** Online-mode gate per §9, requires live chain data, not offline harness.

### `commit_reveal_effectiveness` (synthetic value 0.76, threshold 0.667)
- **Description:** Commit-reveal effectiveness >= 0.667 floor (>= 0.76 target) (online mode)
- **Note:** Online-mode gate per §9, requires live chain data, not offline harness.

### `consecutive_clean_validations` (synthetic value 6.0, threshold 6)
- **Description:** Consecutive clean validations >= 6 (online mode)
- **Note:** Online-mode gate per §9, requires live chain data, not offline harness.

### `convergence_contract` (synthetic value pending, threshold unanimously_met)
- **Description:** Convergence contract (§7) unanimously met + grace period (online mode)
- **Note:** Online-mode gate per §9, requires live chain data, not offline harness.

### `sentinel_posture` (synthetic value TARGET_ACHIEVED, threshold SECURE_AND_IMPROVING)
- **Description:** Sentinel posture SECURE_AND_IMPROVING or stronger (online mode)
- **Note:** Online-mode gate per §9, requires live chain data, not offline harness.

## Verdict

✅ **V14-R1 harness-mode gates: 4/4 passed, 0 failed. Online-mode gates: 6 pending (require live chain).**

All offline-checkable gates pass. The 6 pending gates require live chain verification per §9 ("in `online` mode, across ≥ 2 reruns with different seeds") before V14-R1 can be promoted to production reference.

**Next step:** Re-dispatch the orchestrator with the live V14-R1-CORRECTED-KP config from MongoDB to verify the 6 online-mode gates on-chain. If all clear across ≥ 2 reruns, proceed to HITL promotion.

_JSON report: `researcher_v14_r1_gate_check_2026-07-04T14-33-59.json`_