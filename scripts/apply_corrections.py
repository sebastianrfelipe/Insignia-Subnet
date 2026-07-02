#!/usr/bin/env python3
"""Apply corrected ground-truth to MCP state.

#5 — Overwrite the two poisoned memory keys:
   - emulator_spec_execution_state  (cited hallucinated computeAgentScore + surrogate)
   - v14_r1_corrected_config        (add grounded_root_cause + known_false_leads)

#10 — File a new corrected coder task that names SimulationHarness.run
      as the real target and tells the coder NOT to search for computeAgentScore.

Re-runnable. Idempotent: memory upserts use {key: ...} filter; task insert
is guarded by a title check (won't re-insert if a task with the same title
already exists and is pending).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Reuse the MCP HTTP client from sync_documents.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sync_documents import mcp_initialize, mcp_call, _SESSION_ID  # noqa: E402

COLLECTION_MEMORY = "agent_memory"
COLLECTION_TASKS = "tasks"

NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------------------------------------------------------------------------
# #5a — Corrected emulator_spec_execution_state
# ---------------------------------------------------------------------------

EMULATOR_SPEC_EXECUTION_STATE = {
    "session": "EMULATOR_SPEC_execution_2026_07_02_v3_corrected",
    "phase": "v14_r1_code_fix_and_chain_reverify_grounded",
    "timestamp": NOW,
    "grounded_root_cause": (
        "V13-R3 failed empirical validation because SimulationHarness.run in "
        "subnet/tuning/simulation.py (scoring inline around lines 860-930) "
        "applies _scaled(multiplier=0.10) only to CopycatMiner and CopyTrader. "
        "SybilMiner, OverfittingMiner, SingleMetricGamer, and PartnerGamer have "
        "NO penalty path. SybilMiner scores HIGHER than honest (0.9163 vs 0.9151)."
    ),
    "known_false_leads": [
        ("computeAgentScore() - does NOT exist. Verified absent: "
         "mongodb_find({content:{$regex:'computeAgentScore'}}) on documents "
         "collection returns 0 source matches (only CHANGELOG.md mentions it)."),
        ("GP surrogate - does NOT exist. optimizer.py InsigniaTuningProblem."
         "_evaluate (lines 172-213) calls SimulationHarness.run() directly. "
         "No surrogate model in the loop."),
        "surrogate_separation=0.963 - fabricated. There is no surrogate to produce this number.",
    ],
    "code_location": {
        "file": "subnet/tuning/simulation.py",
        "class": "SimulationHarness",
        "method": "run",
        "scoring_lines": "860-930",
        "existing_penalties": ["CopycatMiner (0.10x)", "CopyTrader (0.10x)"],
        "missing_penalties": ["SybilMiner", "OverfittingMiner", "SingleMetricGamer", "PartnerGamer"],
    },
    "documents_collection_synced": True,
    "documents_sync_note": (
        "subnet/ source files are mirrored in the documents collection "
        "(sync_source=filesystem-sync-script-v1). Use "
        "mongodb_find({path:'subnet/tuning/simulation.py'}) to read the real code."
    ),
    "dispatched_tasks": {
        "coder_anti_gaming_fix_v2": {
            "status": "pending",
            "description": (
                "Add penalty paths for SybilMiner, OverfittingMiner, "
                "SingleMetricGamer, PartnerGamer in SimulationHarness.run. "
                "DO NOT search for computeAgentScore - it does not exist."
            ),
            "note": (
                "Previous task 6a45b0fbd163a13dc7ec9371 hit max_tool_rounds "
                "searching for non-existent computeAgentScore. This task replaces it."
            ),
        },
        "deployer_chain_reverify": {
            "taskId": "6a45b0fbd163a13dc7ec9373",
            "status": "completed",
            "result": (
                "LOCAL_CHAIN_STILL_UNREACHABLE - V14-R1 online validation BLOCKED. "
                "Host 10.0.0.193 not pingable. Proceed offline-only."
            ),
        },
    },
    "execution_plan": {
        "step1_coder": "Add 4 missing penalty paths in SimulationHarness.run (PENDING - new task filed)",
        "step1b_deployer": "Re-verify chain readiness (COMPLETED - chain unreachable, ops required)",
        "step2_simulator": "Run empirical validation of V14-R1 with fixed code (PENDING - after coder)",
        "step3_sentinel": "Re-evaluate 19 attack vectors (PENDING - after simulator)",
        "step4_tuner": "Update convergence state, fold fitness into NSGA-II (PENDING - after sentinel)",
        "step5_researcher": "Assess convergence, prepare production-reference if gates clear (PENDING - after tuner)",
    },
    "hitl_gated_actions": ["btcli hyperparameter apply", "promote converged config as production reference"],
    "last_not_contradicted_checkpoint": (
        "V13-R2-KP-020-a7f2 (honest=0.9795, breach=3.5e-6, separation=0.953) - "
        "NOTE: researcher warns this would ALSO fail empirical validation due to "
        "same structural code gap"
    ),
    "v13_r3_validation_summary": {
        "seed_1042": {"honest": 0.9774, "separation": 0.2441, "gates_pass": False,
                      "failing_gate": "score_separation (0.2441 < 0.90)"},
        "seed_2042": {"honest": 0.9782, "separation": 0.2226, "gates_pass": False,
                      "failing_gate": "score_separation (0.2226 < 0.90)"},
    },
    "chain_status": (
        "LOCAL_CHAIN_UNREACHABLE - V14-R1 online validation BLOCKED. "
        "Proceed offline-only per deployer recommendation."
    ),
}


# ---------------------------------------------------------------------------
# #10 — New corrected coder task
# ---------------------------------------------------------------------------

CODER_TASK_TITLE = (
    "Add anti-gaming penalty paths for 4 adversary types in SimulationHarness.run "
    "(DO NOT search for computeAgentScore)"
)

CODER_TASK_DESCRIPTION = """## CORRECTED CODE FIX - Root Cause of V13-R3 Divergence (grounded)

This task supersedes the previous task `6a45b0fbd163a13dc7ec9371`, which hit
`max_tool_rounds` after 37 calls searching for a function named
`computeAgentScore()`. **That function does not exist.** Verified absent:
`mongodb_find({content:{$regex:'computeAgentScore'}})` on the `documents`
collection returns 0 source-code matches (only `CHANGELOG.md` mentions it).

### Read the real code first

The `subnet/` source tree is now mirrored in the `documents` collection
(sync_source=`filesystem-sync-script-v1`). Start by reading the actual file:

```
mongodb_find({path: 'subnet/tuning/simulation.py'}, {projection: {path:1, content:1, _id:0}, limit: 1, detail_level: 'L2'})
```

Or use `get_details` with `resource_type: mongodb_document` and the doc id.

### The real target

The scoring lives in **`SimulationHarness.run`** in `subnet/tuning/simulation.py`
(inline scoring around lines 860-930 - not a separate function). It currently
applies `_scaled(multiplier=0.10)` to exactly two adversary types:

- `CopycatMiner` - `model_eff = _scaled(model_eff, 0.10)`
- `CopyTrader`   - `trading_eff = _scaled(trading_eff, 0.10)`

The other 4 adversary types have **NO penalty path** and score ~0.90 (same as
honest). `SybilMiner` actually scores HIGHER than honest (0.9163 vs 0.9151)
because the harness computes `sybil_pressure` and `ensemble_signals` but never
feeds them back into `miner_scores`.

### What to add

Extend the inline scoring in `SimulationHarness.run` so every adversary
archetype from EMULATOR_SPEC §5.1/§5.2 has a corresponding penalty path:

1. **SybilMiner** - penalty scaled by `sybil_pressure` / `ensemble_signals`
   (already computed, not yet fed back). Target: strictly below honest mean.
2. **OverfittingMiner** - penalty from IS/OOS gap vs `overfitting.gap_threshold`
   and `overfitting.decay_rate` from the config. Target: strictly below honest mean.
3. **SingleMetricGamer** - penalty for single-metric deviation from the 7-metric
   composite. Target: strictly below honest mean.
4. **PartnerGamer** - collusion/partner-selection penalty via variance-penalized
   marginal-contribution credit. Target: strictly below honest mean.
5. **CopycatMiner** (existing) - verify the 0.10x multiplier is still applied.
6. **CopyTrader** (existing) - verify the 0.10x multiplier is still applied.

### Acceptance gate

- `tests/test_simulation_separation.py::test_no_adversary_outscores_honest_mean`
  must pass: no adversary type may score higher than the honest mean.
- `tests/test_simulation_separation.py::test_harness_separation_meets_gate` must
  pass: honest-vs-adversary separation >= 0.90 across seeds 1042 and 2042.
- Current separation: **0.22** (seed 2042). Target: **>= 0.90**.

### Implementation notes

- The V14-R1-CORRECTED-KP config is in memory key `v14_r1_corrected_config`. Its
  `anti_gaming` block has `gaming_detection_sensitivity=0.92`,
  `penalty_escalation_rate=0.85`, `cross_metric_correlation_threshold=0.45`.
  Use these config values, not hardcoded constants.
- Per §6.6: "Each adversary archetype in §5.1/§5.2 must have a corresponding
  penalty path in `simulation.py`'s scoring loop."
- Per §6.3: penalties must be structural (the mechanism itself prevents gaming),
  not just post-hoc score reduction.
- Create a PR at `sebastianrfelipe/Insignia-Subnet` with the changes.
- All MongoDB writes must include `procedure="insignia_subnet_tuner"`.

### DO NOT

- Do NOT search for `computeAgentScore` - it does not exist.
- Do NOT search for a "GP surrogate" or "surrogate model" - none exists;
  `optimizer.py` calls `SimulationHarness.run()` directly.
- Do NOT re-derive the root cause from memory keys - read the actual source from
  the `documents` collection first.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def upsert_memory(key: str, value_obj: dict, agent_type: str, written_by: str) -> None:
    value_str = json.dumps(value_obj, ensure_ascii=False)
    print(f"  upserting memory key={key!r} ({len(value_str)} chars)...")
    res = mcp_call("mongodb_update_one", {
        "collection": COLLECTION_MEMORY,
        "filter": {"key": key},
        "update": {"$set": {
            "key": key,
            "value": value_str,
            "agentType": agent_type,
            "updatedAt": NOW,
            "writtenBy": written_by,
        }},
        "upsert": True,
    })
    print(f"    -> {res}")


def find_pending_task_by_title(title: str) -> bool:
    """Return True if a task with this title already exists (any status)."""
    try:
        res = mcp_call("mongodb_find", {
            "collection": COLLECTION_TASKS,
            "filter": {"title": title},
            "projection": {"_id": 0, "title": 1, "status": 1},
            "limit": 1,
            "detail_level": "L1",
        })
        items = res.get("items", []) if isinstance(res, dict) else []
        return len(items) > 0
    except Exception as e:
        print(f"    find_existing warn: {e}", file=sys.stderr)
        return False


def insert_task(title: str, description: str, assigned_to: str,
                priority: int, created_by: str) -> None:
    print(f"  inserting task: {title!r}...")
    res = mcp_call("mongodb_insert_one", {
        "collection": COLLECTION_TASKS,
        "document": {
            "title": title,
            "description": description,
            "assignedTo": assigned_to,
            "priority": priority,
            "createdBy": created_by,
            "status": "pending",
            "createdAt": NOW,
            "updatedAt": NOW,
            "iterations": [],
        },
    })
    print(f"    -> {res}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("Applying corrected ground-truth to MCP state...\n")
    mcp_initialize()
    print(f"  initialized (session_id={_SESSION_ID or 'none'})\n")

    # #5a — overwrite emulator_spec_execution_state
    print("#5a: overwrite emulator_spec_execution_state")
    upsert_memory(
        "emulator_spec_execution_state",
        EMULATOR_SPEC_EXECUTION_STATE,
        agent_type="orchestrator",
        written_by="fs-sync-script-apply-corrections",
    )
    print()

    # #5b — augment v14_r1_corrected_config with grounded fields (preserve params)
    print("#5b: augment v14_r1_corrected_config with grounded_root_cause + known_false_leads")
    try:
        res = mcp_call("mongodb_find", {
            "collection": COLLECTION_MEMORY,
            "filter": {"key": "v14_r1_corrected_config"},
            "limit": 1,
            "detail_level": "L2",
        })
        docs = res.get("documents", []) if isinstance(res, dict) else []
        if docs:
            existing = json.loads(docs[0]["value"])
            existing["grounded_root_cause"] = EMULATOR_SPEC_EXECUTION_STATE["grounded_root_cause"]
            existing["known_false_leads"] = EMULATOR_SPEC_EXECUTION_STATE["known_false_leads"]
            existing["code_location"] = EMULATOR_SPEC_EXECUTION_STATE["code_location"]
            existing["documents_collection_synced"] = True
            existing["correction_timestamp"] = NOW
            upsert_memory(
                "v14_r1_corrected_config",
                existing,
                agent_type="coder",
                written_by="fs-sync-script-apply-corrections",
            )
        else:
            print("    WARN: v14_r1_corrected_config not found; skipping (no prior value to augment).")
    except Exception as e:
        print(f"    FAILED: {e}", file=sys.stderr)
    print()

    # #10 — file the corrected coder task (guard against duplicate inserts)
    print("#10: file corrected coder task")
    if find_pending_task_by_title(CODER_TASK_TITLE):
        print(f"  SKIP: a task with this title already exists: {CODER_TASK_TITLE!r}")
    else:
        insert_task(
            CODER_TASK_TITLE,
            CODER_TASK_DESCRIPTION,
            assigned_to="coder",
            priority=10,
            created_by="fs-sync-script-apply-corrections",
        )
    print()

    # Verify
    print("Verifying...")
    try:
        res = mcp_call("mongodb_find", {
            "collection": COLLECTION_MEMORY,
            "filter": {"key": "emulator_spec_execution_state"},
            "projection": {"_id": 0, "key": 1, "updatedAt": 1},
            "limit": 1,
            "detail_level": "L1",
        })
        print(f"  emulator_spec_execution_state: {res}")
    except Exception as e:
        print(f"  verify failed: {e}", file=sys.stderr)
    try:
        res = mcp_call("mongodb_find", {
            "collection": COLLECTION_TASKS,
            "filter": {"title": CODER_TASK_TITLE},
            "projection": {"_id": 0, "title": 1, "status": 1, "priority": 1, "assignedTo": 1},
            "limit": 1,
            "detail_level": "L1",
        })
        print(f"  coder_task: {res}")
    except Exception as e:
        print(f"  verify failed: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
