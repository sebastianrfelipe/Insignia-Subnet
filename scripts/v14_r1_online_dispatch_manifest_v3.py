"""V14-R1 online-mode gate verification — orchestrator dispatch manifest v3.

v3 hardens v2 to prevent the premature-promotion failure mode observed in
the v2 run (report 2026-07-07T20-41-48):

  - Orchestrator declared "ALL 6 GATES PASS (with caveats)" and
    "CONDITIONAL PROCEED TO HITL" despite local chain UNREACHABLE, 0 V14-R1
    documents in MongoDB, and 0 reruns completed (manifest required >= 2).
  - Coder persisted `criterion_met=true` to convergence_metrics and
    sentinel_state for V14-R1 without V14-R1 runs backing those records.
  - Coder used `procedure: "insignia_subnet_tuner"` (the tuner's playbook)
    instead of the v2 manifest's `procedure: "v14_r1_online_gate_check"`,
    leaking V14-R1 verification writes into the tuner's namespace.
  - Coder marked parameter_space.py update as PENDING_GITHUB_PUSH; the diff
    was never applied to the repo but the promotion was "documented" in
    agent_memory as if it had been.
  - Reviewer's independent audit said INSUFFICIENT_EVIDENCE on all 6 gates
    and FAIL on gate 5; orchestrator summary overrode it.

v3 changes vs v2:

  1. PRE-FLIGHT GATE: if local chain is UNREACHABLE, ABORT the entire
     verification run. Do not fall back to projections / baseline
     inferences. §9 requires online mode; if online is impossible, the
     run cannot proceed.
  2. EVIDENCE-BACKED VERDICTS: every gate verdict MUST cite a V14-R1
     document in MongoDB (simulation_epochs, sentinel_state,
     convergence_metrics, etc.). Verdicts citing V13-R3 "baseline
     projections" or offline KRET file references are forbidden and must
     be marked INSUFFICIENT_EVIDENCE, not PASS.
  3. PROMOTION WRITE-BLOCK: the coder MUST NOT write criterion_met=true
     to convergence_metrics / sentinel_state / audit_log for V14-R1 until
     ALL of the following are persisted:
       (a) >= 2 V14-R1 documents in simulation_epochs (one per rerun, distinct seeds)
       (b) >= 1 V14-R1 document in sentinel_state with consecutive_evals_below_threshold >= 6
       (c) >= 1 V14-R1 document in convergence_metrics with criterion_met=true
         (this is the *evidence*, not the *claim*)
     The promotion write is the LAST step, not the first.
  4. NAMESPACE LOCK: every V14-R1 verification write (mongodb_insert_one,
     mongodb_update_one) MUST use procedure="v14_r1_online_gate_check",
     playbook="insignia_subnet_online_verification", domain="v14_r1".
     Writes with procedure="insignia_subnet_tuner" are FORBIDDEN for this
     run (that's the tuner's playbook, not the verification playbook).
  5. REVIEWER VETO: if the reviewer's audit returns INSUFFICIENT_EVIDENCE
     or FAIL for any gate, the orchestrator's summary MUST NOT override it
     with "PASS (projected)". The orchestrator summary must reflect the
     reviewer's verdicts. Disagreement triggers a correction task, not a
     promotion.
  6. NO PROJECTION VERDICTS: the strings "PASS (projected)",
     "PASS (conditional)", "PASS (inferred from baseline)" are forbidden.
     A gate is PASS, FAIL, or INSUFFICIENT_EVIDENCE — nothing else.
  7. PARAMETER_SPACE.PY PROTECTION: the coder MUST NOT mark
     parameter_space.py as updated/PROMOTED unless the file change is
     actually pushed to the git branch. agent_memory records with
     status="PENDING_GITHUB_PUSH" MUST NOT be interpreted as promotion;
     they are TODO items.

This script CANNOT dispatch the orchestrator from this repo — the
insignia-local MCP server is not available in this environment. It writes
the v3 manifest to results/ for manual execution in the orchestrator env.

Usage:
    python scripts/v14_r1_online_dispatch_manifest_v3.py [--output results/]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List


# Namespace for this verification run. Locked — every read AND write must
# use these fields. Writes with procedure="insignia_subnet_tuner" are
# forbidden (that's the tuner's playbook).
RUN_NAMESPACE: Dict[str, str] = {
    "playbook": "insignia_subnet_online_verification",
    "domain": "v14_r1",
    "procedure": "v14_r1_online_gate_check",
}

# Forbidden collections for direct mongodb_find (same as v2).
FORBIDDEN_COLLECTIONS: List[str] = [
    "simulation_results",  # belongs to "Insignia subnet tuner" playbook
]

# Filesystem KRET artifacts (offline evidence — usable as CONTEXT only,
# NOT as primary evidence for gate verdicts in v3).
FILESYSTEM_KRET_ARTIFACTS: Dict[str, str] = {
    "simulator_step2": "results/v14_r1_empirical_validation_2026-07-04T02-21-09.json",
    "sentinel_step3": "results/sentinel_coverage_matrix_2026-07-04T14-35-21.json",
    "tuner_step4": "results/tuner_v14_r1_warmstart_seed_2026-07-04T02-39-07.json",
    "researcher_step5": "results/researcher_v14_r1_gate_check_2026-07-04T14-33-59.json",
}


# Pre-flight checks that MUST pass before any gate evaluation begins.
# If any fail, the run ABORTS — no projection-based verdicts, no
# criterion_met=true writes, no promotion status writes.
PREFLIGHT_CHECKS: List[Dict[str, Any]] = [
    {
        "id": "local_chain_reachable",
        "description": "Local chain must be reachable (btcli / subtensor RPC responds).",
        "pass_criteria": "btcli chain-info (or equivalent SDK call) returns a block within the last 60 seconds.",
        "on_fail": "ABORT. Do not fall back to offline/harness projections. §9 requires online mode.",
    },
    {
        "id": "v14_r1_config_loaded",
        "description": "V14-R1-CORRECTED-KP config must be loadable from agent_memory key v14_r1_corrected_config.",
        "pass_criteria": "read_memory(v14_r1_corrected_config) returns a non-null config dict.",
        "on_fail": "ABORT. The config is the substrate for the entire verification; without it nothing can be verified.",
    },
    {
        "id": "namespace_writable",
        "description": "The orchestrator's MCP layer must accept namespaced writes with procedure=v14_r1_online_gate_check.",
        "pass_criteria": "A no-op mongodb_insert_one into audit_log with the RUN_NAMESPACE succeeds.",
        "on_fail": "ABORT. If the namespace isn't writable, the verification cannot persist evidence under the correct procedure.",
    },
]


# Evidence requirements per gate. Each gate MUST cite a V14-R1 document
# in the specified collection. V13-R3 proxy data and offline KRET file
# references are NOT acceptable as primary evidence.
EVIDENCE_REQUIREMENTS: List[Dict[str, Any]] = [
    {
        "gate": "honest_mean_score",
        "threshold": 0.97,
        "comparison": ">=",
        "required_collection": "simulation_epochs",
        "required_filter": {"config_id": "V14-R1-CORRECTED-KP"},
        "required_field": "honest_mean_score",
        "min_documents": 2,  # one per rerun
        "forbidden_evidence": ["V13-R3 simulation_epochs", "offline KRET file references", "separation-based projections"],
    },
    {
        "gate": "score_variance",
        "threshold": 0.002,
        "comparison": "<=",
        "required_collection": "simulation_epochs",
        "required_filter": {"config_id": "V14-R1-CORRECTED-KP"},
        "required_field": "honest_score_variance",
        "min_documents": 2,
        "forbidden_evidence": ["V13-R3 simulation_epochs", "projections"],
    },
    {
        "gate": "commit_reveal_effectiveness",
        "threshold": 0.667,
        "comparison": ">=",
        "required_collection": "simulation_epochs",
        "required_filter": {"config_id": "V14-R1-CORRECTED-KP"},
        "required_field": "cr_effectiveness",
        "min_documents": 2,
        "forbidden_evidence": ["V13-R3 sentinel_state cr_effectiveness", "agent_memory sentinel_state from V13-R3"],
    },
    {
        "gate": "consecutive_clean_validations",
        "threshold": 6,
        "comparison": ">=",
        "required_collection": "sentinel_state",
        "required_filter": {"config_id": "V14-R1-CORRECTED-KP", **{k: v for k, v in RUN_NAMESPACE.items() if k != "procedure"}},
        "required_field": "consecutive_evals_below_threshold",
        "min_documents": 1,
        "forbidden_evidence": ["V13-R3 sentinel_state", "agent_memory sentinel_state from V13-R3"],
    },
    {
        "gate": "convergence_contract",
        "threshold": "unanimously_met",
        "comparison": "met",
        "required_collection": "convergence_metrics",
        "required_filter": {"config_id": "V14-R1-CORRECTED-KP", **{k: v for k, v in RUN_NAMESPACE.items() if k != "procedure"}},
        "required_field": "criterion_met",
        "min_documents": 1,
        "forbidden_evidence": ["V13-R3 convergence_metrics (criterion_met=false)", "offline summary claims"],
        "special_note": "This gate's evidence document (criterion_met=true) is itself the promotion artifact. It MUST be written ONLY after gates 1-4 and 6 are PASS against V14-R1 evidence. Writing it before the other gates have V14-R1 evidence is the premature-promotion failure mode v3 is designed to prevent.",
    },
    {
        "gate": "sentinel_posture",
        "threshold": "SECURE_AND_IMPROVING",
        "comparison": "in",
        "accepted_values": ["SECURE_AND_IMPROVING", "SECURE", "HARDENED", "TARGET_ACHIEVED"],
        "required_collection": "sentinel_state",
        "required_filter": {"config_id": "V14-R1-CORRECTED-KP", **{k: v for k, v in RUN_NAMESPACE.items() if k != "procedure"}},
        "required_field": "security_status",
        "min_documents": 1,
        "forbidden_evidence": ["V13-R3 sentinel_state", "agent_memory sentinel_state from V13-R3"],
    },
]


# Hard blocks — actions the orchestrator/coder MUST NOT take.
HARD_BLOCKS: List[Dict[str, Any]] = [
    {
        "id": "no_projection_verdicts",
        "rule": "Gate verdicts MUST be PASS, FAIL, or INSUFFICIENT_EVIDENCE. The strings 'PASS (projected)', 'PASS (conditional)', 'PASS (inferred from baseline)', 'PASS (with caveats)' are FORBIDDEN.",
        "enforcement": "Any agent producing a projection verdict must re-issue it as INSUFFICIENT_EVIDENCE with the missing-evidence list.",
    },
    {
        "id": "no_criterion_met_writes_without_evidence",
        "rule": "mongodb_insert_one / mongodb_update_one into convergence_metrics, sentinel_state, audit_log with criterion_met=true or status=PROMOTED for V14-R1 is FORBIDDEN until the EVIDENCE_REQUIREMENTS for all 6 gates are satisfied with V14-R1 documents.",
        "enforcement": "The coder's HITL task must read simulation_epochs (>= 2 V14-R1 docs), sentinel_state (>= 1 V14-R1 doc), convergence_metrics (NOT pre-existing) BEFORE any promotion write. If any prerequisite is missing, the HITL task returns BLOCKED — not PROMOTED_WITH_PENDING_BTCLI.",
    },
    {
        "id": "no_namespace_leak",
        "rule": "V14-R1 verification writes (mongodb_insert_one / mongodb_update_one) MUST use procedure='v14_r1_online_gate_check'. Writes with procedure='insignia_subnet_tuner' are FORBIDDEN for this run.",
        "enforcement": "The coder's mongodb_insert_one calls must include the RUN_NAMESPACE fields. Any write missing them, or using the tuner procedure, must be rejected by the orchestrator's MCP layer (treat as a guardrail, same as NAMESPACE_FILTER_REQUIRED).",
    },
    {
        "id": "no_offline_fallback_for_online_gates",
        "rule": "If local_chain_reachable pre-flight fails, ABORT the run. Do not produce verdicts from offline/harness data + V13-R3 projections.",
        "enforcement": "The orchestrator's first action is the pre-flight. On fail, write agent_memory v14_r1_online_verification_result with status=ABORTED_CHAIN_UNREACHABLE and exit. No gate verdicts, no HITL dispatch.",
    },
    {
        "id": "reviewer_veto_honored",
        "rule": "If the reviewer's audit returns INSUFFICIENT_EVIDENCE or FAIL for any gate, the orchestrator's summary MUST reflect that verdict. The orchestrator MUST NOT override the reviewer with 'PASS (projected)'.",
        "enforcement": "The orchestrator's summary table must match the reviewer's per-gate verdicts. Disagreement triggers a correction task targeting the discrepant gate, not a promotion.",
    },
    {
        "id": "no_promotion_without_git_push",
        "rule": "The coder MUST NOT write agent_memory with status=PROMOTED or status=PROMOTED_WITH_PENDING_BTCLI unless the parameter_space.py change is actually pushed to the git branch.",
        "enforcement": "If GitHub push tools are unavailable, the coder writes status=BLOCKED_NO_GIT_PUSH with the diff description, and the orchestrator files a follow-up task for manual push. PENDING_GITHUB_PUSH is a TODO, not a promotion.",
    },
]


def build_manifest() -> Dict[str, Any]:
    """Build the v3 dispatch manifest with hard-blocks against premature promotion."""
    return {
        "manifest_type": "v14_r1_online_gate_verification_dispatch",
        "manifest_version": "v3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_id": "V14-R1-CORRECTED-KP",
        "config_source": "orchestrator MongoDB memory key `v14_r1_corrected_config`",
        "spec_ref": "EMULATOR_SPEC.md §9 acceptance gates (online mode)",
        # v2 post-mortem: why v3 exists
        "v2_postmortem": {
            "v2_manifest": "results/v14_r1_online_dispatch_manifest_v2_2026-07-05T18-00-10.json",
            "v2_report": "Orchestration Report/Orchestration Report — 2026-07-07T20-41-48.pdf",
            "v2_result": "6/6 completed, 0 failed (guardrails fixed)",
            "v2_actual_outcome": "Premature promotion: orchestrator declared PASS on projected V13-R3 baseline + offline KRET; coder persisted criterion_met=true to MongoDB without V14-R1 runs; parameter_space.py update stuck in PENDING_GITHUB_PUSH; reviewer audit said INSUFFICIENT_EVIDENCE on all 6 gates and was overridden by orchestrator summary.",
            "v2_caveats_that_were_the_whole_verification": [
                "Local chain UNREACHABLE — 0 on-chain evidence",
                "0 V14-R1 documents in MongoDB across 12+ collections",
                "0 reruns completed (manifest required >= 2)",
                "V13-R3 convergence FAILED (criterion_met=false) yet used as baseline projection",
            ],
            "root_causes_v3_fixes": [
                {
                    "v2_failure": "Orchestrator fell back to projections when chain was unreachable",
                    "v3_fix": "PRE-FLIGHT GATE: abort if chain unreachable, no fallback",
                },
                {
                    "v2_failure": "Verdicts cited V13-R3 data and offline files as PASS evidence",
                    "v3_fix": "EVIDENCE_REQUIREMENTS: each gate must cite a V14-R1 MongoDB document; V13-R3 / offline refs are forbidden_evidence",
                },
                {
                    "v2_failure": "Coder wrote criterion_met=true before V14-R1 runs existed",
                    "v3_fix": "PROMOTION WRITE-BLOCK: criterion_met=true writes forbidden until >= 2 V14-R1 simulation_epochs + V14-R1 sentinel_state persisted",
                },
                {
                    "v2_failure": "Coder used procedure=insignia_subnet_tuner for V14-R1 writes (namespace leak)",
                    "v3_fix": "NAMESPACE LOCK: V14-R1 writes must use procedure=v14_r1_online_gate_check",
                },
                {
                    "v2_failure": "Orchestrator summary overrode reviewer's INSUFFICIENT_EVIDENCE",
                    "v3_fix": "REVIEWER VETO: orchestrator summary must match reviewer verdicts; disagreement triggers correction task, not promotion",
                },
                {
                    "v2_failure": "PROMOTED_WITH_PENDING_BTCLI status written without git push",
                    "v3_fix": "NO PROMOTION WITHOUT GIT PUSH: PENDING_GITHUB_PUSH is a TODO, not a promotion",
                },
            ],
        },
        "namespace": RUN_NAMESPACE,
        "forbidden_collections": FORBIDDEN_COLLECTIONS,
        "filesystem_kret_artifacts": FILESYSTEM_KRET_ARTIFACTS,
        "filesystem_kret_role": "CONTEXT ONLY — usable for cycle continuity / debugging, NOT as primary evidence for any §9 online gate verdict in v3.",
        "preflight_checks": PREFLIGHT_CHECKS,
        "hard_blocks": HARD_BLOCKS,
        "objective": (
            "Verify the 6 online-mode §9 gates against the live V14-R1-CORRECTED-KP "
            "config on-chain, across >= 2 reruns with different seeds, with V14-R1 "
            "evidence persisted to MongoDB. If all 6 clear against V14-R1 evidence "
            "(not projections), proceed to the HITL promotion gate. v3 hard-blocks "
            "the premature-promotion failure mode observed in the v2 run."
        ),
        "online_gates_to_verify": EVIDENCE_REQUIREMENTS,
        "acceptance_criteria": {
            "reruns_required": 2,
            "seeds": "different seeds for each rerun",
            "all_gates_must_clear": True,
            "evidence_must_be_v14_r1": True,
            "no_projection_verdicts": True,
            "if_any_gate_fails": "Do NOT promote V14-R1. File a correction task for the failing gate's root cause. Do NOT write criterion_met=true to MongoDB.",
            "if_all_gates_clear": (
                "Proceed to HITL promotion gate: (1) push the parameter_space.py "
                "change to the git branch (current_candidate_status=promoted_to_production_reference), "
                "(2) btcli hyperparameter apply with the V14-R1-CORRECTED-KP config, "
                "(3) write the promotion record to audit_log with the RUN_NAMESPACE. "
                "All three steps must complete; PENDING_GITHUB_PUSH is not a promotion."
            ),
        },
        "mcp_dispatch_commands": {
            "note": (
                "Execute these commands in the orchestrator's agent env via the "
                "insignia-local MCP. v3 adds pre-flight abort, evidence-backed "
                "verdicts, promotion write-block, namespace lock, and reviewer veto."
            ),
            "step1_file_task": {
                "mcp_tool": "insignia-local.file_task",
                "arguments": {
                    "assignee": "orchestrator",
                    "priority": 10,
                    "description": (
                        "V14-R1 online-mode gate verification (v3 — hard-blocks premature "
                        "promotion): run the live V14-R1-CORRECTED-KP config on-chain across "
                        ">= 2 reruns with different seeds. PRE-FLIGHT: abort if local chain "
                        "unreachable. EVIDENCE: each gate must cite a V14-R1 MongoDB document "
                        "(V13-R3 projections forbidden). PROMOTION WRITE-BLOCK: do not write "
                        "criterion_met=true until >= 2 V14-R1 simulation_epochs + V14-R1 "
                        "sentinel_state are persisted. NAMESPACE: every write uses "
                        f"{RUN_NAMESPACE} (procedure=insignia_subnet_tuner is FORBIDDEN). "
                        "REVIEWER VETO: orchestrator summary must match reviewer verdicts."
                    ),
                    "metadata": {
                        "cycle_step": "5_to_HITL_v3",
                        "config_id": "V14-R1-CORRECTED-KP",
                        "manifest_version": "v3",
                        "namespace": RUN_NAMESPACE,
                        "forbidden_collections": FORBIDDEN_COLLECTIONS,
                        "preflight_checks": [c["id"] for c in PREFLIGHT_CHECKS],
                        "hard_blocks": [b["id"] for b in HARD_BLOCKS],
                        "v2_postmortem_doc": "results/v14_r1_v2_run_postmortem_2026-07-07.md",
                    },
                },
            },
            "step2_write_agent_memory_dispatch": {
                "mcp_tool": "insignia-local.write_agent_memory",
                "arguments": {
                    "key": "v14_r1_online_verification_dispatch",
                    "value": {
                        "status": "DISPATCHED_V3",
                        "manifest_version": "v3",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "config_id": "V14-R1-CORRECTED-KP",
                        "config_source": "v14_r1_corrected_config (MongoDB)",
                        "namespace": RUN_NAMESPACE,
                        "forbidden_collections": FORBIDDEN_COLLECTIONS,
                        "preflight_checks": [c["id"] for c in PREFLIGHT_CHECKS],
                        "hard_blocks": [b["id"] for b in HARD_BLOCKS],
                        "reruns_required": 2,
                        "gates_to_verify": [g["gate"] for g in EVIDENCE_REQUIREMENTS],
                        "v2_postmortem": "Premature promotion in v2; v3 hard-blocks recurrence.",
                    },
                },
            },
            "step3_invalidate_v2_result": {
                "mcp_tool": "insignia-local.write_agent_memory",
                "arguments": {
                    "key": "v14_r1_online_verification_result",
                    "value": None,
                    "note": "Clear the v2 result (GATES_VERIFIED_WITH_CAVEATS / PROMOTED_WITH_PENDING_BTCLI) so the orchestrator starts v3 fresh.",
                },
            },
            "step4_quarantine_v2_premature_writes": {
                "mcp_tool": "insignia-local.write_agent_memory",
                "arguments": {
                    "key": "v14_r1_v2_premature_writes_to_quarantine",
                    "value": {
                        "reason": "v2 persisted criterion_met=true / SECURE_AND_IMPROVING for V14-R1 without V14-R1 runs backing them (procedure=insignia_subnet_tuner, namespace leak).",
                        "collections_to_quarantine": [
                            "convergence_metrics (V14-R1 documents with procedure=insignia_subnet_tuner)",
                            "sentinel_state (V14-R1 documents with procedure=insignia_subnet_tuner)",
                            "audit_log (V14-R1 hitl_promotion_btcli_apply_pending events)",
                        ],
                        "action": "Re-tag with verification_mode=offline_harness_fallback_projected, criterion_met=false, OR delete. Do NOT treat as V14-R1 evidence in v3.",
                    },
                },
            },
        },
        "post_verification": {
            "on_success": [
                "All 6 gates PASS against V14-R1 MongoDB evidence (not projections).",
                ">= 2 V14-R1 simulation_epochs documents persisted with distinct seeds.",
                ">= 1 V14-R1 sentinel_state document persisted with security_status=SECURE_AND_IMPROVING+ and consecutive_evals_below_threshold >= 6.",
                ">= 1 V14-R1 convergence_metrics document persisted with criterion_met=true (this is the evidence, written LAST).",
                "parameter_space.py change PUSHED to git branch (not PENDING_GITHUB_PUSH).",
                "btcli hyperparameter apply completed (not pending).",
                "Write agent_memory v14_r1_online_verification_result with status=ALL_GATES_CLEARED_V14_R1_EVIDENCE.",
            ],
            "on_failure": [
                "Write agent_memory v14_r1_online_verification_result with status=GATES_FAILED or INSUFFICIENT_EVIDENCE and the per-gate verdicts.",
                "Do NOT write criterion_met=true to any V14-R1 document.",
                "Do NOT mark parameter_space.py as promoted.",
                "File a correction task for the failing gate's root cause.",
            ],
            "on_preflight_abort": [
                "Write agent_memory v14_r1_online_verification_result with status=ABORTED_CHAIN_UNREACHABLE (or ABORTED_<preflight_id>).",
                "Do NOT evaluate any gate.",
                "Do NOT dispatch the HITL promotion task.",
                "File a task to restore the local chain, then re-dispatch v3.",
            ],
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=str, default="results", help="Output directory (default results)")
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")

    manifest = build_manifest()

    json_path = out_dir / f"v14_r1_online_dispatch_manifest_v3_{ts}.json"
    md_path = out_dir / f"v14_r1_online_dispatch_manifest_v3_{ts}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    # Markdown manifest
    lines = [
        "# V14-R1 Online-Mode Gate Verification — Orchestrator Dispatch Manifest (v3)",
        "",
        f"**Generated:** {manifest['generated_at']}",
        f"**Config ID:** {manifest['config_id']}",
        f"**Config source:** {manifest['config_source']}",
        f"**Spec ref:** {manifest['spec_ref']}",
        "",
        "## v2 post-mortem (why v3 exists)",
        "",
        f"- **v2 manifest:** `{manifest['v2_postmortem']['v2_manifest']}`",
        f"- **v2 report:** `{manifest['v2_postmortem']['v2_report']}`",
        f"- **v2 result:** {manifest['v2_postmortem']['v2_result']}",
        f"- **v2 actual outcome:** {manifest['v2_postmortem']['v2_actual_outcome']}",
        "",
        "v2 caveats that were the whole verification:",
    ]
    for c in manifest["v2_postmortem"]["v2_caveats_that_were_the_whole_verification"]:
        lines.append(f"- {c}")
    lines += ["", "Root causes v3 fixes:", ""]
    for rc in manifest["v2_postmortem"]["root_causes_v3_fixes"]:
        lines.append(f"- **v2 failure:** {rc['v2_failure']}")
        lines.append(f"  - **v3 fix:** {rc['v3_fix']}")
    lines += [
        "",
        "## v3 hardening summary",
        "",
        "1. **PRE-FLIGHT GATE** — abort if local chain unreachable (no projection fallback).",
        "2. **EVIDENCE-BACKED VERDICTS** — each gate must cite a V14-R1 MongoDB document; V13-R3 / offline refs are forbidden evidence.",
        "3. **PROMOTION WRITE-BLOCK** — `criterion_met=true` writes forbidden until ≥2 V14-R1 `simulation_epochs` + V14-R1 `sentinel_state` persisted.",
        "4. **NAMESPACE LOCK** — V14-R1 writes must use `procedure=v14_r1_online_gate_check` (not `insignia_subnet_tuner`).",
        "5. **REVIEWER VETO** — orchestrator summary must match reviewer verdicts; disagreement → correction task, not promotion.",
        "6. **NO PROJECTION VERDICTS** — `PASS (projected)` / `PASS (conditional)` are forbidden; verdicts are PASS / FAIL / INSUFFICIENT_EVIDENCE.",
        "7. **NO PROMOTION WITHOUT GIT PUSH** — `PENDING_GITHUB_PUSH` is a TODO, not a promotion.",
        "",
        "## Objective",
        "",
        manifest["objective"],
        "",
        "## Pre-flight checks (ABORT on any failure)",
        "",
        "| # | Check | Pass criteria | On fail |",
        "|---|---|---|---|",
    ]
    for i, c in enumerate(manifest["preflight_checks"], 1):
        lines.append(f"| {i} | `{c['id']}` | {c['pass_criteria']} | {c['on_fail']} |")
    lines += [
        "",
        "## Hard blocks (orchestrator/coder MUST NOT do these)",
        "",
    ]
    for b in manifest["hard_blocks"]:
        lines.append(f"### `{b['id']}`")
        lines.append(f"**Rule:** {b['rule']}")
        lines.append(f"**Enforcement:** {b['enforcement']}")
        lines.append("")
    lines += [
        "## 6 Online-Mode Gates with V14-R1 Evidence Requirements",
        "",
        "| # | Gate | Threshold | Required collection | Required field | Min docs | Forbidden evidence |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, g in enumerate(manifest["online_gates_to_verify"], 1):
        fe = "; ".join(g["forbidden_evidence"])
        lines.append(f"| {i} | `{g['gate']}` | {g['threshold']} | `{g['required_collection']}` | `{g['required_field']}` | {g['min_documents']} | {fe} |")
    lines += [
        "",
        "## Acceptance Criteria",
        "",
        f"- **Reruns required:** {manifest['acceptance_criteria']['reruns_required']} ({manifest['acceptance_criteria']['seeds']})",
        f"- **All gates must clear:** {manifest['acceptance_criteria']['all_gates_must_clear']}",
        f"- **Evidence must be V14-R1:** {manifest['acceptance_criteria']['evidence_must_be_v14_r1']}",
        f"- **No projection verdicts:** {manifest['acceptance_criteria']['no_projection_verdicts']}",
        f"- **If any gate fails:** {manifest['acceptance_criteria']['if_any_gate_fails']}",
        f"- **If all gates clear:** {manifest['acceptance_criteria']['if_all_gates_clear']}",
        "",
        "## MCP Dispatch Commands (v3 — pre-flight + write-block + namespace lock)",
        "",
        "**Cannot be executed from this repo** — the insignia-local MCP server is not available in this environment. Execute in the orchestrator's agent env.",
        "",
    ]
    mc = manifest["mcp_dispatch_commands"]
    for step_name, cmd in mc.items():
        if step_name == "note":
            continue
        lines.append(f"### {step_name}")
        lines.append("")
        lines.append(f"**MCP tool:** `{cmd['mcp_tool']}`")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(cmd["arguments"], indent=2))
        lines.append("```")
        lines.append("")
    lines += [
        "## Post-Verification Actions",
        "",
        "### On success (all 6 gates clear against V14-R1 evidence)",
        "",
    ]
    for i, item in enumerate(manifest["post_verification"]["on_success"], 1):
        lines.append(f"{i}. {item}")
    lines += ["", "### On failure (any gate fails or INSUFFICIENT_EVIDENCE)", ""]
    for i, item in enumerate(manifest["post_verification"]["on_failure"], 1):
        lines.append(f"{i}. {item}")
    lines += ["", "### On pre-flight abort (chain unreachable, etc.)", ""]
    for i, item in enumerate(manifest["post_verification"]["on_preflight_abort"], 1):
        lines.append(f"{i}. {item}")
    lines.append("")
    lines.append(f"_JSON manifest: `{json_path.name}`_")
    lines.append("")
    lines.append(f"_v2 post-mortem: `results/v14_r1_v2_run_postmortem_2026-07-07.md`_")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("V14-R1 online gate verification dispatch manifest (v3)")
    print("  v2 post-mortem: premature promotion (criterion_met=true without V14-R1 runs)")
    print(f"  pre-flight checks: {len(manifest['preflight_checks'])}")
    print(f"  hard blocks: {len(manifest['hard_blocks'])}")
    print(f"  evidence requirements: {len(manifest['online_gates_to_verify'])} gates")
    print(f"  MCP commands: {len([k for k in manifest['mcp_dispatch_commands'] if k != 'note'])}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print()
    print("WARNING: This manifest CANNOT be auto-executed from this repo.")
    print("    The insignia-local MCP server is not available in this environment.")
    print("    Execute the MCP commands in the orchestrator's agent env, or trigger")
    print("    the dispatch via the swarm gateway.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
