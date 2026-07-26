"""V14-R1 online-mode gate verification — orchestrator dispatch manifest v2.

v2 fixes two guardrail rejections that cancelled 4/6 gates in the v1 run
(dashboard 2026-07-05: 2/6 done, 4 cancelled, 0 failed):

  1. NAMESPACE_FILTER_REQUIRED on `convergence_metrics`:
     The orchestrator's MCP layer refuses unscoped mongodb_find on shared
     collections. v2 declares the run's namespace up front and adds a
     namespace filter (`agent_type`, `playbook`, `domain`, or `procedure`)
     to every MCP read.

  2. PLAYBOOK_COLLECTION_FORBIDDEN on `simulation_results`:
     `simulation_results` belongs to the "Insignia subnet tuner" playbook,
     not the orchestrator's. v2 explicitly forbids mongodb_find on that
     collection and redirects the orchestrator to the filesystem KRET
     artifacts in `results/` (the step 2-5 offline evidence package).

This script CANNOT dispatch the orchestrator from this repo — the
insignia-local MCP server is not available in this environment. It writes
the v2 manifest to results/ for manual execution in the orchestrator env.

Usage:
    python scripts/v14_r1_online_dispatch_manifest_v2.py [--output results/]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List


# Namespace for this verification run. Every MCP read must be scoped with
# at least one of these fields to pass the NAMESPACE_FILTER_REQUIRED guardrail.
RUN_NAMESPACE: Dict[str, str] = {
    "playbook": "insignia_subnet_online_verification",
    "domain": "v14_r1",
    "procedure": "v14_r1_online_gate_check",
}

# Collections the orchestrator is NOT allowed to mongodb_find on directly.
# These belong to other playbooks — read their KRET artifacts from the
# filesystem (results/) instead.
FORBIDDEN_COLLECTIONS: List[str] = [
    "simulation_results",  # belongs to "Insignia subnet tuner" playbook
]

# Filesystem KRET artifacts that substitute for the forbidden collections.
# These are the step 2-5 offline evidence files already on disk in results/.
FILESYSTEM_KRET_ARTIFACTS: Dict[str, str] = {
    "simulator_step2": "results/v14_r1_empirical_validation_2026-07-04T02-21-09.json",
    "sentinel_step3": "results/sentinel_coverage_matrix_2026-07-04T14-35-21.json",
    "tuner_step4": "results/tuner_v14_r1_warmstart_seed_2026-07-04T02-39-07.json",
    "researcher_step5": "results/researcher_v14_r1_gate_check_2026-07-04T14-33-59.json",
}


# The 6 online-mode §9 gates with namespaced evidence sources.
ONLINE_GATES: List[Dict[str, Any]] = [
    {
        "gate": "honest_mean_score",
        "threshold": 0.97,
        "comparison": ">=",
        "evidence_source": {
            "type": "on_chain",
            "method": "live miner composite scores on-chain (mean of honest miners)",
            "namespace_filter": {**RUN_NAMESPACE, "agent_type": "miner"},
        },
        "note": "V13-R2 knee claimed 0.9795 (surrogate-predicted, never empirically confirmed). Offline harness produces 0.9007 via synthetic scorer; live chain must verify >= 0.97.",
    },
    {
        "gate": "score_variance",
        "threshold": 0.002,
        "comparison": "<=",
        "evidence_source": {
            "type": "on_chain",
            "method": "live miner composite score variance across honest miners",
            "namespace_filter": {**RUN_NAMESPACE, "agent_type": "miner"},
        },
        "note": "Offline harness produces 0.0013 (synthetic); live variance must be <= 0.002.",
    },
    {
        "gate": "commit_reveal_effectiveness",
        "threshold": 0.667,
        "comparison": ">=",
        "target": 0.76,
        "evidence_source": {
            "type": "on_chain",
            "method": "commit-reveal telemetry (commit_timestamps, reveal_timestamps, no_reveal_streaks)",
            "namespace_filter": {**RUN_NAMESPACE, "agent_type": "validator"},
        },
        "note": "Offline harness hardcodes 0.76; live CR effectiveness must be >= 0.667 floor (>= 0.76 target).",
    },
    {
        "gate": "consecutive_clean_validations",
        "threshold": 6,
        "comparison": ">=",
        "evidence_source": {
            "type": "mcp_read",
            "collection": "sentinel_state",
            "method": "sentinel breach-free validation streak",
            "namespace_filter": {**RUN_NAMESPACE, "agent_type": "sentinel"},
        },
        "note": "Offline harness hardcodes 6; live streak must be >= 6 consecutive clean validations.",
    },
    {
        "gate": "convergence_contract",
        "threshold": "unanimously_met",
        "comparison": "met",
        "evidence_source": {
            "type": "mcp_read",
            "collection": "convergence_metrics",
            "method": "§7 contract: all agents agree + grace period elapsed",
            # v1 failure: NAMESPACE_FILTER_REQUIRED — v2 mandates the filter.
            "namespace_filter": {**RUN_NAMESPACE, "agent_type": "convergence_monitor"},
            "required_filter_fields": ["procedure", "agent_type", "playbook", "domain"],
        },
        "note": "Requires orchestrator-side convergence_metrics read WITH a namespace filter (v1 was rejected for missing it). The §7 contract must be unanimously met with the grace period elapsed.",
    },
    {
        "gate": "sentinel_posture",
        "threshold": "SECURE_AND_IMPROVING",
        "comparison": "in",
        "accepted_values": ["SECURE_AND_IMPROVING", "SECURE", "HARDENED", "TARGET_ACHIEVED"],
        "evidence_source": {
            "type": "mcp_read",
            "collection": "sentinel_state",
            "method": "security_status field",
            "namespace_filter": {**RUN_NAMESPACE, "agent_type": "sentinel"},
        },
        "note": "Offline harness hardcodes TARGET_ACHIEVED; live sentinel posture must be SECURE_AND_IMPROVING or stronger.",
    },
]


def build_manifest() -> Dict[str, Any]:
    """Build the v2 dispatch manifest with namespace + forbidden-collection fixes."""
    return {
        "manifest_type": "v14_r1_online_gate_verification_dispatch",
        "manifest_version": "v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_id": "V14-R1-CORRECTED-KP",
        "config_source": "orchestrator MongoDB memory key `v14_r1_corrected_config`",
        "spec_ref": "EMULATOR_SPEC.md §9 acceptance gates (online mode)",
        # v1 post-mortem: why 4/6 gates were cancelled
        "v1_postmortem": {
            "v1_manifest": "results/v14_r1_online_dispatch_manifest_2026-07-04T14-45-08.json",
            "v1_result": "2/6 done, 4 cancelled, 0 failed (dashboard 2026-07-05)",
            "cancelled_gates": [
                "convergence_contract (NAMESPACE_FILTER_REQUIRED)",
                "sentinel_posture (downstream of convergence_contract cancellation)",
                "consecutive_clean_validations (downstream)",
                "one additional gate (downstream)",
            ],
            "root_causes": [
                {
                    "error": "NAMESPACE_FILTER_REQUIRED",
                    "trigger": "mongodb_find on convergence_metrics without namespace filter",
                    "fix_in_v2": "RUN_NAMESPACE declared + namespace_filter on every MCP read",
                },
                {
                    "error": "PLAYBOOK_COLLECTION_FORBIDDEN",
                    "trigger": "mongodb_find on simulation_results (belongs to Insignia subnet tuner playbook)",
                    "fix_in_v2": "FORBIDDEN_COLLECTIONS list + FILESYSTEM_KRET_ARTIFACTS redirect",
                },
            ],
        },
        "namespace": RUN_NAMESPACE,
        "forbidden_collections": FORBIDDEN_COLLECTIONS,
        "filesystem_kret_artifacts": FILESYSTEM_KRET_ARTIFACTS,
        "objective": (
            "Verify the 6 online-mode §9 gates against the live V14-R1-CORRECTED-KP "
            "config on-chain, across >= 2 reruns with different seeds. If all 6 clear, "
            "proceed to the HITL promotion gate (btcli hyperparameter apply + promote "
            "converged config as production reference)."
        ),
        "offline_evidence_package": {
            "step2_simulator": FILESYSTEM_KRET_ARTIFACTS["simulator_step2"],
            "step3_sentinel": FILESYSTEM_KRET_ARTIFACTS["sentinel_step3"],
            "step4_tuner_seed": FILESYSTEM_KRET_ARTIFACTS["tuner_step4"],
            "step5_gate_check": FILESYSTEM_KRET_ARTIFACTS["researcher_step5"],
            "offline_summary": (
                "All 4 harness-mode gates PASS (attack_breach_rate=0.0, separation=0.9004, "
                "validator_latency_severity=0.0351, prediction_timing_severity=0.025). "
                "0 adversary leaks across the 19-vector sentinel surface. "
                "2 non-penalty-path breaches remain (random_baseline_discrimination, "
                "collusion_temporal_pattern) — both synthetic harness artifacts, not adversary leaks."
            ),
            "read_instruction": (
                "Read these from the FILESYSTEM, not MongoDB. The orchestrator's "
                "mongodb_find on `simulation_results` is forbidden (belongs to the "
                "Insignia subnet tuner playbook). These JSON files are the KRET "
                "artifacts that substitute for that collection."
            ),
        },
        "online_gates_to_verify": ONLINE_GATES,
        "acceptance_criteria": {
            "reruns_required": 2,
            "seeds": "different seeds for each rerun",
            "all_gates_must_clear": True,
            "if_any_gate_fails": (
                "Do NOT promote V14-R1. File a correction task for the failing gate's "
                "root cause and re-run the cycle from the appropriate step."
            ),
            "if_all_gates_clear": (
                "Proceed to HITL promotion gate: (1) btcli hyperparameter apply with the "
                "V14-R1-CORRECTED-KP config, (2) promote the converged config as the "
                "production reference in research_targets."
            ),
        },
        "mcp_dispatch_commands": {
            "note": (
                "Execute these commands in the orchestrator's agent env via the "
                "insignia-local MCP. v2 adds namespace filters and forbids "
                "simulation_results — fixes for the v1 guardrail rejections."
            ),
            "step1_file_task": {
                "mcp_tool": "insignia-local.file_task",
                "arguments": {
                    "assignee": "orchestrator",
                    "priority": 10,
                    "description": (
                        "V14-R1 online-mode gate verification (v2 — fixes v1 guardrail "
                        "rejections): run the live V14-R1-CORRECTED-KP config on-chain "
                        "across >= 2 reruns with different seeds. Verify the 6 online-mode "
                        "§9 gates. Scope every MCP read with the namespace "
                        f"{RUN_NAMESPACE}. Do NOT mongodb_find on {FORBIDDEN_COLLECTIONS} "
                        "— read filesystem KRET artifacts in results/ instead."
                    ),
                    "metadata": {
                        "cycle_step": "5_to_HITL",
                        "config_id": "V14-R1-CORRECTED-KP",
                        "manifest_version": "v2",
                        "namespace": RUN_NAMESPACE,
                        "forbidden_collections": FORBIDDEN_COLLECTIONS,
                        "filesystem_kret_artifacts": FILESYSTEM_KRET_ARTIFACTS,
                        "offline_evidence_branch": "feat/signal-driven-adversary-penalties",
                        "offline_gate_check": FILESYSTEM_KRET_ARTIFACTS["researcher_step5"],
                    },
                },
            },
            "step2_write_agent_memory_dispatch": {
                "mcp_tool": "insignia-local.write_agent_memory",
                "arguments": {
                    "key": "v14_r1_online_verification_dispatch",
                    "value": {
                        "status": "DISPATCHED_V2",
                        "manifest_version": "v2",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "config_id": "V14-R1-CORRECTED-KP",
                        "config_source": "v14_r1_corrected_config (MongoDB)",
                        "namespace": RUN_NAMESPACE,
                        "forbidden_collections": FORBIDDEN_COLLECTIONS,
                        "filesystem_kret_artifacts": FILESYSTEM_KRET_ARTIFACTS,
                        "offline_evidence": {
                            "harness_mode_gates_passed": 4,
                            "harness_mode_gates_failed": 0,
                            "online_mode_gates_pending": 6,
                            "adversary_leaks": 0,
                            "separation": 0.9004,
                        },
                        "reruns_required": 2,
                        "gates_to_verify": [g["gate"] for g in ONLINE_GATES],
                        "v1_postmortem": "4/6 cancelled by NAMESPACE_FILTER_REQUIRED + PLAYBOOK_COLLECTION_FORBIDDEN; v2 fixes both.",
                    },
                },
            },
            "step3_invalidate_prior_result": {
                "mcp_tool": "insignia-local.write_agent_memory",
                "arguments": {
                    "key": "v14_r1_online_verification_result",
                    "value": None,
                    "note": "Clear the v1 result (2/6 done, 4 cancelled) so the orchestrator starts v2 fresh.",
                },
            },
        },
        "post_verification": {
            "on_success": [
                "Write agent_memory key `v14_r1_online_verification_result` with status=ALL_GATES_CLEARED.",
                "File a HITL task for `btcli hyperparameter apply` with the V14-R1-CORRECTED-KP config.",
                "Update `research_targets` in parameter_space.py: target_achieved=True, current_candidate_status=promoted_to_production_reference.",
            ],
            "on_failure": [
                "Write agent_memory key `v14_r1_online_verification_result` with status=GATES_FAILED and the failing gate(s).",
                "File a correction task for the failing gate's root cause.",
                "Do NOT promote V14-R1.",
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

    json_path = out_dir / f"v14_r1_online_dispatch_manifest_v2_{ts}.json"
    md_path = out_dir / f"v14_r1_online_dispatch_manifest_v2_{ts}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    # Markdown manifest
    lines = [
        "# V14-R1 Online-Mode Gate Verification — Orchestrator Dispatch Manifest (v2)",
        "",
        f"**Generated:** {manifest['generated_at']}",
        f"**Config ID:** {manifest['config_id']}",
        f"**Config source:** {manifest['config_source']}",
        f"**Spec ref:** {manifest['spec_ref']}",
        "",
        "## v1 post-mortem (why v2 exists)",
        "",
        f"- **v1 manifest:** `{manifest['v1_postmortem']['v1_manifest']}`",
        f"- **v1 result:** {manifest['v1_postmortem']['v1_result']}",
        "",
        "v1 cancelled 4/6 gates due to two MCP guardrail rejections:",
        "",
    ]
    for rc in manifest["v1_postmortem"]["root_causes"]:
        lines.append(f"- **`{rc['error']}`** — trigger: {rc['trigger']}")
        lines.append(f"  - v2 fix: {rc['fix_in_v2']}")
    lines += [
        "",
        "## v2 fixes",
        "",
        f"1. **Namespace declared up front:** `{manifest['namespace']}` — every MCP read must include one of these filter fields.",
        f"2. **Forbidden collections:** `{manifest['forbidden_collections']}` — orchestrator must NOT `mongodb_find` on these.",
        f"3. **Filesystem KRET redirect:** offline evidence read from filesystem (`results/`), not MongoDB.",
        "",
        "## Objective",
        "",
        manifest["objective"],
        "",
        "## Offline Evidence Package (filesystem KRET artifacts — read from disk, NOT MongoDB)",
        "",
        f"- **Step 2 (simulator):** `{manifest['offline_evidence_package']['step2_simulator']}`",
        f"- **Step 3 (sentinel):** `{manifest['offline_evidence_package']['step3_sentinel']}`",
        f"- **Step 4 (tuner seed):** `{manifest['offline_evidence_package']['step4_tuner_seed']}`",
        f"- **Step 5 (gate check):** `{manifest['offline_evidence_package']['step5_gate_check']}`",
        "",
        "**Read instruction:** " + manifest["offline_evidence_package"]["read_instruction"],
        "",
        "**Offline summary:** " + manifest["offline_evidence_package"]["offline_summary"],
        "",
        "## 6 Online-Mode Gates to Verify On-Chain (with namespaced evidence sources)",
        "",
        "| # | Gate | Threshold | Evidence source | Namespace filter |",
        "|---|---|---|---|---|",
    ]
    for i, g in enumerate(manifest["online_gates_to_verify"], 1):
        es = g["evidence_source"]
        nsf = es.get("namespace_filter", {})
        nsf_str = ", ".join(f"{k}={v}" for k, v in nsf.items())
        lines.append(f"| {i} | `{g['gate']}` | {g['threshold']} | {es['type']}: {es['method']} ({es.get('collection', 'on-chain')}) | `{nsf_str}` |")
    lines += [
        "",
        "## Acceptance Criteria",
        "",
        f"- **Reruns required:** {manifest['acceptance_criteria']['reruns_required']} ({manifest['acceptance_criteria']['seeds']})",
        f"- **All gates must clear:** {manifest['acceptance_criteria']['all_gates_must_clear']}",
        f"- **If any gate fails:** {manifest['acceptance_criteria']['if_any_gate_fails']}",
        f"- **If all gates clear:** {manifest['acceptance_criteria']['if_all_gates_clear']}",
        "",
        "## MCP Dispatch Commands (v2 — namespaced + forbidden-collection-aware)",
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
        "### On success (all 6 gates clear across >= 2 reruns)",
        "",
    ]
    for i, item in enumerate(manifest["post_verification"]["on_success"], 1):
        lines.append(f"{i}. {item}")
    lines += ["", "### On failure (any gate fails)", ""]
    for i, item in enumerate(manifest["post_verification"]["on_failure"], 1):
        lines.append(f"{i}. {item}")
    lines.append("")
    lines.append(f"_JSON manifest: `{json_path.name}`_")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("V14-R1 online gate verification dispatch manifest (v2)")
    print("  v1 post-mortem: 2/6 done, 4 cancelled (NAMESPACE_FILTER_REQUIRED + PLAYBOOK_COLLECTION_FORBIDDEN)")
    print(f"  namespace: {manifest['namespace']}")
    print(f"  forbidden_collections: {manifest['forbidden_collections']}")
    print(f"  6 online-mode gates, each with namespaced evidence source")
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
