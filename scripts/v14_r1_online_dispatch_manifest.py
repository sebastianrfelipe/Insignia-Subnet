"""V14-R1 online-mode gate verification — orchestrator dispatch manifest (cycle step 5→HITL).

This script produces the dispatch manifest the user (or the orchestrator's
agent env) must execute via the insignia-local MCP to trigger the V14-R1
online-mode gate verification. It cannot dispatch the orchestrator directly
from this repo — the insignia-local MCP server is not available in this
environment, so the manifest is written to a file for manual execution.

The manifest contains:
  1. The 6 online-mode §9 gates that require live chain verification
  2. The MCP commands to file the task + write the agent_memory key
  3. The evidence the orchestrator must collect on-chain
  4. The acceptance criteria (all 6 gates clear across >= 2 reruns)
  5. References to the offline evidence package (steps 2-5 artifacts)

Usage:
    python scripts/v14_r1_online_dispatch_manifest.py [--output results/]

After the manifest is produced, the user must execute the MCP commands in
the orchestrator's agent env (or trigger the dispatch via the swarm gateway)
to start the online verification.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List


# The 6 online-mode §9 gates that require live chain verification.
# These cannot be checked against the offline harness — per §9 the gates
# must hold "in `online` mode, across >= 2 reruns with different seeds".
ONLINE_GATES: List[Dict[str, Any]] = [
    {
        "gate": "honest_mean_score",
        "threshold": 0.97,
        "comparison": ">=",
        "evidence_source": "live miner composite scores on-chain (mean of honest miners)",
        "note": "The V13-R2 knee claimed 0.9795 (surrogate-predicted, never empirically confirmed). The offline harness produces 0.9007 via its synthetic scorer; the live chain must verify >= 0.97.",
    },
    {
        "gate": "score_variance",
        "threshold": 0.002,
        "comparison": "<=",
        "evidence_source": "live miner composite score variance across honest miners",
        "note": "Offline harness produces 0.0013 (synthetic); live variance must be <= 0.002.",
    },
    {
        "gate": "commit_reveal_effectiveness",
        "threshold": 0.667,
        "comparison": ">=",
        "target": 0.76,
        "evidence_source": "commit-reveal telemetry on-chain (commit_timestamps, reveal_timestamps, no_reveal_streaks)",
        "note": "Offline harness hardcodes 0.76; live CR effectiveness must be >= 0.667 floor (>= 0.76 target).",
    },
    {
        "gate": "consecutive_clean_validations",
        "threshold": 6,
        "comparison": ">=",
        "evidence_source": "sentinel breach-free validation streak on-chain",
        "note": "Offline harness hardcodes 6; live streak must be >= 6 consecutive clean validations.",
    },
    {
        "gate": "convergence_contract",
        "threshold": "unanimously_met",
        "comparison": "met",
        "evidence_source": "convergence_metrics MCP (§7 contract: all agents agree + grace period)",
        "note": "Requires orchestrator-side convergence_metrics read. The §7 contract must be unanimously met with the grace period elapsed.",
    },
    {
        "gate": "sentinel_posture",
        "threshold": "SECURE_AND_IMPROVING",
        "comparison": "in",
        "accepted_values": ["SECURE_AND_IMPROVING", "SECURE", "HARDENED", "TARGET_ACHIEVED"],
        "evidence_source": "sentinel_state MCP (security_status field)",
        "note": "Offline harness hardcodes TARGET_ACHIEVED; live sentinel posture must be SECURE_AND_IMPROVING or stronger.",
    },
]


def build_manifest() -> Dict[str, Any]:
    """Build the dispatch manifest for the V14-R1 online gate verification."""
    return {
        "manifest_type": "v14_r1_online_gate_verification_dispatch",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_id": "V14-R1-CORRECTED-KP",
        "config_source": "orchestrator MongoDB memory key `v14_r1_corrected_config`",
        "spec_ref": "EMULATOR_SPEC.md §9 acceptance gates (online mode)",
        "objective": (
            "Verify the 6 online-mode §9 gates against the live V14-R1-CORRECTED-KP "
            "config on-chain, across >= 2 reruns with different seeds. If all 6 clear, "
            "proceed to the HITL promotion gate (btcli hyperparameter apply + promote "
            "converged config as production reference)."
        ),
        "offline_evidence_package": {
            "step2_simulator": "results/v14_r1_empirical_validation_2026-07-04T02-21-09.json",
            "step3_sentinel": "results/sentinel_coverage_matrix_2026-07-04T14-35-21.json",
            "step4_tuner_seed": "results/tuner_v14_r1_warmstart_seed_2026-07-04T02-39-07.json",
            "step5_gate_check": "results/researcher_v14_r1_gate_check_2026-07-04T14-33-59.json",
            "offline_summary": (
                "All 4 harness-mode gates PASS (attack_breach_rate=0.0, separation=0.9004, "
                "validator_latency_severity=0.0351, prediction_timing_severity=0.025). "
                "0 adversary leaks across the 19-vector sentinel surface. "
                "2 non-penalty-path breaches remain (random_baseline_discrimination, "
                "collusion_temporal_pattern) — both synthetic harness artifacts, not adversary leaks."
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
                "insignia-local MCP. These cannot be executed from the offline repo "
                "— the insignia-local MCP server is not available in this environment."
            ),
            "step1_file_task": {
                "mcp_tool": "insignia-local.file_task",
                "arguments": {
                    "assignee": "orchestrator",
                    "priority": 10,
                    "description": (
                        "V14-R1 online-mode gate verification: run the live V14-R1-CORRECTED-KP "
                        "config on-chain across >= 2 reruns with different seeds. Verify the 6 "
                        "online-mode §9 gates (honest_mean_score >= 0.97, score_variance <= 0.002, "
                        "commit_reveal_effectiveness >= 0.667, consecutive_clean_validations >= 6, "
                        "convergence_contract unanimously met, sentinel_posture "
                        "SECURE_AND_IMPROVING+). If all clear, proceed to HITL promotion."
                    ),
                    "metadata": {
                        "cycle_step": "5_to_HITL",
                        "config_id": "V14-R1-CORRECTED-KP",
                        "offline_evidence_branch": "feat/signal-driven-adversary-penalties",
                        "offline_gate_check": "results/researcher_v14_r1_gate_check_2026-07-04T14-33-59.json",
                    },
                },
            },
            "step2_write_agent_memory": {
                "mcp_tool": "insignia-local.write_agent_memory",
                "arguments": {
                    "key": "v14_r1_online_verification_dispatch",
                    "value": {
                        "status": "DISPATCHED",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "config_id": "V14-R1-CORRECTED-KP",
                        "config_source": "v14_r1_corrected_config (MongoDB)",
                        "offline_evidence": {
                            "harness_mode_gates_passed": 4,
                            "harness_mode_gates_failed": 0,
                            "online_mode_gates_pending": 6,
                            "adversary_leaks": 0,
                            "separation": 0.9004,
                        },
                        "reruns_required": 2,
                        "gates_to_verify": [g["gate"] for g in ONLINE_GATES],
                    },
                },
            },
            "step3_invalidate_prior": {
                "mcp_tool": "insignia-local.write_agent_memory",
                "arguments": {
                    "key": "v14_r1_online_verification_result",
                    "value": None,
                    "note": "Clear any prior result so the orchestrator starts fresh.",
                },
            },
        },
        "post_verification": {
            "on_success": [
                "Write the verification result to agent_memory key `v14_r1_online_verification_result` with status=ALL_GATES_CLEARED.",
                "File a HITL task for `btcli hyperparameter apply` with the V14-R1-CORRECTED-KP config.",
                "Update `research_targets` in parameter_space.py: set target_achieved=True, current_candidate_status=promoted_to_production_reference.",
            ],
            "on_failure": [
                "Write the verification result to agent_memory key `v14_r1_online_verification_result` with status=GATES_FAILED and the failing gate(s).",
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

    json_path = out_dir / f"v14_r1_online_dispatch_manifest_{ts}.json"
    md_path = out_dir / f"v14_r1_online_dispatch_manifest_{ts}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    # Markdown manifest
    lines = [
        "# V14-R1 Online-Mode Gate Verification — Orchestrator Dispatch Manifest",
        "",
        f"**Generated:** {manifest['generated_at']}",
        f"**Config ID:** {manifest['config_id']}",
        f"**Config source:** {manifest['config_source']}",
        f"**Spec ref:** {manifest['spec_ref']}",
        "",
        "## Objective",
        "",
        manifest["objective"],
        "",
        "## Offline Evidence Package (from `feat/signal-driven-adversary-penalties`)",
        "",
        f"- **Step 2 (simulator):** `{manifest['offline_evidence_package']['step2_simulator']}`",
        f"- **Step 3 (sentinel):** `{manifest['offline_evidence_package']['step3_sentinel']}`",
        f"- **Step 4 (tuner seed):** `{manifest['offline_evidence_package']['step4_tuner_seed']}`",
        f"- **Step 5 (gate check):** `{manifest['offline_evidence_package']['step5_gate_check']}`",
        "",
        "**Offline summary:** " + manifest["offline_evidence_package"]["offline_summary"],
        "",
        "## 6 Online-Mode Gates to Verify On-Chain",
        "",
        "| # | Gate | Threshold | Evidence source | Note |",
        "|---|---|---|---|---|",
    ]
    for i, g in enumerate(manifest["online_gates_to_verify"], 1):
        lines.append(f"| {i} | `{g['gate']}` | {g['threshold']} | {g['evidence_source']} | {g['note']} |")
    lines.append("")
    lines.append("## Acceptance Criteria")
    lines.append("")
    ac = manifest["acceptance_criteria"]
    lines.append(f"- **Reruns required:** {ac['reruns_required']} ({ac['seeds']})")
    lines.append(f"- **All gates must clear:** {ac['all_gates_must_clear']}")
    lines.append(f"- **If any gate fails:** {ac['if_any_gate_fails']}")
    lines.append(f"- **If all gates clear:** {ac['if_all_gates_clear']}")
    lines.append("")
    lines.append("## MCP Dispatch Commands")
    lines.append("")
    lines.append("**⚠️ Cannot be executed from this repo** — the insignia-local MCP server is not available in this environment. Execute these in the orchestrator's agent env (or trigger via the swarm gateway).")
    lines.append("")
    mc = manifest["mcp_dispatch_commands"]
    lines.append(f"_{mc['note']}_")
    lines.append("")
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
    lines.append("## Post-Verification Actions")
    lines.append("")
    pv = manifest["post_verification"]
    lines.append("### On success (all 6 gates clear across >= 2 reruns)")
    lines.append("")
    for item in pv["on_success"]:
        lines.append(f"1. {item}")
    lines.append("")
    lines.append("### On failure (any gate fails)")
    lines.append("")
    for item in pv["on_failure"]:
        lines.append(f"1. {item}")
    lines.append("")
    lines.append(f"_JSON manifest: `{json_path.name}`_")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"V14-R1 online gate verification dispatch manifest")
    print(f"  6 online-mode gates to verify on-chain")
    print(f"  {manifest['acceptance_criteria']['reruns_required']} reruns required (different seeds)")
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
