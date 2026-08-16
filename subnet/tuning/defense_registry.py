"""Defense wiring registry — the V13-R3 copycat lesson, made permanent.

V13-R3 failed empirical validation because the surrogate model assumed
anti-gaming defenses that the scoring path never implemented (copycat scored
0.7333 against a 0.90 separation gate; see `results/` postmortems and the
§Incentive Design warning in `docs/SUBNET_SPEC.md`). This registry maps every
anti-gaming control to:

1. its live-path implementation symbol (import-checked), and
2. the runtime evidence proving the simulator actually exercises it — penalty
   multipliers applied in the scoring loop and telemetry keys emitted in the
   result. Catches defenses "assumed but unimplemented".

The reverse direction is enforced too: every penalty-multiplier constant and
ensemble detector key the simulator defines must be claimed by a registry
entry here. Catches controls "defined but never used" (the unused
`cross_metric_correlation_threshold` failure mode).

All three directions are enforced by `subnet/tests/test_defense_wiring.py`.
Standing rule: no defense enters a knee-point config unless the wiring test
passes (see `docs/sentinel.md`, Known issues).

Evidence occurrence counts: a multiplier must appear at least twice in
`simulation.py` (definition + application in the scoring loop); a telemetry
key once (emission).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple


@dataclass(frozen=True)
class DefenseEntry:
    """One anti-gaming control and its proof of exercise."""

    control_id: str
    implementation: str                       # "module:Symbol", import-checked
    sim_evidence: Tuple[Tuple[str, int], ...] = field(default_factory=tuple)
    # (identifier in simulation.py source, min occurrences)
    live_path_only_rationale: str = ""        # required when sim_evidence is empty

    def __post_init__(self):
        if not self.sim_evidence and not self.live_path_only_rationale:
            raise ValueError(
                f"{self.control_id}: empty sim_evidence requires a "
                "live_path_only_rationale (an explicit, auditable decision)"
            )


DEFENSE_REGISTRY: Dict[str, DefenseEntry] = {
    # --- Scoring-loop penalty multipliers (EXP-ADVERSARY-COVERAGE-002) ---
    "COPYCAT-SUPPRESS": DefenseEntry(
        control_id="COPYCAT-SUPPRESS",
        implementation="insignia.incentive:ModelFingerprinter",
        sim_evidence=(("_COPYCAT_MULTIPLIER", 2),),
    ),
    "COPYTRADE-SUPPRESS": DefenseEntry(
        control_id="COPYTRADE-SUPPRESS",
        implementation="insignia.incentive:CopyTradeDetector",
        sim_evidence=(("_COPYTRADER_MULTIPLIER", 2),),
    ),
    "OVERFIT-GATE": DefenseEntry(
        control_id="OVERFIT-GATE",
        implementation="insignia.scoring:ReferenceOverfittingDetector",
        sim_evidence=(("_OVERFITTER_MULTIPLIER", 2),),
    ),
    "MULTI-OBJECTIVE-GUARD": DefenseEntry(
        control_id="MULTI-OBJECTIVE-GUARD",
        implementation="insignia.scoring:CompositeScorer",
        sim_evidence=(("_SINGLE_METRIC_MULTIPLIER", 2),),
    ),
    "COLLUSION-GUARD": DefenseEntry(
        control_id="COLLUSION-GUARD",
        implementation="insignia.pairing:PairingConfig",
        sim_evidence=(("_COLLUDER_MULTIPLIER", 2),),
    ),
    "PARTNER-GAMING-GUARD": DefenseEntry(
        control_id="PARTNER-GAMING-GUARD",
        implementation="insignia.pairing:PairFitness",
        sim_evidence=(("_PARTNER_GAMER_MULTIPLIER", 2),),
    ),
    # --- Sybil defenses (structural holdout post-commit-reveal) ---
    "SYBIL-FLOOR": DefenseEntry(
        control_id="SYBIL-FLOOR",
        implementation="tuning.pc_vh_006_symbol_diversity:SymbolDiversityEnforcer",
        sim_evidence=(
            ("_SYBIL_FLOOR_MULTIPLIER", 2),
            ("pc_vh_006_symbol_diversity", 1),
        ),
    ),
    "SYBIL-SIGNAL": DefenseEntry(
        control_id="SYBIL-SIGNAL",
        implementation="tuning.sentinel_symbol_monitor:SentinelSymbolMonitor",
        sim_evidence=(("sybil_diversity_detector", 1),),
    ),
    # --- Ensemble detector telemetry ---
    "TEMPORAL-ANOMALY": DefenseEntry(
        control_id="TEMPORAL-ANOMALY",
        implementation="insignia.incentive:CommitRevealManager",
        sim_evidence=(("temporal_anomaly_detector", 1),),
    ),
    "CROSS-CORRELATION": DefenseEntry(
        control_id="CROSS-CORRELATION",
        implementation="insignia.incentive:CopyTradeDetector",
        sim_evidence=(("cross_correlation_detector", 1),),
    ),
    "BEHAVIORAL-FINGERPRINT": DefenseEntry(
        control_id="BEHAVIORAL-FINGERPRINT",
        implementation="insignia.incentive:ModelFingerprinter",
        sim_evidence=(("behavioral_fingerprinting", 1),),
    ),
    # --- Commit-reveal and timing ---
    "COMMIT-REVEAL": DefenseEntry(
        control_id="COMMIT-REVEAL",
        implementation="insignia.incentive:CommitRevealManager",
        sim_evidence=(
            ("selective_reveal_penalties", 1),
            ("no_reveal_streaks", 1),
        ),
    ),
    "TIMING-SLA": DefenseEntry(
        control_id="TIMING-SLA",
        implementation="insignia.incentive:CommitRevealConfig",
        sim_evidence=(("timing_attack_composite_severity", 1),),
    ),
    # --- Live-path only (explicit, audited decision) ---
    "RATE-LIMIT": DefenseEntry(
        control_id="RATE-LIMIT",
        implementation="insignia.incentive:SubmissionRateLimit",
        sim_evidence=(),
        live_path_only_rationale=(
            "Rate limiting gates submission intake in neurons/model_validator.py; "
            "the simulator models population-level adversaries, not submission spam."
        ),
    ),
    "NATIVE-COLLATERAL-GATE": DefenseEntry(
        control_id="NATIVE-COLLATERAL-GATE",
        implementation="insignia.native_collateral:apply_collateral_gate",
        sim_evidence=(),
        live_path_only_rationale=(
            "Native Subtensor registration collateral is an on-chain time-bond "
            "(lock_share / drain_ratio) enforced by zeroing Yuma weights. The "
            "simulator scores paper/live books, not registration locks or "
            "emission-drain horizons; FreezeLedger + the paired validator gate "
            "are the live path (docs/COLLATERAL.md)."
        ),
    ),
}


def claimed_evidence() -> Set[str]:
    """Union of all simulator evidence identifiers claimed by the registry."""
    return {identifier for entry in DEFENSE_REGISTRY.values() for identifier, _ in entry.sim_evidence}


def discover_simulator_controls(sim_source: str) -> Dict[str, Set[str]]:
    """Auto-discover the simulator's anti-gaming surface from its source.

    Two families are discovered, mirroring how controls are actually wired:
      - penalty multiplier constants (`_X_MULTIPLIER = <float>`), applied in
        the scoring loop;
      - ensemble detector telemetry keys (`"*_detector"`, `"*fingerprinting*"`),
        emitted into `result.ensemble_signals`.
    """
    multipliers = set(re.findall(r"^(_[A-Z_]+_MULTIPLIER)\s*=\s*[\d.]", sim_source, re.M))
    detectors = set(re.findall(r'"(\w+(?:_detector|_fingerprinting)\w*)"', sim_source))
    return {"penalty_multipliers": multipliers, "ensemble_detectors": detectors}
