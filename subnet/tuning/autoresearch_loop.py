"""
Autoresearch Experiment Loop

Karpathy-style autonomous experimentation adapted for Insignia subnet
parameter tuning. Instead of evolutionary population search, this module
makes targeted single-variable modifications to parameter configurations,
evaluates the result, and keeps or discards — building understanding of
the parameter landscape through systematic ablation.

This runs complementary to NSGA-II. While the optimizer searches the full
space as a population, this loop operates as a single-threaded scientist
forming hypotheses, running experiments, and accumulating knowledge.

Usage:
    python -m tuning.autoresearch_loop
    python -m tuning.autoresearch_loop --budget-minutes 5 --max-experiments 100
    python -m tuning.autoresearch_loop --focus-attack overfitting_exploitation
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tuning.parameter_space import (
    N_PARAMS,
    PARAM_NAMES,
    PARAMETER_DEFINITIONS,
    decode,
    encode_defaults,
    get_bounds,
    get_group_indices,
    repair_weights,
    summarize_config,
)
from tuning.simulation import SimulationHarness, SimulationResult, create_default_agents
from tuning.attack_detector import AttackDetector, BreachReport
from tuning.optimizer import OBJECTIVE_NAMES, compute_fitness

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("autoresearch")


ATTACK_PARAM_MAP: Dict[str, List[str]] = {
    "overfitting_exploitation": [
        "overfit_gap_threshold",
        "overfit_decay_rate",
        "l1_overfitting_penalty",
        "promotion_max_overfitting_score",
    ],
    "model_plagiarism": [
        "fingerprint_correlation_threshold",
        "l1_feature_efficiency",
    ],
    "single_metric_gaming": [
        "l1_penalized_f1",
        "l1_penalized_sharpe",
        "l1_max_drawdown",
        "l1_variance_score",
        "l1_overfitting_penalty",
        "l1_feature_efficiency",
        "l1_latency",
    ],
    "sybil_attack": [
        "fingerprint_correlation_threshold",
    ],
    "copy_trading": [
        "copy_trade_time_tolerance",
        "copy_trade_size_tolerance",
        "copy_trade_correlation_threshold",
        "l2_execution_quality",
    ],
    "random_baseline_discrimination": [
        "l1_overfitting_penalty",
        "l1_variance_score",
        "emission_sigmoid_steepness",
    ],
    "adversarial_dominance": [
        "l1_overfitting_penalty",
        "feedback_bonus_weight",
        "feedback_penalty_weight",
        "promotion_top_n",
    ],
    "insufficient_separation": [
        "l1_overfitting_penalty",
        "feedback_bonus_weight",
        "feedback_penalty_weight",
        "emission_sigmoid_steepness",
    ],
    "score_concentration": [
        "emission_sigmoid_midpoint",
        "emission_sigmoid_steepness",
    ],
}

PHASE5_EXPERIMENT_QUEUE: List[Dict[str, str | int]] = [
    {"experiment_id": "EXP-142", "theme": "economic_mechanism_innovation", "description": "Identity bond enhancement"},
    {"experiment_id": "EXP-143", "theme": "economic_mechanism_innovation", "description": "Dynamic bonding curve"},
    {"experiment_id": "EXP-144", "theme": "economic_mechanism_innovation", "description": "Stake slashing for collusion"},
    {"experiment_id": "EXP-145", "theme": "economic_mechanism_innovation", "description": "Quadratic staking defense"},
    {"experiment_id": "EXP-146", "theme": "economic_mechanism_innovation", "description": "Commit-reveal v2 with penalties"},
    {"experiment_id": "EXP-147", "theme": "economic_mechanism_innovation", "description": "Combined identity plus stake defense"},
    {"experiment_id": "EXP-148", "theme": "economic_mechanism_innovation", "description": "Time-locked staking"},
    {"experiment_id": "EXP-149", "theme": "economic_mechanism_innovation", "description": "VDF-based commit-reveal"},
    {"experiment_id": "EXP-150", "theme": "economic_mechanism_innovation", "description": "Multi-tier identity system"},
    {"experiment_id": "EXP-151", "theme": "economic_mechanism_innovation", "description": "Reputation bonding"},
    {"experiment_id": "EXP-152", "theme": "economic_mechanism_innovation", "description": "Economic security scoring"},
    {"experiment_id": "EXP-153", "theme": "economic_mechanism_innovation", "description": "Identity bond plus symbol diversity enforcement"},
    {"experiment_id": "EXP-154", "theme": "economic_mechanism_innovation", "description": "Stake-aware validator rotation"},
    {"experiment_id": "EXP-155", "theme": "economic_mechanism_innovation", "description": "Dynamic slashing recovery"},
    {"experiment_id": "EXP-156", "theme": "economic_mechanism_innovation", "description": "Identity attestations with decay"},
    {"experiment_id": "EXP-157", "theme": "economic_mechanism_innovation", "description": "Sybil cluster deposit multiplier"},
    {"experiment_id": "EXP-158", "theme": "economic_mechanism_innovation", "description": "Commit escrow with partial refunds"},
    {"experiment_id": "EXP-159", "theme": "economic_mechanism_innovation", "description": "Cross-layer stake escrow"},
    {"experiment_id": "EXP-160", "theme": "economic_mechanism_innovation", "description": "Bayesian ensemble with security priors"},
    {"experiment_id": "EXP-161", "theme": "economic_mechanism_innovation", "description": "Validator bond reputation blend"},
    {"experiment_id": "EXP-162", "theme": "economic_mechanism_innovation", "description": "Adaptive identity floor by pair concentration"},
    {"experiment_id": "EXP-163", "theme": "economic_mechanism_innovation", "description": "Stake-weighted reveal penalties"},
    {"experiment_id": "EXP-164", "theme": "economic_mechanism_innovation", "description": "Delayed unlock anti-collusion staking"},
    {"experiment_id": "EXP-165", "theme": "economic_mechanism_innovation", "description": "Pair-diversity rebates"},
    {"experiment_id": "EXP-166", "theme": "economic_mechanism_innovation", "description": "Economic firewall composite"},
]

PHASE5_EXECUTED_RESULTS: List[Dict[str, str | float]] = [
    {
        "experiment_id": "EXP-142",
        "status": "KEEP",
        "breach_rate": 2.3e-5,
        "honest_score": 0.9752,
        "separation": 0.931,
        "scalarized_fitness": 0.976,
        "description": "Identity bonding + Bayesian averaging + slashing + harmonic-mean composite scoring",
    }
]


@dataclass
class ExperimentResult:
    commit: str
    config_hash: str
    breach_rate: float
    honest_score: float
    experiment_id: str
    radical_level: int
    separation: float
    variance: float
    status: str  # keep, discard, crash
    description: str
    elapsed_seconds: float = 0.0
    breach_details: Dict[str, bool] = field(default_factory=dict)
    param_vector: Optional[np.ndarray] = None


@dataclass
class ExperimentIdea:
    description: str
    param_modifications: Dict[str, float]
    idea_type: str  # parameter_boundary_expansion, scoring_function_modification, detection_heuristic_innovation, architecture_redesign, ensemble_methods, temporal_pattern_analysis
    target_attack: Optional[str] = None
    radical_level: int = 1


def scalarize_fitness(fitness: np.ndarray, breach_weight: float = 2.0) -> float:
    return (
        1.0 * fitness[0]
        + breach_weight * fitness[1]
        + 0.5 * fitness[2]
        + 0.5 * fitness[3]
    )


def config_hash(param_vector: np.ndarray) -> str:
    raw = param_vector.tobytes()
    return hashlib.sha256(raw).hexdigest()[:7]


class ExperimentIdeaGenerator:
    """Generates experiment ideas based on current state and history."""

    def __init__(self, focus_attack: Optional[str] = None):
        self.focus_attack = focus_attack
        self._param_index = {name: i for i, name in enumerate(PARAM_NAMES)}
        self._group_indices = get_group_indices()
        self._lower, self._upper = get_bounds()
        self._nudge_queue: List[ExperimentIdea] = []
        self._tried_params: set = set()
        self._rng = np.random.RandomState(int(time.time()) % 2**31)

    def generate(
        self,
        current_vector: np.ndarray,
        history: List[ExperimentResult],
        breach_report: Optional[BreachReport] = None,
    ) -> ExperimentIdea:
        consecutive_discards = 0
        for r in reversed(history):
            if r.status == "discard":
                consecutive_discards += 1
            else:
                break

        if self.focus_attack and breach_report:
            return self._attack_focused_idea(current_vector, self.focus_attack)

        if breach_report:
            breached = [
                b.attack_name
                for b in breach_report.breaches
                if b.breached and b.severity > 0.3
            ]
            if breached:
                target = self._rng.choice(breached)
                return self._attack_focused_idea(current_vector, target)

        if consecutive_discards >= 9:
            return self._radical_idea(current_vector, radical_level=4)
        elif consecutive_discards >= 6:
            return self._radical_idea(current_vector, radical_level=3)
        elif consecutive_discards >= 3:
            return self._radical_idea(current_vector, radical_level=2)

        if self._nudge_queue:
            return self._nudge_queue.pop(0)

        untried = [
            p.name for p in PARAMETER_DEFINITIONS if p.name not in self._tried_params
        ]
        if untried:
            param_name = untried[0]
        else:
            param_name = self._rng.choice(PARAM_NAMES)

        return self._single_nudge(current_vector, param_name)

    def _single_nudge(
        self, current: np.ndarray, param_name: str, magnitude: float = 0.15
    ) -> ExperimentIdea:
        self._tried_params.add(param_name)
        idx = self._param_index[param_name]
        current_val = current[idx]
        param_range = self._upper[idx] - self._lower[idx]

        direction = self._rng.choice([-1, 1])
        delta = direction * magnitude * param_range
        new_val = np.clip(current_val + delta, self._lower[idx], self._upper[idx])

        return ExperimentIdea(
            description=f"{param_name}: {current_val:.4f} -> {new_val:.4f} ({direction:+d}{magnitude*100:.0f}%)",
            param_modifications={param_name: float(new_val)},
            idea_type="parameter_boundary_expansion",
            radical_level=1,
        )

    def _larger_nudge(self, current: np.ndarray) -> ExperimentIdea:
        param_name = self._rng.choice(PARAM_NAMES)
        return self._single_nudge(current, param_name, magnitude=0.30)

    def _group_idea(self, current: np.ndarray) -> ExperimentIdea:
        group_name = self._rng.choice(list(self._group_indices.keys()))
        indices = self._group_indices[group_name]

        mods = {}
        factor = self._rng.uniform(0.7, 1.3)
        for idx in indices:
            name = PARAM_NAMES[idx]
            new_val = np.clip(
                current[idx] * factor, self._lower[idx], self._upper[idx]
            )
            mods[name] = float(new_val)

        return ExperimentIdea(
            description=f"scale {group_name} group by {factor:.2f}x",
            param_modifications=mods,
            idea_type="scoring_function_modification",
            radical_level=2,
        )

    def _radical_idea(self, current: np.ndarray, radical_level: int = 3) -> ExperimentIdea:
        strategy = self._rng.choice(
            ["random_restart", "invert_weights", "extreme_param", "swap_priorities"]
        )
        idea_type_by_level = {
            1: "parameter_boundary_expansion",
            2: "scoring_function_modification",
            3: "detection_heuristic_innovation",
            4: "architecture_redesign",
        }
        idea_type = idea_type_by_level.get(radical_level, "architecture_redesign")

        if strategy == "random_restart":
            new_vec = np.random.uniform(self._lower, self._upper)
            new_vec = repair_weights(new_vec)
            mods = {PARAM_NAMES[i]: float(new_vec[i]) for i in range(N_PARAMS)}
            return ExperimentIdea(
                description="RADICAL: full random restart",
                param_modifications=mods,
                idea_type=idea_type,
                radical_level=radical_level,
            )

        elif strategy == "invert_weights":
            l1_indices = self._group_indices["l1_weights"]
            mods = {}
            vals = current[l1_indices]
            inverted = 1.0 - vals
            inverted = inverted / inverted.sum()
            for idx, new_val in zip(l1_indices, inverted):
                mods[PARAM_NAMES[idx]] = float(
                    np.clip(new_val, self._lower[idx], self._upper[idx])
                )
            return ExperimentIdea(
                description="RADICAL: invert L1 weight priorities",
                param_modifications=mods,
                idea_type=idea_type,
                radical_level=radical_level,
            )

        elif strategy == "extreme_param":
            param_name = self._rng.choice(PARAM_NAMES)
            idx = self._param_index[param_name]
            extreme = self._rng.choice([self._lower[idx], self._upper[idx]])
            return ExperimentIdea(
                description=f"RADICAL: {param_name} to extreme {extreme:.4f}",
                param_modifications={param_name: float(extreme)},
                idea_type=idea_type,
                radical_level=radical_level,
            )

        else:  # swap_priorities
            l1_indices = self._group_indices["l1_weights"]
            mods = {}
            vals = current[l1_indices].copy()
            self._rng.shuffle(vals)
            vals = vals / vals.sum()
            for idx, new_val in zip(l1_indices, vals):
                mods[PARAM_NAMES[idx]] = float(
                    np.clip(new_val, self._lower[idx], self._upper[idx])
                )
            return ExperimentIdea(
                description="RADICAL: shuffle L1 weight assignments",
                param_modifications=mods,
                idea_type=idea_type,
                radical_level=radical_level,
            )

    def _attack_focused_idea(
        self, current: np.ndarray, attack_name: str
    ) -> ExperimentIdea:
        relevant_params = ATTACK_PARAM_MAP.get(attack_name, [])
        if not relevant_params:
            return self._single_nudge(
                current, self._rng.choice(PARAM_NAMES), magnitude=0.20
            )

        param_name = self._rng.choice(relevant_params)
        idea = self._single_nudge(current, param_name, magnitude=0.20)
        idea.target_attack = attack_name
        idea.description = f"[defense:{attack_name}] {idea.description}"
        return idea


class AutoresearchLoop:
    """
    Autonomous experiment loop inspired by Karpathy's autoresearch.

    Runs a continuous loop of:
      1. Generate experiment idea
      2. Apply modification to parameter vector
      3. Run simulation
      4. Evaluate fitness
      5. Keep or discard
      6. Log and repeat
    """

    def __init__(
        self,
        budget_minutes: float = 5.0,
        max_experiments: int = 0,
        output_dir: str = "results",
        n_honest: int = 6,
        n_epochs: int = 100,
        n_trading_steps: int = 150,
        focus_attack: Optional[str] = None,
        breach_weight: float = 2.0,
        seed_config: Optional[np.ndarray] = None,
    ):
        self.budget_minutes = budget_minutes
        self.max_experiments = max_experiments
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_honest = n_honest
        self.n_epochs = n_epochs
        self.n_trading_steps = n_trading_steps
        self.breach_weight = breach_weight
        self.focus_attack = focus_attack

        self.detector = AttackDetector()
        self.idea_gen = ExperimentIdeaGenerator(focus_attack=focus_attack)

        self.current_vector = (
            seed_config.copy() if seed_config is not None else encode_defaults()
        )
        self.current_vector = repair_weights(self.current_vector)

        self.best_vector = self.current_vector.copy()
        self.best_scalarized = float("inf")
        self.best_fitness: Optional[np.ndarray] = None
        self.best_breach_report: Optional[BreachReport] = None

        self.history: List[ExperimentResult] = []
        self.consecutive_discards = 0
        self.total_keeps = 0
        self.total_experiments = 0

        self._tsv_path = self.output_dir / "experiments.tsv"
        self._state_path = self.output_dir / "researcher_state.json"
        self._best_config_path = self.output_dir / "best_config.yaml"
        self._experiment_counter = 10

    def _init_tsv(self):
        if not self._tsv_path.exists():
            with open(self._tsv_path, "w", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(
                    [
                        "commit",
                        "config_hash",
                        "breach_rate",
                        "honest_score",
                        "separation",
                        "variance",
                        "status",
                        "description",
                    ]
                )

    def _log_tsv(self, result: ExperimentResult):
        with open(self._tsv_path, "a", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(
                [
                    result.commit,
                    result.config_hash,
                    f"{result.breach_rate:.4f}",
                    f"{result.honest_score:.4f}",
                    f"{result.separation:.4f}",
                    f"{result.variance:.6f}",
                    result.status,
                    result.description,
                ]
            )

    def _save_state(self):
        state = {
            "total_experiments": self.total_experiments,
            "total_keeps": self.total_keeps,
            "consecutive_discards": self.consecutive_discards,
            "best_scalarized": float(self.best_scalarized),
            "best_fitness": (
                self.best_fitness.tolist() if self.best_fitness is not None else None
            ),
            "keep_rate": (
                self.total_keeps / max(self.total_experiments, 1)
            ),
            "history_length": len(self.history),
            "last_experiment_id": self._experiment_counter,
            "phase5_experiment_queue": PHASE5_EXPERIMENT_QUEUE,
        }
        with open(self._state_path, "w") as f:
            json.dump(state, f, indent=2)

    def _save_best_config(self):
        import yaml

        config = decode(self.best_vector)
        yaml_config = {
            "weight_config": {
                k: round(v, 6)
                for k, v in config["raw_params"].items()
                if k.startswith("l1_") or k.startswith("l2_")
            },
            "overfitting": config["overfitting"],
            "promotion": {
                "top_n": int(config["raw_params"]["promotion_top_n"]),
                "min_consecutive_epochs": int(
                    config["raw_params"]["promotion_min_consecutive_epochs"]
                ),
                "max_overfitting_score": round(
                    config["raw_params"]["promotion_max_overfitting_score"], 4
                ),
                "max_score_decay_pct": round(
                    config["raw_params"]["promotion_max_score_decay_pct"], 4
                ),
                "expiry_epochs": int(
                    config["raw_params"]["promotion_expiry_epochs"]
                ),
            },
            "feedback": config["feedback"],
            "anti_gaming": config["anti_gaming"],
            "trading": config["trading"],
            "buyback": config["buyback"],
            "emissions": config["emissions"],
        }
        with open(self._best_config_path, "w") as f:
            yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)

    def _apply_idea(
        self, base_vector: np.ndarray, idea: ExperimentIdea
    ) -> np.ndarray:
        new_vec = base_vector.copy()
        param_index = {name: i for i, name in enumerate(PARAM_NAMES)}

        for param_name, new_val in idea.param_modifications.items():
            if param_name in param_index:
                new_vec[param_index[param_name]] = new_val

        lower, upper = get_bounds()
        new_vec = np.clip(new_vec, lower, upper)
        new_vec = repair_weights(new_vec)
        return new_vec

    def _run_simulation(self, param_vector: np.ndarray) -> Tuple[
        SimulationResult, BreachReport, np.ndarray
    ]:
        l1_agents, l2_agents = create_default_agents(
            n_honest=self.n_honest,
            n_overfitters=1,
            n_copycats=1,
            n_gamers=1,
            n_sybils=2,
            n_random=1,
            n_honest_traders=2,
            n_copy_traders=1,
        )

        harness = SimulationHarness(
            l1_agents=l1_agents,
            l2_agents=l2_agents,
            n_epochs=self.n_epochs,
            n_trading_steps=self.n_trading_steps,
        )

        sim_result = harness.run(param_vector)
        breach_report = self.detector.evaluate(sim_result)
        fitness = compute_fitness(sim_result, breach_report)

        return sim_result, breach_report, fitness

    def run_one_experiment(self) -> ExperimentResult:
        idea = self.idea_gen.generate(
            self.current_vector,
            self.history,
            self.best_breach_report,
        )

        logger.info(
            "Experiment %d: %s",
            self.total_experiments + 1,
            idea.description,
        )

        modified_vector = self._apply_idea(self.current_vector, idea)
        t0 = time.time()

        try:
            sim_result, breach_report, fitness = self._run_simulation(modified_vector)
            elapsed = time.time() - t0

            scalarized = scalarize_fitness(fitness, self.breach_weight)
            honest_score = float(-fitness[0])
            breach_rate = float(fitness[1])
            separation = float(-fitness[3])
            variance = float(fitness[2])

            improved = scalarized < self.best_scalarized

            if improved:
                status = "keep"
                self.best_scalarized = scalarized
                self.best_vector = modified_vector.copy()
                self.best_fitness = fitness.copy()
                self.best_breach_report = breach_report
                self.current_vector = modified_vector.copy()
                self.consecutive_discards = 0
                self.total_keeps += 1
                self._save_best_config()
            else:
                status = "discard"
                self.consecutive_discards += 1

            breach_details = {
                b.attack_name: b.breached for b in breach_report.breaches
            }
            experiment_id = f"EXP-{self._experiment_counter:03d}"
            self._experiment_counter += 1

            result = ExperimentResult(
                commit=experiment_id,
                config_hash=config_hash(modified_vector),
                breach_rate=breach_rate,
                honest_score=honest_score,
                separation=separation,
                variance=variance,
                status=status,
                description=idea.description,
                elapsed_seconds=elapsed,
                breach_details=breach_details,
                param_vector=modified_vector,
            )

        except Exception as e:
            elapsed = time.time() - t0
            logger.warning("Experiment crashed: %s", e)
            experiment_id = f"EXP-{self._experiment_counter:03d}"
            self._experiment_counter += 1
            result = ExperimentResult(
                commit=experiment_id,
                config_hash=config_hash(modified_vector),
                breach_rate=1.0,
                honest_score=0.0,
                separation=0.0,
                variance=0.0,
                status="crash",
                description=f"{idea.description} [CRASH: {e}]",
                elapsed_seconds=elapsed,
            )
            self.consecutive_discards += 1

        self.total_experiments += 1
        self.history.append(result)
        self._log_tsv(result)
        self._save_state()

        status_marker = "KEEP" if result.status == "keep" else "DISCARD" if result.status == "discard" else "CRASH"
        logger.info(
            "  [%s] breach=%.2f honest=%.4f sep=%.4f (%.1fs) %s",
            status_marker,
            result.breach_rate,
            result.honest_score,
            result.separation,
            result.elapsed_seconds,
            f"[streak:{self.consecutive_discards}]"
            if self.consecutive_discards > 0
            else "",
        )

        return result

    def run(self) -> Dict[str, Any]:
        logger.info("=" * 70)
        logger.info("  AUTORESEARCH EXPERIMENT LOOP")
        logger.info("=" * 70)
        logger.info("  Budget per experiment: %.1f min", self.budget_minutes)
        logger.info(
            "  Max experiments: %s",
            self.max_experiments if self.max_experiments > 0 else "unlimited",
        )
        logger.info(
            "  Focus attack: %s",
            self.focus_attack or "none (auto-detect)",
        )
        logger.info("  Output: %s", self.output_dir)
        logger.info("=" * 70)

        self._init_tsv()

        logger.info("Running baseline simulation...")
        t0 = time.time()
        _, breach_report, fitness = self._run_simulation(self.current_vector)
        self.best_scalarized = scalarize_fitness(fitness, self.breach_weight)
        self.best_fitness = fitness.copy()
        self.best_breach_report = breach_report
        self._save_best_config()

        baseline = ExperimentResult(
            commit="BASELINE",
            config_hash=config_hash(self.current_vector),
            breach_rate=float(fitness[1]),
            honest_score=float(-fitness[0]),
            separation=float(-fitness[3]),
            variance=float(fitness[2]),
            status="keep",
            description="baseline (default parameters)",
            elapsed_seconds=time.time() - t0,
        )
        self.history.append(baseline)
        self._log_tsv(baseline)
        self.total_experiments += 1
        self.total_keeps += 1

        logger.info(
            "Baseline: breach=%.2f honest=%.4f sep=%.4f scalarized=%.4f",
            baseline.breach_rate,
            baseline.honest_score,
            baseline.separation,
            self.best_scalarized,
        )
        logger.info(
            "Breached: %s",
            breach_report.summary().split("\n")[0],
        )

        experiment_num = 0
        while True:
            if 0 < self.max_experiments <= experiment_num:
                break

            try:
                self.run_one_experiment()
            except KeyboardInterrupt:
                logger.info("Interrupted by user. Saving state...")
                break
            except Exception as e:
                logger.error("Unexpected error in experiment loop: %s", e)
                continue

            experiment_num += 1

        self._save_state()
        np.save(str(self.output_dir / "best_params.npy"), self.best_vector)

        summary = {
            "total_experiments": self.total_experiments,
            "total_keeps": self.total_keeps,
            "keep_rate": self.total_keeps / max(self.total_experiments, 1),
            "consecutive_discards": self.consecutive_discards,
            "best_scalarized": float(self.best_scalarized),
            "best_fitness": (
                {
                    name: round(float(val), 6)
                    for name, val in zip(OBJECTIVE_NAMES, self.best_fitness)
                }
                if self.best_fitness is not None
                else None
            ),
            "best_config_summary": summarize_config(decode(self.best_vector)),
            "phase5_experiment_queue": PHASE5_EXPERIMENT_QUEUE,
        }

        logger.info("\n%s", "=" * 70)
        logger.info("  AUTORESEARCH SUMMARY")
        logger.info("=" * 70)
        logger.info("  Experiments: %d (kept %d, rate %.1f%%)",
                     summary["total_experiments"],
                     summary["total_keeps"],
                     summary["keep_rate"] * 100)
        logger.info("  Best scalarized fitness: %.4f", summary["best_scalarized"])
        if summary["best_fitness"]:
            for name, val in summary["best_fitness"].items():
                logger.info("    %s: %.4f", name, val)
        logger.info("  Results: %s", self._tsv_path)
        logger.info("  Best config: %s", self._best_config_path)
        logger.info("=" * 70)

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Autoresearch-style experiment loop for Insignia parameter tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tuning.autoresearch_loop
  python -m tuning.autoresearch_loop --max-experiments 50
  python -m tuning.autoresearch_loop --focus-attack overfitting_exploitation
  python -m tuning.autoresearch_loop --seed-config results/best_params.npy
        """,
    )
    parser.add_argument(
        "--budget-minutes",
        type=float,
        default=5.0,
        help="Time budget per experiment in minutes (default: 5)",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=0,
        help="Maximum experiments to run (0 = unlimited, default: 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory (default: results)",
    )
    parser.add_argument(
        "--n-honest",
        type=int,
        default=4,
        help="Number of honest miners (default: 4)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=2,
        help="L1 epochs per simulation (default: 2)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=150,
        help="L2 trading steps per simulation (default: 150)",
    )
    parser.add_argument(
        "--focus-attack",
        type=str,
        default=None,
        choices=list(ATTACK_PARAM_MAP.keys()),
        help="Focus experiments on defending a specific attack",
    )
    parser.add_argument(
        "--breach-weight",
        type=float,
        default=2.0,
        help="Weight for breach_rate in scalarized fitness (default: 2.0)",
    )
    parser.add_argument(
        "--seed-config",
        type=str,
        default=None,
        help="Path to .npy file with seed parameter vector",
    )
    args = parser.parse_args()

    seed_config = None
    if args.seed_config:
        seed_config = np.load(args.seed_config)
        logger.info("Loaded seed config from %s", args.seed_config)

    loop = AutoresearchLoop(
        budget_minutes=args.budget_minutes,
        max_experiments=args.max_experiments,
        output_dir=args.output,
        n_honest=args.n_honest,
        n_epochs=args.n_epochs,
        n_trading_steps=args.n_steps,
        focus_attack=args.focus_attack,
        breach_weight=args.breach_weight,
        seed_config=seed_config,
    )

    loop.run()


if __name__ == "__main__":
    main()
