"""
Insignia Paired Genetic Incentive Engine

Implements the single-mechanism replacement for the old two-layer
(L1 model -> promotion -> L2 strategy) pipeline. A candidate strategy is a
``(researcher_uid, trader_uid)`` pair (a *genome*); the set of active pairs in
an epoch is one *generation* of a population. Pairs are:

  1. assigned deterministically from chain block state (``ChainSeededPairing``),
  2. jointly evaluated into a multi-objective fitness vector (the validator
     builds these from ``insignia.scoring.combine_pair_scores``),
  3. ranked with NSGA-II non-dominated sorting + crowding (``NSGA2Matchmaker``),
  4. screened for collusion (``CollusionGraphDetector``), and
  5. converted to a single per-miner emission weight via a variance-penalized
     marginal contribution (``MarginalContributionCredit``).

This module depends only on numpy so it can be unit-tested in isolation. The
validator (``neurons/validator.py``) and the simulation harness
(``tuning/simulation.py``) drive it.

See ``docs/PAIRING_MECHANISM.md`` for the full design.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairGenome:
    """A candidate strategy: one researcher miner matched to one trader miner."""

    researcher_uid: str
    trader_uid: str

    @property
    def key(self) -> str:
        return f"{self.researcher_uid}::{self.trader_uid}"


@dataclass
class PairFitness:
    """
    Evaluated fitness of a single pair.

    ``objectives`` is the NSGA-II vector (all MINIMIZED). The validator builds
    it from the joint model + trading scores, conventionally:
        [-model_composite, -trading_composite, trading_max_drawdown,
         -trading_consistency]

    ``pair_composite`` is the scalar blend used for collusion analysis and
    human-readable ranking.
    """

    genome: PairGenome
    objectives: np.ndarray
    model_composite: float = 0.0
    trading_composite: float = 0.0
    pair_composite: float = 0.0
    raw: Dict[str, float] = field(default_factory=dict)

    # Filled in by the matchmaker
    rank: int = -1
    crowding: float = 0.0
    selection_score: float = 0.0
    collusion_flagged: bool = False

    def __post_init__(self):
        self.objectives = np.asarray(self.objectives, dtype=float)


@dataclass
class PairingConfig:
    """Tunable parameters for the paired genetic mechanism."""

    partners_per_miner: int = 3       # K: min partners each miner is judged against
    elite_fraction: float = 0.30      # fraction of pairs retained as elites
    mutation_rate: float = 0.20       # probability mass for random re-pairings
    pair_blend_alpha: float = 0.50    # weight on model composite in pair composite
    marginal_contribution_weight: float = 0.50  # lambda in mean - lambda*std credit
    fixed_pair_correlation_threshold: float = 0.85  # collusion interaction threshold
    pairing_seed_source: str = "chain_block_hash"
    max_pairs: int = 64               # population cap per generation
    crossover_prob: float = 0.70      # prob of keeping each elite x elite recombination


@dataclass
class CollusionReport:
    """Result of collusion-graph screening over a generation."""

    flagged: List[Tuple[str, float]] = field(default_factory=list)  # (pair_key, severity)
    details: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def n_flagged(self) -> int:
        return len(self.flagged)


# ---------------------------------------------------------------------------
# NSGA-II non-dominated sorting + crowding distance (minimization)
# ---------------------------------------------------------------------------

def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """Return True if objective vector ``a`` Pareto-dominates ``b`` (minimization)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(objectives: np.ndarray) -> List[List[int]]:
    """
    Deb's fast non-dominated sort. Returns a list of fronts; front 0 is the
    Pareto-optimal (non-dominated) set. Each front is a list of row indices.
    """
    n = len(objectives)
    if n == 0:
        return []

    dominated_by: List[List[int]] = [[] for _ in range(n)]
    domination_count = np.zeros(n, dtype=int)
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objectives[p], objectives[q]):
                dominated_by[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in dominated_by[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    fronts.pop()  # drop trailing empty front
    return fronts


def crowding_distance(objectives: np.ndarray, front: List[int]) -> Dict[int, float]:
    """Standard NSGA-II crowding distance for one front."""
    distances = {i: 0.0 for i in front}
    if len(front) <= 2:
        for i in front:
            distances[i] = float("inf")
        return distances

    n_obj = objectives.shape[1]
    for k in range(n_obj):
        ordered = sorted(front, key=lambda i: objectives[i, k])
        distances[ordered[0]] = float("inf")
        distances[ordered[-1]] = float("inf")
        v_min = objectives[ordered[0], k]
        v_max = objectives[ordered[-1], k]
        span = v_max - v_min
        if span < 1e-12:
            continue
        for idx in range(1, len(ordered) - 1):
            prev_v = objectives[ordered[idx - 1], k]
            next_v = objectives[ordered[idx + 1], k]
            distances[ordered[idx]] += float((next_v - prev_v) / span)
    return distances


class NSGA2Matchmaker:
    """Ranks a generation of pairs by Pareto front (85%) and crowding (15%)."""

    def __init__(self, front_weight: float = 0.85):
        self.front_weight = front_weight

    def rank(self, fitnesses: List[PairFitness]) -> List[PairFitness]:
        if not fitnesses:
            return []

        objectives = np.array([f.objectives for f in fitnesses], dtype=float)
        fronts = fast_non_dominated_sort(objectives)
        n_fronts = max(len(fronts), 1)

        for front_idx, front in enumerate(fronts):
            cds = crowding_distance(objectives, front)
            finite = [c for c in cds.values() if np.isfinite(c)]
            max_finite = max(finite) if finite else 1.0
            for i in front:
                f = fitnesses[i]
                f.rank = front_idx
                f.crowding = cds[i]
                if np.isinf(f.crowding):
                    crowd_norm = 1.0
                elif max_finite > 1e-12:
                    crowd_norm = float(f.crowding / max_finite)
                else:
                    crowd_norm = 0.5
                rank_score = 1.0 if n_fronts == 1 else (n_fronts - 1 - front_idx) / (n_fronts - 1)
                f.selection_score = float(
                    np.clip(
                        self.front_weight * rank_score + (1.0 - self.front_weight) * crowd_norm,
                        0.0,
                        1.0,
                    )
                )
        return fitnesses


# ---------------------------------------------------------------------------
# Chain-seeded pairing (assignment + genetic reproduction)
# ---------------------------------------------------------------------------

class ChainSeededPairing:
    """
    Deterministic, miner/validator-unchooseable pairing seeded from chain
    block state. Generation 0 (or a fresh population) uses ``assign``; later
    generations use ``reproduce`` (elite retention + crossover + mutation),
    always enforcing the K-partner floor.
    """

    def __init__(self, config: Optional[PairingConfig] = None):
        self.config = config or PairingConfig()

    def _rng(self, block_seed, salt: int = 0) -> np.random.RandomState:
        digest = hashlib.sha256(f"{block_seed}:{salt}".encode()).hexdigest()
        return np.random.RandomState(int(digest[:8], 16))

    def assign(
        self,
        researcher_uids: Sequence[str],
        trader_uids: Sequence[str],
        block_seed,
    ) -> List[PairGenome]:
        researchers = list(researcher_uids)
        traders = list(trader_uids)
        if not researchers or not traders:
            return []

        k = max(1, int(self.config.partners_per_miner))
        span = max(len(researchers), len(traders))
        seen: set = set()
        genomes: List[PairGenome] = []

        for round_idx in range(k):
            rng = self._rng(block_seed, round_idx)
            r_perm = list(rng.permutation(len(researchers)))
            t_perm = list(rng.permutation(len(traders)))
            for i in range(span):
                r = researchers[r_perm[i % len(researchers)]]
                t = traders[t_perm[i % len(traders)]]
                if (r, t) in seen:
                    continue
                seen.add((r, t))
                genomes.append(PairGenome(r, t))
                if len(genomes) >= self.config.max_pairs:
                    return genomes
        return self._enforce_partner_floor(genomes, researchers, traders, block_seed)

    def reproduce(
        self,
        prev_fitnesses: List[PairFitness],
        researcher_uids: Sequence[str],
        trader_uids: Sequence[str],
        block_seed,
    ) -> List[PairGenome]:
        researchers = list(researcher_uids)
        traders = list(trader_uids)
        if not prev_fitnesses or not researchers or not traders:
            return self.assign(researchers, traders, block_seed)

        rng = self._rng(block_seed, 1000)
        ranked = sorted(prev_fitnesses, key=lambda f: f.selection_score, reverse=True)
        n_elite = max(1, int(round(len(ranked) * self.config.elite_fraction)))
        elites = ranked[:n_elite]

        seen: set = set()
        genomes: List[PairGenome] = []

        def add(r: str, t: str) -> bool:
            if (r, t) in seen:
                return False
            if r not in researchers or t not in traders:
                return False
            seen.add((r, t))
            genomes.append(PairGenome(r, t))
            return True

        # Elite retention
        for f in elites:
            add(f.genome.researcher_uid, f.genome.trader_uid)
            if len(genomes) >= self.config.max_pairs:
                return genomes

        # Crossover: recombine elite researchers with elite traders
        elite_r = list(dict.fromkeys(f.genome.researcher_uid for f in elites))
        elite_t = list(dict.fromkeys(f.genome.trader_uid for f in elites))
        for r in elite_r:
            for t in elite_t:
                if len(genomes) >= self.config.max_pairs:
                    break
                if rng.random_sample() < self.config.crossover_prob:
                    add(r, t)

        # Mutation: random re-pairings across the whole population
        n_mut = int(
            self.config.mutation_rate
            * max(len(researchers), len(traders))
            * max(1, self.config.partners_per_miner)
        )
        for _ in range(max(0, n_mut)):
            if len(genomes) >= self.config.max_pairs:
                break
            add(researchers[rng.randint(len(researchers))], traders[rng.randint(len(traders))])

        return self._enforce_partner_floor(genomes, researchers, traders, block_seed)

    def _enforce_partner_floor(
        self,
        genomes: List[PairGenome],
        researchers: List[str],
        traders: List[str],
        block_seed,
    ) -> List[PairGenome]:
        k = max(1, int(self.config.partners_per_miner))
        seen = {(g.researcher_uid, g.trader_uid) for g in genomes}
        r_count: Dict[str, int] = defaultdict(int)
        t_count: Dict[str, int] = defaultdict(int)
        for g in genomes:
            r_count[g.researcher_uid] += 1
            t_count[g.trader_uid] += 1

        rng = self._rng(block_seed, 2000)

        def top_up(uid: str, is_researcher: bool):
            counter = r_count if is_researcher else t_count
            others = traders if is_researcher else researchers
            attempts = 0
            while counter[uid] < k and attempts < 10 * k and len(genomes) < self.config.max_pairs:
                attempts += 1
                partner = others[rng.randint(len(others))]
                pair = (uid, partner) if is_researcher else (partner, uid)
                if pair in seen:
                    continue
                seen.add(pair)
                genomes.append(PairGenome(pair[0], pair[1]))
                r_count[pair[0]] += 1
                t_count[pair[1]] += 1

        for r in researchers:
            top_up(r, True)
        for t in traders:
            top_up(t, False)

        return genomes[: self.config.max_pairs]


# ---------------------------------------------------------------------------
# Collusion-graph detection (interaction anomaly)
# ---------------------------------------------------------------------------

class CollusionGraphDetector:
    """
    Detects miner collusion as *non-transferable lift*: a pair whose joint
    composite greatly exceeds BOTH partners' mean composite with their other
    partners. Honest skill is transferable (helps many partners); a colluding
    pair only performs when matched together.
    """

    def __init__(self, fixed_pair_correlation_threshold: float = 0.85):
        # Reinterpreted as the minimum lift (over each partner's other-partner
        # mean) required to flag an interaction anomaly. Scaled into [0, 1].
        self.lift_threshold = max(0.0, 1.0 - float(fixed_pair_correlation_threshold))

    def detect(self, fitnesses: List[PairFitness]) -> CollusionReport:
        report = CollusionReport()
        if len(fitnesses) < 3:
            return report

        # Map each miner to (genome_key -> pair_composite) so a pair can look up
        # its partners' performance with *other* counterparts.
        researcher_pairs: Dict[str, Dict[str, float]] = defaultdict(dict)
        trader_pairs: Dict[str, Dict[str, float]] = defaultdict(dict)
        for f in fitnesses:
            researcher_pairs[f.genome.researcher_uid][f.genome.key] = f.pair_composite
            trader_pairs[f.genome.trader_uid][f.genome.key] = f.pair_composite

        for f in fitnesses:
            r = f.genome.researcher_uid
            t = f.genome.trader_uid
            # Best score each partner achieves WITHOUT this exact pairing.
            r_others = [v for k, v in researcher_pairs[r].items() if k != f.genome.key]
            t_others = [v for k, v in trader_pairs[t].items() if k != f.genome.key]
            # Need transferability evidence on both sides to judge.
            if not r_others or not t_others:
                continue
            # Lift over each partner's BEST alternative pairing. A colluder's one
            # high pair sits far above its next-best; an honest strong miner has
            # several comparably good pairings, so this lift is near zero.
            lift_r = f.pair_composite - max(r_others)
            lift_t = f.pair_composite - max(t_others)

            if lift_r > self.lift_threshold and lift_t > self.lift_threshold:
                severity = float(min(1.0, min(lift_r, lift_t)))
                f.collusion_flagged = True
                report.flagged.append((f.genome.key, severity))
                report.details[f.genome.key] = {
                    "lift_over_researcher_best_other": lift_r,
                    "lift_over_trader_best_other": lift_t,
                    "severity": severity,
                }
        return report

    def apply_discount(
        self,
        fitnesses: List[PairFitness],
        report: CollusionReport,
        discount: float = 0.5,
    ) -> List[PairFitness]:
        """Reduce the selection_score of flagged pairs (collusion is unprofitable)."""
        flagged = {key for key, _ in report.flagged}
        for f in fitnesses:
            if f.genome.key in flagged:
                f.selection_score *= (1.0 - discount)
        return fitnesses


# ---------------------------------------------------------------------------
# Marginal-contribution credit (variance-penalized, single emission vector)
# ---------------------------------------------------------------------------

class MarginalContributionCredit:
    """
    Turns per-pair selection scores into a single per-miner weight using the
    repository's ``mean - lambda * std`` philosophy across a miner's partners.
    Transferable skill (high mean, low variance across partners) is rewarded;
    one-hit / partner-dependent miners are penalized.
    """

    def __init__(self, marginal_weight: float = 0.50):
        self.marginal_weight = float(marginal_weight)

    def compute(self, fitnesses: List[PairFitness]) -> Dict[str, float]:
        by_miner: Dict[str, List[float]] = defaultdict(list)
        for f in fitnesses:
            by_miner[f.genome.researcher_uid].append(f.selection_score)
            by_miner[f.genome.trader_uid].append(f.selection_score)

        credit: Dict[str, float] = {}
        for uid, scores in by_miner.items():
            arr = np.asarray(scores, dtype=float)
            mean = float(arr.mean())
            std = float(arr.std()) if len(arr) > 1 else 0.0
            credit[uid] = max(0.0, mean - self.marginal_weight * std)

        total = sum(credit.values())
        if total < 1e-12:
            n = len(credit)
            return {uid: 1.0 / n for uid in credit} if n else {}
        return {uid: c / total for uid, c in credit.items()}


# ---------------------------------------------------------------------------
# Population orchestrator
# ---------------------------------------------------------------------------

class PairingPopulation:
    """
    High-level driver tying the pieces together for the validator and harness.

    Usage per generation:
        pop = PairingPopulation(config)
        genomes = pop.assign(researcher_uids, trader_uids, block_seed)
        # ... build PairFitness for each genome via joint evaluation ...
        result = pop.select(fitnesses)   # -> {"weights", "fitnesses", "collusion"}
    """

    def __init__(self, config: Optional[PairingConfig] = None):
        self.config = config or PairingConfig()
        self.matchmaker = NSGA2Matchmaker()
        self.pairing = ChainSeededPairing(self.config)
        self.credit = MarginalContributionCredit(self.config.marginal_contribution_weight)
        self.collusion = CollusionGraphDetector(self.config.fixed_pair_correlation_threshold)
        self._last_fitnesses: Optional[List[PairFitness]] = None
        self.generation: int = 0

    def assign(
        self,
        researcher_uids: Sequence[str],
        trader_uids: Sequence[str],
        block_seed,
    ) -> List[PairGenome]:
        if self._last_fitnesses:
            return self.pairing.reproduce(
                self._last_fitnesses, researcher_uids, trader_uids, block_seed
            )
        return self.pairing.assign(researcher_uids, trader_uids, block_seed)

    def select(self, fitnesses: List[PairFitness]) -> Dict[str, object]:
        ranked = self.matchmaker.rank(fitnesses)
        report = self.collusion.detect(ranked)
        adjusted = self.collusion.apply_discount(ranked, report)
        weights = self.credit.compute(adjusted)
        self._last_fitnesses = ranked
        self.generation += 1
        return {
            "weights": weights,
            "fitnesses": ranked,
            "collusion": report,
            "generation": self.generation,
        }
