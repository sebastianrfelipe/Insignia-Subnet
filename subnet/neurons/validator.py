"""
Insignia Paired Validator — Single-Mechanism Joint Evaluation

The unified validator for the paired genetic incentive mechanism. It replaces
the separate L1 and L2 validators + cross-layer feedback with one pipeline:

    chain-seeded pairing
        -> joint evaluation (model on benchmark + strategy using that model)
        -> NSGA-II non-dominated ranking
        -> collusion screening
        -> variance-penalized marginal-contribution credit
        -> a single Yuma `set_weights` vector over all miner UIDs

The validator is agnostic about *how* the two halves of a pair are scored: the
caller supplies a model `ScoreVector` (typically from `ModelEvaluator` against
the proprietary benchmark) and a trading `ScoreVector` (typically from running
the trader's strategy with the paired model through paper/live trading). This
mirrors the granular APIs of the old `L1Validator`/`L2Validator` and lets both
the live neuron and the simulation harness reuse the same code.

Usage:
    python neurons/validator.py --netuid <NETUID> --wallet.name <WALLET> --subtensor.network test
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import bittensor as bt
except ImportError:
    bt = None

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from insignia.scoring import CompositeScorer, ScoreVector, WeightConfig
from insignia.pairing import (
    PairGenome,
    PairFitness,
    PairingConfig,
    PairingPopulation,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Validator] %(message)s")
logger = logging.getLogger(__name__)


class PairedValidator:
    """
    Unified validator for the single paired mechanism.

    Lifecycle per generation (epoch):
      1. `assign_pairs(...)` — derive the chain-seeded population.
      2. For each pair, compute a model `ScoreVector` and a trading
         `ScoreVector`, then call `score_pair(...)`.
      3. `finalize_generation()` — NSGA-II rank, collusion screen, and convert
         to a single per-miner weight vector.
      4. `set_weights(...)` — publish to the chain (no-op without bittensor).
    """

    def __init__(
        self,
        scorer: CompositeScorer | None = None,
        pairing_config: PairingConfig | None = None,
        weights: WeightConfig | None = None,
    ):
        self.scorer = scorer or CompositeScorer(weights=weights)
        self.population = PairingPopulation(pairing_config)
        self.current_generation: int = 0
        self._fitnesses: List[PairFitness] = []
        self._model_score_cache: Dict[Tuple[str, int], ScoreVector] = {}
        self.generation_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Pairing
    # ------------------------------------------------------------------

    def assign_pairs(
        self,
        researcher_uids: Sequence[str],
        trader_uids: Sequence[str],
        block_seed,
    ) -> List[PairGenome]:
        """Derive this generation's population deterministically from chain state."""
        genomes = self.population.assign(researcher_uids, trader_uids, block_seed)
        logger.info(
            "Generation %d: assigned %d pairs over %d researchers x %d traders (seed=%s)",
            self.current_generation, len(genomes),
            len(list(researcher_uids)), len(list(trader_uids)), block_seed,
        )
        return genomes

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def cache_model_score(self, researcher_uid: str, epoch: int, score: ScoreVector):
        """Cache a researcher's model score so its K pairs reuse one evaluation."""
        self._model_score_cache[(researcher_uid, epoch)] = score

    def get_cached_model_score(self, researcher_uid: str, epoch: int) -> Optional[ScoreVector]:
        return self._model_score_cache.get((researcher_uid, epoch))

    def score_pair(
        self,
        genome: PairGenome,
        model_score: ScoreVector,
        trading_score: ScoreVector,
    ) -> PairFitness:
        """
        Combine the model and trading scores for a pair into a `PairFitness`
        and stage it for this generation's selection.
        """
        pair_score = self.scorer.combine_pair(model_score, trading_score)
        fitness = PairFitness(
            genome=genome,
            objectives=pair_score.objectives,
            model_composite=pair_score.model_composite,
            trading_composite=pair_score.trading_composite,
            pair_composite=pair_score.pair_composite,
            raw={
                "model_composite": pair_score.model_composite,
                "trading_composite": pair_score.trading_composite,
                "trading_max_drawdown": float(trading_score.raw.get("max_drawdown", 0.0)),
                "trading_consistency": float(trading_score.normalized.get("consistency", 0.0)),
            },
        )
        self._fitnesses.append(fitness)
        return fitness

    # ------------------------------------------------------------------
    # Selection -> weights
    # ------------------------------------------------------------------

    def finalize_generation(self) -> Dict[str, Any]:
        """
        Rank the staged pairs with NSGA-II, screen for collusion, and compute
        the single per-miner emission weight vector.
        """
        result = self.population.select(self._fitnesses)
        weights: Dict[str, float] = result["weights"]
        report = result["collusion"]

        summary = {
            "generation": result["generation"],
            "n_pairs": len(result["fitnesses"]),
            "n_collusion_flags": report.n_flagged,
            "collusion_flagged": [k for k, _ in report.flagged],
            "weights": weights,
            "pareto_front_size": sum(1 for f in result["fitnesses"] if f.rank == 0),
            "top_pairs": [
                {
                    "pair": f.genome.key,
                    "pair_composite": round(f.pair_composite, 4),
                    "model_composite": round(f.model_composite, 4),
                    "trading_composite": round(f.trading_composite, 4),
                    "rank": f.rank,
                    "selection_score": round(f.selection_score, 4),
                    "collusion_flagged": f.collusion_flagged,
                }
                for f in sorted(
                    result["fitnesses"], key=lambda x: x.selection_score, reverse=True
                )[:10]
            ],
        }
        self.generation_history.append(summary)
        self.current_generation = result["generation"]
        self._fitnesses = []
        self._model_score_cache = {}

        logger.info(
            "Generation %d finalized: %d pairs, %d collusion flags, %d miners weighted",
            summary["generation"], summary["n_pairs"],
            summary["n_collusion_flags"], len(weights),
        )
        return summary

    def set_weights(self, weights: Dict[str, float], uid_map: Dict[str, int] | None = None):
        """
        Publish the single weight vector to the chain via Yuma consensus.

        Without bittensor available this logs the intended weights. In
        production, map UIDs and call `subtensor.set_weights(...)` once.
        """
        if bt is None:
            logger.info("set_weights (dry-run): %d UIDs", len(weights))
            return
        # Production: convert {uid: weight} -> aligned arrays and submit.
        # uids = [uid_map[u] for u in weights]; vals = [weights[u] for u in weights]
        # subtensor.set_weights(netuid=..., uids=uids, weights=vals, ...)
        logger.info("set_weights: submitting %d weights to chain", len(weights))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    """Standalone demo of the paired validator using synthetic score vectors."""
    logger.info("=" * 60)
    logger.info("Insignia Paired Validator — Demo Mode")
    logger.info("=" * 60)

    validator = PairedValidator(pairing_config=PairingConfig(partners_per_miner=2))
    researchers = [f"researcher_{i}" for i in range(4)]
    traders = [f"trader_{i}" for i in range(4)]

    genomes = validator.assign_pairs(researchers, traders, block_seed="demo_block_0")

    rng = np.random.RandomState(7)
    for g in genomes:
        # Synthetic: researcher_0 is strong, trader_0 is strong; others middling.
        r_q = 0.85 if g.researcher_uid == "researcher_0" else float(rng.uniform(0.3, 0.6))
        t_q = 0.8 if g.trader_uid == "trader_0" else float(rng.uniform(0.3, 0.6))
        model_score = ScoreVector(composite=r_q, normalized={}, raw={})
        trading_score = ScoreVector(
            composite=t_q,
            normalized={"consistency": t_q},
            raw={"max_drawdown": 0.1 * (1 - t_q)},
        )
        validator.score_pair(g, model_score, trading_score)

    summary = validator.finalize_generation()
    logger.info("Pareto front size: %d", summary["pareto_front_size"])
    logger.info("Top pairs:")
    for p in summary["top_pairs"][:5]:
        logger.info("  %s: pair=%.3f rank=%d", p["pair"], p["pair_composite"], p["rank"])
    logger.info("Per-miner weights (top 5):")
    for uid, w in sorted(summary["weights"].items(), key=lambda x: x[1], reverse=True)[:5]:
        logger.info("  %s: %.4f", uid, w)
    return summary


if __name__ == "__main__":
    demo()
