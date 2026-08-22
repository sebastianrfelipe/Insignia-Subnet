"""
Insignia Scoring-Function R&D Loop

Treats the scoring function itself as a first-class R&D target, not a fixed
input. The subnet's scoring weights and normalization transforms are the
highest-leverage mechanism for shaping miner behavior; improving the scoring
function compounds across every downstream evaluation.

This module provides:

1. **ScoringRNDLoop**: a structured loop for proposing, evaluating, and
   keeping/dropping scoring-function variants. Each experiment proposes a
   change to ``WeightConfig`` or the normalization transforms, evaluates it
   against a held-out outcome, and records whether it improved discrimination
   (separation between honest and adversarial submissions) without
   regressing the honest baseline.

2. **ExploitSignalCollector**: captures "scientific gaming" events - cases
   where a miner optimized for a metric rather than the underlying objective -
   and converts them into candidate scoring-function revisions. This is the
   hybrid exploit philosophy: economic attacks (sybil, collusion, timing) are
   still hard-gated, but metric-gaming is treated as a signal to revise the
   metric, not merely an attack to kill.

3. **RetrospectiveValidator**: evaluates a proposed scoring variant against a
   held-out set of known honest and adversarial submissions, computing
   enrichment-style metrics (AUC, separation, honest-floor, adversary-ceiling)
   analogous to the hit-identification validation used in drug-discovery
   scoring. A variant is only promoted if it improves the held-out
   discrimination metrics without collapsing the honest baseline.

Design principles:
    - Scoring-function R&D is as important as model R&D.
    - Exploits of a metric are diagnostic: they reveal where the oracle lies.
    - Hard gates for economic attacks; signal-driven revision for scientific gaming.
    - Retrospective validation before promotion: no scoring change ships
      without evidence it improves discrimination on held-out data.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .scoring import CompositeScorer, ScoreVector, WeightConfig
from .enrichment import LiveEnrichmentMetrics


# ---------------------------------------------------------------------------
# Experiment data model
# ---------------------------------------------------------------------------

@dataclass
class ScoringExperiment:
    """A single scoring-function R&D experiment."""

    experiment_id: str
    description: str
    proposed_weights: WeightConfig
    # Optional transform overrides (keys are metric names, values are
    # callables mapping raw -> normalized).
    proposed_transforms: Dict[str, Callable[[float], float]] = field(default_factory=dict)
    # What kind of signal motivated this experiment.
    motivation: str = ""
    # Which attack vector or metric it targets.
    target_vector: str = ""
    # Filled in after evaluation.
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    candidate_metrics: Dict[str, float] = field(default_factory=dict)
    kept: bool = False
    reason: str = ""
    timestamp: str = ""
    # Ground-truth enrichment factor at evaluation time (None when no
    # live deployment data is available).
    enrichment_factor: Optional[float] = None


@dataclass
class DiscriminationMetrics:
    """
    Held-out discrimination metrics for a scoring variant, analogous to
    hit-identification enrichment in drug-discovery scoring.

    All metrics are computed on a held-out population of submissions with
    known honest/adversarial labels (from the simulation harness).
    """

    # Honest/adversarial separation: mean(honest) - mean(adversarial).
    separation: float = 0.0
    # Honest floor: the minimum honest score (lower = better, fewer false negatives).
    honest_floor: float = 0.0
    # Adversary ceiling: the maximum adversarial score (lower = better).
    adversary_ceiling: float = 0.0
    # AUC of the honest-vs-adversarial ranking (0.5 = random, 1.0 = perfect).
    auc: float = 0.0
    # Fraction of adversaries that outscore the honest floor (lower = better).
    adversary_leak_rate: float = 0.0
    # Mean honest score (must not regress below baseline).
    honest_mean: float = 0.0
    # Mean adversarial score (lower = better).
    adversary_mean: float = 0.0


# ---------------------------------------------------------------------------
# Exploit signal collection
# ---------------------------------------------------------------------------

@dataclass
class ExploitSignal:
    """
    A captured "scientific gaming" event - a miner optimizing for a metric
    rather than the underlying objective. This is a signal to revise the
    scoring function, not merely an attack to kill.
    """

    miner_uid: str
    epoch: int
    signal_type: str  # "metric_concentration", "size_bias", "regime_overfit", ...
    affected_metric: str
    severity: float  # [0, 1]
    description: str
    suggested_revision: str = ""


class ExploitSignalCollector:
    """
    Collects and interprets exploit signals from submission telemetry.

    Each signal type corresponds to a known class of scientific gaming:

    - **metric_concentration**: a miner's score is dominated by one metric
      (e.g., high F1 but poor drawdown), suggesting the metric weight is too
      high or the normalization is too lenient.
    - **size_bias**: a miner's score scales with a confound (model size, feature
      count, trade count) rather than skill, suggesting the metric needs
      size normalization.
    - **regime_overfit**: a miner scores well in one regime but collapses in
      others, suggesting the variance penalty is too weak.
    - **oracle_blind_spot**: a miner scores high on the composite but fails
      an orthogonal quality check, suggesting a missing metric.

    Collected signals are converted into candidate scoring revisions by
    :meth:`propose_revisions`.
    """

    def __init__(self) -> None:
        self._signals: List[ExploitSignal] = []

    def record(self, signal: ExploitSignal) -> None:
        self._signals.append(signal)

    def record_metric_concentration(
        self,
        miner_uid: str,
        epoch: int,
        score_vector: ScoreVector,
        concentration_threshold: float = 0.60,
    ) -> Optional[ExploitSignal]:
        """
        Detect when a miner's composite is dominated by a single metric.

        ``concentration_threshold`` is the fraction of the composite that
        one normalized metric contributes; above it, the metric weight is
        likely too high or the normalization too lenient.
        """
        if not score_vector.normalized or not score_vector.composite:
            return None

        contributions = {
            k: v for k, v in score_vector.normalized.items()
            if k != "win_rate"  # diagnostic, not weighted
        }
        if not contributions:
            return None

        max_metric = max(contributions, key=contributions.get)
        max_value = contributions[max_metric]
        if max_value <= 0:
            return None

        # Fraction of the composite attributable to the top metric.
        concentration = max_value / max(sum(contributions.values()), 1e-12)
        if concentration < concentration_threshold:
            return None

        signal = ExploitSignal(
            miner_uid=miner_uid,
            epoch=epoch,
            signal_type="metric_concentration",
            affected_metric=max_metric,
            severity=float(min(1.0, concentration)),
            description=(
                f"miner {miner_uid} composite dominated by {max_metric} "
                f"({concentration:.0%} of normalized sum)"
            ),
            suggested_revision=f"reduce weight or tighten normalization for {max_metric}",
        )
        self._signals.append(signal)
        return signal

    def record_size_bias(
        self,
        miner_uid: str,
        epoch: int,
        composite: float,
        size_metric: float,
        size_name: str = "n_features",
        correlation_threshold: float = 0.70,
    ) -> Optional[ExploitSignal]:
        """
        Detect when a composite score correlates with a size confound.

        This requires accumulating samples across miners; pass per-epoch
        composites and size metrics. When the correlation across the
        population exceeds the threshold, the metric is size-biased.
        """
        # This is called per-miner but the correlation is computed across
        # the population; the caller is expected to aggregate. Here we just
        # record the raw signal for the collector to evaluate in batch.
        signal = ExploitSignal(
            miner_uid=miner_uid,
            epoch=epoch,
            signal_type="size_bias",
            affected_metric=size_name,
            severity=0.0,  # set in batch evaluation
            description=(
                f"composite may correlate with {size_name}={size_metric}"
            ),
            suggested_revision=f"add size normalization for the affected metric",
        )
        self._signals.append(signal)
        return signal

    def evaluate_size_bias_batch(
        self,
        composites: Sequence[float],
        sizes: Sequence[float],
        epoch: int,
        correlation_threshold: float = 0.70,
    ) -> Optional[ExploitSignal]:
        """
        Evaluate size-bias across the population. Call after collecting
        per-miner composites and size metrics for an epoch.
        """
        if len(composites) < 5 or len(sizes) < 5:
            return None
        if len(composites) != len(sizes):
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_matrix = np.corrcoef(composites, sizes)
        corr = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
        if abs(corr) < correlation_threshold:
            return None

        signal = ExploitSignal(
            miner_uid="<population>",
            epoch=epoch,
            signal_type="size_bias",
            affected_metric="composite",
            severity=float(min(1.0, abs(corr))),
            description=(
                f"composite correlates {corr:.2f} with size across population"
            ),
            suggested_revision="normalize metrics by size confound",
        )
        self._signals.append(signal)
        return signal

    def record_oracle_blind_spot(
        self,
        miner_uid: str,
        epoch: int,
        composite: float,
        orthogonal_check: float,
        orthogonal_name: str = "orthogonal_quality",
        gap_threshold: float = 0.30,
    ) -> Optional[ExploitSignal]:
        """
        Detect when a miner scores high on the composite but fails an
        orthogonal quality check, suggesting a missing metric.
        """
        gap = composite - orthogonal_check
        if gap < gap_threshold:
            return None

        signal = ExploitSignal(
            miner_uid=miner_uid,
            epoch=epoch,
            signal_type="oracle_blind_spot",
            affected_metric=orthogonal_name,
            severity=float(min(1.0, gap)),
            description=(
                f"miner {miner_uid} composite {composite:.2f} vs "
                f"{orthogonal_name} {orthogonal_check:.2f} (gap {gap:.2f})"
            ),
            suggested_revision=f"add a metric capturing {orthogonal_name}",
        )
        self._signals.append(signal)
        return signal

    def record_sim_live_gap(
        self,
        pair_id: str,
        epoch: int,
        sim_composite: float,
        live_pnl_rank: float,
        gap_threshold: float = 0.30,
    ) -> Optional[ExploitSignal]:
        """
        Detect when a pair scores high in sim but ranks low in live P&L.

        This is the ground-truth oracle-blind-spot signal: the sim oracle
        says the pair is good, but live deployment says otherwise. It fires
        an ``oracle_blind_spot`` signal with
        ``orthogonal_name="live_pnl_rank"`` so the R&D loop can propose a
        metric revision that addresses the root cause.

        Args:
            pair_id: The pair identifier.
            epoch: The deployment epoch.
            sim_composite: The sim composite score (0-1, higher = better).
            live_pnl_rank: The live P&L rank normalized to [0, 1]
                (1 = best, 0 = worst). A pair that is sim-high but
                live-low has a large ``sim_composite - live_pnl_rank`` gap.
            gap_threshold: Minimum gap to trigger the signal.
        """
        gap = sim_composite - live_pnl_rank
        if gap < gap_threshold:
            return None

        signal = ExploitSignal(
            miner_uid=pair_id,
            epoch=epoch,
            signal_type="oracle_blind_spot",
            affected_metric="live_pnl_rank",
            severity=float(min(1.0, gap)),
            description=(
                f"pair {pair_id} sim composite {sim_composite:.2f} vs "
                f"live_pnl_rank {live_pnl_rank:.2f} (gap {gap:.2f})"
            ),
            suggested_revision=(
                "add or weight a metric that correlates with live P&L "
                "rank (current sim composite does not predict live "
                "outcome for this pair)"
            ),
        )
        self._signals.append(signal)
        return signal

    def propose_revisions(self) -> List[str]:
        """
        Convert collected signals into candidate scoring-function revision
        descriptions. Each is a human-readable proposal that the R&D loop
        can turn into a ``ScoringExperiment``.
        """
        proposals: List[str] = []
        seen_types: set = set()

        for signal in self._signals:
            key = (signal.signal_type, signal.affected_metric)
            if key in seen_types:
                continue
            seen_types.add(key)
            proposals.append(
                f"[{signal.signal_type}:{signal.affected_metric}] "
                f"{signal.suggested_revision} "
                f"(severity={signal.severity:.2f})"
            )
        return proposals

    def reset(self) -> None:
        self._signals.clear()

    @property
    def signals(self) -> List[ExploitSignal]:
        return list(self._signals)


# ---------------------------------------------------------------------------
# Retrospective validation
# ---------------------------------------------------------------------------

class RetrospectiveValidator:
    """
    Evaluates a proposed scoring variant against a held-out population of
    submissions with known honest/adversarial labels.

    This is the "hit-identification" analog: a scoring change is only promoted
    if it improves discrimination (separation, AUC, adversary ceiling)
    without regressing the honest floor.
    """

    def __init__(
        self,
        promotion_separation_delta: float = 0.02,
        promotion_auc_delta: float = 0.02,
        max_honest_floor_regression: float = 0.01,
    ):
        self.promotion_separation_delta = promotion_separation_delta
        self.promotion_auc_delta = promotion_auc_delta
        self.max_honest_floor_regression = max_honest_floor_regression

    def compute_metrics(
        self,
        honest_scores: Sequence[float],
        adversarial_scores: Sequence[float],
    ) -> DiscriminationMetrics:
        """Compute discrimination metrics from labeled scores."""
        h = np.asarray(honest_scores, dtype=float)
        a = np.asarray(adversarial_scores, dtype=float)

        if len(h) == 0 or len(a) == 0:
            return DiscriminationMetrics()

        honest_mean = float(h.mean())
        adversary_mean = float(a.mean())
        separation = honest_mean - adversary_mean
        honest_floor = float(h.min())
        adversary_ceiling = float(a.max())
        adversary_leak_rate = float(np.mean(a > honest_floor))
        auc = self._auc(h, a)

        return DiscriminationMetrics(
            separation=separation,
            honest_floor=honest_floor,
            adversary_ceiling=adversary_ceiling,
            auc=auc,
            adversary_leak_rate=adversary_leak_rate,
            honest_mean=honest_mean,
            adversary_mean=adversary_mean,
        )

    @staticmethod
    def _auc(honest: np.ndarray, adversarial: np.ndarray) -> float:
        """
        ROC-AUC where honest = positive class, adversarial = negative.
        1.0 = perfect separation, 0.5 = random, <0.5 = inverted.
        """
        if len(honest) == 0 or len(adversarial) == 0:
            return 0.5
        # Mann-Whitney U statistic.
        count = 0
        for h_val in honest:
            count += np.sum(h_val > adversarial)
            count += 0.5 * np.sum(h_val == adversarial)
        total = len(honest) * len(adversarial)
        return float(count / total) if total > 0 else 0.5

    def should_promote(
        self,
        baseline: DiscriminationMetrics,
        candidate: DiscriminationMetrics,
    ) -> Tuple[bool, str]:
        """
        Decide whether a candidate scoring variant should be promoted.

        A variant is promoted if it improves separation OR AUC by at least
        the configured delta, without regressing the honest floor by more
        than the configured tolerance.
        """
        sep_improved = (
            candidate.separation - baseline.separation
            >= self.promotion_separation_delta
        )
        auc_improved = (
            candidate.auc - baseline.auc >= self.promotion_auc_delta
        )
        floor_regression = (
            baseline.honest_floor - candidate.honest_floor
            > self.max_honest_floor_regression
        )

        if floor_regression:
            return False, (
                f"honest floor regressed {baseline.honest_floor:.4f} -> "
                f"{candidate.honest_floor:.4f}"
            )
        if not (sep_improved or auc_improved):
            return False, (
                f"no improvement: separation {baseline.separation:.4f} -> "
                f"{candidate.separation:.4f}, AUC {baseline.auc:.4f} -> "
                f"{candidate.auc:.4f}"
            )
        return True, "promoted"


# ---------------------------------------------------------------------------
# Scoring R&D Loop
# ---------------------------------------------------------------------------

class ScoringRNDLoop:
    """
    Structured loop for scoring-function R&D.

    Each iteration:
        1. Propose a scoring-function variant (from exploit signals or manual).
        2. Evaluate it against the current baseline on held-out data.
        3. Keep or drop based on the :class:`RetrospectiveValidator` verdict.
        4. Record the experiment for audit and future warm-starting.

    Usage:
        rnd = ScoringRNDLoop(baseline_scorer=CompositeScorer())
        experiment = rnd.propose(
            description="reduce F1 weight, raise variance score",
            weights=WeightConfig(model_penalized_f1=0.15, model_variance_score=0.22),
            motivation="metric_concentration on penalized_f1",
            target_vector="single_metric_gaming",
        )
        verdict = rnd.evaluate(
            experiment,
            honest_scores_baseline=honest_scores,
            adversarial_scores_baseline=adversarial_scores,
            honest_scores_candidate=honest_scores_new,
            adversarial_scores_candidate=adversarial_scores_new,
        )
        if experiment.kept:
            rnd.promote(experiment)
    """

    def __init__(
        self,
        baseline_scorer: Optional[CompositeScorer] = None,
        validator: Optional[RetrospectiveValidator] = None,
        min_enrichment_factor: float = 1.5,
    ):
        self.current_scorer = baseline_scorer or CompositeScorer()
        self.current_weights = self.current_scorer.weights
        self.validator = validator or RetrospectiveValidator()
        self.min_enrichment_factor = min_enrichment_factor
        self.history: List[ScoringExperiment] = []
        self._experiment_counter = 0

    def propose(
        self,
        description: str,
        weights: WeightConfig,
        motivation: str = "",
        target_vector: str = "",
        transforms: Optional[Dict[str, Callable[[float], float]]] = None,
    ) -> ScoringExperiment:
        """Create a new scoring-function experiment."""
        self._experiment_counter += 1
        return ScoringExperiment(
            experiment_id=f"EXP-SCR-{self._experiment_counter:03d}",
            description=description,
            proposed_weights=weights,
            proposed_transforms=transforms or {},
            motivation=motivation,
            target_vector=target_vector,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    def evaluate(
        self,
        experiment: ScoringExperiment,
        honest_scores_baseline: Sequence[float],
        adversarial_scores_baseline: Sequence[float],
        honest_scores_candidate: Sequence[float],
        adversarial_scores_candidate: Sequence[float],
        enrichment_metrics: Optional[LiveEnrichmentMetrics] = None,
    ) -> Tuple[bool, str]:
        """
        Evaluate a proposed experiment against held-out labeled scores.
        Populates the experiment's metrics and kept/reason fields.

        When ``enrichment_metrics`` is provided (live deployment outcomes
        available), a second promotion gate is applied: the shrunk
        enrichment factor must not fall below ``min_enrichment_factor``.
        This ensures a scoring variant that improves sim separation but
        destroys live P&L predictivity is not promoted.
        """
        baseline = self.validator.compute_metrics(
            honest_scores_baseline, adversarial_scores_baseline
        )
        candidate = self.validator.compute_metrics(
            honest_scores_candidate, adversarial_scores_candidate
        )

        experiment.baseline_metrics = asdict(baseline)
        experiment.candidate_metrics = asdict(candidate)

        kept, reason = self.validator.should_promote(baseline, candidate)

        # Ground-truth enrichment gate (second gate).
        if kept and enrichment_metrics is not None:
            shrunk_ef = enrichment_metrics.shrunk_enrichment_factor
            experiment.enrichment_factor = round(
                enrichment_metrics.enrichment_factor, 6
            )
            if shrunk_ef < self.min_enrichment_factor:
                kept = False
                ef_reason = (
                    f"enrichment gate: shrunk EF {shrunk_ef:.3f} < "
                    f"min {self.min_enrichment_factor:.3f} "
                    f"(raw EF {enrichment_metrics.enrichment_factor:.3f}, "
                    f"shrinkage {enrichment_metrics.shrinkage:.3f}, "
                    f"N={enrichment_metrics.n_promoted + enrichment_metrics.n_baseline})"
                )
                reason = f"{reason}; {ef_reason}"

        experiment.kept = kept
        experiment.reason = reason
        self.history.append(experiment)
        return kept, reason

    def promote(self, experiment: ScoringExperiment) -> CompositeScorer:
        """Promote a kept experiment to the current scorer."""
        if not experiment.kept:
            raise ValueError(f"experiment {experiment.experiment_id} was not kept")
        self.current_weights = experiment.proposed_weights
        self.current_scorer = CompositeScorer(
            weights=self.current_weights,
            overfitting_detector=self.current_scorer.overfitting_detector,
        )
        return self.current_scorer

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the R&D loop history."""
        kept = [e for e in self.history if e.kept]
        dropped = [e for e in self.history if not e.kept]
        return {
            "total_experiments": len(self.history),
            "kept": len(kept),
            "dropped": len(dropped),
            "current_weights": asdict(self.current_weights),
            "kept_experiments": [
                {
                    "id": e.experiment_id,
                    "description": e.description,
                    "motivation": e.motivation,
                    "target_vector": e.target_vector,
                    "reason": e.reason,
                    "baseline_separation": e.baseline_metrics.get("separation", 0.0),
                    "candidate_separation": e.candidate_metrics.get("separation", 0.0),
                    "baseline_auc": e.baseline_metrics.get("auc", 0.0),
                    "candidate_auc": e.candidate_metrics.get("auc", 0.0),
                }
                for e in kept
            ],
        }

    @property
    def kept_count(self) -> int:
        return sum(1 for e in self.history if e.kept)

    @property
    def dropped_count(self) -> int:
        return sum(1 for e in self.history if not e.kept)
