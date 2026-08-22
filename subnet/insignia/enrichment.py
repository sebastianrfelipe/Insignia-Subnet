"""
Insignia Ground-Truth Enrichment Tracking

Closes the loop between simulation scoring and real deployment outcomes.
The subnet's scoring function is tuned on sim separation (honest vs.
adversarial), but the *real* objective is live desk P&L. This module
records the journey of each promoted pair from sim score to live result
and computes enrichment metrics that feed back into the scoring R&D loop
and the tuner objectives.

The analog in drug-discovery hit identification:

    virtual screen → wet-lab assay → hit rate
    sim score       → live desk P&L → enrichment factor

A scoring function that separates honest from adversarial in sim is
necessary but not sufficient. The enrichment factor — promoted hit rate
divided by baseline hit rate — measures whether the sim oracle has real
predictive power over live outcomes. If the enrichment factor drops below
1.0, the sim is no better than random selection, and the scoring function
must be revised.

This module is deliberately separable from the scoring engine so the
tuning harness can sweep enrichment parameters independently and
validators can run it as a post-deployment gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EnrichmentConfig:
    """
    Parameters controlling enrichment computation.

    The enrichment factor is the headline metric:

        EF = promoted_hit_rate / baseline_hit_rate

    where a "hit" is a pair whose live P&L exceeds the baseline threshold
    (default 0.0 = break-even). EF > 1 means the sim selects better than
    random; EF = 1 means no enrichment; EF < 1 means the sim is
    anti-predictive.

    The shrinkage factor ``sqrt(N / (N + confidence_k))`` prevents
    thin-sample enrichment from dominating tuner objectives. At N >> k
    the factor approaches 1; at small N the enrichment is shrunk toward 0.
    """

    # P&L threshold above which a pair is a "hit" (default: break-even).
    hit_threshold: float = 0.0

    # Number of epochs to retain in the outcome history window.
    history_window_epochs: int = 12

    # Sample-size confidence shrinkage constant (same pattern as the
    # ratio metrics: penalized_sharpe, omega, sharpe, sortino).
    confidence_k: float = 30.0


# ---------------------------------------------------------------------------
# Outcome records
# ---------------------------------------------------------------------------

@dataclass
class PairOutcome:
    """A single pair's journey from sim score to live result."""

    pair_id: str
    epoch: int
    sim_composite: float
    live_pnl: float
    live_sharpe: float
    deployed: bool

    @property
    def is_hit(self) -> bool:
        """True if live P&L exceeds the hit threshold (break-even by default)."""
        return self.live_pnl > 0.0


# ---------------------------------------------------------------------------
# Enrichment metrics
# ---------------------------------------------------------------------------

@dataclass
class LiveEnrichmentMetrics:
    """
    Enrichment metrics measuring the sim oracle's predictive power over
    live deployment outcomes.

    These are the ground-truth analog of the sim-only
    :class:`~insignia.scoring_rnd.DiscriminationMetrics`. Sim separation
    measures honest-vs-adversarial ranking; enrichment measures whether
    that ranking predicts real P&L.
    """

    # Spearman rank correlation between sim composite and live P&L.
    # High = sim ranking predicts live ranking; low = sim oracle is blind.
    sim_vs_live_correlation: float = 0.0

    # Fraction of promoted (sim-top) pairs that were live hits.
    promoted_hit_rate: float = 0.0

    # Fraction of baseline (sim-bottom or random) pairs that were live hits.
    baseline_hit_rate: float = 0.0

    # Enrichment factor: promoted_hit_rate / baseline_hit_rate.
    # 1.0 = no enrichment, >1 = sim selects better than random.
    enrichment_factor: float = 1.0

    # Fraction of sim-bottom-quartile pairs that were live losers
    # (negative predictive value of the sim floor).
    sim_floor_accuracy: float = 0.0

    # Sample sizes for confidence assessment.
    n_promoted: int = 0
    n_baseline: int = 0
    epochs_observed: int = 0

    # Confidence shrinkage factor sqrt(N / (N + k)).
    # Applied to enrichment_factor when used as a tuner objective.
    shrinkage: float = 1.0

    @property
    def shrunk_enrichment_factor(self) -> float:
        """Enrichment factor after sample-size confidence shrinkage."""
        return self.enrichment_factor * self.shrinkage

    def to_dict(self) -> Dict:
        d = {
            "sim_vs_live_correlation": round(self.sim_vs_live_correlation, 6),
            "promoted_hit_rate": round(self.promoted_hit_rate, 6),
            "baseline_hit_rate": round(self.baseline_hit_rate, 6),
            "enrichment_factor": round(self.enrichment_factor, 6),
            "sim_floor_accuracy": round(self.sim_floor_accuracy, 6),
            "n_promoted": self.n_promoted,
            "n_baseline": self.n_baseline,
            "epochs_observed": self.epochs_observed,
            "shrinkage": round(self.shrinkage, 6),
            "shrunk_enrichment_factor": round(self.shrunk_enrichment_factor, 6),
        }
        return d


# ---------------------------------------------------------------------------
# Enrichment Tracker
# ---------------------------------------------------------------------------

class EnrichmentTracker:
    """
    Tracks deployment outcomes across epochs to compute ground-truth
    enrichment metrics.

    Usage per deployment window:
        tracker.record_outcome(pair_id, epoch, sim_composite, live_pnl, live_sharpe, deployed=True)
        tracker.record_outcome(baseline_pair_id, epoch, sim_composite, live_pnl, live_sharpe, deployed=False)
        metrics = tracker.compute_enrichment(current_epoch)
        tracker.time_decay(current_epoch)
    """

    def __init__(self, config: Optional[EnrichmentConfig] = None):
        self.config = config or EnrichmentConfig()
        self._outcomes: List[PairOutcome] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        pair_id: str,
        epoch: int,
        sim_composite: float,
        live_pnl: float,
        live_sharpe: float = 0.0,
        deployed: bool = True,
    ) -> None:
        """Record a single pair's deployment outcome."""
        self._outcomes.append(
            PairOutcome(
                pair_id=pair_id,
                epoch=epoch,
                sim_composite=sim_composite,
                live_pnl=live_pnl,
                live_sharpe=live_sharpe,
                deployed=deployed,
            )
        )

    def record_batch(
        self,
        outcomes: Sequence[Dict],
        epoch: int,
    ) -> None:
        """
        Bulk-record a cohort of promoted + baseline pairs.

        Each dict in ``outcomes`` should have keys:
            pair_id, sim_composite, live_pnl, live_sharpe, deployed
        """
        for o in outcomes:
            self.record_outcome(
                pair_id=o["pair_id"],
                epoch=epoch,
                sim_composite=o["sim_composite"],
                live_pnl=o["live_pnl"],
                live_sharpe=o.get("live_sharpe", 0.0),
                deployed=o.get("deployed", True),
            )

    # ------------------------------------------------------------------
    # Enrichment computation
    # ------------------------------------------------------------------

    def compute_enrichment(self, current_epoch: int) -> LiveEnrichmentMetrics:
        """
        Compute enrichment metrics over the windowed outcome history.

        Returns :class:`LiveEnrichmentMetrics`. When there are no outcomes
        or insufficient data, returns metrics with default values (EF=1.0,
        shrinkage=0.0) so downstream gates fall back to sim-only behavior.
        """
        cfg = self.config
        cutoff = current_epoch - cfg.history_window_epochs
        window = [o for o in self._outcomes if o.epoch >= cutoff]

        if not window:
            return LiveEnrichmentMetrics(shrinkage=0.0, enrichment_factor=1.0)

        sim_scores = np.array([o.sim_composite for o in window])
        live_pnls = np.array([o.live_pnl for o in window])
        deployed_flags = np.array([o.deployed for o in window])

        # Promoted = deployed pairs (sim-top, selected for live trading).
        # Baseline = non-deployed pairs (sim-bottom or random controls).
        promoted_mask = deployed_flags
        baseline_mask = ~deployed_flags

        n_promoted = int(np.sum(promoted_mask))
        n_baseline = int(np.sum(baseline_mask))

        # Hit rates.
        if n_promoted > 0:
            promoted_hits = np.sum(live_pnls[promoted_mask] > cfg.hit_threshold)
            promoted_hit_rate = float(promoted_hits / n_promoted)
        else:
            promoted_hit_rate = 0.0

        if n_baseline > 0:
            baseline_hits = np.sum(live_pnls[baseline_mask] > cfg.hit_threshold)
            baseline_hit_rate = float(baseline_hits / n_baseline)
        else:
            baseline_hit_rate = 0.0

        # Enrichment factor.
        if baseline_hit_rate > 1e-12:
            enrichment_factor = promoted_hit_rate / baseline_hit_rate
        elif n_promoted > 0:
            # No baseline hits but promoted pairs exist: if promoted hit
            # rate > 0, EF is infinite in principle; cap at a large value.
            # If promoted also has 0 hits, EF = 1.0 (no signal either way).
            enrichment_factor = 10.0 if promoted_hit_rate > 0 else 1.0
        else:
            enrichment_factor = 1.0

        # Sim-vs-live Spearman rank correlation.
        if len(window) >= 3:
            corr = self._spearman(sim_scores, live_pnls)
        else:
            corr = 0.0

        # Sim floor accuracy: fraction of sim-bottom-quartile pairs that
        # were live losers (PnL <= 0).
        if len(window) >= 4:
            quartile_cutoff = np.percentile(sim_scores, 25)
            bottom = live_pnls[sim_scores <= quartile_cutoff]
            if len(bottom) > 0:
                sim_floor_accuracy = float(np.mean(bottom <= 0.0))
            else:
                sim_floor_accuracy = 0.0
        else:
            sim_floor_accuracy = 0.0

        # Sample-size confidence shrinkage.
        total_n = n_promoted + n_baseline
        shrinkage = math.sqrt(total_n / (total_n + cfg.confidence_k)) if total_n > 0 else 0.0

        epochs_observed = len(set(o.epoch for o in window))

        return LiveEnrichmentMetrics(
            sim_vs_live_correlation=float(corr),
            promoted_hit_rate=promoted_hit_rate,
            baseline_hit_rate=baseline_hit_rate,
            enrichment_factor=float(enrichment_factor),
            sim_floor_accuracy=sim_floor_accuracy,
            n_promoted=n_promoted,
            n_baseline=n_baseline,
            epochs_observed=epochs_observed,
            shrinkage=float(shrinkage),
        )

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def time_decay(self, current_epoch: int) -> None:
        """Prune outcomes older than the configured window."""
        cutoff = current_epoch - self.config.history_window_epochs
        self._outcomes = [o for o in self._outcomes if o.epoch >= cutoff]

    def reset(self) -> None:
        """Clear all recorded outcomes (used between tuning runs)."""
        self._outcomes.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _spearman(x: np.ndarray, y: np.ndarray) -> float:
        """Spearman rank correlation between two arrays."""
        if len(x) != len(y) or len(x) < 3:
            return 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            rx = _rankdata(x)
            ry = _rankdata(y)
            corr_matrix = np.corrcoef(rx, ry)
        corr = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
        return corr


def _rankdata(a: np.ndarray) -> np.ndarray:
    """
    Rank data (average ranks for ties), matching scipy.stats.rankdata
    behavior without the scipy dependency.
    """
    arr = np.asarray(a, dtype=float)
    sorter = np.argsort(arr, kind="mergesort")
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)
    arr = arr[sorter]
    obs = np.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum()[inv]
    # Average ranks for ties.
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)
