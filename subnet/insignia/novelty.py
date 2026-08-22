"""
Insignia Novelty & Diversity Tracking

Rewards exploration of genuinely new feature families and execution styles
while aggressively penalizing near-duplicate models and strategy clones. The
mechanism has two parts:

1. **Novelty bonus**: a time-decaying bonus applied to a miner's composite
   score when its submission is structurally or behaviorally distinct from
   everything seen in prior epochs. The bonus shrinks as the same approach
   is re-submitted across epochs, so a once-novel strategy becomes baseline.

2. **Duplicate invalidation**: near-duplicate model fingerprints and cloned
   execution styles are penalized more aggressively than the existing
   "share rewards" approach. A submission whose fingerprint or prediction
   correlation is too close to a prior submission receives a composite
   penalty, making copy-mining and re-serialization unprofitable.

The tracker is epoch-aware: novelty decays over time so the system keeps
exploring rather than collapsing onto a single locally-optimal approach.
It is deliberately separable from the scoring engine so validators can
run it as a pre-scoring gate and the tuning harness can sweep its
parameters independently.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NoveltyConfig:
    """
    Parameters controlling novelty bonuses and duplicate penalties.

    The novelty bonus is a multiplicative boost on the composite score:
        composite *= (1 + novelty_bonus_weight * novelty_score)
    where ``novelty_score`` decays each epoch a submission is re-seen.

    The duplicate penalty is a multiplicative reduction:
        composite *= (1 - duplicate_penalty * duplicate_score)
    where ``duplicate_score`` is 1.0 for exact-hash duplicates and scales
    with prediction correlation for near-duplicates.
    """

    # Maximum bonus for a fully novel submission (added to composite).
    novelty_bonus_weight: float = 0.10

    # Number of epochs over which novelty halves for a re-submitted approach.
    novelty_decay_epochs: int = 4

    # Composite penalty for exact or near-duplicates.
    duplicate_penalty: float = 0.50

    # Jaccard threshold above which two feature sets are considered duplicates.
    feature_novelty_threshold: float = 0.85

    # Prediction correlation threshold above which two models are near-duplicates.
    prediction_correlation_threshold: float = 0.90

    # Position-correlation threshold above which two trading styles are clones.
    style_correlation_threshold: float = 0.85

    # Number of epochs to retain in the seen-history window. Older entries
    # are pruned so the tracker does not grow unbounded and so novelty is
    # measured against recent submissions, not the full historical corpus.
    history_window_epochs: int = 12

    # If True, exact-hash duplicates are zeroed (invalidated) rather than
    # merely penalized. This is the hard invalidation path.
    invalidate_exact_duplicates: bool = False


# ---------------------------------------------------------------------------
# Registration records
# ---------------------------------------------------------------------------

@dataclass
class ModelRegistration:
    """A model submission recorded for novelty tracking."""

    fingerprint: str
    feature_set: frozenset
    predictions: Optional[np.ndarray]
    first_seen_epoch: int
    last_seen_epoch: int
    resubmission_count: int = 1


@dataclass
class TradingRegistration:
    """A trading style recorded for novelty tracking."""

    style_signature: str
    position_correlation_vector: Optional[np.ndarray]
    first_seen_epoch: int
    last_seen_epoch: int
    resubmission_count: int = 1


# ---------------------------------------------------------------------------
# Novelty Tracker
# ---------------------------------------------------------------------------

class NoveltyTracker:
    """
    Tracks model fingerprints, feature sets, and execution styles across
    epochs to compute time-decaying novelty scores and detect duplicates.

    Usage per epoch:
        tracker.register_model(miner_uid, fingerprint, feature_set, predictions, epoch)
        tracker.register_trading(miner_uid, style_sig, position_vec, epoch)
        model_novelty = tracker.model_novelty(miner_uid, fingerprint, feature_set, predictions, epoch)
        trading_novelty = tracker.trading_novelty(miner_uid, style_sig, position_vec, epoch)
        tracker.time_decay(epoch)
    """

    def __init__(self, config: Optional[NoveltyConfig] = None):
        self.config = config or NoveltyConfig()
        self._model_history: Dict[str, ModelRegistration] = {}
        self._trading_history: Dict[str, TradingRegistration] = {}
        # Per-miner latest registrations for cross-miner duplicate detection.
        # Each miner maps to the *set* of fingerprints/styles it has registered
        # within the window, so an earlier-but-still-windowed artifact is not
        # lost when a newer one is registered.
        self._miner_models: Dict[str, Set[str]] = {}  # miner_uid -> fingerprints
        self._miner_styles: Dict[str, Set[str]] = {}  # miner_uid -> style_sigs
        # All fingerprints seen this window for cross-miner checks.
        self._all_fingerprints: Dict[str, int] = {}  # fingerprint -> last_seen_epoch

    # ------------------------------------------------------------------
    # Model novelty
    # ------------------------------------------------------------------

    def register_model(
        self,
        miner_uid: str,
        fingerprint: str,
        feature_set: Sequence[str],
        predictions: Optional[np.ndarray],
        epoch: int,
    ) -> None:
        """Record a model submission for novelty tracking."""
        fs = frozenset(feature_set)
        key = self._model_key(miner_uid, fingerprint)

        existing = self._model_history.get(key)
        if existing is not None:
            existing.last_seen_epoch = epoch
            existing.resubmission_count += 1
        else:
            self._model_history[key] = ModelRegistration(
                fingerprint=fingerprint,
                feature_set=fs,
                predictions=predictions,
                first_seen_epoch=epoch,
                last_seen_epoch=epoch,
            )

        self._miner_models.setdefault(miner_uid, set()).add(fingerprint)
        self._all_fingerprints[fingerprint] = epoch

    def model_novelty(
        self,
        miner_uid: str,
        fingerprint: str,
        feature_set: Sequence[str],
        predictions: Optional[np.ndarray],
        epoch: int,
    ) -> Tuple[float, float]:
        """
        Compute the novelty score and duplicate score for a model submission.

        Returns ``(novelty_score, duplicate_score)``:
            - ``novelty_score`` in [0, 1]: 1 = fully novel, decays with re-submission.
            - ``duplicate_score`` in [0, 1]: 1 = exact or near-duplicate, 0 = unique.

        The novelty score decays exponentially across epochs:
            novelty = 0.5 ** (epochs_since_first_seen / decay_epochs)
        A first-time submission scores 1.0; the same approach re-submitted
        after ``decay_epochs`` scores 0.5; after ``2*decay_epochs`` scores 0.25.
        """
        cfg = self.config
        fs = frozenset(feature_set)
        key = self._model_key(miner_uid, fingerprint)

        # --- Exact duplicate detection (cross-miner) ---
        exact_dup = self._is_exact_duplicate(miner_uid, fingerprint, epoch)

        if exact_dup and cfg.invalidate_exact_duplicates:
            return 0.0, 1.0

        # --- Novelty decay based on first-seen epoch ---
        if exact_dup:
            # An exact fingerprint match from a different miner is never
            # novel, regardless of which epoch it was seen in. This matches
            # trading_novelty's handling of exact style duplicates.
            novelty = 0.0
        else:
            existing = self._model_history.get(key)
            if existing is not None:
                epochs_since = max(0, epoch - existing.first_seen_epoch)
                novelty = 0.5 ** (epochs_since / max(1, cfg.novelty_decay_epochs))
            else:
                # Check cross-miner fingerprint collision.
                other_epoch = self._all_fingerprints.get(fingerprint)
                if other_epoch is not None:
                    # Same fingerprint seen from a different miner: not novel.
                    epochs_since = max(0, epoch - other_epoch)
                    novelty = 0.5 ** (epochs_since / max(1, cfg.novelty_decay_epochs))
                else:
                    novelty = 1.0

        # --- Near-duplicate detection via feature-set Jaccard ---
        feature_dup = self._feature_set_duplicate_score(miner_uid, fs, epoch)

        # --- Near-duplicate detection via prediction correlation ---
        pred_dup = 0.0
        if predictions is not None:
            pred_dup = self._prediction_duplicate_score(
                miner_uid, predictions, epoch
            )

        duplicate_score = max(feature_dup, pred_dup)
        if exact_dup:
            duplicate_score = 1.0

        return float(novelty), float(duplicate_score)

    def _is_exact_duplicate(
        self, miner_uid: str, fingerprint: str, epoch: int
    ) -> bool:
        """True if the exact fingerprint was seen from a different miner."""
        for other_uid, other_fps in self._miner_models.items():
            if other_uid != miner_uid and fingerprint in other_fps:
                return True
        return False

    def _feature_set_duplicate_score(
        self,
        miner_uid: str,
        feature_set: frozenset,
        epoch: int,
    ) -> float:
        """
        Jaccard similarity against the most similar prior feature set from
        a *different* miner. Returns 1.0 if a near-identical feature set was
        seen, scaled by the configured threshold.
        """
        cfg = self.config
        best_jaccard = 0.0
        for key, reg in self._model_history.items():
            # Skip same-miner registrations (novelty decay handles those).
            if key.startswith(f"{miner_uid}::"):
                continue
            if not reg.feature_set:
                continue
            intersection = len(feature_set & reg.feature_set)
            union = len(feature_set | reg.feature_set)
            if union == 0:
                continue
            j = intersection / union
            if j > best_jaccard:
                best_jaccard = j

        if best_jaccard >= cfg.feature_novelty_threshold:
            # Scale above the threshold: at threshold -> 0.5, at 1.0 -> 1.0
            excess = (best_jaccard - cfg.feature_novelty_threshold) / max(
                1e-12, 1.0 - cfg.feature_novelty_threshold
            )
            return float(min(1.0, 0.5 + 0.5 * excess))
        return 0.0

    def _prediction_duplicate_score(
        self,
        miner_uid: str,
        predictions: np.ndarray,
        epoch: int,
    ) -> float:
        """
        Correlation-based near-duplicate detection. Returns 1.0 if
        predictions are nearly identical to a prior submission from a
        different miner, scaled by the configured threshold.
        """
        cfg = self.config
        if predictions is None or len(predictions) == 0:
            return 0.0

        best_corr = 0.0
        for key, reg in self._model_history.items():
            if key.startswith(f"{miner_uid}::"):
                continue
            if reg.predictions is None or len(reg.predictions) != len(predictions):
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                corr_matrix = np.corrcoef(predictions, reg.predictions)
            corr = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
            if corr > best_corr:
                best_corr = corr

        if best_corr >= cfg.prediction_correlation_threshold:
            excess = (best_corr - cfg.prediction_correlation_threshold) / max(
                1e-12, 1.0 - cfg.prediction_correlation_threshold
            )
            return float(min(1.0, 0.5 + 0.5 * excess))
        return 0.0

    # ------------------------------------------------------------------
    # Trading novelty
    # ------------------------------------------------------------------

    def register_trading(
        self,
        miner_uid: str,
        style_signature: str,
        position_correlation_vector: Optional[np.ndarray],
        epoch: int,
    ) -> None:
        """Record a trading style for novelty tracking."""
        key = self._trading_key(miner_uid, style_signature)

        existing = self._trading_history.get(key)
        if existing is not None:
            existing.last_seen_epoch = epoch
            existing.resubmission_count += 1
        else:
            self._trading_history[key] = TradingRegistration(
                style_signature=style_signature,
                position_correlation_vector=position_correlation_vector,
                first_seen_epoch=epoch,
                last_seen_epoch=epoch,
            )

        self._miner_styles.setdefault(miner_uid, set()).add(style_signature)

    def trading_novelty(
        self,
        miner_uid: str,
        style_signature: str,
        position_correlation_vector: Optional[np.ndarray],
        epoch: int,
    ) -> Tuple[float, float]:
        """
        Compute novelty and duplicate scores for a trading style.

        Returns ``(novelty_score, duplicate_score)`` with the same semantics
        as :meth:`model_novelty`.
        """
        cfg = self.config
        key = self._trading_key(miner_uid, style_signature)

        # Exact style duplicate across miners.
        exact_dup = False
        for other_uid, other_sigs in self._miner_styles.items():
            if other_uid != miner_uid and style_signature in other_sigs:
                exact_dup = True
                break

        if exact_dup and cfg.invalidate_exact_duplicates:
            return 0.0, 1.0

        # Novelty decay.
        if exact_dup:
            # An exact style match from a different miner is never novel,
            # regardless of epoch. This must be checked before the
            # self-history lookup: after the documented register_trading
            # then trading_novelty sequence, the copier's own row already
            # exists in _trading_history, so the existing-is-not-None
            # branch would otherwise fire and yield novelty=1.0 for a
            # same-epoch clone. This mirrors model_novelty's ordering.
            novelty = 0.0
        else:
            existing = self._trading_history.get(key)
            if existing is not None:
                epochs_since = max(0, epoch - existing.first_seen_epoch)
                novelty = 0.5 ** (epochs_since / max(1, cfg.novelty_decay_epochs))
            else:
                novelty = 1.0

        # Near-duplicate via position correlation.
        style_dup = 0.0
        if position_correlation_vector is not None:
            style_dup = self._style_duplicate_score(
                miner_uid, position_correlation_vector
            )

        duplicate_score = 1.0 if exact_dup else style_dup
        return float(novelty), float(duplicate_score)

    def _style_duplicate_score(
        self,
        miner_uid: str,
        position_vec: np.ndarray,
    ) -> float:
        """Correlation-based clone detection for trading styles."""
        cfg = self.config
        if position_vec is None or len(position_vec) == 0:
            return 0.0

        best_corr = 0.0
        for key, reg in self._trading_history.items():
            if key.startswith(f"{miner_uid}::"):
                continue
            if (
                reg.position_correlation_vector is None
                or len(reg.position_correlation_vector) != len(position_vec)
            ):
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                corr_matrix = np.corrcoef(position_vec, reg.position_correlation_vector)
            corr = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
            if corr > best_corr:
                best_corr = corr

        if best_corr >= cfg.style_correlation_threshold:
            excess = (best_corr - cfg.style_correlation_threshold) / max(
                1e-12, 1.0 - cfg.style_correlation_threshold
            )
            return float(min(1.0, 0.5 + 0.5 * excess))
        return 0.0

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def time_decay(self, current_epoch: int) -> None:
        """Prune history older than the configured window."""
        cutoff = current_epoch - self.config.history_window_epochs
        self._model_history = {
            k: v
            for k, v in self._model_history.items()
            if v.last_seen_epoch >= cutoff
        }
        self._trading_history = {
            k: v
            for k, v in self._trading_history.items()
            if v.last_seen_epoch >= cutoff
        }
        self._all_fingerprints = {
            fp: ep for fp, ep in self._all_fingerprints.items() if ep >= cutoff
        }
        # Rebuild the per-miner lookup maps from surviving history so
        # exact-duplicate detection does not keep referencing submissions
        # that have aged out of the window. Without this, a fingerprint or
        # style that left the history window would still force novelty=0
        # and duplicate_score=1 for other miners indefinitely.
        # Each miner maps to the *set* of fingerprints/styles it has
        # registered within the window, so an earlier-but-still-windowed
        # artifact is not lost when a newer one is registered.
        rebuilt_models: Dict[str, Set[str]] = {}
        for k, v in self._model_history.items():
            uid = k.split("::")[0]
            rebuilt_models.setdefault(uid, set()).add(v.fingerprint)
        self._miner_models = rebuilt_models

        rebuilt_styles: Dict[str, Set[str]] = {}
        for k, v in self._trading_history.items():
            uid = k.split("::")[0]
            rebuilt_styles.setdefault(uid, set()).add(v.style_signature)
        self._miner_styles = rebuilt_styles

    def reset(self) -> None:
        """Clear all tracked history (used between tuning runs)."""
        self._model_history.clear()
        self._trading_history.clear()
        self._miner_models.clear()
        self._miner_styles.clear()
        self._all_fingerprints.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _model_key(miner_uid: str, fingerprint: str) -> str:
        return f"{miner_uid}::{fingerprint}"

    @staticmethod
    def _trading_key(miner_uid: str, style_signature: str) -> str:
        return f"{miner_uid}::{style_signature}"


# ---------------------------------------------------------------------------
# Style signature helpers
# ---------------------------------------------------------------------------

def compute_style_signature(
    positions: Sequence[Dict],
    instruments: Sequence[str],
) -> Tuple[str, np.ndarray]:
    """
    Derive a style signature and position-correlation vector from a list of
    positions.

    The signature is a hash of the dominant execution characteristics
    (instrument distribution, side bias, holding-period bucket, sizing
    quantile). The correlation vector is a per-instrument signed-position
    series suitable for clone detection across traders.
    """
    if not positions:
        return "empty", np.array([])

    # Instrument distribution.
    inst_counts: Dict[str, int] = {}
    side_balance = 0
    holding_buckets: List[int] = []
    sizes: List[float] = []

    for p in positions:
        inst = p.get("instrument", "unknown")
        inst_counts[inst] = inst_counts.get(inst, 0) + 1
        side = p.get("side", "long")
        side_balance += 1 if side == "long" else -1
        dur = p.get("holding_seconds", 0)
        if dur < 60:
            holding_buckets.append(0)
        elif dur < 3600:
            holding_buckets.append(1)
        elif dur < 14400:
            holding_buckets.append(2)
        else:
            holding_buckets.append(3)
        sizes.append(float(p.get("size", 0)))

    total = max(len(positions), 1)
    inst_dist = tuple(
        sorted((k, round(v / total, 3)) for k, v in inst_counts.items())
    )
    side_ratio = round(side_balance / total, 3)
    holding_hist = tuple(
        holding_buckets.count(b) for b in range(4)
    )
    size_quantile = round(float(np.median(sizes)) if sizes else 0.0, 4)

    sig_str = f"{inst_dist}|{side_ratio}|{holding_hist}|{size_quantile}"
    signature = hashlib.sha256(sig_str.encode()).hexdigest()[:16]

    # Per-instrument signed-position vector for correlation.
    inst_series: Dict[str, float] = {inst: 0.0 for inst in instruments}
    for p in positions:
        inst = p.get("instrument", "unknown")
        if inst not in inst_series:
            inst_series[inst] = 0.0
        sign = 1.0 if p.get("side", "long") == "long" else -1.0
        inst_series[inst] += sign * float(p.get("size", 0))

    vec = np.array([inst_series.get(i, 0.0) for i in instruments])
    return signature, vec


def feature_set_signature(features: Sequence[str]) -> str:
    """Hash a feature set for compact fingerprinting."""
    return hashlib.sha256(
        "|".join(sorted(features)).encode()
    ).hexdigest()[:16]
