"""
Researcher Miner — ML Model Generation (single paired mechanism)

A researcher miner trains predictive ML models and submits artifacts for
validator evaluation, exactly as in the original Layer 1 miner task. Under the
single paired mechanism the model is not promoted into a pool; instead the
validator pairs this researcher with trader miners (chain-seeded) and evaluates
the resulting `(model, strategy)` pairs jointly.

This module is a thin, role-aware wrapper over the existing `L1Miner` training
pipeline so the model-building code is reused verbatim. The only addition is the
declared `MinerRole.RESEARCHER` and a `role` field on submissions.

Usage:
    python neurons/researcher_miner.py --netuid <NETUID> --wallet.name <WALLET> --subtensor.network test
"""

from __future__ import annotations

from typing import Any, Dict

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neurons.l1_miner import (  # noqa: F401  (re-exported for convenience)
    L1Miner,
    L1ModelTrainer,
    generate_demo_data,
    PUBLIC_FEATURE_REGISTRY,
)
from insignia.protocol import MinerRole


class ResearcherMiner(L1Miner):
    """Role-aware researcher miner. Reuses the L1 training pipeline."""

    role = MinerRole.RESEARCHER

    def train_and_submit(self, data) -> Dict[str, Any]:
        submission = super().train_and_submit(data)
        submission["role"] = MinerRole.RESEARCHER.value
        return submission


def demo():
    data = generate_demo_data(n_samples=3000)
    miner = ResearcherMiner()
    submission = miner.train_and_submit(data)
    print(f"Researcher submission: role={submission['role']} "
          f"artifact={submission['artifact_size_bytes']} bytes "
          f"hash={submission['artifact_hash'][:12]}")
    return submission


if __name__ == "__main__":
    demo()
