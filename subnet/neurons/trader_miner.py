"""
Trader Miner — Trading Operations (single paired mechanism)

A trader miner builds and operates a trading strategy that consumes a model's
signals, exactly as in the original Layer 2 miner task. The key change under the
single paired mechanism: the trader does NOT self-select models from a promotion
pool. Instead the validator assigns a researcher partner (chain-seeded) and the
trader runs its strategy on that paired model. This removes the self-selection
surface that previously enabled researcher/trader collusion.

This module is a thin, role-aware wrapper over the existing `L2StrategyMiner`
paper-trading engine so the execution code is reused verbatim. The only addition
is the declared `MinerRole.TRADER` and a `load_assigned_model` helper that makes
the pairing-driven model load explicit.

Usage:
    python neurons/trader_miner.py --netuid <NETUID> --wallet.name <WALLET> --subtensor.network test
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neurons.l2_miner import (  # noqa: F401  (re-exported for convenience)
    L2StrategyMiner,
    PaperTradingEngine,
    SlippageConfig,
    Side,
)
from insignia.protocol import MinerRole


class TraderMiner(L2StrategyMiner):
    """Role-aware trader miner. Reuses the L2 paper-trading engine."""

    role = MinerRole.TRADER

    def load_assigned_model(self, model_id: str, model_artifact: bytes):
        """
        Load the model assigned by the validator's pairing for this generation.

        Unlike the old self-selection flow, the trader cannot pick which model
        it runs — it is the researcher half of the chain-seeded pair.
        """
        self.load_model(model_id, model_artifact)


def demo():
    from neurons.researcher_miner import ResearcherMiner, generate_demo_data
    import numpy as np
    import time

    researcher = ResearcherMiner()
    submission = researcher.train_and_submit(generate_demo_data(n_samples=2000))

    trader = TraderMiner()
    trader.load_assigned_model("paired_model", submission["model_artifact"])

    rng = np.random.RandomState(11)
    price = 50000.0
    for step in range(200):
        ret = rng.normal(0.0001, 0.003)
        price *= (1 + ret)
        features = rng.normal(0, 1, 15)
        features[0] = ret
        trader.execute_step("BTC-USDT-PERP", price, features, time.time() + step * 3600)

    perf = trader.engine.get_performance_summary()
    print(f"Trader role={trader.role.value} trades={perf['total_trades']} pnl={perf['realized_pnl']:.2f}")
    return perf


if __name__ == "__main__":
    demo()
