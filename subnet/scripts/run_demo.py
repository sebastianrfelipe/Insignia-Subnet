#!/usr/bin/env python
"""
Insignia Subnet — Full End-to-End Demo

Demonstrates the complete two-layer pipeline:

  1. Layer 1: Multiple miners train ML models; validator evaluates and ranks them
  2. Cross-Layer: Top models are promoted to the L2 pool
  3. Layer 2: Strategy miners build paper trading strategies around promoted models
  4. Layer 2 Validator: Scores strategies on real (simulated) trading outcomes
  5. Feedback: L2 performance feeds back to adjust L1 model scores

Run this script to see the full pipeline in action with synthetic data.
No external dependencies required beyond numpy, pandas, sklearn, joblib.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --n-miners 10 --n-epochs 5 --n-l2-miners 6
"""

from __future__ import annotations

import sys
import os
import argparse
import time
import logging

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neurons.l1_miner import L1Miner, L1ModelTrainer, generate_demo_data, PUBLIC_FEATURE_REGISTRY
from neurons.l1_validator import L1Validator, ModelEvaluator, DemoBenchmarkProvider
from neurons.l2_miner import L2StrategyMiner, PaperTradingEngine, Side
from neurons.l2_validator import L2Validator
from insignia.cross_layer import CrossLayerOrchestrator, PromotionEngine, PromotionConfig
from insignia.scoring import CompositeScorer, WeightConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")


def section(title: str):
    logger.info("")
    logger.info("=" * 70)
    logger.info("  %s", title)
    logger.info("=" * 70)


def run_full_demo(
    n_l1_miners: int = 8,
    n_l2_miners: int = 4,
    n_epochs: int = 3,
    n_trading_steps: int = 400,
    top_n_promote: int = 4,
):
    section("INSIGNIA SUBNET — FULL PIPELINE DEMO")
    logger.info("  L1 Miners: %d | L2 Miners: %d | Epochs: %d", n_l1_miners, n_l2_miners, n_epochs)
    logger.info("  Trading steps per epoch: %d | Promotion pool: top-%d", n_trading_steps, top_n_promote)

    # ----- Phase 1: Layer 1 Model Generation -----
    section("PHASE 1: Layer 1 — Model Generation")

    l1_miners = {}
    l1_submissions = {}
    for i in range(n_l1_miners):
        data = generate_demo_data(
            n_samples=300,
            n_features=5 + (i % 3),
            seed=42 + i * 7,
        )
        trainer = L1ModelTrainer(
            n_estimators=5 + i * 2,
            max_depth=2,
            learning_rate=0.1,
            min_samples_leaf=10,
            max_bins=32,
            features=PUBLIC_FEATURE_REGISTRY[: 5 + (i % 3)],
            random_state=42 + i,
        )
        miner = L1Miner(trainer=trainer)
        submission = miner.train_and_submit(data)
        uid = f"l1_miner_{i:02d}"
        l1_miners[uid] = miner
        l1_submissions[uid] = submission
        logger.info(
            "  %s: %d features, IS=%.3f, OOS=%.3f, gap=%.3f",
            uid,
            len(trainer.features),
            miner.trainer.training_metrics["in_sample_accuracy"],
            miner.trainer.training_metrics["out_of_sample_accuracy"],
            miner.trainer.training_metrics["overfitting_gap"],
        )

    # ----- Phase 2: Layer 1 Validation -----
    section("PHASE 2: Layer 1 — Validator Evaluation")

    l1_validator = L1Validator(top_n_promote=top_n_promote)
    orchestrator = CrossLayerOrchestrator(
        promotion_engine=PromotionEngine(
            PromotionConfig(
                top_n=top_n_promote,
                min_consecutive_epochs=1,
            )
        )
    )

    all_epoch_results = []
    for epoch in range(n_epochs):
        logger.info("\n--- L1 Epoch %d ---", epoch)
        epoch_result = l1_validator.run_epoch(l1_submissions, force=True)
        all_epoch_results.append(epoch_result)

        epoch_scores = {
            uid: {
                "composite_score": r.get("composite_score", 0.0),
                "overfitting_score": r.get("raw_metrics", {}).get("overfitting_penalty", 0.0),
                "artifact_hash": l1_submissions[uid].get(
                    "preprocessing_hash", ""
                ),
            }
            for uid, r in epoch_result["results"].items()
            if r.get("accepted")
        }
        promo_result = orchestrator.process_l1_epoch(epoch_scores, epoch)
        logger.info("  Promoted: %d models", promo_result["n_promoted"])

    logger.info("\n--- L1 Final Rankings (Last Epoch) ---")
    last_results = all_epoch_results[-1]["results"]
    ranked = sorted(
        [(uid, r) for uid, r in last_results.items() if r.get("accepted")],
        key=lambda x: x[1]["composite_score"],
        reverse=True,
    )
    for rank, (uid, r) in enumerate(ranked):
        promoted = "[PROMOTED]" if r.get("promoted_to_l2") else ""
        logger.info(
            "  #%d %s: %.4f %s",
            rank + 1, uid, r["composite_score"], promoted,
        )

    # ----- Phase 3: Model Promotion -----
    section("PHASE 3: Cross-Layer — Model Promotion")

    pool = orchestrator.promotion.get_pool_summary()
    logger.info("  Active models in L2 pool: %d", pool["active_models"])
    for m in pool["models"]:
        logger.info(
            "    %s (%s): score=%.4f, epochs=%d",
            m["model_id"], m["miner_uid"], m["composite_score"], m["consecutive_epochs"],
        )

    promoted_artifacts = {}
    for model_info in pool["models"]:
        uid = model_info["miner_uid"]
        if uid in l1_submissions:
            promoted_artifacts[model_info["model_id"]] = l1_submissions[uid]["model_artifact"]

    # ----- Phase 4: Layer 2 Strategy Deployment -----
    section("PHASE 4: Layer 2 — Strategy Deployment (Paper Trading)")

    l2_validator = L2Validator()
    l2_miners = {}

    for j in range(n_l2_miners):
        engine = PaperTradingEngine(
            initial_capital=100_000,
            max_position_pct=0.03 + j * 0.015,
            max_drawdown_pct=0.20,
        )
        l2 = L2StrategyMiner(engine=engine)

        model_subset = list(promoted_artifacts.items())[: 2 + (j % 3)]
        for mid, artifact in model_subset:
            l2.load_model(mid, artifact)

        uid = f"l2_miner_{j:02d}"
        l2_miners[uid] = l2
        l2_validator.register_strategy(
            miner_uid=uid,
            strategy_id=l2.strategy_id,
            model_ids=list(l2.models.keys()),
        )
        logger.info("  %s: %d models loaded, max_pos=%.1f%%", uid, len(l2.models), engine.max_position_pct * 100)

    # ----- Phase 5: Paper Trading Simulation -----
    section("PHASE 5: Layer 2 — Paper Trading Simulation")

    rng = np.random.RandomState(777)
    price = 50000.0

    for step in range(n_trading_steps):
        regime_phase = step / n_trading_steps
        if regime_phase < 0.3:
            drift, vol = 0.0003, 0.003  # trending up
        elif regime_phase < 0.5:
            drift, vol = -0.0002, 0.005  # volatile pullback
        elif regime_phase < 0.7:
            drift, vol = 0.0, 0.002  # ranging
        else:
            drift, vol = 0.0001, 0.004  # recovery

        ret = rng.normal(drift, vol)
        price *= (1 + ret)
        features = rng.normal(0, 1, 15)
        features[0] = ret
        ts = time.time() + step * 3600

        for uid, l2 in l2_miners.items():
            update = l2.execute_step("BTC-USDT-PERP", price, features, ts)
            if update:
                l2_validator.process_position_update(uid, update)

    logger.info("  Simulation complete: %d steps, final price=%.2f", n_trading_steps, price)

    # ----- Phase 6: L2 Scoring -----
    section("PHASE 6: Layer 2 — Strategy Scoring")

    l2_epoch = l2_validator.score_epoch()

    logger.info("\n--- L2 Strategy Rankings ---")
    for uid, score_info in l2_epoch["scores"].items():
        tracker = l2_validator.trackers.get(uid)
        status = "ELIMINATED" if tracker and tracker.eliminated else "active"
        logger.info(
            "  %s: composite=%.4f (%s) trades=%d",
            uid,
            score_info["composite"],
            status,
            len(tracker.trades) if tracker else 0,
        )
        if score_info.get("breakdown"):
            for metric, val in score_info["breakdown"].items():
                logger.info("    %s: %.4f", metric, val)

    # ----- Phase 7: Cross-Layer Feedback -----
    section("PHASE 7: Cross-Layer Feedback")

    l1_feedback = l2_validator.get_l1_feedback()
    logger.info("  Feedback adjustments for L1 models:")
    for mid, adj in l1_feedback.items():
        direction = "BONUS" if adj > 1.0 else "PENALTY" if adj < 1.0 else "NEUTRAL"
        logger.info("    %s: %.4f (%s)", mid, adj, direction)

    logger.info("\n  Model Attribution:")
    for mid, attr in l2_epoch.get("model_attribution", {}).items():
        logger.info(
            "    %s: total_pnl=%.2f, uses=%d, avg=%.2f",
            mid,
            attr["total_pnl_contribution"],
            attr["n_strategy_uses"],
            attr["avg_pnl_per_use"],
        )

    # ----- Summary -----
    section("PIPELINE SUMMARY")

    logger.info("  Layer 1:")
    logger.info("    Miners evaluated: %d", n_l1_miners)
    logger.info("    Epochs completed: %d", n_epochs)
    logger.info("    Models promoted: %d", pool["active_models"])

    logger.info("  Layer 2:")
    logger.info("    Strategy miners: %d", n_l2_miners)
    logger.info("    Active: %d", l2_epoch["n_active"])
    logger.info("    Eliminated: %d", l2_epoch["n_eliminated"])

    logger.info("  Cross-Layer:")
    logger.info("    Feedback records: %d", len(l1_feedback))
    logger.info("    Models with L2 data: %d", len(l2_epoch.get("model_attribution", {})))

    logger.info("")
    logger.info("  DEMO COMPLETE — Full two-layer pipeline demonstrated")
    logger.info("  In production: proprietary tick data replaces synthetic benchmark")
    logger.info("  In production: real exchange WebSocket replaces simulated prices")
    logger.info("")

    return {
        "l1_results": all_epoch_results,
        "l2_results": l2_epoch,
        "pool_summary": pool,
        "feedback": l1_feedback,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insignia Subnet Full Demo")
    parser.add_argument("--n-miners", type=int, default=5, help="Number of L1 miners")
    parser.add_argument("--n-l2-miners", type=int, default=3, help="Number of L2 miners")
    parser.add_argument("--n-epochs", type=int, default=2, help="Number of L1 epochs")
    parser.add_argument("--n-steps", type=int, default=200, help="Trading simulation steps")
    parser.add_argument("--top-n", type=int, default=3, help="Top-N models to promote")
    args = parser.parse_args()

    run_full_demo(
        n_l1_miners=args.n_miners,
        n_l2_miners=args.n_l2_miners,
        n_epochs=args.n_epochs,
        n_trading_steps=args.n_steps,
        top_n_promote=args.top_n,
    )
