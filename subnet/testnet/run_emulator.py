"""
Emulator CLI Entry Point

Provides a command-line interface for the Insignia testnet emulator
with multiple run modes:

  setup       – Create wallets, subnet, register neurons, fund wallets
  single      – Run a single epoch with default parameters
  sweep       – Run a parameter sweep across N random configurations
  evolve      – Run evolutionary optimization with testnet integration
  status      – Check subnet and wallet status
  full        – Run complete pipeline: setup + evolve + export

Usage:
    python -m testnet.run_emulator --mode single --network local
    python -m testnet.run_emulator --mode evolve --generations 20 --population 15
    python -m testnet.run_emulator --mode full --network local --output testnet_results/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testnet.config import EmulatorConfig, NetworkTarget, WalletConfig
from testnet.wallet_manager import WalletManager
from testnet.subnet_manager import SubnetManager
from testnet.emulator import InsigniaEmulator

from tuning.parameter_space import encode_defaults, get_bounds, repair_weights, decode
from tuning.metrics_exporter import start_metrics_server, stop_metrics_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_emulator")


def run_setup(config: EmulatorConfig) -> bool:
    """Setup phase: wallets, subnet, registration, funding."""
    logger.info("=" * 70)
    logger.info("  SETUP: %s network", config.network.value)
    logger.info("=" * 70)

    wallet_mgr = WalletManager(config)
    wallets = wallet_mgr.setup_all_wallets()

    subnet_mgr = SubnetManager(config)

    if config.netuid is None:
        netuid = subnet_mgr.create_subnet()
        if netuid is None:
            logger.error("Failed to create subnet")
            return False
        config.netuid = netuid
    else:
        subnet_mgr.netuid = config.netuid
        logger.info("Using existing subnet: netuid=%d", config.netuid)

    if config.network == NetworkTarget.LOCAL:
        wallet_mgr.fund_from_alice(amount_per_wallet=10000.0)

    reg_results = subnet_mgr.register_all_neurons(wallets)

    for i in range(config.wallets.n_validators):
        vname = config.wallets.validator_coldkey(i)
        subnet_mgr.stake_validator(vname, amount=1000.0)

    subnet_mgr.configure_hyperparameters()

    if config.network == NetworkTarget.LOCAL:
        subnet_mgr.start_emissions()

    info = subnet_mgr.get_subnet_info()
    logger.info("Subnet info: %s", info)

    balances = wallet_mgr.check_balances()
    logger.info("Wallet balances: %s", json.dumps(balances, indent=2))

    logger.info("Setup complete!")
    return True


def run_single(config: EmulatorConfig):
    """Run a single emulator epoch with default parameters."""
    emulator = InsigniaEmulator(config)
    emulator.initialize()

    result = emulator.run_single_epoch(epoch_idx=0)

    out_path = emulator.save_results()
    logger.info("Single epoch complete. Results saved to %s", out_path)

    return result


def run_sweep(config: EmulatorConfig, n_configs: int = 10):
    """Run a parameter sweep across random configurations."""
    emulator = InsigniaEmulator(config)
    emulator.initialize()

    lower, upper = get_bounds()
    param_vectors = [encode_defaults()]
    for _ in range(n_configs - 1):
        x = np.random.uniform(lower, upper)
        x = repair_weights(x)
        param_vectors.append(x)

    run_result = emulator.run_parameter_sweep(param_vectors)

    out_path = emulator.save_results()

    run_file = Path(config.output_dir) / "sweep_results.json"
    with open(run_file, "w") as f:
        json.dump(run_result.to_dict(), f, indent=2, default=str)

    logger.info(
        "Parameter sweep complete: %d configs, best at epoch %d",
        len(param_vectors),
        run_result.best_epoch,
    )

    if run_result.best_config:
        from tuning.parameter_space import summarize_config
        logger.info("Best configuration:\n%s", summarize_config(run_result.best_config))

    return run_result


def run_evolve(config: EmulatorConfig, n_generations: int = 20, population: int = 15):
    """Run evolutionary optimization with testnet integration."""
    emulator = InsigniaEmulator(config)
    emulator.initialize()

    run_result = emulator.run_evolutionary_tuning(
        n_generations=n_generations,
        population_size=population,
    )

    out_path = emulator.save_results()

    run_file = Path(config.output_dir) / "evolution_results.json"
    with open(run_file, "w") as f:
        json.dump(run_result.to_dict(), f, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("  EVOLUTIONARY TUNING COMPLETE")
    logger.info("=" * 70)
    logger.info("  Generations: %d", n_generations)
    logger.info("  Population: %d", population)
    logger.info("  Total time: %.1fs", run_result.total_elapsed)
    logger.info("  Best epoch: %d", run_result.best_epoch)

    if run_result.best_fitness is not None:
        for name, val in zip(OBJECTIVE_NAMES, run_result.best_fitness):
            logger.info("  %s: %.4f", name, val)

    if run_result.best_config:
        from tuning.parameter_space import summarize_config
        logger.info("\nBest configuration:\n%s", summarize_config(run_result.best_config))

    return run_result


def run_status(config: EmulatorConfig):
    """Check current subnet and wallet status."""
    logger.info("=" * 70)
    logger.info("  STATUS CHECK: %s network", config.network.value)
    logger.info("=" * 70)

    if config.netuid:
        subnet_mgr = SubnetManager(config)
        subnet_mgr.netuid = config.netuid
        info = subnet_mgr.get_subnet_info()
        logger.info("Subnet info: %s", info)

        metagraph = subnet_mgr.get_metagraph()
        if metagraph:
            logger.info("Metagraph:\n%s", metagraph[:2000])

    wallet_mgr = WalletManager(config)
    wallet_mgr.setup_all_wallets()
    balances = wallet_mgr.check_balances()
    logger.info("Wallet balances:")
    for key, balance in balances.items():
        logger.info("  %s: %.4f TAO", key, balance)


def run_full(config: EmulatorConfig, n_generations: int = 20, population: int = 15):
    """Full pipeline: setup + evolve + export."""
    logger.info("=" * 70)
    logger.info("  FULL PIPELINE: setup → evolve → export")
    logger.info("=" * 70)

    ok = run_setup(config)
    if not ok:
        logger.warning("Setup encountered issues — continuing with emulator in offline mode")

    return run_evolve(config, n_generations=n_generations, population=population)


from tuning.optimizer import OBJECTIVE_NAMES


def main():
    parser = argparse.ArgumentParser(
        description="Insignia Testnet Emulator — Hyperparameter & Incentive Mechanism Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  setup     Create wallets, subnet, register neurons (local chain)
  single    Run one epoch with default parameters (quick validation)
  sweep     Run N random parameter configurations
  evolve    Run evolutionary optimization loop
  status    Check subnet and wallet status
  full      Complete pipeline: setup → evolve → export

Examples:
  # Quick local test
  python -m testnet.run_emulator --mode single --network local

  # Full pipeline on local chain
  python -m testnet.run_emulator --mode full --network local --generations 10

  # Evolutionary tuning against public testnet
  python -m testnet.run_emulator --mode evolve --network testnet --netuid 42 --generations 30

  # Parameter sweep with custom agent counts
  python -m testnet.run_emulator --mode sweep --n-configs 20 --n-honest 8 --n-adversarial 6
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["setup", "single", "sweep", "evolve", "status", "full"],
        default="single",
        help="Emulator run mode",
    )
    parser.add_argument(
        "--network",
        choices=["local", "testnet", "devnet"],
        default="local",
        help="Target Bittensor network",
    )
    parser.add_argument("--netuid", type=int, default=None, help="Existing subnet netuid")
    parser.add_argument("--output", type=str, default="testnet_results", help="Output directory")

    parser.add_argument("--generations", type=int, default=20, help="Evolutionary generations")
    parser.add_argument("--population", type=int, default=15, help="Population size per generation")
    parser.add_argument("--n-configs", type=int, default=10, help="Number of configurations (sweep mode)")

    parser.add_argument("--n-honest", type=int, default=6, help="Number of honest L1 agents")
    parser.add_argument("--n-adversarial", type=int, default=4, help="Number of adversarial L1 agents")
    parser.add_argument("--n-epochs", type=int, default=3, help="L1 epochs per simulation")
    parser.add_argument("--n-steps", type=int, default=200, help="L2 trading steps per simulation")

    parser.add_argument("--metrics-port", type=int, default=8001, help="Prometheus metrics port")
    parser.add_argument("--no-metrics", action="store_true", help="Disable Prometheus metrics")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")

    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))
    logging.getLogger("simulation").setLevel(logging.WARNING)

    config = EmulatorConfig(
        network=NetworkTarget(args.network),
        netuid=args.netuid,
        n_l1_epochs=args.n_epochs,
        n_l2_trading_steps=args.n_steps,
        n_honest_l1=args.n_honest,
        n_adversarial_l1=args.n_adversarial,
        output_dir=args.output,
        metrics_port=args.metrics_port,
    )

    if not args.no_metrics:
        try:
            start_metrics_server(args.metrics_port)
            logger.info("Prometheus metrics at http://localhost:%d/metrics", args.metrics_port)
        except Exception as e:
            logger.warning("Could not start metrics server: %s", e)

    try:
        if args.mode == "setup":
            run_setup(config)

        elif args.mode == "single":
            run_single(config)

        elif args.mode == "sweep":
            run_sweep(config, n_configs=args.n_configs)

        elif args.mode == "evolve":
            run_evolve(config, n_generations=args.generations, population=args.population)

        elif args.mode == "status":
            run_status(config)

        elif args.mode == "full":
            run_full(config, n_generations=args.generations, population=args.population)

    except KeyboardInterrupt:
        logger.info("Interrupted — saving partial results...")
    finally:
        if not args.no_metrics:
            stop_metrics_server()


if __name__ == "__main__":
    main()
