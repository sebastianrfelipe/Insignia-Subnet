"""
Testnet Configuration

Central configuration for Insignia subnet testnet deployments.
Supports three target environments:

  local     – Docker-based subtensor with fast (250ms) blocks
  testnet   – Bittensor public testnet (wss://test.finney.opentensor.ai)
  devnet    – Custom devnet endpoint (bring your own)

All tunable subnet hyperparameters that get registered on-chain are
defined here alongside wallet naming conventions and network endpoints.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional


class NetworkTarget(str, Enum):
    LOCAL = "local"
    TESTNET = "testnet"
    DEVNET = "devnet"


NETWORK_ENDPOINTS: Dict[NetworkTarget, str] = {
    NetworkTarget.LOCAL: os.environ.get(
        "SUBTENSOR_LOCAL_ENDPOINT", "ws://127.0.0.1:9945"
    ),
    NetworkTarget.TESTNET: os.environ.get(
        "SUBTENSOR_TESTNET_ENDPOINT", "wss://test.finney.opentensor.ai:443"
    ),
    NetworkTarget.DEVNET: os.environ.get(
        "SUBTENSOR_DEVNET_ENDPOINT", "ws://127.0.0.1:9944"
    ),
}

EVM_ENDPOINTS: Dict[NetworkTarget, str] = {
    NetworkTarget.LOCAL: "http://127.0.0.1:9933",
    NetworkTarget.TESTNET: "https://test.chain.opentensor.ai",
    NetworkTarget.DEVNET: "http://127.0.0.1:9933",
}

EVM_CHAIN_IDS: Dict[NetworkTarget, int] = {
    NetworkTarget.LOCAL: 945,
    NetworkTarget.TESTNET: 945,
    NetworkTarget.DEVNET: 945,
}

BITTENSOR_HOME = Path(os.environ.get("BITTENSOR_HOME", "~/.bittensor")).expanduser()
DEFAULT_TRADING_PAIRS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "AVAXUSDT",
    "ADAUSDT",
]


@dataclass
class WalletConfig:
    """Naming conventions for the wallets used in emulation."""

    owner_coldkey: str = "insignia-owner"
    owner_hotkey: str = "default"

    validator_coldkey_prefix: str = "insignia-validator"
    validator_hotkey: str = "default"

    miner_coldkey_prefix: str = "insignia-miner"
    miner_hotkey: str = "default"

    n_validators: int = 1
    n_miners: int = 12

    def validator_coldkey(self, idx: int) -> str:
        if self.n_validators == 1:
            return self.validator_coldkey_prefix
        return f"{self.validator_coldkey_prefix}-{idx}"

    def miner_coldkey(self, idx: int) -> str:
        return f"{self.miner_coldkey_prefix}-{idx}"


@dataclass
class SubnetHyperparameters:
    """
    On-chain hyperparameters registered when creating/configuring the subnet.

    These are the Bittensor network-level parameters, separate from the
    Insignia scoring weights (which live in the parameter_space module).
    """

    tempo: int = 360
    immunity_period: int = 5000
    max_validators: int = 64
    min_allowed_weights: int = 1
    max_weight_limit: int = 65535
    adjustment_alpha: int = 0
    registration_allowed: bool = True
    target_regs_per_interval: int = 1
    min_burn: int = 0
    max_burn: int = 100_000_000_000
    bonds_moving_avg: int = 900000
    commit_reveal_weights_enabled: bool = True
    commit_reveal_weights_interval: int = 1000
    alpha_high: int = 58982
    alpha_low: int = 45875
    liquid_alpha_enabled: bool = True


@dataclass
class CommitRevealConfig:
    """
    Application-level commit-reveal scheme configuration.

    Approach B (off-chain with validator attestation) as recommended by
    the deployer agent's feasibility assessment (CR-FEAS-001).
    """

    enabled: bool = True
    commit_window_seconds: float = 30.0
    reveal_window_seconds: float = 15.0
    hash_algorithm: str = "sha256"
    nonce_bits: int = 128
    min_attestations: int = 1
    max_reveal_attempts: int = 3
    late_reveal_penalty: float = 1.0
    grace_period_seconds: float = 2.0


@dataclass
class ValidationTimingConfig:
    """
    Parameters defending against validator latency exploitation (Vector 8)
    and prediction timing manipulation (Vector 11).
    """

    min_prediction_lead_time_seconds: float = 35.0
    validator_latency_penalty_weight: float = 0.28
    high_latency_threshold_ms: float = 1800.0
    commit_rate_threshold: float = 0.75
    commitment_violation_weight: float = 0.012
    expected_commit_reveal_effectiveness: float = 0.700
    required_validation_streak: int = 6
    selective_reveal_warning_streak: int = 1
    selective_reveal_penalty_streak: int = 2
    selective_reveal_zero_streak: int = 3


@dataclass
class ConsensusIntegrityConfig:
    """
    Parameters defending against miner-validator collusion (Vector 9).
    Values from V3-PF-007 knee point and autoresearch optimal.
    """

    weight_entropy_minimum: float = 1.45
    cross_validator_score_variance_max: float = 0.18
    validator_rotation_max_consecutive_epochs: int = 4
    validator_agreement_threshold: float = 0.17
    collusion_detection_lookback_epochs: int = 12


@dataclass
class EnsembleDetectionConfig:
    """
    Ensemble detector parameters from EXP-103 / EXP-113.
    """

    correlation_threshold: float = 0.80
    entropy_threshold_lower: float = 0.20
    symbol_diversity_threshold: float = 0.33
    fusion_strategy: str = "bayesian_model_averaging"
    response_vote_threshold: int = 3
    confirmation_window: int = 2


@dataclass
class ConvergenceMonitoringConfig:
    """
    Sentinel convergence monitoring configuration used by autoresearch.
    """

    moving_average_window: int = 5
    consecutive_increase_threshold: int = 3
    info_threshold: float = 0.05
    critical_threshold: float = 0.15
    emergency_breach_count: int = 5
    monitoring_frequency_seconds: int = 60


@dataclass
class MarketDataConfig:
    """
    Market data diversification settings.
    """

    trading_pairs: list[str] = field(default_factory=lambda: list(DEFAULT_TRADING_PAIRS))
    dominant_pair_warning_ratio: float = 1.35


@dataclass
class EmulatorConfig:
    """Top-level emulator configuration."""

    network: NetworkTarget = NetworkTarget.LOCAL
    netuid: Optional[int] = None

    wallets: WalletConfig = field(default_factory=WalletConfig)
    subnet_params: SubnetHyperparameters = field(default_factory=SubnetHyperparameters)

    commit_reveal: CommitRevealConfig = field(default_factory=CommitRevealConfig)
    validation_timing: ValidationTimingConfig = field(default_factory=ValidationTimingConfig)
    consensus_integrity: ConsensusIntegrityConfig = field(default_factory=ConsensusIntegrityConfig)
    ensemble_detection: EnsembleDetectionConfig = field(default_factory=EnsembleDetectionConfig)
    convergence_monitoring: ConvergenceMonitoringConfig = field(default_factory=ConvergenceMonitoringConfig)
    market_data: MarketDataConfig = field(default_factory=MarketDataConfig)

    n_l1_epochs: int = 5
    n_l2_trading_steps: int = 300
    epoch_interval_seconds: int = 120

    n_honest_l1: int = 6
    n_adversarial_l1: int = 4
    n_honest_l2: int = 3
    n_adversarial_l2: int = 1

    output_dir: str = "testnet_results"
    enable_metrics: bool = True
    metrics_port: int = 8001

    log_level: str = "INFO"
    fast_mode: bool = True

    @property
    def endpoint(self) -> str:
        return NETWORK_ENDPOINTS[self.network]

    @property
    def btcli_network_flag(self) -> str:
        if self.network == NetworkTarget.TESTNET:
            return "test"
        return self.endpoint

    def to_dict(self) -> Dict[str, Any]:
        return {
            "network": self.network.value,
            "endpoint": self.endpoint,
            "netuid": self.netuid,
            "n_l1_epochs": self.n_l1_epochs,
            "n_l2_trading_steps": self.n_l2_trading_steps,
            "n_honest_l1": self.n_honest_l1,
            "n_adversarial_l1": self.n_adversarial_l1,
            "n_honest_l2": self.n_honest_l2,
            "n_adversarial_l2": self.n_adversarial_l2,
            "wallets": {
                "owner": self.wallets.owner_coldkey,
                "n_validators": self.wallets.n_validators,
                "n_miners": self.wallets.n_miners,
            },
            "market_data": {
                "trading_pairs": list(self.market_data.trading_pairs),
                "dominant_pair_warning_ratio": self.market_data.dominant_pair_warning_ratio,
            },
            "output_dir": self.output_dir,
        }


def load_config_from_env() -> EmulatorConfig:
    """Build an EmulatorConfig from environment variables."""
    network_str = os.environ.get("INSIGNIA_NETWORK", "local")
    try:
        network = NetworkTarget(network_str)
    except ValueError:
        network = NetworkTarget.LOCAL

    return EmulatorConfig(
        network=network,
        netuid=int(os.environ["INSIGNIA_NETUID"]) if "INSIGNIA_NETUID" in os.environ else None,
        n_l1_epochs=int(os.environ.get("INSIGNIA_L1_EPOCHS", "5")),
        n_l2_trading_steps=int(os.environ.get("INSIGNIA_L2_STEPS", "300")),
        n_honest_l1=int(os.environ.get("INSIGNIA_HONEST_L1", "6")),
        n_adversarial_l1=int(os.environ.get("INSIGNIA_ADVERSARIAL_L1", "4")),
        output_dir=os.environ.get("INSIGNIA_OUTPUT_DIR", "testnet_results"),
    )
