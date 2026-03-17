"""
Insignia Testnet Deployment & Emulator

Provides tooling for deploying the Insignia subnet on a Bittensor testnet
(local or public) and running the incentive-mechanism emulator for
hyperparameter tuning with real Yuma consensus weight-setting.

Modules:
    config          – Network endpoints, wallet names, subnet hyperparameters
    wallet_manager  – btcli wallet creation and funding helpers
    subnet_manager  – Subnet lifecycle (create, register, configure)
    emulator        – Full L1/L2 emulator bridging simulation agents to the chain
    run_emulator    – CLI entry point
"""

__all__ = [
    "config",
    "wallet_manager",
    "subnet_manager",
    "emulator",
]
