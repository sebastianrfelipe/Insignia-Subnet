"""Shared chain-parameter and pool-snapshot read layer for the fund modules.

Every fund module reads protocol parameters through this package (SPEC §0.16):
all parameters are root-mutable, so nothing downstream may embed chain constants.
Pure-math modules take `ChainParams` / `PoolSnapshot` values as arguments; live
providers are the only code that talks to the chain or taostats.
"""

from chainio.collateral import (
    CollateralPolicy,
    MinerCollateralPosition,
    SettleResult,
    apply_collateral_gate,
    insignia_default_policy,
    native_locked_fraction,
    registration_split,
    settle_miner_collateral,
    total_native_locked,
    unlock_horizon_days,
)
from chainio.params import (
    ChainParams,
    PoolSnapshot,
    ParamsProvider,
    StaticProvider,
    SubtensorProvider,
    TaostatsProvider,
    ValidatorBasket,
    reference_pool,
    stake_weighted_insignia_weight,
    total_escrow_alpha,
)

__all__ = [
    "ChainParams",
    "CollateralPolicy",
    "MinerCollateralPosition",
    "PoolSnapshot",
    "ParamsProvider",
    "SettleResult",
    "StaticProvider",
    "SubtensorProvider",
    "TaostatsProvider",
    "ValidatorBasket",
    "apply_collateral_gate",
    "insignia_default_policy",
    "native_locked_fraction",
    "reference_pool",
    "registration_split",
    "settle_miner_collateral",
    "stake_weighted_insignia_weight",
    "total_escrow_alpha",
    "total_native_locked",
    "unlock_horizon_days",
]
