"""Shared chain-parameter and pool-snapshot read layer for the fund modules.

Every fund module reads protocol parameters through this package (SPEC §0.16):
all parameters are root-mutable, so nothing downstream may embed chain constants.
Pure-math modules take `ChainParams` / `PoolSnapshot` values as arguments; live
providers are the only code that talks to the chain or taostats.
"""

from chainio.params import (
    ChainParams,
    PoolSnapshot,
    ParamsProvider,
    StaticProvider,
    SubtensorProvider,
    TaostatsProvider,
    reference_pool,
)

__all__ = [
    "ChainParams",
    "PoolSnapshot",
    "ParamsProvider",
    "StaticProvider",
    "SubtensorProvider",
    "TaostatsProvider",
    "reference_pool",
]
