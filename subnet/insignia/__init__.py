"""
Insignia Subnet — Decentralized Predictive Modeling for On-Chain Markets

A Bittensor subnet implementing a single paired genetic incentive mechanism:
researcher miners (ML models) and trader miners (trading operations) share one
UID space and one weight vector. A candidate strategy is a `(researcher, trader)`
pair; pairs are chain-seeded, jointly evaluated on the same model + trading
metrics as before, ranked with NSGA-II, screened for collusion, and rewarded via
a variance-penalized marginal contribution.

See `insignia.pairing` and `docs/PAIRING_MECHANISM.md`.
"""

__version__ = "0.2.0"
__subnet_name__ = "Insignia"
