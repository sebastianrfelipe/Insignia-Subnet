# Changelog

## 2026-06-16 - Single paired genetic incentive mechanism

- migrated from the two-layer (L1 model -> promotion -> L2 strategy) design to a single incentive mechanism in which researcher and trader miners are matched into `(model, strategy)` pairs, jointly evaluated, and selected with an NSGA-II-style genetic algorithm (`docs/PAIRING_MECHANISM.md`)
- added `insignia/pairing.py`: `PairGenome`/`PairFitness`/`PairingConfig`, chain-seeded `ChainSeededPairing` (assignment + crossover/mutation reproduction with a K-partner floor), `NSGA2Matchmaker` (non-dominated sort + crowding + scalar-composite blend), `MarginalContributionCredit` (variance-penalized `mean - lambda*std` across partners), `CollusionGraphDetector` (non-transferable-lift interaction anomaly), and `PairingPopulation`
- added `CompositeScorer.combine_pair`/`combine_pair_scores` and `WeightConfig.pair_blend_alpha`; the 7 model + 10 trading metrics and their weights are unchanged
- added `MinerRole` and the `PairAssignment`/`PairEvaluationRequest`/`PairScoreReport` synapses; made the `bittensor` import optional so the tuning/simulation stack runs without the chain SDK
- added unified `neurons/validator.py` (`PairedValidator`: assign -> joint eval -> NSGA-II -> credit -> single `set_weights`) and role-aware `neurons/researcher_miner.py` / `neurons/trader_miner.py`; deprecated `insignia/cross_layer.py`
- rewrote `tuning/simulation.py` to run pair-based generations with `ColludingResearcher`/`ColludingTrader`/`PartnerGamingTrader` adversaries; fixed the `route_manifest()` vs `routing_metadata()` mismatch
- added `pair_collusion`, `partner_selection_gaming`, and `latency_arbitrage_pairing` attack vectors and reframed `weight_manipulation` as role-emission balance
- replaced the `l1_l2_emission_split` / cross-layer parameters with a `pairing` parameter group (`partners_per_miner`, `elite_fraction`, `mutation_rate`, `pair_blend_alpha`, `marginal_contribution_weight`, `fixed_pair_correlation_threshold`, `max_pairs`)
- updated the offline NSGA-II tuner, the end-to-end demo (`scripts/run_demo.py`), and added `tests/test_pairing.py`

## 2026-04-14 - Orchestration report synchronization

- upgraded the tuning stack toward AttackDetector v8 behavior with commitment violation scoring, selective revelation escalation, and expanded anomaly/correlation checks
- updated default weights and validation timing hyperparameters from the latest orchestration findings
- added `composite_integrity_scorer.py` for EXP-023 style integrity scoring
- expanded protocol and testnet config with SOL, AVAX, and ADA market diversification support
- enriched simulator telemetry for commit/reveal timestamps, no-reveal streaks, validator timing, ensemble signals, and convergence tracking
- expanded Prometheus metrics with commit/reveal timestamps, timing attack composite, trading-pair activity, and ensemble signals
- updated autoresearch logging for experiment ids, radical levels, and report-aligned TSV output
- documented Phase 4 status, Phase 5 transition viability, NSGA-II defaults, and researcher/sentinel workflows
