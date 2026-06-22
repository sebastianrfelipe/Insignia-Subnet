# Changelog

## 2026-06-22 - Reproducible code submission for researcher miners

- researcher miners now submit the **source code that produced/serves their model** alongside the serialized artifact, mirroring Metanova Labs' NOVA subnet (SN68): validators re-execute the code in an isolated sandbox and confirm it reproduces the artifact's predictions before scoring, making submissions auditable and ungameable by opaque/hard-coded/tampered artifacts
- added `insignia/code_submission.py`: deterministic `tar.gz` bundle packaging (`build_code_bundle`/`build_code_bundle_from_dir`) with a per-file manifest; `CodeBundleVerifier` (hash + manifest + entrypoint checks, extraction safety against traversal/symlinks/zip-bombs, static scan rejecting sandbox-escaping source); `SandboxRunner` (subprocess isolation via POSIX rlimits, wall-clock budget, scrubbed env, and best-effort network-namespace drop with `unshare -n`); `ReproducibilityChecker` (re-runs the entrypoint and compares predictions); and `CodeFingerprinter` (normalized-source plagiarism detection)
- `protocol.py`: `ModelSubmission` and `PairEvaluationRequest` carry `code_bundle`, `code_bundle_hash`, `code_entrypoint`, `code_manifest`, and `code_signature`, plus `code_verified`/`code_reproducible`/`reproducibility_score`/`code_rejection_reason` response fields
- `neurons/researcher_miner.py`: `ModelTrainer.build_code_bundle()` packages an `inference.py` entrypoint + model + training source; `train_and_submit` attaches the bundle to every submission (NOVA `input.json` → `result.json` I/O convention)
- `neurons/model_validator.py`: added `CodeSubmissionValidator` (verify → fingerprint → sandbox-reproduce); `ModelEvaluator.evaluate` accepts an optional `capture` for repro inputs and exposes a shared `predict` convention; `ModelValidator` gains `require_code` and `gate_on_reproducibility` (non-reproducible submissions are scored zero)
- `insignia/incentive.py`: new attack/defense entry for opaque/unreproducible/tampered artifacts, and code-fingerprinting added to the model-plagiarism defense
- added `tests/test_code_submission.py` (26 tests) and updated `docs/SUBNET_SPEC.md` + `docs/researcher.md`

## 2026-06-22 - Deterministic tiebreakers in pair selection

- made cross-validator pair selection fully order-independent so every honest validator reproducing the same generation derives byte-identical pairings/weights (a Yuma-consensus requirement)
- `ChainSeededPairing.reproduce`: elite selection now sorts by `(-selection_score, genome.key)` instead of `selection_score` alone; previously equal-score pairs kept whatever order `prev_fitnesses` arrived in, so two validators holding the same fitnesses in a different order could pick different elites and diverge
- `crowding_distance`: objective-value ties now break on the row index so the NSGA-II crowding ordering is identical regardless of incidental front ordering
- added `test_reproduce_tiebreak_is_order_independent` to `tests/test_pairing.py`

## 2026-06-17 - Rename L1/L2 nomenclature to researcher/trader

- there are no L1/L2 miners under the single paired mechanism, so the two-layer naming was renamed to the role/skill it denotes throughout the code and docs
- scoring: `score_l1`/`score_l2` → `score_model`/`score_trading`; `WeightConfig.l1_*`/`l2_*` → `model_*`/`trading_*`; normalization helpers renamed
- neurons: consolidated `l1_miner.py` → `researcher_miner.py` (`L1Miner` → `ResearcherMiner`, `L1ModelTrainer` → `ModelTrainer`) and `l2_miner.py` → `trader_miner.py` (`L2StrategyMiner` → `TraderMiner`); renamed `l1_validator.py` → `model_validator.py` (`L1Validator` → `ModelValidator`) and `l2_validator.py` → `trading_validator.py` (`L2Validator` → `TradingValidator`)
- protocol: `L1ModelSubmission`/`L1EvaluationRequest`/`L1ScoreReport`/`L2StrategySubmission`/`L2ModelPool`/`L2PositionUpdate` → `ModelSubmission`/`ModelEvaluationRequest`/`ModelScoreReport`/`TradingStrategySubmission`/`ModelPool`/`TradingPositionUpdate`
- tuning/testnet: parameter group names (`l1_weights`/`l2_weights` → `model_weights`/`trading_weights`) and weight param names, `SimulationResult`/`SimulationHarness` fields (`l1_agents`/`l2_agents` → `researcher_agents`/`trader_agents`, `honest_l1_scores` → `honest_researcher_scores`, etc.), `EmulatorConfig` fields, Prometheus metric names (`insignia_l1_*`/`insignia_l2_*` → `insignia_researcher_*`/`insignia_trader_*`), and the Grafana dashboard
- the deprecated cross-layer/promotion subsystem keeps its legacy `l1`/`l2`/`cross_layer` names since it no longer exists in the paired design

## 2026-06-17 - Remove Model Attribution from trading score

- removed the `model_attribution` metric from the L2/trading scorer: under the single paired genetic mechanism the model is *assigned* to a trader by the chain-seeded genetic algorithm rather than self-selected, so a miner cannot influence which model it is paired with and should not be rewarded or penalized for that assignment
- redistributed the removed weight across the remaining performance metrics; the L2 scorer is now 9 metrics (`realized_pnl` 0.20, `omega` 0.13, `max_drawdown` 0.14, `win_rate` 0.06, `consistency` 0.20, `execution_quality` 0.10, `annualized_volatility` 0.05, `sharpe_ratio` 0.06, `sortino_ratio` 0.06)
- dropped `model_attribution_score` from `CompositeScorer.score_l2`, removed `WeightConfig.l2_model_attribution`, and removed the `l2_model_attribution` tuning parameter (`tuning/parameter_space.py`)
- removed the now-unused `ModelAttributionEngine` from `neurons/l2_validator.py`
- cross-partner model quality is still expressed structurally via NSGA-II pair selection and the variance-penalized marginal-contribution credit, not as a per-miner scoring dimension
- updated `docs/INCENTIVE_MECHANISM.md`, `docs/SUBNET_SPEC.md`, `docs/PARAMETER_TUNING_PLAN.md`, `docs/tuner.md`, `README.md`, and `program.md` accordingly

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
