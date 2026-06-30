# Changelog

## 2026-06-30 - Fix critical RCE: allowlisted deserialization of untrusted model artifacts

- **Critical (RCE):** `model_artifact` bytes arrive over the wire from untrusted researcher miners and were deserialized with `joblib.load` (pickle) directly in the validator process — in `ModelEvaluator.evaluate` *before* the code-submission sandbox ever ran — so a hostile artifact with a crafted `__reduce__` achieved arbitrary code execution on every validator that scored it (and on every paired trader, which `joblib.load`s the model assigned to it). The reproducibility sandbox in `insignia/code_submission.py` was fully bypassed by this path.
- added `insignia/safe_model_loader.py`: `safe_load_model(artifact)` drives joblib's own numpy-aware unpickler but overrides `find_class` with a strict **allowlist** (inert builtins + `numpy`/`scipy`/`sklearn`/`joblib` classes only). Gadget sources a pickle RCE needs — `os`/`posix`/`nt`, `subprocess`, `sys`, `builtins.eval`/`exec`/`getattr`/`__import__`, `operator`, `functools` — are never resolvable, so a malicious `REDUCE`/`STACK_GLOBAL` raises `UnsafeArtifactError` *before any callable runs*. Compatible with existing `joblib.dump` artifacts (incl. compression); **fails closed** (never falls back to `joblib.load`).
- `neurons/model_validator.py`: `_deserialize` now uses `safe_load_model`; `process_submission` catches `UnsafeArtifactError` and rejects the submission with score 0 instead of crashing/executing.
- `neurons/trader_miner.py` (`load_model`) and `tuning/simulation.py` (replay path) routed through the same safe loader for consistency/defense-in-depth.
- added `tests/test_safe_model_loader.py` (8 tests): legit/compressed round-trip, and `os.system`/`eval`/`subprocess` reduce payloads blocked with a sentinel asserting non-execution, including an end-to-end `ModelValidator.process_submission` rejection test.
- note: this narrows the artifact attack surface to the numerical stack rather than eliminating pickle execution categorically; the durable fix is a non-executable artifact format (e.g. ONNX). Tracked separately.

## 2026-06-29 - Retract V13-R3 promotion: failed empirical separation gate

- empirical validation (Orchestration Report 2026-06-29T01-35-48, session 6a419a72) **invalidated** the V13-R3 knee promoted on 2026-06-26. The surrogate reported `0.963` honest/adversarial separation; validating against the full 14-agent adversarial population measured `~0.23` (runs `0.244` / `0.223`). The best adversary (Copycat) scores `0.733` — vs the `~0.018` the surrogate implied (~40× underestimate) — and adversaries capture ~64.7% of chain weight. V13-R3 **fails** the `≥0.90` separation gate; honest_score (`~0.977`) and breach held, separation did not. **Production promotion BLOCKED.**
- root cause: the NSGA-II optimizer scores adversaries with a **simplified analytical model** that under-scores Copycat/CopyTrader (`~0.02-0.05` vs `~0.73` empirically), so the GP surrogate (`R²=0.96` to those biased targets) optimized a false objective; the Pareto front collapsed into a narrow `0.958-0.967` separation band (false local optimum)
- reverted the "state of record" downgrade across `README.md`, `program.md` §1, `docs/SUBNET_SPEC.md`, `docs/tuner.md`, `docs/PARAMETER_TUNING_PLAN.md`, `docs/sentinel.md`, `docs/EMULATOR_SPEC.md` §9 — the last non-contradicted checkpoint is again the R2 knee `V13-R2-KP-020-a7f2`. V13-R3 is recorded as an invalidated offline candidate, not a promoted config
- `reference_configs/knee_point_V13-R3.json`: `status` → `INVALIDATED_...`, added an `empirical_validation` block (verdict, empirical metrics, root cause), relabelled `objectives` as surrogate-predicted
- `docs/EMULATOR_SPEC.md`: added §6.6 "Surrogate vs. empirical adversary model (the validation contract)" — the optimizer's adversary scores must come from the same evaluation path as the acceptance gates; no surrogate-only promotion; surrogate-vs-empirical divergence is a `diverged` sentinel signal
- secondary finding (logged, not a repo change): the local chain at `<chain-host>` was unreachable (`ECONNREFUSED` on all ports; "not running or network-isolated"), so on-chain validation remains blocked and the empirical numbers above are from the simulation harness

## 2026-06-26 - Promote V13-R3 knee point as state of record

- the orchestration study (Orchestration Report — 2026-06-27T03-11-52) records a new surrogate-guided knee point, **V13-R3-KP-020-a3c7**, that strictly dominates the prior R2 knee on all four NSGA-II objectives: breach_rate `2.6e-6` (was `3.5e-6`), honest_score `0.9808` (was `0.9795`), separation `0.963` (was `0.953`), variance `0.00081` (was `0.0009`); knee stable since generation 7 (13 consecutive generations), ~48% below the `5e-6` target
- added `reference_configs/knee_point_V13-R3.json` capturing the knee point objectives, run spec (NSGA-II + GP surrogate, 30 pop × 20 gen, 41 vars, R² `0.96`), elite-seed lineage (EXP-140/141/134/132/133/135), Pareto extremes, and provenance (sourced from MCP `agent_memory` keys `tuner_v13_r3_gen20_final` / `tuner_v13_r3_optimization_run`); the full 41-dim decoded vector was not persisted as a flat artifact and remains reconstructable from MCP run `nsga_ii_v13_r3_surrogate`
- updated the state-of-record references from R2→R3 across `README.md`, `program.md` §1, `docs/SUBNET_SPEC.md`, `docs/tuner.md`, `docs/PARAMETER_TUNING_PLAN.md`, `docs/sentinel.md`, and `docs/EMULATOR_SPEC.md` §9 (surrogate quality `R^2` `0.93`→`0.96`, Pareto front `21`→`26`, hypervolume `0.0161`→`0.0189`)
- `docs/EMULATOR_SPEC.md` (the agent-orchestrated emulator spec) is added to version control as part of this change

## 2026-06-22 - Reproducible code submission for researcher miners

- researcher miners now submit the **source code that produced/serves their model** alongside the serialized artifact: validators re-execute the code in an isolated sandbox and confirm it reproduces the artifact's predictions before scoring, making submissions auditable and ungameable by opaque/hard-coded/tampered artifacts
- added `insignia/code_submission.py`: deterministic `tar.gz` bundle packaging (`build_code_bundle`/`build_code_bundle_from_dir`) with a per-file manifest; `CodeBundleVerifier` (hash + manifest + entrypoint checks, extraction safety against traversal/symlinks/zip-bombs, static scan rejecting sandbox-escaping source); `SandboxRunner` (subprocess isolation via POSIX rlimits, wall-clock budget, scrubbed env, and best-effort network-namespace drop with `unshare -n`); `ReproducibilityChecker` (re-runs the entrypoint and compares predictions); and `CodeFingerprinter` (normalized-source plagiarism detection)
- `protocol.py`: `ModelSubmission` and `PairEvaluationRequest` carry `code_bundle`, `code_bundle_hash`, `code_entrypoint`, `code_manifest`, and `code_signature`, plus `code_verified`/`code_reproducible`/`reproducibility_score`/`code_rejection_reason` response fields
- `neurons/researcher_miner.py`: `ModelTrainer.build_code_bundle()` packages an `inference.py` entrypoint + model + training source; `train_and_submit` attaches the bundle to every submission (`input.json` → `result.json` I/O convention)
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
