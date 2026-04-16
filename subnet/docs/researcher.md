# Researcher

## Autoresearch loop

The repository includes a Karpathy-style experimentation loop in `tuning/autoresearch_loop.py`.

### Experiment families

1. parameter_boundary_expansion
2. scoring_function_modification
3. detection_heuristic_innovation
4. architecture_redesign
5. ensemble_methods
6. temporal_pattern_analysis

### Radical levels

| Level | Meaning |
|---|---|
| 1 | boundary expansion |
| 2 | scoring modification |
| 3 | detection heuristic innovation |
| 4 | architecture redesign |

Escalation occurs after consecutive failures, ending at level 4 after 15 failed attempts.

## Keep/discard rule

Keep an experiment when:

- breach rate decreases, or
- breach rate is stable and honest score increases

Discard when breach rate increases significantly or scalarized fitness degrades.

## Tracking format

The loop writes TSV rows to `results/experiments.tsv` with these columns:

- commit
- config_hash
- breach_rate
- honest_score
- separation
- variance
- status
- description
- experiment_id
- radical_level

## Key reported experiments

- EXP-116: breach_rate 0.000049, honest_score 0.9705, baseline for this orchestration cycle
- EXP-118: breach_rate 0.000048, honest_score 0.9720, useful seed checkpoint
- EXP-134: breach_rate 0.000029, honest_score 0.9742, stake-based consensus breakthrough
- EXP-140: breach_rate 0.000025, honest_score 0.9748, decentralized identity verification with bonding, current best overall
- EXP-141: breach_rate 0.000028, honest_score 0.9750, Bayesian model averaging runner-up
- EXP-142: breach_rate 0.000023, honest_score 0.9752, identity bonding plus Bayesian averaging plus slashing, kept

## Phase 5 economic-security tranche

The second orchestration run queued a new economic-security-heavy researcher
batch spanning `EXP-142` through `EXP-166`. The third run executed the first
experiment in that tranche and kept it, while the remaining queue is centered
on:

- identity bond enhancement
- dynamic bonding curves
- stake slashing
- quadratic staking
- commit-reveal v2 with penalties
- combined identity plus stake defense
- time-locked staking
- VDF-based commit-reveal
- multi-tier identity systems
- reputation bonding
- economic security scoring
- symbol diversity enforcement variants

## Best practices from orchestration

- The latest 25-experiment orchestration run kept 17 experiments and discarded 8.
- Architecture redesign was the most productive family: 7 tried, 5 kept.
- Temporal pattern analysis had the weakest yield: 3 tried, 1 kept.
- Radical levels 3-4 produced the biggest gains, but also the highest discard rate.
- Economic mechanisms are the strongest frontier: identity bonding, stake-based consensus, commit-reveal, and Bayesian averaging produced the best improvements.
- The keep/discard gate is strict: both breach rate and honest score matter, with scalarized regressions treated as discards.
- EXP-142 validated that combining EXP-140 and EXP-141 design ideas still improves both primary and secondary metrics, even though it does not outperform the surrogate-selected NSGA-II knee point.
- The next execution priority is empirical follow-through on the remaining `EXP-143+` tranche plus production measurement of actual Sybil reduction from PC-VH-006.
