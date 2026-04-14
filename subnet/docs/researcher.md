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
- breach_rate
- honest_score
- status
- description
- experiment_id
- radical_level
- separation
- variance

## Key reported experiments

- EXP-050: breach_rate 0.00058, honest_score 0.955
- EXP-103: breach_rate 0.000095, honest_score 0.97, ensemble breakthrough
- EXP-116: breach_rate 0.000049, honest_score 0.9705, current best overall

## Best practices from orchestration

- Moderate radical levels 1-2 outperform extreme redesigns.
- Boundary expansion plus sensitivity tuning yields the most reliable gains.
- Ensemble detection and temporal stability remain the most productive research directions.
