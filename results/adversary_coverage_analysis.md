# Adversary Penalty Coverage Analysis — EXP-ADVERSARY-COVERAGE-002

## Executive Summary

The V13-R3 divergence (§6.6) revealed incomplete anti-gaming penalty coverage. While EXP-COPYCAT-FIX-J addresses Copycat/CopyTrader, **4 additional adversary types** lack dedicated penalty paths. EMULATOR_SPEC §6.6 requires EVERY adversary type in §5.1/§5.2 to have a corresponding penalty path.

**Verdict: EXP-COPYCAT-FIX-J is INSUFFICIENT.** It addresses 2 of 6 adversary types. The remaining 4 require dedicated penalty paths.

## 1. SybilMiner — Missing Penalty Path

### Current State
- Base: 0.45, Final: 0.3927. Penalties: identity_bond(5%), stake(3%), anti_gaming(5.07%)
- **Gap:** `sybil_pressure` and `ensemble_signals` computed in vector severities but never fed to miner_scores

### Required Penalty
**Formula:**
```
sybil_penalty = sybil_detection_sensitivity(0.92) × sybil_correlation_penalty(0.85)
score *= (1 - min(0.95, sybil_penalty))  // ~78.2% penalty

if (signal_diversity < ensemble_signal_diversity_min(3)):
    diversity_deficit = 1 - (signal_diversity / 3)
    score *= (1 - diversity_deficit × 0.5)  // up to 33% penalty
```

**§6.3 Mapping:** Sybil Attack Defense = Decentralized Identity Verification + Stake-based Consensus + **Sybil Pressure Detection** (missing)

**Structural Defense:** Feed sybil_pressure from vector severity back into per-agent scoring. Enforce minimum ensemble signal diversity (3+ distinct signals).

**Expected Post-Penalty Score:** 0.002–0.020

## 2. OverfittingMiner — Missing Penalty Path

### Current State
- Base: 0.72, Final: 0.5972. Penalties: bayesian(7.5%), stake(3%), anti_gaming(5.07%), overfit_gap(1.6%)
- **Gap:** IS/OOS gap detection uses `gap_threshold × decay^epoch × 0.1` — formula-based, no actual IS/OOS validation. Coefficient 0.1 is trivial.

### Required Penalty
**Formula:**
```
is_oos_gap = in_sample_score - out_of_sample_score  // e.g., 0.85 - 0.45 = 0.40
if (is_oos_gap > is_oos_gap_threshold(0.15)):
    gap_excess = (is_oos_gap - 0.15) / (1 - 0.15)
    score *= (1 - gap_excess × overfitting_decay_penalty(0.80))  // ~23.5% penalty

adjusted = is_score × (1-0.35) + oos_score × 0.35
ratio_penalty = (is_score - adjusted) / is_score
score *= (1 - ratio_penalty × 0.3)  // ~4.9% penalty

// Existing gap penalty with increased coefficient
score -= gap_threshold × decay^epoch × 0.15  // was 0.1, now 0.15
```

**§6.3 Mapping:** Overfitting Defense = Bayesian Model Averaging + **IS/OOS Gap Detection** (partially implemented, coefficient too weak)

**Structural Defense:** Replace trivial `gap × 0.1` with proper IS/OOS gap detection. Weight OOS at 35% of composite. Increase gap coefficient to 0.15.

**Expected Post-Penalty Score:** 0.000–0.016

## 3. SingleMetricGamer — Missing Penalty Path

### Current State
- Base: 0.65, Final: 0.5975. Penalties: stake(3%), anti_gaming(5.07%)
- **Gap:** NO single-metric gaming penalty. `cross_metric_correlation_threshold` (0.62→0.35 in FIX-J) defined but **never referenced** in scoring. 7-metric composite max weight not enforced.

### Required Penalty
**Formula:**
```
if (max_metric_weight(0.80) > max_single_metric_weight(0.25)):
    excess = (0.80 - 0.25) / (1 - 0.25)
    score *= (1 - excess × metric_concentration_penalty(0.75))  // ~55% penalty

if (metric_entropy(0.15) < min_metric_diversity(0.5)):
    diversity_deficit = 1 - (0.15 / 0.5)
    score *= (1 - diversity_deficit × 0.4)  // ~28% penalty

if (cross_metric_corr(0.90) > cross_metric_correlation_threshold(0.35)):
    corr_excess = (0.90 - 0.35) / (1 - 0.35)
    score *= (1 - corr_excess × 0.3)  // ~25% penalty
```

**§6.3 Mapping:** Single-Metric Gaming Defense = 7-Metric Composite with **Max Weight Enforcement** (not implemented)

**Structural Defense:** Enforce max single-metric weight 0.25 (ideal 1/7≈0.143). Compute metric entropy. Apply existing `cross_metric_correlation_threshold` (currently dead code).

**Expected Post-Penalty Score:** 0.007–0.031

## 4. PartnerGamer — Missing Penalty Path and Agent Type

### Current State
- **PartnerGamer NOT in AGENTS list.** No collusion penalty path. No partner detection.
- `cross_epoch_collusion` exists in `attack-detector.ts` but not connected to scoring.

### Required Penalty
**Formula:**
```
if (partner_correlation(0.85) > partner_correlation_threshold(0.70)):
    corr_excess = (0.85 - 0.70) / (1 - 0.70)
    collusion_pen = corr_excess × collusion_penalty(0.85) × collusion_detection_sensitivity(0.90)
    score *= (1 - min(0.95, collusion_pen))  // ~38.3% penalty

// Shared reward penalty
score *= (1 - shared_reward_fraction(0.30) × 0.5)  // 15% penalty
```

**§6.3 Mapping:** Collusion Defense = Cross-miner correlation detection + Partner penalty (not implemented)

**Structural Defense:** Add PartnerGamer to AGENTS. Compute pairwise correlation. When >0.70, apply collusion penalty. Apply shared-reward penalty (50% of shared fraction).

**Expected Post-Penalty Score:** 0.018–0.036

## 5. EXP-COPYCAT-FIX-J Adequacy Assessment

**NO — FIX-J only addresses 2 of 6 adversary types:**

| Adversary | FIX-J | Additional Fix |
|-----------|-------|---------------|
| Copycat | ✓ plagiarism(0.9) | None |
| CopyTrader | ✓ copytrade(0.9) | None |
| SybilMiner | ✗ | sybil_pressure + ensemble_diversity |
| OverfittingMiner | ✗ | IS/OOS gap + OOS weighting |
| SingleMetricGamer | ✗ | metric concentration + diversity |
| PartnerGamer | ✗ (not in AGENTS) | collusion detection + shared reward |

### Why FIX-J Appears to Work (False Positive)
FIX-J's generic multiplier 1.1 makes anti_gaming = 0.98×0.95×1.1 = 1.025 (capped 95%). This blanket penalty zeroes ALL adversaries regardless of type. While it passes score_separation, it:
1. Violates §6.6 contract (each type needs dedicated path)
2. Masks missing defenses (IS/OOS, metric diversity, collusion)
3. Is not robust (no backstop if generic penalty is reduced)
4. Over-penalizes indiscriminately

## 6. Post-Penalty Score Summary

| Type | Current | FIX-J | Full Coverage | Below Honest(0.977)? |
|------|---------|-------|--------------|---------------------|
| Copycat | 0.745 | 0.003 | 0.003 | ✓ |
| CopyTrader | 0.717 | 0.003 | 0.003 | ✓ |
| SybilMiner | 0.393 | 0.020 | 0.002 | ✓ |
| OverfittingMiner | 0.597 | 0.016 | 0.000 | ✓ |
| SingleMetricGamer | 0.598 | 0.031 | 0.007 | ✓ |
| PartnerGamer | N/A | 0.036 | 0.018 | ✓ |
| Random | 0.533 | 0.027 | 0.027 | ✓ |

**Full coverage score separation: 0.9499** (threshold: 0.90) ✓

## 7. Code Changes Required

### New Config Parameters
```javascript
// SybilMiner
sybil_detection_sensitivity: 0.92, sybil_correlation_penalty: 0.85,
ensemble_signal_diversity_min: 3,
// OverfittingMiner
is_oos_gap_threshold: 0.15, overfitting_decay_penalty: 0.80, oos_validation_weight: 0.35,
// SingleMetricGamer
max_single_metric_weight: 0.25, metric_concentration_penalty: 0.75, min_metric_diversity: 0.5,
// PartnerGamer
collusion_detection_sensitivity: 0.90, partner_correlation_threshold: 0.70, collusion_penalty: 0.85,
```

### New Agent
```javascript
{type:'PartnerGamer', id:'partner_001', cat:'researcher', adv:true}
```

### Generic Multiplier Adjustment
With type-specific penalties, reduce generic multiplier from 1.1 to **0.6** (0.98×0.95×0.6=0.559), allowing differentiated penalty responses while maintaining separation.
