# Insignia Fund Structure — LP Wrapper and Root Reborn

**Status:** draft · 2026-08-16 · companion to [INSIGNIA_SYSTEM_EQUATIONS.md](INSIGNIA_SYSTEM_EQUATIONS.md) and [docs/SPEC.md](docs/SPEC.md) §0.5 (wrapper) and §0.16 (Root Reborn mechanics)
**Chain context:** Root Reborn, runtime v441, mainnet 2026-08-03 (subtensor PR #2968; guide `docs/guides/root-reborn.mdx`). **Every parameter below is root-mutable, read from chain at runtime, never hardcode.**
**Presentation:** chart tokens per [Insignia-Deck-Design-System.md](Insignia-Deck-Design-System.md); interactive configs target the engine in `Insignia-Website/charts.js`.

**The one-line thesis.** Alpha is the investment wrapper for the prop-trading desk. LP capital converts into locked, staked Insignia alpha. Trading performance transmits to LP return through alpha price; LP conversion and revenue buy-flow transmit to alpha demand. Root Reborn changes how the root slice of issuance *realizes* (escrow, then claim), not who the LP is and not which asset carries principal.

Verified mechanics this rests on: LP alpha is contractually staked (SPEC §4); return is \((1+y_\alpha)(1+g_p)-1\) ([SYSTEM_EQUATIONS](INSIGNIA_SYSTEM_EQUATIONS.md) §9); owner-hotkey locks grant instant conviction; Root Reborn dividends accrue as real stake under a keyless pallet escrow and pay out only on claim, at realizable quote.

---

## 1 · The structure: one LP asset, one conduit

1. **LP position — locked, staked alpha.** Converted OTC, delivered `move_stake` to the subnet-owner hotkey, 12-month perpetual lock then 60-day exponential decay. Principal is alpha. Root stake is not the LP product.
2. **Alpha — miner incentive and the wrapper.** The 41% miner tranche, plus the LP unit. Alpha price is how desk P&L reaches LPs. The NAV band is the discipline: buy below 0.9× realizable NAV from routed revenue, never above 1.1×.
3. **Root validators — protocol overlay, not LP capital.** Each root validator runs a beta-basket of escrowed subnet alpha (SPEC §0.16). That flow is extra bid or extra claim-sell on *our* pool. Delegated TAO on an Insignia root seat is mercenary / IR capital: its coupon can be redeployed into Insignia, but the principal never enters the wrapper and is not sold as desk exposure.

Everything in [SYSTEM_EQUATIONS](INSIGNIA_SYSTEM_EQUATIONS.md) §§1–3, 6–8 (emission share, EMA, issuance, pool mechanics, move/hold costs) is unchanged and still governs.

---

## 2 · LP return (the transmission)

Desk P&L → band buy-flow (+ conversion demand) → \(g_p\) → LP wealth on the **full principal**:

$$R_{\text{LP}} \;=\; (1+y_\alpha)\,(1+g_p)-1, \qquad y_\alpha \;=\; \frac{7{,}200 \times 0.41 \times (1-\text{rp}) \times 365}{S}$$

\(y_\alpha\) is protocol-fixed in alpha terms (dilution recapture, not profit). \(g_p\) is the price term the desk actually moves. Both halves must appear together in investor materials.

A staked LP recaptures the staker tranche. Equivalently, as drag on the desk's gross return ([SYSTEM_EQUATIONS](INSIGNIA_SYSTEM_EQUATIONS.md) §9):

$$R_{\text{LP}} \;\approx\; R_{\text{trading}} - \underbrace{\frac{7{,}200 \times 365}{S}\Big[0.41\,\text{rp} + 0.41\,\sigma\Big]}_{\text{leakage}}$$

| Miner sell-through \(\sigma\) (at \(S=12\)M) | 0% | 30% | 60% | 100% |
|---|---|---|---|---|
| Drag | 0.7 pp | 3.4 pp | 6.1 pp | 9.6 pp |
| Unstaked hurdle | 21.9 pp | 21.9 pp | 21.9 pp | 21.9 pp |

Only unstaked positions eat the full dilution hurdle \(h = 7{,}200 \times 365/T\). The owner cut (18%) accrues to the treasury and backs \(\text{NAV}_\alpha\). The residuals that actually bind a staked LP are miner sell-through \(\sigma\) (compensation leaving the wrapper) and the root slice \(\text{rp}\) (deferred escrow flow, SPEC §0.16).

---

## 3 · What counters miner sell

$$B \;=\; \underbrace{\text{LP conversion buys}}_{\text{principal hitting the pool}} + \underbrace{\varphi\,R\cdot\text{AUM routed below the band}}_{\text{revenue buy-flow}} + \underbrace{\bar{w}_{\text{ins}}\cdot 983 \cdot K_{\text{ext}}/\tau_{\text{root}}}_{\text{external basket flow, SPEC §0.16}}$$

Conversion is one-shot demand from LP entry (DCA'd). Revenue routing is the standing bid, gated by \(\delta < -0.1\). External (and optional own-validator) basket flow is protocol dividend redeployment — it scales with *root* stake, not with LP notional, and must not be counted as if the LP's principal were buying alpha.

Against maintenance \(7{,}200\,\sigma p\) (93.6 τ/day at \(\sigma=0.65\), \(p=0.02\)), the load-bearing terms are conversion + routed revenue. Basket flow is incremental.

**Epoch timing:** basket origin sells and redeploy buys land at epoch boundaries. Treasury TWAP stays randomized *away* from them (SPEC §0.5 rule 5). The desk never trades ahead of its own validator's deployments if it runs a root seat.

---

## 4 · Root Reborn: realization timing, not a new LP

Root Reborn retired the per-block auto-sell of the root slice. Each root validator's dividends accrue as **real staked alpha under a keyless escrow**. A claim redeems a fund fraction, selling a pro-rata slice of every holding at realizable quote.

Consequences for the wrapper (SPEC §0.16):

- **Still leaked.** Escrow is economically owned by root stakers. Never count it as retained NAV. Model exit as claim flow (no deadline), not a constant per-block drain.
- **Inflates the staked base.** Escrow earns every epoch and counts in SubnetAlphaOut, so per-unit \(y_\alpha\) is reported against \(S + E\).
- **Zero conviction.** The escrow coldkey cannot sign `lock_stake`. Growth raises the king-activation denominator without a challenger.
- **Claim overhang (R16).** Claims cluster in TAO drawdowns, correlated with LP redemption demand (R1). Factsheet reports \(E\), per-validator weights toward Insignia, and trailing claim flow (SPEC §8).

Steady-state escrow from a standing dividend bid \(F\) (τ/day), price \(p\), annual claim rate \(c\):

$$\frac{dE}{dt} = \frac{F}{p} - c\,E \qquad E^{*} = \frac{F}{p\,c}$$

An optional Insignia root validator with weight \(w_{\text{ins}}\) and delegated stake \(K_v\) contributes \(F_{\text{own}} = w_{\text{ins}} \cdot 983 \cdot K_v/\tau_{\text{root}}\) τ/day. That coupon is not LP yield. Root stakers keep TAO principal; only the ~6.7%/yr network coupon (net of take) is deployed into the basket. Do not market that coupon as desk exposure.

---

## 5 · Conviction defense

LP locks to the owner hotkey grant **instant** conviction equal to locked mass ([SYSTEM_EQUATIONS](INSIGNIA_SYSTEM_EQUATIONS.md) §12). That is the king-defense mass. Root stake contributes none. Escrow inflates SubnetAlphaOut, so defensive ratios are recomputed against the escrow-inclusive denominator. LP-miner delegation (SPEC §10.6) remains the supplementary lock source if early-warning ratios lag.

---

## 6 · Interactive charts (website)

Engine, slider wiring, `Chart` class unchanged (`charts.js`). LP math lives in `treasury/emissions.py` and `lockmgr/schedules.py` — charts must not invent a second yield.

```js
const lpAlphaYield = p => yieldNumerator(p.age) / p.staked;          // y_α, recapture
const lpReturn     = (R, p) => (1 + lpAlphaYield(p)) * (1 + priceFromDesk(R, p)) - 1;
const leakageDrag  = p => (M.ALPHA_OUT * 365 / p.staked) *
                          (M.VALSTAKE * rootProp(p.staked) + M.MINER * p.sigma);
const maintFlow    = p => M.ALPHA_OUT * p.sigma * p.alphaPrice;
const convBid      = p => p.conversionTao;                            // one-off, DCA'd
const revBid       = p => p.delta < -0.1 ? p.phi * p.tradingR * p.aum / p.taoUsd / 365 : 0;
const extBid       = p => p.wExt * ROOT_DIV_DAY;                      // other validators
const escrowSS     = p => extBid(p) / (p.alphaPrice * p.claimRate);

// DEFAULTS +=
//   staked: 12e6, lpLocked: 4e6, phi: 0.90, tradingR: 0.20, delta: -0.15,
//   sigma: 0.65, conversionTao: 0, wExt: 0.00, claimRate: 0.5
```

**`chart-lp-return` — the wrapper.** x: \(R_{\text{trading}}\) ∈ [0, 40%]. Series: `lpReturn`. Message: dollar return is staking yield times what the desk earns on price.

**`chart-dilution` — hurdle vs staked drag.** x: \(\sigma\) ∈ [0, 1]. Series: `leakageDrag` at \(S\) ∈ {6M, 12M, 24M}; dashed unstaked hurdle. Message: staking + reinvested owner cut caps LP drag at the leakage term.

**`chart-flow` — who counters the miners.** x: \(\sigma\) ∈ [0, 1]. Series: `maintFlow` (need); stacked `convBid`, `revBid`, `extBid`. Marker: \(\sigma^{*}\) from standing bids.

**`chart-vesting` — LP lock lifecycle.** Owner-hotkey conviction instant; 12-month perpetual then 60-day decay. Caption: the LP position is locked alpha, not root TAO.

**`chart-conviction` — king defense.** Auto-lock + LP owner-hotkey path vs escrow-inflated SubnetAlphaOut (dashed), 10% marker.

**`chart-escrow` — claim overhang.** x: claim rate \(c\); series \(E^{*}\) at observed basket \(F\). Message: escrow is leaked, conviction-inert, and a drawdown-correlated sell stock.

---

## Reading order for an investor

1. §1 — LPs hold locked alpha; root baskets are protocol flow, not the LP
2. §2 — \(R_{\text{LP}} = (1+y_\alpha)(1+g_p)-1\): yield is fixed, dollars come from the desk
3. §3 — conversion + NAV-band buy-flow absorb miner supply; basket flow is extra
4. §4 — Root Reborn defers the root slice into escrow; it is still leakage
5. Falsifiers — what we watch

## What would falsify this design

- Sustained buy-flow with no EMA/share response after two quarters → transmission assumption wrong
- \(\delta\) persistently > +10% while treasury keeps buying → discipline not being followed
- Retention ([SYSTEM_EQUATIONS](INSIGNIA_SYSTEM_EQUATIONS.md) §10) below ~60% → miner alignment failing, LP returns leaking
- Claim clusters (R16) driving drawdown-correlated sell flow past the reflexivity engine's 95th percentile → raise reserve coverage
- Root changes take, dividend formula, claim mechanics, \(w_\tau\), or the share formula → re-derive from chain

## Open items

1. **Counsel** on the wrapper (OTC alpha, lock, Howey surface) — Phase 0 still gates investor-facing features on `LEGAL_SIGNOFF.md`.
2. **Own root seat:** IR and coupon-bid only; never marketed as the LP. If run, disclose public weights and a standing rule that the desk never trades ahead of epoch deployments.
3. **Re-run `risk/reflexivity.py`** on the wrapper baseline (LP lock cohorts on): conversion + revenue routing, plus R15 basket rotation and R16 claim clustering as overlay shocks.
4. **Factsheet:** \(E\), per-validator weights toward Insignia, trailing claim flow (SPEC §8).
