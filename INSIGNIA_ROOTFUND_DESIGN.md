# Insignia, Root-Basket Fund Structure — Design Spec (v3)

**Status:** v3 draft · 2026-08-07 · supersedes the v2 (full-separation) draft, which is retained below as a compared alternative · companion to [INSIGNIA_SYSTEM_EQUATIONS.md](INSIGNIA_SYSTEM_EQUATIONS.md) (v1 model) and [docs/SPEC.md](docs/SPEC.md) §0.16 (Root Reborn mechanics)
**Chain context:** Root Reborn, runtime v441, mainnet 2026-08-03 (subtensor PR #2968; guide `docs/guides/root-reborn.mdx` @ `c02a376`). **Every parameter below is root-mutable, read from chain at runtime, never hardcode.**
**Presentation:** chart tokens per [Insignia-Deck-Design-System.md](Insignia-Deck-Design-System.md); interactive configs target the engine in `Insignia-Website/charts.js`.

> *"6.6% becomes the starting point, not the story, and we expect the effective yield distributed to TAO holders to rise with allocation skill."* — v441 release notes

**The one-line thesis.** LP capital stakes TAO to the Insignia root validator, whose basket concentrates in Insignia alpha. The validator does not compete on rotation skill — it competes by **making its home subnet perform**, which is the desk's actual job. Trading performance transmits to LP yield through the basket's alpha exposure; LP stake transmits to alpha demand through dividend redeployment. Miner sell is countered by **product demand — a dividend-funded structural bid that scales with delegated stake** — not by treasury buyback.

Verified mechanics this rests on (release notes + root-reborn.mdx): dividends are **pro-rata to delegated stake**; unclaimed entitlements are **real stake that keep compounding** with the validator's own dividends; delegation is **frictionless at face value** (any staker, any validator, `btcli root show` exposes NAV, basket, lifetime return); weights are normalized over **≥ 8 positive destinations with no documented per-destination cap**; all valuation is at **realizable quote** at current pool depth.

---

## 0 · The three designs

| | **v1 — same-token wrapper** | **v2 — full separation** | **v3 — root-basket coupling** |
|---|---|---|---|
| LP position | locked alpha (12-mo perpetual → 60-d decay) | fund equity only | TAO delegated to the Insignia root validator (+ fund equity for USD institutions) |
| LP principal risk | alpha beta + pool depth | none (off-chain) | **none on-chain — plain TAO, moves at face value, no lock** |
| LP link to desk P&L | full, reflexive: $(1+y_\alpha)(1+g_p)-1$ | full, linear: $R-\text{fees}$ | **on yield + compounding accrual stack**: $(1-t)\,y_{\text{root}}(1+w\,g_\alpha(R))$, unclaimed accrual compounds in alpha |
| What counters miner sell | LP conversion buys + revenue buy-flow | treasury buyback $\varphi_c$ (**the v2 flaw: circular, fund-funded**) | **dividend bid $w \cdot 983 \cdot K/\tau_{\text{root}}$ — perpetual, stake-scaled, protocol-executed** + band buyback as discipline only |
| Miner hold case | implicit shared boat | buyback promise | higher per-unit yield + structural bid that *grows with product demand* + escrow float shrink |
| Conviction from LPs | instant (owner-hotkey locks) | none | none (root stake ≠ subnet lock) |
| Alpha's role | LP wrapper + miner incentive | miner incentive + revenue share | miner incentive + **the performance conduit** between desk and root stakers |
| Growth engine | OTC pipeline | IR to institutions | **the public scoreboard**: outperform → attract root stake → bigger bid → outperform |

Everything in v1 §§1–3, 6–8 (emission share, EMA, issuance, pool mechanics, move/hold costs) is unchanged and still governs.

---

## 0.5 · The dilution objection, re-examined (v1 rehabilitated)

The advisor objection (McGrath, 2026-08) — LPs diluted by paying signal providers in the fund's own token — **conflates issuance with dilution**. The v1 equations already contained the rebuttal; it was lost in presentation:

1. **Only unstaked positions eat the full hurdle.** The dilution hurdle $h = 7{,}200 \times 365/T$ (≈ 21.9%/yr at 12M) binds on *non-staking* holders. LP alpha is contractually staked (SPEC §4 invariant).
2. **A staked LP recaptures the staker tranche**: $0.41(1-\text{rp})$ of issuance accrues to staked positions pro-rata.
3. **The owner cut is not leakage.** The 18% accrues to the treasury, which backs $\text{NAV}_\alpha$ — reinvested, it returns to every alpha holder including LPs. With rp = 0.155: $0.18 + 0.41(1-\text{rp}) = 52.7\%$ of issuance flows straight back to the wrapper's staker + treasury side, before counting unsold miner alpha (much of which is itself staked by miners-as-co-investors).
4. **The residual is the v1 §9 leakage term, nothing more:**

$$\text{drag}_{\text{staked LP}} \;=\; \frac{7{,}200 \times 365}{S}\Big[\,0.41\,\text{rp} + 0.41\,\sigma\,\Big]$$

| $\sigma$ (at $S=12$M, rp = 0.155) | 0% | 30% | 65% | 100% |
|---|---|---|---|---|
| Staked LP drag | 1.4 pp | 4.1 pp | 7.2 pp | 10.4 pp |
| Unstaked holder | 21.9 pp | 21.9 pp | 21.9 pp | 21.9 pp |

**Verdict: the dilution concern is invalid as stated; the valid residuals are** (a) miner sell-through $\sigma$ — which is compensation expense leaving the wrapper, the product's COGS, bounded and reportable — and (b) the root slice rp, now deferred escrow flow (SPEC §0.16). What v1 *actually* couldn't answer institutionally: **principal alpha-beta, the 12-month lock + decay exit, and same-token optics.** Those, not dilution, are what v3 fixes — which is why v3 keeps the wrapper math and relocates principal, rather than severing the token from performance as v2 did.

---

## 1 · The structure: three audiences, three assets, one conduit

1. **Fund equity — USD institutions.** Shares in desk P&L, standard fees, Cayman + Delaware feeder, listing for a balance-sheet price. Unchanged from v2. The pure-desk-exposure product for allocators who will never touch a wallet.
2. **Root sleeve — TAO-native LPs.** Delegate TAO to the Insignia root validator. Principal stays plain TAO (no fees/slippage/MEV on entry/exit, no unbonding, no lock — switching validators moves at face value). Yield = the validator's dividend stream, deployed into a basket concentrated in Insignia alpha, compounding untouched until claimed. **Self-custodied and permissionless** — no OTC desk, no lock manager, no proxy toggles for this cohort.
3. **Alpha — miners.** The 41% tranche, plus the conduit role: alpha price is the transmission medium between desk performance and root-sleeve yield (§2). NAV band unchanged as *discipline* (buy < 0.9×NAV from $\varphi_c$ of P&L, never above 1.1×) — demoted from primary sell-counter to floor-setter.
4. **The validator — the coupling device.** Burn-registered seat, take $t$ (default 18%) on staker yields = a fund revenue line that replaces management fees for the sleeve. Basket: $w_{\text{ins}}$ concentrated in Insignia (normalized weights, ≥ 8 positive destinations → 7 dust slots + uid 0 as the stability slot; **no documented cap — testnet-verify the effective max, assume $w_{\text{ins}} \approx 0.9$ herein**).

## 2 · The transmission chain (the core mechanism)

**Downstream — performance to LP yield.** Desk P&L → band buy-flow (+ organic demand) → $g_\alpha$ → basket NAV at realizable quote → the validator's realized TAO yield:

$$y_{\text{eff}} \;=\; (1-t)\; y_{\text{root}} \left(1 + w_{\text{ins}}\, g_\alpha + \textstyle\sum_{j \neq \text{ins}} w_j g_j \right), \qquad y_{\text{root}} = \frac{983 \times 365}{\tau_{\text{root}}} \approx 6.7\%$$

| $g_\alpha$ (at $w_{\text{ins}}=0.9$, $t=0.18$) | −30% | 0 | +50% | +100% | +150% |
|---|---|---|---|---|---|
| $y_{\text{eff}}$ | 4.0% | 5.5% | 7.9% | 10.4% | 12.9% |

The release's own framing — 6.6% is the floor, ~18% the skilled-allocation scenario — is this table with compounding. **The accrual stack is the real performance position:** unclaimed entitlement is real stake, compounding with the validator's dividends and earning every epoch. A sleeve LP who never claims is running an automatic, fee-less DCA out of protocol yield into Insignia alpha at realizable NAV — desk performance compounds the *stack*, while principal stays TAO. Claiming any time realizes pro-rata to root stake or free balance; nothing expires.

**Upstream — LP stake to alpha demand.** Dividends are pro-rata to delegated stake $K_v$, redeployed per weights each epoch:

$$F_{\text{ins}} \;=\; w_{\text{ins}} \cdot 983 \cdot \frac{K_v}{\tau_{\text{root}}} \ \ \tau/\text{day} \;\; + \;\; \underbrace{\textstyle\sum_{u \neq v} w^{(u)}_{\text{ins}} \cdot 983 \cdot \frac{K_u}{\tau_{\text{root}}}}_{\text{external validators (§3)}}$$

Against the maintenance cost $7{,}200\,\sigma p$ (93.6 τ/day at $\sigma=0.65$, $p=0.02$), the fund's own validator at $w_{\text{ins}}=0.9$:

| Sleeve $K_v$ | 100k τ | 250k τ | 500k τ | 1M τ |
|---|---|---|---|---|
| $F_{\text{ins}}$ | 16.5 τ/day | 41.1 τ/day | 82.3 τ/day | 164.6 τ/day |
| % of maintenance | 18% | 44% | **88%** | 176% |

**This is the answer to the v2 flaw.** A 500k τ sleeve funds ~88% of the miner-sell counter out of *protocol dividends directed by product demand* — perpetual, compounding (escrowed alpha itself earns and stays), executed by the chain, and scaling with every new τ delegated. The buyback $\varphi_c$ drops to a discount-gated discipline, not the load-bearing bid.

**The closed loop.** Perform → scoreboard rank ("who made their stakers the most TAO", NAV and lifetime return queryable per validator) → stake inflow (LPs *and* mercenary delegators chasing the ranking, on whom the fund earns take $t$) → dividends up → $F_{\text{ins}}$ up → depth and EMA up → perform. The flywheel of v1, rebuilt with TAO principal at the rim instead of locked alpha.

## 3 · Why this beats rotation — and recruits the rotators

Every other curator's "allocation skill" is picking subnets that will perform. Insignia's validator holds an edge no rotator can copy: **its curator operates the subnet and the desk that drives the subnet's price.** It does not forecast the home slot — it *produces* it. When the scoreboard shows it, the rational move for competing validators is to add Insignia weight to their own baskets — every follower converts their stakers' dividends into additional $F_{\text{ins}}$ the fund neither funds nor controls. The scoreboard is simultaneously the growth engine, the distribution channel, and the recruiting mechanism for external bid.

The same lever cuts backwards: a desk drawdown ranks the validator down and rotates stake out, shrinking the bid exactly when the pool is weak (R15, sharpened — see falsifiers).

## 4 · Miner hold case, v3

$$\text{hold}_{\text{miner}} \;=\; \underbrace{y_\alpha' = \frac{7{,}200 \times 0.41(1-\text{rp}) \times 365}{S - L + E}}_{\text{LP alpha out of the denominator: } 7.6\% \to \sim 11\% \text{ at ref}} \;+\; \underbrace{F_{\text{ins}}/p \ \ \alpha/\text{day}}_{\text{stake-scaled structural bid}} \;+\; \underbrace{\varphi_c \text{ band floor}}_{\text{discipline, not engine}}$$

The bid the miner sells into is no longer the fund's own P&L recycled (v2's circularity) — it is dividend flow that **exists because outsiders want the product** and grows when the subnet performs. Escrow accumulation shrinks float: at $F_{\text{ins}} = 82$ τ/day, $p = 0.02$, claim rate $c = 0.5$/yr, steady-state escrow $E^{*} = F_{\text{ins}}/(pc) \approx 3.0$M α — one of the largest holders on the subnet, conviction-inert, price-insensitive, and never selling except pro-rata on staker claims. Non-financial retention (deployment pipeline, token-gated API) unchanged.

## 5 · Escrow dynamics and the overhang

$$\frac{dE}{dt} = \frac{F_{\text{ins}}}{p} - c\,E \qquad E^{*} = \frac{F_{\text{ins}}}{p\,c}$$

Escrow positions are real stake under a keyless pallet sub-account: they earn every epoch, count in the staker-yield denominator and in **SubnetAlphaOut**, and can never sign `lock_stake` — zero conviction, permanently. v3's concentrated basket makes $E$ larger than v2 projected, so both effects scale up: the king-activation denominator inflates faster (attacker's bar rises), and the **claim-flow overhang** grows (R16: claims sell pro-rata across holdings, clustering in TAO drawdowns, exactly when sleeve rotation also bites). Factsheet must report $E$, per-validator weights toward Insignia, and trailing claim flow (docs/SPEC.md §8).

## 6 · Pool flow accounting

$$B_{\text{v3}} \;=\; \underbrace{w_{\text{ins}} \cdot 983 \cdot K_v/\tau_{\text{root}}}_{\text{own-validator dividend bid}} + \underbrace{\bar{w}^{\text{ext}}_{\text{ins}} \cdot 983 \cdot K_{\text{ext}}/\tau_{\text{root}}}_{\text{follower flow}} + \underbrace{\frac{\varphi_c R\,\text{AUM}}{\tau_{\text{usd}} \cdot 365}\bigg|_{\delta < -0.1}}_{\text{band buyback (gated)}}$$

Break-even sell-through $\sigma^{*} = B_{\text{v3}}/(7{,}200\,p)$: at $K_v = 500$k, $w=0.9$, no followers, no buyback → $\sigma^{*} = 0.57$; a 500k sleeve alone absorbs a 57% sell-through with zero treasury spend. Adding $\varphi_c = 0.3$ at $R = 20\%$, AUM \$50M: $\sigma^{*} = 0.87$. v1's φ ≈ 0.9 revenue routing is retired; revenue above $\varphi_c$ stays in the fund where the equity LPs expect it. **Epoch timing:** dividend bids land at epoch boundaries — treasury TWAP stays randomized *away* from them (SPEC §0.5 rule 5) and the desk never front-runs its own validator's deployments (counsel item, §10).

## 7 · Conviction defense (unchanged posture, bigger moat)

Root stake carries no subnet conviction, so v3's defense = owner-cut auto-lock (1,296 α/day, instant on owner hotkey) + treasury perpetual locks, same as v2: ~15 months from a 2.6M-α base for auto-lock alone to reach the 10% king line, 18% asymptote. The larger v3 escrow raises SubnetAlphaOut faster — the attacker's absolute conviction requirement grows with every dividend epoch while escrow itself can never challenge. LP-miner delegation (SPEC §10.6) remains the supplementary lock source if early-warning ratios lag.

## 8 · LP products, honestly compared

$$R^{\text{equity}} = R_{\text{trading}} - \text{fees} \qquad\qquad R^{\text{sleeve}} = (1-t)\,y_{\text{root}}\big(1 + w_{\text{ins}}\,g_\alpha(R)\big) \;\text{on yield, principal in τ}$$

The sleeve is **not** levered desk exposure — first-year yield spans ~4–13% (§2) regardless of how well the desk trades, because principal never enters the basket. Its honest pitch: TAO-denominated capital preservation + performance-linked yield + a compounding alpha stack, claimable any time, no lock. Institutions wanting desk returns buy equity; TAO holders wanting principal safety with upside participation delegate to the sleeve; nobody is sold reflexive exposure without choosing it (the v1 wrapper remains available to a crypto-native cohort that wants exactly that — §0.5 shows its drag is 1.4–10.4 pp, not the headline hurdle).

---

## 9 · Compare and contrast, revised

**v3 vs v1 (the first design spec):**
- *Keeps:* the flywheel (performance → price → yield → demand → price), the NAV band, the retention economics, the miner-alignment thesis. §0.5 shows v1's dilution objection was answered by its own equations — v1 is not abandoned for being wrong about dilution.
- *Fixes:* LP principal no longer carries alpha beta or pool-depth exit risk (plain TAO, face-value moves, vs 12-mo lock + 60-d decay + staged exits); the LP asset is no longer the same unit miners sell (optics + Howey surface both improve); LP onboarding needs no OTC/custody/lock tooling.
- *Loses:* LP conviction mass (instant owner-hotkey locks → none; §7 carries the defense) and the LPs' full-principal exposure to the flywheel's upside — the sleeve caps performance participation at the yield + accrual stack.

**v3 vs v2 (the separation draft):**
- *Keeps:* the three-audience separation, fund equity for institutions, realizable-NAV discipline, the escrow/conviction analysis.
- *Fixes the central flaw:* v2 countered miner sell with $\varphi_c$ buyback — fund money recycled, capped by the fund's own P&L, and circular ("we pay miners in a token we then buy back"). v3's counter is dividend flow that scales with **outside demand for the product** (delegated stake), is executed by the protocol at epoch cadence, compounds in escrow, and recruits follower validators. v2 also severed every LP from on-chain performance; v3 restores the link through the basket without re-exposing principal.
- *Inherits and sharpens:* R15/R16 — scoreboard-driven stake rotation and a larger claim overhang are now first-order risks, priced in the falsifiers.

**Cost accounting across all three** (pool depth and conviction, per the standing question):
- Pool bid at reference params — v1: ~129 τ/day (φ≈0.9 routing) + one-off conversion buys; v2: ~43 τ/day + basket dust; **v3: ~82 τ/day per 500k τ sleeve + followers + gated buyback, growing with AUM-of-sleeve rather than spending desk P&L.**
- Conviction — v1: instant and massive; v2/v3: auto-lock path (~15 mo to threshold), v3 with the fastest-inflating attacker denominator.
- LP yield benefit — v1: full reflexive wrapper return (drag 1.4–10.4 pp); v2: clean $R-\text{fees}$, zero on-chain; v3: 5.5% base scaling to ~13%+ with performance, principal untouched.
- Miner yield benefit — v1: baseline (LP alpha dilutes the staked base); v2/v3: ~1.2–1.5× per-unit staking yield, v3 adding the stake-scaled bid on top.

---

## 10 · Interactive charts (website) — v3 set

Engine, slider wiring, `Chart` class unchanged (`charts.js`). Model functions and `DEFAULTS`:

```js
const ROOT_DIV_DAY = 983;                          // τ/day, read live
const rootBaseYield = () => ROOT_DIV_DAY * 365 / M.ROOT_TAO;            // ≈ 6.7%
const divBid   = p => p.wIns * ROOT_DIV_DAY * p.rootSleeve / M.ROOT_TAO; // τ/day
const extBid   = p => p.wExt * ROOT_DIV_DAY;                             // τ/day (followers, stake-weighted)
const bandBid  = p => p.delta < -0.1 ? p.phiC * p.tradingR * p.aum / p.taoUsd / 365 : 0;
const escrowSS = p => (divBid(p) + extBid(p)) / (p.alphaPrice * p.claimRate);
const minerYieldV3 = p => yieldNumerator(p.age) / (p.staked - p.lpAlpha + escrowSS(p));
const maintFlow = p => M.ALPHA_OUT * p.sigma * p.alphaPrice;
const sleeveYield = (g, p) => (1 - p.take) * rootBaseYield() * (1 + p.wIns * g);
const stakedDrag  = p => (M.ALPHA_OUT * 365 / p.staked) * M.VALSTAKE *
                         (rootProp(issuanceAt(p.age)) + p.sigma);
const lpEquity = (R, p) => (1 - p.perfFee) * R - p.mgmtFee;

// DEFAULTS +=
//   rootSleeve: 5e5, wIns: 0.90, wExt: 0.00, take: 0.18, claimRate: 0.5,
//   lpAlpha: 4e6, phiC: 0.30, tradingR: 0.20, delta: -0.15,
//   perfFee: 0.20, mgmtFee: 0.02, alphaOut0: 2.6e6
```

**`chart-root-yield` — "6.6% is the starting point"** *(revised, now the flagship)*
x: $g_\alpha$ ∈ [−50%, +150%]. Series: `sleeveYield(g)` per $w_{\text{ins}}$ ∈ {0.3, 0.6, 0.9}; dashed network base at $y_{\text{root}}$ and net-of-take floor. Marker at $g=0$: "principal unaffected — plain TAO". Sliders: `wIns`, `take`. Message: effective yield rises with the home subnet's performance; the release's 6.6→18% chart, made Insignia-specific.

**`chart-transmission` — desk return → validator scoreboard** *(new)*
x: $R_{\text{trading}}$ ∈ [0, 40%]. Chain per point: $B = $ `divBid + extBid + bandBid`; $g = $ `priceMultiple(B·365 − sellFlowTao, poolTao)` − 1; y = `sleeveYield(g)`. Dashed: network base 6.7% (scoreboard parity). Marker: $R$ where the validator overtakes the median. Sliders: `rootSleeve`, `wIns`, `phiC`, `sigma`, `aum`, `taoUsd`. Message: the full loop — trading performance becomes scoreboard rank.

**`chart-flow-v3` — who counters the miners** *(replaces flow-v2)*
x: $\sigma$ ∈ [0, 1]. Series: `maintFlow` (need, rising); stacked sources `divBid`, `extBid`, `bandBid` (flat). Marker: $\sigma^{*}$. Sliders: `rootSleeve`, `wIns`, `wExt`, `phiC`, `alphaPrice`. Message: a 500k τ sleeve absorbs σ ≈ 0.57 with zero treasury spend — the v2 buyback circularity, retired visibly.

**`chart-dilution` — is the dilution objection valid?** *(new; answers §0.5)*
x: $\sigma$ ∈ [0, 1]. Series: `stakedDrag` at $S$ ∈ {6M, 12M, 24M} (rising, 1.4–10.4 pp band); dashed flat line: unstaked hurdle `ALPHA_OUT·365/staked`. Sliders: `staked`, `age`. Message: staking + reinvested owner cut caps LP drag at the leakage term; the debate is σ, not issuance.

**`chart-miner-v3` — miner yield, three designs** *(revised)*
x: diverted LP alpha $L$ ∈ [0, 8M]. Series: v1 flat `yieldNumerator/staked`; v3 rising `minerYieldV3`. Sliders: `staked`, `age`, `rootSleeve`, `wIns`, `claimRate`. Message: the miner's mechanical gain, now escrow-adjusted with the stake-scaled bid.

**`chart-conviction` — king defense** *(retained from v2)*
As before: auto-lock-only path vs v1 instant-lock path vs escrow-inflated denominator (dashed), 10% marker. v3 note: the dashed denominator rises fastest here.

**`chart-products` — three LP products, one desk** *(replaces chart-revenue-v2)*
x: $R_{\text{trading}}$ ∈ [0, 40%]. Series: v1 wrapper `lpReturn(R,p).total`; equity `lpEquity(R)`; sleeve via `chart-transmission` chain. Markers: wrapper/equity crossover; sleeve/base crossover. Message: reflexive, linear, and principal-protected exposure priced side by side — each audience picks its risk.

**`chart-vesting`** — retained, rescoped to treasury/desk locks and the optional v1-wrapper cohort. Caption notes the root sleeve has **no lock at all**.

**Presentation tokens.** Website keeps the `CL` palette + crosshair engine. Deck exports per the design system: accent `#856ED1` emphasized series, other series white at reduced opacity, 11% glass panels radius 48, Arial scale (axis 90/72), one chart per panel, one emphasized series per slide.

---

## Reading order for an investor

1. §1 — three assets: equity buys the desk, the sleeve keeps principal in TAO with performance-linked yield, alpha pays the miners
2. §2 — the yield table and the accrual stack: how desk performance reaches you without touching your principal
3. §6 — the bid that absorbs miner supply is dividend flow scaled by demand for the product, not our own P&L recycled
4. §0.5 — why "LPs get diluted" was the wrong objection, with the drag actually priced
5. §3 + falsifiers — the scoreboard cuts both ways, and here is what we watch

## What would falsify this design

- **Weight concentration blocked**: testnet shows an effective per-destination cap ≪ 0.9 → transmission and bid tables re-derive; if $w_{\text{ins}}$ caps near emission weight, v3 degenerates to v2 → revisit
- **Transmission too weak**: sustained $R > 20\%$ for two quarters with $y_{\text{eff}}$ statistically indistinguishable from the network base → the $g_\alpha$ link (band flow + organic demand) is undersized; sleeve pitch fails
- **Scoreboard rotation dominates**: sleeve stake outflow elasticity to a single bad quarter exceeds modeled R15 paths → concentrated-basket volatility is repricing the product; consider raising the uid-0 stability slot
- **σ does not fall** within two quarters of the higher per-unit yield + visible structural bid → miner-retention mechanism weaker than modeled
- **Escrow overhang realizes**: claim clusters (R16) drive drawdown-correlated sell flow exceeding the reflexivity engine's 95th percentile → cap effective $w_{\text{ins}}$, raise reserve coverage
- Root changes take, seat count, ≥ 8-destination rule, dividend formula, or claim mechanics → re-derive §2–§6 from chain

## Open items (blocking adoption)

1. **Effective max $w_{\text{ins}}$** on testnet: normalization behaviour with 7 dust slots + uid 0; whether dust weights round to zero (breaking the ≥ 8 rule) — determines every table above.
2. **Counsel:** (a) marketing the sleeve as performance-linked yield ties returns to managerial efforts — Howey analysis for the *delegation relationship*, not just alpha; (b) self-dealing and MNPI posture for a validator whose curator controls the subnet and the buy program — public-weights disclosure regime, and a standing rule that the desk never trades ahead of its own validator's epoch deployments.
3. **Re-run `risk/reflexivity.py`** with the v3 baseline: sleeve-rotation shock (R15, elasticity-parameterized), claim-cluster overhang at v3 escrow scale (R16), no LP lock cohorts (R8/R9 retire), band buyback gated by δ.
4. **Take-rate policy** ($t$): default 18% vs custom — it is the sleeve's fee schedule and the fund's revenue line; set it deliberately and disclose it in the factsheet.
5. **Sleeve growth targets**: $K_v$ tiers at which $\sigma^{*}$ crosses 0.65 and 1.0 (≈ 575k τ and 1.0M τ at reference $p$) — these are now the fundraising milestones that replace v1's Phase-2 conversion.
