# Insignia, System Equations

**The one-line thesis.** Emission share drives `tao_in` (pool depth, hence *price*), **not** `alpha_out` (the *quantity* paid to stakers). LP yield is protocol-fixed in alpha terms; trading performance acts on the price term. Both halves of that sentence must appear together in investor materials, or the model reads as either a free lunch or a pump.

LP principal is **locked, staked alpha** — \(R_{\text{LP}}=(1+y_\alpha)(1+g_p)-1\). Root Reborn (SPEC §0.16) defers the root slice into basket escrow and changes claim timing; it does not move LP principal into TAO. See [INSIGNIA_ROOTFUND_DESIGN.md](INSIGNIA_ROOTFUND_DESIGN.md).

Verified against `bittensor.com/docs/concepts/emissions` and `/staking-pools`. **Every parameter below is root-mutable, read from chain at runtime, never hardcode.**

---

## 1 · Emission share

$$s_i \;=\; \frac{p_i\,(1-b_i)}{\sum_j p_j\,(1-b_j)}$$

| Symbol | Meaning |
|---|---|
| $p_i$ | `SubnetMovingPrice`, **EMA** of spot alpha price, capped at 1.0 |
| $b_i$ | fraction of last tempo's miner incentive withheld to subnet-owner hotkeys |

Two consequences:

- Share is driven by an **EMA of price**, not by flow. Buy-flow matters only insofar as it moves spot, which moves the EMA.
- $b_i$ is taxed **1:1**. Routing miner incentive to owner hotkeys to boost the owner cut destroys the share the entire thesis rests on. **Design rule: $b_i = 0$.**

## 2 · EMA responsiveness (the built-in delay)

$$\text{ema}_\alpha \;=\; \text{base}_\alpha \cdot \frac{\text{blocks\_since\_start}}{\text{blocks\_since\_start} + 201{,}600}$$

| Age | 7d | 28d | 90d | 180d | 1yr |
|---|---|---|---|---|---|
| Responsiveness | 20% | 50% | 76% | 87% | 93% |

This exists explicitly to blunt launch pumps and coordinated buying. **Budget 3–6 months between sustained buy-flow and emission-share response.** It rewards *time-at-price*, not peak price.

## 3 · Issuance

$$\text{alpha\_out} = 1\ \alpha/\text{block} = 7{,}200\ \alpha/\text{day}\quad\textbf{flat for every subnet}$$

$$\text{tao\_in}_i = s_i \times 3{,}600\ \tau/\text{day},\qquad \text{alpha\_in} = \min\!\left(\frac{\text{tao\_in}}{p},\; 1\ \alpha/\text{block}\right)$$

Split of `alpha_out`: **owner 18% · miners 41% · validators+stakers 41%**.

`tao_in` flows into the **pool reserve**, not to stakers. Injections shift balancer *weights*, so protocol emission is price-neutral.

## 4 · Root proportion (the year-one haircut)

$$\text{rp} \;=\; \frac{\tau_{\text{root}} \cdot w_\tau}{\tau_{\text{root}} \cdot w_\tau + \alpha_{\text{issuance}}}, \qquad \tau_{\text{root}} \approx 5{,}374{,}582,\; w_\tau = 0.18$$

A young subnet has small $\alpha_{\text{issuance}}$, so rp is large and most of the validator half is diverted to **root TAO stakers**:

| Subnet age | 1 mo | 3 mo | 6 mo | 1 yr | 2 yr |
|---|---|---|---|---|---|
| Alpha stakers receive | 12.7% | 23.5% | 29.9% | 34.6% | 37.5% |

Against the 41% everyone quotes. Worst precisely during the 12-month LP lock.

## 5 · LP alpha yield, **the fixed term**

$$y_\alpha \;=\; \frac{7{,}200 \times 0.41 \times (1-\text{rp}) \times 365}{S} \;=\; \frac{910{,}091}{S}\Big|_{\text{rp}=0.155}$$

$S$ = total staked alpha in the subnet.

| $S$ | 3M | 6M | 12M | 24M |
|---|---|---|---|---|
| $y_\alpha$ | 30.3% | 15.2% | 7.6% | 3.8% |

**Independent of trading revenue, subnet rank, and the fund's share of staked alpha** (that share cancels). This is *dilution recapture*, not profit, stakers are receiving newly issued supply. Never quote it without the dilution figure alongside.

## 6 · Pool mechanics (balancer)

$$p = \frac{w_1\,y}{w_2\,x}\qquad(y=\text{TAO reserve},\; x=\text{alpha reserve},\; w\in[0.01,0.99])$$

Buying alpha with $\Delta y$ TAO:

$$\Delta x \;=\; x\left(1-\left(\frac{y}{y+\Delta y}\right)^{w_2/w_1}\right)$$

Fee ≈ 0.05% (`FeeRate` 33/65535) off the input side. **No unbonding period.** Swaps > 1,000× the TAO reserve are rejected. `move_stake` between hotkeys on the same subnet is *not* a swap, no fee, no price impact.

## 7 · Cost to move price, the "optimal rate"

$$\Delta y \;=\; y\left(\left(\frac{p'}{p}\right)^{w_1}-1\right) \;\;\xrightarrow{\;w_1=0.5\;}\;\; \frac{\Delta y}{y} = \sqrt{\tfrac{p'}{p}}-1$$

| Target price increase | +10% | +25% | +50% | +100% |
|---|---|---|---|---|
| Net buy-flow (% of TAO reserve) | 4.9% | 11.8% | 22.5% | 41.4% |

**This formula has no time term, and it is path-independent.** Moving $p \to p'$ in ten steps costs exactly the same as one step. Splitting execution does *not* reduce AMM cost; do it for MEV protection and slippage bounds, not for price.

The time-dependent cost is *holding* the level (§8), and it dominates. Against the reference pool ($131{,}662\,\tau / 2{,}431{,}633\,\alpha$, $\sigma=0.65$):

| Target | One-off move | Hold for 12 months |
|---|---|---|
| flat | 0% of reserve | **70.2%/yr** |
| +10% | 4.9% | 73.7%/yr |
| +50% | 22.5% | 86.0%/yr |
| +100% | 41.4% | 99.3%/yr |

Reaching a level is cheap; defending it against 7,200 α/day of issuance is the real expense, roughly **3–4× the one-off cost, every year**. Optimal policy:

1. Never move price faster than NAV per alpha grows, NAV is the speed limit.
2. Budget for the hold, not the move. The move is a rounding error against the standing cost.
3. Hold levels rather than chase them, the EMA credits time-at-price, not peak price.
4. Spend only the discount: if spot ≥ NAV, correct buy-flow is **zero**.
5. Cut $\sigma$ (miner sell-through), it scales the hold cost linearly and is the only term the fund can directly influence.

## 8 · Price maintenance (standing cost)

Emission sell pressure that must be absorbed just to hold price flat:

$$\Delta y_{\text{maint}} \;=\; 7{,}200 \times \sigma \times p \quad [\tau/\text{day}]$$

$\sigma$ = miner/validator sell-through. At $p=0.02\,\tau$, $\sigma=0.65$, $\tau=\$191$: ≈ **$0.54M/month before any appreciation**, and it *rises* with price, so success raises the running cost.

## 9 · LP return

$$\boxed{\;R_{\text{LP}} \;=\; (1+y_\alpha)\,(1+g_p) - 1\;}$$

- $y_\alpha$, protocol-fixed (§5)
- $g_p$, alpha price change over the period

Equivalently, as drag on the desk's gross return:

$$R_{\text{LP}} \approx R_{\text{trading}} - \underbrace{\frac{7{,}200 \times 365}{S}\Big[0.41\,\text{rp} + 0.41\,\sigma\Big]}_{\text{leakage}}$$

| Miner sell-through $\sigma$ | 0% | 30% | 60% | 100% |
|---|---|---|---|---|
| Drag at $S$ = 12M | 0.7pp | 3.4pp | 6.1pp | 9.6pp |

**Miner alignment is worth ≈9pp/yr of LP return.** That is the argument for the holding incentive, stronger than price support.

## 10 · Issuance retention (NAV leakage)

$$\text{retained} \;=\; \underbrace{0.18}_{\text{owner}\,\to\,\text{fund}} + \underbrace{0.41(1-\text{rp})}_{\text{LP stakers}} + \underbrace{0.41(1-\sigma)}_{\text{unsold miner }\alpha}$$

At 1-year age: **93.6%** retained at $\sigma=0$; **81.3%** at 30%; **69.0%** at 60%; **52.6%** at 100%.

## 11 · NAV band (the discipline)

$$\text{NAV}_\alpha = \frac{\text{trading AUM} + \text{treasury}}{\text{circulating }\alpha},\qquad \delta = \frac{p_{\text{spot}}}{\text{NAV}_\alpha}-1$$

| $\delta$ | Action |
|---|---|
| $< -10\%$ | **Buy**, accretes NAV per alpha for all holders |
| $-10\%\ldots+10\%$ | Hold; absorb sell-flow only |
| $> +10\%$ | **Stop buying.** Selling above NAV transfers value from incoming LPs |

$\delta$ is the headline monthly metric. It converts "we support our token" into "we run a closed-end fund at a disciplined band."

**Dilution hurdle**, trading return required to hold NAV per alpha flat:

$$h = \frac{7{,}200 \times 365}{\text{circulating }\alpha}$$

263%/yr at 1M α · 52.6% at 5M · 26.3% at 10M · 13.1% at 20M. **A larger, well-distributed alpha base makes the structure easier to sustain, not harder.**

## 12 · Conviction & LP lock

Perpetual lock to a **non-owner** hotkey: $C(t) = M_0(1-e^{-t\ln 2/60\text{d}})$.
Lock to the **subnet-owner hotkey**: $C = M_0$ **instantly**, this is what Insignia uses.

Decaying (default) mode: $M(t) = M_0 e^{-\lambda t}$, $\lambda = \ln 2/60\text{d}$; redeemable $= M_0 - M(t)$.

Roll-forward from chain code:

$$C_{\text{new}} = e^{-\Delta t/R_z}C_{\text{old}} + \gamma M_{\text{old}},\qquad \gamma = \frac{R_x(e^{-\Delta t/R_x}-e^{-\Delta t/R_z})}{R_x-R_z}$$

with $R_x$ = `UnlockRate`, $R_z$ = `ConvictionMaturityRate` (both 648,000 blocks by default; when equal, $\gamma \to (\Delta t/R_x)e^{-\Delta t/R_x}$).

**Exit is quoted, not marked.** Spot valuation ignores your own price impact: unwinding 500k α against the reference pool realises −17.1% vs spot in one transaction, but −1.6% staged across the decay curve. The exponential release is a slippage-control mechanism, not merely a lockup.

---

## Reading order for an investor

1. §5, your alpha yield is fixed and knowable
2. §9, your dollar return is that yield times what the desk earns on price
3. §7, moving price has a defined, square-root cost the desk pays out of revenue
4. §2, it takes 3–6 months to transmit; year one is deliberately low-yield
5. §11, the desk is constrained by a published NAV band, not discretion

## What would falsify this

- Sustained buy-flow with no EMA/share response after two quarters → transmission assumption wrong
- $\delta$ persistently > +10% while treasury keeps buying → discipline not being followed
- Retention (§10) below ~60% → miner alignment failing, LP returns leaking
- Root changes $w_\tau$, `UnlockRate`, or the share formula → re-derive §1, §4, §12 from chain
