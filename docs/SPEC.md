# Insignia Subnet, Fund-Backed Alpha Flywheel: Build Specification

**Status:** v0.3 · adopted 2026-07-26 · Root Reborn (runtime v441) implications incorporated 2026-08-05
**Audience:** engineering agents building this repository
**Chain context:** Bittensor mainnet, dTAO + TAO Flow V2 + Conviction v2 + Root Reborn (subtensor PRs #2658, #2687, #2696, #2968; verify current state before implementing, conviction v2 was on devnet/testnet as of late May 2026, Root Reborn shipped mainnet 2026-08-03)

**References:**
- Bittensor docs, https://www.bittensor.com/docs (emissions, dTAO, conviction; pull formulas from here, never from memory)
- Taostats, https://taostats.io/ (live chain data + API for pool reserves, prices, emission shares)
- Insignia MCP, internal data service (MongoDB-backed; see [scripts/mcp_probe.py](../scripts/mcp_probe.py) for the wire protocol). Fund accounting and factsheet pipelines may read attested revenue and desk P&L from here.
- Quantitative model source: [dashboards/charts.py](../dashboards/charts.py) (port of `model_v6.py`), regenerating [docs/investor/](investor/).

---

## 0. Corrections to the original thesis (read first)

The design in this repo MUST be built against the **actual** Conviction v2 mechanics, not the older draft description. Key deltas:

1. **Conviction is a governance primitive, not a yield primitive.** Locking alpha earns *conviction* (weight toward subnet ownership / "subnet king" and governance), not extra emissions. Staking yield comes from ordinary alpha emissions to staked alpha. Locked alpha remains staked to a hotkey and continues to earn those emissions, that is where LP yield comes from.
2. **Correct maturity curve.** For a perpetual lock to a **non-owner** hotkey: `conviction(t) = locked_mass × (1 − e^(−t/τ))` with `τ = 648,000 blocks` (≈ 60-day half-life; 50% at ~day 60, 75% at ~day 120, ~98.5% at 1 year). This matches the `1 − e^(−t/τ)` intuition but with protocol-defined τ, tunable by root, do not hardcode.
3. **Locks DECAY by default.** `lock_stake()` creates a decaying lock (locked_mass half-life ~60 days). Durable locks require an explicit `set_perpetual_lock(netuid, true)`.
4. **Instant conviction rule.** Any coldkey locking to the **subnet owner's hotkey** gets `conviction = locked_mass` immediately. This is the strategically correct target for LP locks: instant governance weight AND it fortifies Insignia's ownership against subnet-king challenges (aggregation is per-hotkey).
5. **One lock per coldkey per subnet**, single hotkey target, top-ups must match the same hotkey.
6. **Subnet king is currently disabled** (code present, transfer call commented out). Design defensively as if it will be enabled: subnets ≥ 1 year old, subnet-wide rolled conviction ≥ 10% of SubnetAlphaOut, highest-conviction hotkey takes ownership.
7. Emissions: the formula (shipped June 2026) is:
   ```
   share_i = p_i × (1 − b_i) / Σ_j p_j × (1 − b_j)
   ```
   where `p_i` = `SubnetMovingPrice` (an **EMA of spot alpha price**, capped at 1.0) and `b_i` = the proportion of last tempo's miner incentive withheld by being directed to subnet-owner hotkeys. Flow matters only insofar as it moves **spot price**, which moves the **EMA**, which moves share. Emission-disabled subnets get zero; there is no zero-emission floor otherwise.

8. EMA:
   ```
   ema_alpha = base_alpha × blocks_since_start / (blocks_since_start + 201,600)
   ```
   New subnets start near zero responsiveness: 20% at day 7, 50% at day 28, 76% at day 90, 93% at day 365. The docs state this exists to blunt launch pumps, coordinated buys, and flash attacks on emission shares. **Treasury buy-flow must be DCA**, budget for a 3–6 month lag between sustained buy-flow and emission-share response, and never promise LPs a fast ramp.

9. **`alpha_out` is flat and price-neutral injections mean emission does not move the market.** `alpha_out` = 1 alpha/block for every subnet regardless of rank (subject to that subnet's own halving curve; alpha issuance also has a 21M cap). `alpha_in ≈ tao_in / price` is injected into the pool, and because injections shift the **balancer weights** rather than the price, protocol emission is price-neutral.

10. **Young subnets pay alpha stakers far below the headline 41%.** A `root_proportion` slice of the validator half is reserved for **root TAO stakers**:
    ```
    root_proportion = root_tao × tao_weight / (root_tao × tao_weight + alpha_issuance)
    ```
    with `tao_weight` = 0.18 (governance-set). Because a young subnet's alpha issuance is small, the slice is large. **Note: LP acceptance occurs 12 months after the subnet is deployed on mainnet, so this year-1 haircut falls on the fund's own locked alpha before any external LP is exposed to it, the desk absorbs the worst of the root-proportion ramp, and incoming LPs enter as the curve flattens.** (Paid only while Σ of all subnet EMA prices > 1.0; otherwise that alpha is recycled.) Within a validator's dividends the split is `α / (α + τ·w)` to alpha stakers, `τ·w / (α + τ·w)` to TAO stakers, after the validator's take. **Root Reborn does not change this split — it changes what happens to the root slice afterward: it accrues in beta-basket escrows as staked alpha instead of auto-selling (§0.16). The haircut is unchanged; its market impact is deferred to claim flow.**

11. **Never route miner incentive to owner hotkeys.** `b_i` taxes emission share **one-for-one**

12. **Mature subnets get free buybacks.** Once `alpha_in` hits its cap (`root_proportion × alpha_emission`), the excess `tao_in` is instead **swapped for alpha on the subnet's own pool**, accumulating as protocol-owned alpha. It reduces the treasury's maintenance burden as the subnet ages, partially offsetting the year-1 root-proportion headwind.

13. **Pool mechanics (for `treasury/execution` and `otc/settlement`).** Balancer weighted pool, `p = (w1·TAO_res)/(w2·alpha_res)`, weights start 0.5/0.5 and are bounded [0.01, 0.99]. Buying alpha: `Δx = x·(1 − (y/(y+Δy))^(w2/w1))`. Fee ≈0.05% (`FeeRate` 33/65535, per-subnet) off the **input** side to the block author. Single swaps > 1,000× the TAO reserve are rejected (`InsufficientLiquidity`). **No unbonding period.** Required implementation rules:
    - Use `add-stake-limit` / `remove-stake-limit` with explicit `limit_price` and `allow_partial=false` for treasury execution; max fill at a limit is `Δy = y·((p′/p)^w1 − 1)`.
    - Use **MEV-shielded submission** (`submit_shielded`, ML-KEM-768) for any large swap with a loose limit. Shield *and* bound, they defend against different attacks.
    - `move-stake` between hotkeys **on the same subnet is not a swap**, no fee, no price impact. Use it for OTC delivery and for any owner-hotkey migration. Cross-subnet moves run two swaps.

14. **NAV must be quoted, not marked, this is now a hard requirement.** Value positions via `quote-unstake` against live reserves, and the investor factsheet must report depth-adjusted NAV. A concentrated LP redemption window walks this curve down for every LP behind it in the queue, which is the quantitative justification for the ≤25%-per-60-day cohort cap in M6.

15. **All parameters are mutable by root** (UnlockRate, ConvictionMaturityRate, auto-lock defaults, flow accounting). Every module must read parameters from chain at runtime and tolerate change. In this repo the shared read layer is [chainio/params.py](../chainio/params.py); pure-math modules take parameters as arguments and never embed chain constants.

16. **Root Reborn (runtime v441, mainnet 2026-08-03, PR #2968) replaced the per-block auto-sell of root dividends.** Each root validator (64 seats, burn-based entry, 18% default take) now runs a single on-chain fund ("beta basket"): its root dividends are held as **real staked alpha under a keyless escrow coldkey**, fund shares mint at **realizable NAV** (priced at what holdings would fetch at current pool depth, not spot), and claims are arg-less fund-level pro-rata redemptions that sell fraction `f` of *every* holding. Weight curation launched **disabled**; the default is accumulate-in-place at emission weights, executing zero trades. Consequences for this design:
    - **The root slice is deferred leakage, not retained value.** The escrowed alpha sits physically staked on the subnet (no immediate sell pressure, and it counts in the staked-alpha denominators), but it is **economically owned by root stakers through fund shares**. It exits the wrapper when they claim. Never count it as retained in NAV or retention accounting; model it as leakage whose *realization timing* is claim flow rather than per-block auto-sell.
    - **Per-subnet net flow becomes a competitive allocation game once curation enables:** `−(own root dividend sold at origin) + Σ(validator weight to Insignia × total root dividends, ~983 τ/day network-wide)`. Basket flows land at **epoch boundaries**.
    - **Escrow-held basket alpha is stake, not lock:** it earns emissions (dilutes per-unit staker APY), accrues under root validators' hotkeys (watch stake-weight in the subnet's own consensus), and carries **no conviction** (no direct subnet-king vector) — but verify on-chain whether escrowed alpha enters the SubnetAlphaOut denominator of the ≥10% king-activation threshold.
    - **New RPC surface to wire into [chainio/params.py](../chainio/params.py):** `betaBasket_getValidatorWeights`, per-validator NAV + basket composition, network-wide NAV, staker pending TAO.

---

## 0.5 Governing design principle, alpha is the fund wrapper, not a growth asset

**Alpha is a proxy for the fund.** Returns originate in the prop-trading desk. The subnet's token is the investment wrapper: LPs hold and stake alpha, which gives them fund exposure, deepens pool liquidity, and aligns miners, who hold alpha and thereby become investors rather than mercenary sellers. Emission share is a **byproduct** of a healthy wrapper, never a target to be bought.

This reframe resolves the year-1 headwinds in §0.8 and §0.10. The EMA ramp (20% → 76% responsiveness over 90 days) and the root-proportion haircut (12.7% → 34.6% of alpha_out to stakers over 12 months) both improve over exactly the period the desk is building a prop-trading track record, and LPs generally require that track record before committing. **The protocol's ramp and the fund's fundraising ramp are aligned.** Phase 1 should therefore be run as a deliberate low-yield, track-record-building period, and marketed as such.

### The NAV-tracking policy (the discipline that makes this legitimate)

Treat the structure as a **closed-end fund**. Define:

```
NAV_per_alpha = (trading AUM + realizable treasury holdings) / circulating alpha supply
premium_discount = spot_alpha_price / NAV_per_alpha - 1
```

**NAV is realizable, not marked (hard requirement, see §0.14).** Alpha-denominated treasury holdings are valued via `quote-unstake` against live reserves — what they would actually fetch at current pool depth — never at spot. This is the same standard the protocol itself now applies to root beta-basket funds (§0.16: deposits and claims price at realizable NAV at pool depth), so it is both the honest number and the protocol-native idiom. **Realizable NAV is what gates the DCA:** the band below compares spot against it, so thin pool depth automatically lowers NAV, widens the measured premium, and throttles buy-flow before the treasury overpays into its own illiquidity.

Treasury policy is a band, not a target price:

| State | Action | Rationale |
|---|---|---|
| Spot < 0.9 × NAV | **Buy** (TWAP, limit-bounded, shielded) | Accretes NAV per alpha for every holder; buying real value at a discount |
| 0.9–1.1 × NAV | Hold / minimal maintenance | Absorb miner sell-flow only |
| Spot > 1.1 × NAV | **Stop buying.** Consider OTC issuance | Buying above NAV destroys value; selling alpha above NAV transfers value from incoming LPs to existing holders and is the behaviour that would make this look like a pump |

`premium_discount` is the single headline metric in the monthly factsheet. It converts "we support our token" into "we run a closed-end fund at a disciplined band," which is auditable, defensible in diligence, and the honest description of what the mechanism does.

### "Optimal rate" of EMA increase, specified

Moving price costs `Δy = y·((p′/p)^w1 − 1)`, the same formula the chain uses for `add_stake_limit` fills. Because it is a square root, **the first increments are cheap and large jumps are punitive**: +10% costs 4.9% of the TAO reserve, +50% costs 22.5%, +100% costs 41.4%. Combined with the EMA needing weeks at a level before emission share responds, the optimum is:

1. **Never move price faster than NAV per alpha grows.** NAV is the speed limit; the trading desk's return sets it.
2. **Spread execution.** Continuous small TWAP buys, since cost is convex in step size.
3. **Hold levels, don't chase them.** EMA responds to time-at-price, not to peak price. A level held for 8 weeks beats a spike.
4. **Spend only the discount.** If spot ≥ NAV, the correct buy-flow is zero, bank the revenue as reserve instead.
5. **Net out basket flows (post-Root-Reborn, once curation enables).** Root-basket origin sells and redeploy buys land at **epoch boundaries** and obey the same fill formula, so the Δy required of the treasury to hold a level is `Δy_treasury = Δy_target − Δy_basket_net`. Rebalance the DCA schedule per epoch against observed basket net flow toward/away from Insignia (queryable via `betaBasket_getValidatorWeights` + escrow stake deltas): spend less when baskets are net buyers (their flow raises the EMA for free), expect a larger required Δy when rotation or claim clusters are net sellers. Keep TWAP timing randomized *away from* epoch boundaries so the buy program neither clusters with basket flow nor becomes predictable against it.

### Dilution, staking, and why miner alignment matters

Issuance is 7,200 alpha/day regardless of subnet size, so holding NAV per alpha flat requires an annual trading return of `7,200 × 365 / circulating_supply`: **263% at 1M alpha, 52.6% at 5M, 26.3% at 10M, 13.1% at 20M.** Two consequences:

- **Staking is a dilution defence, not a yield perk.** Stakers *receive* the issuance; the hurdle binds on non-staking holders. LP alpha must never sit unstaked, this belongs in the LP agreement, not just in ops.
- **Miner selling is NAV leakage.** Retention inside the wrapper = owner cut (18%, back to the fund) + alpha-staker share (41% × (1 − root_proportion), to LPs) + unsold miner alpha. At a 1-year-old subnet: **93.6% retained at 0% miner sell-through, 81.3% at 30%, 69.0% at 60%, 52.6% at 100%.** The miner-holding incentive is therefore a first-order NAV lever, which is a far stronger justification for it than price support. Design miner rewards (and any staking bonus for miners) around retention, and report retention monthly. The existing pairing mechanism's deployment pipeline (top `(model, strategy)` pairs deployed by the desk) and token-gated API access are the retention levers already in place, see [subnet/docs/INCENTIVE_MECHANISM.md](../subnet/docs/INCENTIVE_MECHANISM.md).
- **The root-proportion slice is deferred leakage (post-Root-Reborn).** The root slice (`41% × root_proportion` of the staker tranche) no longer auto-sells per block; it accrues as staked alpha in validator beta-basket escrows, economically owned by root stakers (§0.16). Count it as **leaked** in retention accounting — the change is realization *timing*, not ownership. Model its exit as claim flow (no deadline, pro-rata across fund holdings, likely clustering in TAO drawdowns) instead of a constant drain. Two second-order effects: escrowed alpha inflates the staked-alpha denominator (report LP staker APY against the full staked base including escrow), and until curation enables, the deferral means near-zero mechanical sell pressure from root on the pool.
- Grow supply deliberately: the hurdle falls as supply grows, so a larger, well-distributed alpha base makes the structure *easier* to sustain, not harder.

---

## 1. System overview

```
                        ┌────────────────────────────┐
                        │  Insignia Trading Desk      │
                        │  (off-chain prop trading;   │
                        │   deploys subnet's evolved  │
                        │   (model, strategy) pairs)  │
                        └────────────┬───────────────┘
                                     │ realized revenue (USD/stables)
                                     ▼
   LP capital ──► ┌────────────── Treasury Engine ──────────────┐
   (Phase 1:      │ 1. convert revenue → TAO (CEX/DEX, TWAP)     │
    trading cap)  │ 2. stake TAO → SN alpha (on-chain buy-flow)  │
                  │ 3. route alpha: OTC book / reserve / burn    │
                  └───────┬──────────────────────┬──────────────┘
                          │ alpha OTC (discount   │ net TAO inflow
                          ▼  for lock commitment) ▼
                  ┌──────────────┐        ┌──────────────────────┐
                  │  LP Lock Mgr │        │ EMA price → emission  │
                  │ lock_stake → │        │ share ↑ → alpha       │
                  │ owner hotkey │        │ emissions to stakers ↑│
                  │ + perpetual  │        └──────────┬───────────┘
                  └──────┬───────┘                   │
                         │  locked, staked alpha ◄───┘  (yield)
                         ▼
                  instant conviction → subnet ownership defense
```

Wrapper loop (NOT a pump): trading revenue → TAO buy-flow into the Insignia pool → higher EMA price → higher emission share → higher staking yield on LPs' locked alpha → OTC demand for alpha at premium/discount schedule → more locked supply → price support + ownership defense.

**Known failure mode (design for it):** the loop is reflexive. If revenue stalls, buy-flow stops, emission share falls, yield falls, and decaying locks begin releasing supply into a falling market. Mitigations are mandatory (see §7).

---

## 2. Phases

### Phase 0, Legal & entity scaffolding (BLOCKING; not code)
- Engage securities counsel. Pooled LP capital + profits from the firm's trading is very likely an investment contract / regulated fund in most jurisdictions; OTC alpha sales to LPs are likely securities offerings. Structure (e.g., exempt private fund, offshore feeder), KYC/AML, investor accreditation, and CTA/CPO/IA analysis are prerequisites. **Agents: do not ship investor-facing onboarding until a `LEGAL_SIGNOFF.md` exists at repo root.** The gate is enforced in code by [otc/compliance.py](../otc/compliance.py).
- Custody design: institutional-grade coldkey custody (multisig/proxy extrinsics, HSM or qualified custodian).

### Phase 1, Trading-capital bootstrap
- Accept LP capital into the fund as trading capital (off-chain).
- Register/operate the Insignia subnet: **already implemented in this repo**, the researcher/trader pairing mechanism with NSGA-II selection is the genuine-utility requirement (flow-only subnets are penalized by design and by market scrutiny). See [subnet/docs/PAIRING_MECHANISM.md](../subnet/docs/PAIRING_MECHANISM.md) and [subnet/docs/SUBNET_SPEC.md](../subnet/docs/SUBNET_SPEC.md).
- Stand up Treasury Engine in accumulation-only mode (small, scheduled TAO→alpha buys from revenue).

### Phase 2, Cap & conversion
- Close the fund at target AUM.
- Offer LPs conversion: fund units → alpha allocation delivered OTC, locked on-chain per §4.
- Publish the vesting/lock schedule and the redemption policy before conversion.

### Phase 3, Steady-state flywheel
- Revenue routing policy (governable): e.g., X% buy-flow, Y% OTC inventory, Z% reserve buffer, W% ops.
- Continuous reporting: emission share, net flow, conviction table, reserve coverage.
- **CEX listing readiness.** Track centralized-exchange listing requirements for subnet tokens (Kraken, a US venue with hard compliance standards, began listing subnet tokens in 2026). A CEX balance-sheet price is the institutional entry requirement; DEX-only price discovery is a non-starter for the LP base. Target post-Phase-2 conversion, sequenced with counsel.
- **Root-validator IR channel (post-Root-Reborn).** Once basket curation enables, 64 root validators allocating ~983 τ/day of dividends — publicly scored on realized TAO returns — are a named institutional audience for exactly the diligence surface this fund already publishes (realizable NAV, premium/discount band, reserve coverage, retention). Treat basket curators as a distribution channel in IR materials; their inflow raises the EMA without treasury spend. Differentiation to state explicitly: Insignia returns originate in off-chain trading P&L, not passive alpha beta — relevant once root fund shares tokenize (the stated endgame of PR #2968).

---

## 3. Repository layout

The fund layer lives at the repo root alongside the existing `subnet/` package (which retains its own `pyproject.toml`). One deviation from the original draft layout: a shared `chainio/` package holds the chain-parameter/pool-snapshot read layer used by every fund module, so that no module hardcodes chain constants (§0.16).

```
Insignia-Subnet/
├── LEGAL_SIGNOFF.md              # ABSENT until Phase-0 counsel signs off; gates investor-facing features
├── docs/
│   ├── SPEC.md                   # this file
│   ├── RISK_REGISTER.md
│   └── investor/                 # generated charts, factsheets (dashboards/charts.py output)
├── subnet/                       # the actual Bittensor subnet (pre-existing; own pyproject)
│   ├── insignia/                 # pairing, incentive, scoring, protocol
│   ├── neurons/                  # researcher_miner, trader_miner, validators
│   ├── docs/                     # SUBNET_SPEC, PAIRING_MECHANISM, INCENTIVE_MECHANISM, …
│   └── tests/
├── chainio/
│   └── params.py                 # ChainParams / PoolSnapshot dataclasses + live providers
├── treasury/
│   ├── engine.py                 # revenue → TAO → alpha pipeline
│   ├── execution/                # TWAP/limit execution, slippage guards, MEV shield rules
│   ├── pool_math.py              # balancer weighted-pool math (quotes, staged exits, move costs)
│   ├── emissions.py              # root_proportion, EMA responsiveness, retention, dilution hurdle
│   ├── policy.py                 # NAV band, routing % config, circuit breakers
│   └── accounting.py             # lot-level P&L, depth-adjusted NAV, proof-of-reserves
├── lockmgr/
│   ├── locks.py                  # lock_stake / set_perpetual_lock wrappers
│   ├── monitor.py                # get_coldkey_lock, get_hotkey_conviction pollers; king early-warning
│   └── schedules.py              # per-LP vesting state machine, cohort windows, outer-bound toggle
├── otc/
│   ├── compliance.py             # LEGAL_SIGNOFF + KYC gate (imported by all investor-facing code)
│   ├── desk.py                   # quote engine (discount vs lock commitment)
│   └── settlement.py             # move_stake delivery + lock verification
├── dashboards/
│   ├── investor_api/             # read-only: NAV, APY, conviction, redemptions, factsheet
│   └── charts.py                 # regenerates docs/investor/*.png (port of model_v6.py)
├── risk/
│   ├── reflexivity.py            # scenario Monte Carlo (see §7)
│   └── alerts.py                 # pager rules
├── tests/                        # fund-layer tests (subnet tests stay in subnet/tests)
├── infra/                        # keys (proxy/multisig), CI, deployment
└── pyproject.toml                # fund-layer package (insignia-fund)
```

---

## 4. LP lock lifecycle (lockmgr)

**Chart impact note.** The original vesting chart drew conviction as a 60-day `1 − e^(−t/τ)` ramp. That is the **non-owner** curve. Insignia locks to the **subnet owner hotkey**, so conviction is granted **instantly at 100% of locked mass** and thereafter simply tracks locked mass. Superseded by `docs/investor/v6_plot1_vesting_corrected.png`.

Per-LP state machine. Each LP uses (or is assigned) a dedicated **coldkey** (one-lock-per-coldkey-per-subnet constraint).

1. **DELIVERED**, OTC settlement transfers alpha stake to LP coldkey (staked to Insignia owner hotkey).
2. **LOCKED**, `lock_stake(owner_hotkey, netuid, amount)` executed; verify via `get_coldkey_lock`.
3. **PERPETUAL**, `set_perpetual_lock(netuid, true)` within same session. Conviction = locked_mass instantly (owner-hotkey rule). Yield (alpha emissions) accrues throughout.
4. **VESTING_COMPLETE** (month 12 default, per LP agreement), LP (or fund via proxy, per the signed agreement) may toggle `set_perpetual_lock(netuid, false)`.
5. **DECAYING**, locked_mass halves every ~60 days; LP may unstake `original − locked_mass` at any time. ~50% redeemable at +2 months, ~88% at +6, ~98.5% at +12.
6. **CLOSED**, residual dust unstaked; lock removed.

Invariants to enforce/test:
- Lock target MUST be the owner hotkey (instant conviction + ownership defense). Alert if owner hotkey changes.
- **Locked alpha MUST remain staked at all times.** Unstaked LP alpha forgoes issuance recapture and eats the full dilution hurdle (§0.5). Put this in the LP agreement, not just ops runbooks. Monitor and alert on any unstaked LP position.
- **Cliff exits are contractually prohibited.** The exponential decay is a slippage-control mechanism, not merely a lockup: unwinding a 500k-alpha position in one transaction against the reference pool realises −17.1% vs spot, while staging it across the decay curve realises −1.6%. Redemption tooling must default to staged, quote-checked increments (`quote-unstake` before each), never a single `remove_stake`.
- **The 12-month term is justified, not arbitrary:** it spans the EMA responsiveness ramp (20% → 93%) and the root-proportion ramp (12.7% → 34.6% of alpha_out to stakers). Document this rationale in LP materials, it is a far better answer than "market convention".
- Never allow a coldkey swap into a coldkey with active locks (chain will reject; pre-validate).
- Read `UnlockRate` / `ConvictionMaturityRate` from chain each epoch; recompute schedules if changed; alert on change.
- Track aggregate `OwnerLock + DecayingOwnerLock` vs any third-party `HotkeyLock` aggregates → subnet-king early-warning (even while disabled).
- Track beta-basket escrow stake on the Insignia netuid per root-validator hotkey (§0.16): it is stake without conviction (no king vector), but it shifts stake-weight in subnet consensus and is the claim-flow overhang. Verify empirically whether escrowed alpha counts in the SubnetAlphaOut denominator of the ≥10% king-activation threshold, and adjust the early-warning ratio accordingly.

Reference formulas (implemented in [lockmgr/schedules.py](../lockmgr/schedules.py), parameterized by on-chain τ):

```
λ            = ln(2) / half_life_days              # half_life ≈ 60d at current defaults
perpetual:     mass(t) = M0
               conviction(t) = M0                   # if locked to owner hotkey
               conviction(t) = M0·(1 − e^(−λt))     # if non-owner hotkey
decaying:      mass(t) = M0·e^(−λt)
               redeemable(t) = M0 − mass(t)
roll-forward (general, from chain code):
               decay_x = exp(−dt/UnlockRate); decay_z = exp(−dt/MaturityRate)
               γ = UnlockRate·(decay_x − decay_z)/(UnlockRate − MaturityRate)
                   → (dt/UnlockRate)·decay_x when rates are equal
               C_new = decay_z·C_old + γ·M_old
```

---

## 5. Treasury Engine

- **Inputs:** attested revenue deposits (stables/USD), routing policy, market data.
- **Buy-flow execution:** TWAP TAO purchases; on-chain `add_stake` into the Insignia pool sized to avoid > configurable slippage bps; randomized timing (MEV Shield aware).
- **Routing policy (initial defaults, all governable):** buy-flow is CONDITIONAL on premium/discount per the NAV band in §0.5, zero when spot >= NAV. Nominal split when at a discount: 50% buy-flow / 20% OTC inventory / 25% reserve buffer / 5% ops; when at a premium the buy-flow tranche accrues to reserve. Reserve buffer target: ≥ 6 months of trailing median buy-flow (reflexivity brake).
- **Circuit breakers:** halt buy-flow if (a) alpha price > x·30d MA (don't chase), (b) reserve < 3 months, (c) emission share falls > y% WoW despite flow (parameter regime change → investigate). For (c), first check basket weights (`betaBasket_getValidatorWeights`) — root-basket rotation away from Insignia is a benign-mechanics explanation that must be ruled out before declaring a regime change; the responses differ (rotation → IR problem, regime change → model problem).
- **Accounting:** every conversion lot recorded; daily NAV; monthly proof-of-reserves publication.

## 6. OTC desk

- Quotes alpha to incoming LPs at pool-referenced price ± schedule: discount scales with committed perpetual-lock duration; premium for unlocked delivery (discouraged).
- Settlement is atomic-ish: deliver stake → verify `lock_stake` + perpetual flag on the LP coldkey within N blocks, else claw back per agreement.
- All OTC counterparties pass Phase-0 KYC gate.

## 7. Risk engine (mandatory)

[risk/reflexivity.py](../risk/reflexivity.py), Monte Carlo of the flywheel with shocks:
- revenue drawdown paths (−50%, −100% for 3/6/12 months),
- correlated TAO price drawdowns,
- competing-subnet flow growth (share erosion),
- mass decay-mode toggles by LPs (redemption run) into thin pool liquidity,
- **basket rotation** (root validators re-weight away from Insignia after underperformance; basket flow is momentum-amplifying in both directions — today only new-flow direction changes, holdings don't rebalance, but validator-directed rebalancing is stated future work in PR #2968),
- **root-claim clustering** (in a TAO drawdown, root stakers claiming en masse sell fraction `f` of every fund holding including Insignia — correlated sell flow proportional to accumulated escrow holdings, correlated with exactly the states where LP redemption demand also peaks). Baseline note: the constant root-drain assumption is retired; root sell pressure is now episodic claim flow, not a per-block constant.

Output: probability of "spiral" states (emission share < threshold AND redeemable supply > pool depth). Publish quarterly to LPs. Alert rules in [risk/alerts.py](../risk/alerts.py) tied to live chain data (taostats API + direct RPC: `get_coldkey_lock`, `get_hotkey_conviction`, `get_most_convicted_hotkey_on_subnet`).

[docs/RISK_REGISTER.md](RISK_REGISTER.md) must cover, at minimum: reflexive unwind; protocol parameter/root changes; subnet-king enablement; regulatory action; custody/key compromise; pool liquidity/slippage on exit; alpha inflation vs buyback net effect; concentration of LP unlock dates (stagger vesting cliffs across cohorts).

## 8. Reporting & investor dashboard

Read-only API + monthly factsheet:
- **premium/discount to NAV per alpha** (headline metric),
- **issuance retention rate** (share of alpha_out retained in the wrapper; miner sell-through),
- staker APY realized (alpha and TAO/USD terms, clearly separated), shown against the dilution hurdle,
- subnet net TAO flow vs network, emission share trend,
- conviction table (owner aggregate vs top external hotkeys),
- lock cohort schedule (redeemable supply curve, next 24 months),
- reserve coverage ratio, buy-flow executed vs revenue attested,
- **root-basket exposure** (post-Root-Reborn): aggregate escrow-held Insignia alpha, per-validator basket weight toward Insignia, epoch basket net flow, and trailing claim-flow trend — the overhang that realizes as sell pressure when root stakers claim.

Charts in [docs/investor/](investor/) regenerated by [dashboards/charts.py](../dashboards/charts.py) (conviction maturity, decaying default, LP vesting lifecycle, staged-exit slippage control, leakage drag).

## 9. Milestones & acceptance criteria

| # | Milestone | Acceptance |
|---|-----------|------------|
| M0 | Legal signoff, custody live | `LEGAL_SIGNOFF.md`; multisig/proxy tested on testnet |
| M1 | Subnet live with real utility | miners/validators earning; docs; incentive audit |
| M2 | Lockmgr on testnet | full lifecycle §4 executed & verified via RPC on testnet; param-change chaos test passes |
| M3 | Treasury accumulation mode | 30 days of TWAP buys within slippage budget; accounting reconciles to chain |
| M4 | OTC pilot (internal capital only) | settlement + lock verification atomic path proven |
| M5 | Risk engine + dashboard | reflexivity report generated; alerts fire in game-day drill |
| M6 | LP conversion (Phase 2) | staggered cohorts; ≤ 25% of locked supply sharing any 60-day redemption window |

## 10. Open questions (resolve before M6)

1. **DECIDED (v0.2): the perpetual→decay toggle is operated by the Insignia trading desk, per terms agreed with each LP.** Implementation requirements:
   - **Key architecture:** prefer LP-held coldkey + limited proxy granted to the desk covering `lock_stake` / `set_perpetual_lock` only (no transfer/unstake authority to fund addresses). Agents: verify on testnet which subtensor proxy type actually gates the lock extrinsics; if none does, fall back to fund-custodied cohort coldkeys under multisig, and document the custody implications for Phase-0 counsel.
   - **Rules, not discretion:** LP agreement must encode (a) a hard outer bound, toggle auto-flips at month N regardless of desk judgment; (b) enumerated conditions permitting delay within the bound; (c) disclosure of any delay and its reason in the monthly factsheet. [lockmgr/schedules.py](../lockmgr/schedules.py) enforces the outer bound in code (scheduled toggle transaction that the desk can accelerate but not cancel).
   - **Cohort design:** desk-controlled toggling enables cohort coldkeys sharing one schedule; keep ≤ 25% of locked supply in any shared 60-day redemption window (M6 criterion).
   - **Conflict-of-interest note for RISK_REGISTER.md:** desk incentive to delay toggles during drawdowns coincides with peak LP exit demand; the hard outer bound is the mitigation.
2. Auto-lock of owner cut: enable (`sudo_set_owner_cut_auto_lock_enabled`) to compound ownership defense, or keep owner emissions liquid for ops? Recommend: enable, perpetual.
3. Treatment if subnet-king is enabled: target defensive conviction ratio (owner-hotkey aggregate ≥ 2× largest external hotkey aggregate AND ≥ 10% SubnetAlphaOut)?
4. Buyback vs burn split, does burning any alpha improve LP economics more than buy-flow, given TAO Flow V2 accounting? Model in risk engine.
5. Redemption liquidity backstop: does the treasury commit to bid redeemed alpha at NAV-band, and how is that funded without re-inflating the loop?
6. **LP-miner delegation track.** Post-conversion, LPs may run (or delegate to approved operators) miner pods, with mined alpha auto-staked to the owner hotkey. This compounds three existing design goals: lower miner sell-through σ (§0.5 leakage), owner-hotkey conviction aggregation (subnet-king defense, R3), and LP lock-in beyond the 12-month term. Decide before Phase 2 conversion terms are drafted: eligibility, delegation approval, and whether LP-miner rewards carry any retention bonus beyond the standard 41% miner tranche.
7. **Chain shorting (hedging primitive).** Track the subtensor shorting proposal (TAO-collateralized negative positions, no lending, no protocol bad-debt). Once shipped it is a reflexivity relief valve for R1: locked LPs can hedge instead of exiting at the first toggle opportunity. Expect short interest to cluster around predictable events, cohort toggle windows, redemption windows, and factsheet dates; keep treasury TWAP timing randomized around those windows so the buy program cannot be front-run. Model the relief quarterly via `hedge_relief_frac` in [risk/reflexivity.py](../risk/reflexivity.py).

---

*This document describes mechanism design and engineering scope. It is not legal, tax, or investment advice; Phase 0 gates all investor-facing functionality.*
