# Native registration collateral and deployment bonds

**Status:** adopted 2026-08-16 · companion to [SPEC.md](SPEC.md) §0.17 and [subnet/docs/INCENTIVE_MECHANISM.md](../subnet/docs/INCENTIVE_MECHANISM.md)
**Chain:** Subtensor miner-registration collateral pallet (`pallets/subtensor/src/subnets/collateral.rs`); query surface [`collateral-policy`](https://github.com/RaoFoundation/subtensor/blob/1f090af85d1771c5d8ece1f0910576fbd129906e/docs/query/collateral-policy.mdx)
**Do not conflate with** conviction `lock_stake` (SPEC §4) or Insignia deployment bonds (`treasury/collateral.py`)

This document is the engineering source of truth for how Insignia uses the native Subtensor collateral primitive Const described for trading subnets, and how that primitive sits next to the desk-P&L bond this repo already had.

---

## Premise

Until this primitive, miner entry was **burn registration**: TAO destroyed via a Dutch-auction moving average. The only signal you could express was “I paid this much to get in.”

Native collateral splits that into two things:

1. **The ticket** — the burned share of the registration price.
2. **The bet on future mining** — alpha locked as a bond, recoverable **only by earning emission** on that subnet.

The origin is trading subnets that score Sharpe / Sortino / PnL (the SN8 fights). A miner can martingale: lever up, go all-in one direction, look statistically brilliant until they blow up. Short-window scores pay them; the blow-up is someone else’s problem.

**Force them to collateralize performance, and you pull expected return further into the future.** They have to keep mining — and keep not blowing up — long enough to unlock the bond. If validators catch an exploit or a blow-up and set weights to zero, the miner stops earning, remaining collateral **freezes**, and a later re-registration credits the standing lock rather than unlocking it.

Predecessor: the [Church of Rao EVM collateral contracts](https://github.com/bactensor/collateral-contracts) (ComputeHorde SN12). Those allowed validators to `slashCollateral()` but required H160 wallets, gas, per-validator contracts, and reclaim/deny flows. Native collateral is the same economic idea pulled into Subtensor so explorers and `btcli` can see it.

This is **exactly Insignia’s scoring surface** (Sharpe, Sortino, Omega, max drawdown, 7-day consistency). The native primitive is the missing cheap, universal, on-chain layer under the scoring suite. It is **not** a substitute for the deployment bond that slashes against live desk P&L.

---

## Three locks, three jobs

| | Conviction `lock_stake` | Native registration collateral | Deployment collateral |
|---|---|---|---|
| Who | LPs, owner hotkey | Every registering miner | Deployed `(researcher, trader)` pairs only |
| Job | Governance weight / king defense | Statistical-significance time-bond | Live desk P&L downside |
| Unlock | Decay / toggle (SPEC §4) | Earn emission at snapshotted `drain_ratio` | Clean undeployment minus slashes |
| Punishment | None (it’s a lock, not a slash) | Freeze if weights go to 0 / UID pruned | Realized losses slash the bond |
| Fate of taken alpha | Unlocks on decay | Frozen until re-earned | **Burned** via `add_stake_burn` |
| Split between roles | N/A | N/A (one hotkey) | Attribution (`blame_split`), not pro-rata |
| Custody | Miner/LP’s own stake | Pallet lock on the miner’s own stake | `transfer_stake` to fund escrow coldkey |
| Code | `lockmgr/` | `chainio/collateral.py`, `subnet/insignia/native_collateral.py` | `treasury/collateral.py` |

Conviction and native collateral both subtract from `available_to_unstake`, but they are different maps. Native collateral is keyed `(netuid, hotkey, coldkey)` so nominators on the same hotkey are not frozen by the owner’s bond. It has **no transfer exit**: `ensure_transfer_respects_collateral` refuses to let a miner `transfer_stake` their lock to someone else (including the fund escrow). That is the interaction risk with deployment bonds — see below.

---

## Pallet mechanics

Owner-set, per subnet (admin-utils events `CollateralLockShareSet` / `CollateralDrainRatioSet`):

| Param | Meaning |
|---|---|
| `lock_share` \(p\) | Fraction of the registration price locked as collateral instead of burned. `0` disables. Pallet cap 95% (`CollateralLockShareTooHigh`). Encoded as `u16` with `u16::MAX = 100%`. |
| `drain_ratio` \(k\) | Alpha released per alpha of hotkey emission earned. **Snapshotted per miner at registration.** Changing \(k\) later does not reprice existing miners until they re-register. |

Miner-signed extrinsics:

- `add_collateral(netuid, hotkey, alpha, limit_price)` — lock additional alpha on the signer’s own hotkey. Prefers free already-staked alpha; buys the shortfall with TAO. Does **not** re-snapshot drain ratio (a top-up is not a new registration).
- `set_min_collateral(netuid, hotkey, min_locked)` — miner-set floor. Drain stops at the floor; emission fills a shortfall. Zero clears it. **Validators cannot set another miner’s floor on-chain.** They publish a required minimum and zero weights if the metagraph row is short.

Settle (`settle_miner_collateral`, called from emission distribution):

- `emission` drives lifetime `earned` and the release rate \(k \times \text{emission}\).
- `capturable` is the maximum that may be diverted into the lock when below the floor. It must already belong to the owner (full miner incentive, or only the validator’s take). Nominator / root-claimable shares must never be passed as capturable.
- Below `min_locked`: capture `min(capturable, shortfall)` into the lock.
- Above `min_locked`: release `min(k × emission, locked − min_locked)` back to withdrawable stake.
- Fully drained with no floor → row removed.
- Zero emission → no-op. A miner who stops earning keeps the remainder **frozen indefinitely**.

Re-registration credits standing collateral against the new requirement, valued at **EMA price not spot** (so you cannot pump-and-re-register cheap). You pay only the burned share plus any shortfall.

Mirrored in float alpha by `chainio.collateral.settle_miner_collateral` / `unlock_horizon_days`. Live reads: `ParamsProvider.collateral_policy` / `miner_collateral`. Empty rows with an enabled policy is **missing visibility**, never “no collateral” (`lockmgr.monitor.native_collateral_findings`).

Query:

```python
sub = bt.Subtensor()
sub.collateral.collateral_policy(netuid=...)
# or
sub.read("collateral_policy", netuid=...)
```

```bash
btcli query collateral-policy --netuid <n> --json
```

---

## Insignia policy

Engineering defaults, **not** chain commitments. The owner still has to set them on-chain; every production path reads through `ParamsProvider`.

| Knob | Default | Why |
|---|---|---|
| `lock_share` | `0.50` | Half the registration price is a recoverable bond; half still burns. Pure-burn (`0`) disables the primitive. |
| `drain_ratio` | `1.0` | One locked alpha releases per one alpha of emission earned. Lowering \(k\) stretches the martingale horizon (`unlock_horizon_days`). |
| `required_min_alpha` | `0` | Registration lock is the floor unless validators publish a higher one. Raise this for a GPU-style “keep X posted.” |
| Drawdown freeze | `0.20` | Same hard ceiling as trading scoring / `TradingValidator.max_drawdown_limit`. |

Unlock horizon, the actual statistical-significance bond:

```
days = (locked − min_locked) / (drain_ratio × daily_emission)
```

At \(k = 1\), a miner with 1,000 α locked earning 10 α/day takes 100 days to recover the bond. At \(k = 0.5\), 200 days. At daily_emission = 0, the horizon is infinite — that is the freeze.

### Validator enforcement (the teeth)

Validators **cannot** write another miner’s `min_locked`. They enforce by zeroing Yuma weights (`subnet/insignia/native_collateral.py`, applied in `PairedValidator.finalize_generation`):

1. **Floor shortfall** — `locked < required_min_alpha` (when the published floor is > 0).
2. **Martingale / blow-up freeze** — `FreezeLedger` records a trader whose `max_drawdown` breaches 20%. Weights stay zero across subsequent epochs so they cannot immediately farm emission (and drain the lock) on the next lucky window. The record drops when the UID leaves the metagraph (pruned or re-registered).

A frozen miner earns nothing → collateral cannot drain → the standing lock is still there when they re-register. That is Const’s “collateral is a bet on your future mining.”

Scoring still withholds the blown-up epoch’s upside (max-drawdown hard ceiling, consistency, Omega). The freeze is the *downside on the registration bond* that scoring alone cannot create.

### What native collateral does not do

- It does **not** slash against live desk P&L. Drain is tied to Yuma emission, not to the desk’s dollar book. A gamed-but-still-weighted miner keeps unlocking.
- It does **not** burn. Freeze is not a sink. A sybil can sit on frozen alpha and re-enter later paying only the burned share.
- It cannot attribute a joint loss between a researcher and a trader who did not choose each other (PAIRING_MECHANISM.md §2.3).

Those three are why the deployment bond stays.

---

## Deployment collateral (unchanged, stacked)

Acceptance into the live desk pipeline still requires a staked-alpha bond escrowed via `transfer_stake` to a fund coldkey, sized against allocated capital (`treasury.collateral.required_bond_alpha`). Realized losses slash by **attribution** (`blame_split`), and slashed alpha is **burned** (`treasury/execution/burn.py`). See INCENTIVE_MECHANISM.md §Deployment Collateral.

Native collateral is the **screening / emissions-layer** bond for every UID, including undeployed miners who never touch desk capital. Deployment collateral is the **desk-P&L** bond for the subset that does.

### Interaction: the registration lock cannot fund the desk bond

Native collateral has no transfer exit. A deployed miner needs:

1. The native registration lock still sitting on their `(hotkey, coldkey)`.
2. Additional **free** alpha to `transfer_stake` as the deployment bond.

`MinerCollateralPosition.free_alpha = stake − locked`. `lockmgr.monitor` warns `native_collateral_starves_bond` when free alpha is below the posted desk bond. Size the two so the registration lock does not starve the escrow.

Retention (R11): the two stocks are **disjoint**. `emissions.effective_sell_through(base, bonded_fraction, native_locked_fraction)` adds them and caps at 1.

---

## Operations

| Surface | What |
|---|---|
| `chainio/collateral.py` | Pallet-mirroring math, gate, policy defaults |
| `chainio/params.py` | `CollateralPolicy`, `MinerCollateralPosition`, provider reads |
| `subnet/insignia/native_collateral.py` | Live validator gate + `FreezeLedger` |
| `subnet/neurons/validator.py` | Applies the gate in `finalize_generation` |
| `lockmgr/monitor.py` | Policy drift, missing visibility, floor shortfall, bond starvation |
| `risk/alerts.py` | `from_native_collateral` |
| `treasury/emissions.py` | `native_locked_fraction` on the R11 lever |
| `dashboards/investor_api/factsheet.py` | Locked α, lock_share, deployment bonds, cumulative burns |
| Defense registry | `NATIVE-COLLATERAL-GATE` (live-path only — the simulator does not model registration locks) |

Owner runbook (once the SDK/CLI expose the admin extrinsics — verify names on testnet):

1. Set `CollateralLockShare` / `CollateralDrainRatio` to the Insignia defaults (or a deliberate variant).
2. Publish `required_min_alpha` in miner docs if it is not zero.
3. Confirm `PairedValidator` is running with that floor and the 20% drawdown freeze.
4. Watch `native_collateral_*` monitor findings each epoch. Missing visibility is a warn, not a zero.
5. Factsheet reports native locked α next to deployment bonds and slash-settlement burns (SPEC §8).

---

## Legal

Miner-facing, not investor-facing — does not sit behind `LEGAL_SIGNOFF`. Registration-collateral terms (lock share, drain, freeze-on-zero-weight) and deployment-agreement slash terms still need Phase-0 counsel review as enforceable miner agreements. See INCENTIVE_MECHANISM.md §Deployment Collateral legal note.
