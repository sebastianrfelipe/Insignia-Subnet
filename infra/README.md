# infra — keys, CI, deployment

Non-code scaffolding for the fund layer (SPEC §3). Nothing sensitive lives in
the repo; this directory holds runbooks and configuration templates only.

## Custody (Phase 0, blocking)

- Institutional-grade coldkey custody: multisig / proxy extrinsics, HSM or
  qualified custodian (SPEC §2 Phase 0).
- Desk authority over LP locks is a **limited proxy** covering `lock_stake` /
  `set_perpetual_lock` only — no transfer or unstake authority to fund
  addresses. Verify on testnet which subtensor proxy type gates the lock
  extrinsics (SPEC §10.1); if none does, fall back to fund-custodied cohort
  coldkeys under multisig and document the change for counsel.
- Owner hotkey changes must page (lockmgr/monitor.py); rotation is a custody
  ceremony, not an ops task.

## M2 testnet checklist (lockmgr)

1. Full §4 lifecycle on testnet: deliver → lock_stake → set_perpetual_lock →
   toggle → staged unstake, each verified via RPC (`get_coldkey_lock`,
   `get_hotkey_conviction`).
2. Param-change chaos test: sudo-change UnlockRate / ConvictionMaturityRate on
   a local chain; confirm monitor pages and schedules recompute.
3. Confirm extrinsic/storage names against subtensor PRs #2658 / #2687 / #2696 —
   chainio and lockmgr/locks.py carry the assumptions.

## CI

Fund-layer tests run from the repo root: `python -m pytest tests/`.
Subnet tests remain under `subnet/tests/`. Chart regeneration
(`python -m dashboards.charts`) must succeed in CI so docs/investor/ never
drifts from the mechanism code.
