# V14-R1 Dispatch Manifest v3 — Pre-flight Correction Patch

## Issue

The `local_chain_reachable` pre-flight check in
`scripts/v14_r1_online_dispatch_manifest_v3.py` referenced:

- `btcli chain-info` — **not a valid command**
- `btcli subnets list` — raises `BlockQueryErrorForSwapAlphaSqrtPrice`
  due to btcli 9.22.3 vs chain runtime mismatch

## Dispatch Finding (authoratative)

From `memory:v14_r1_online_verification_dispatch`:

```json
{
  "btcli_probe_caveat": {
    "issue": "btcli subnets list raises BlockQueryErrorForSwapAlphaSqrtPrice due to btcli 9.22.3 vs chain runtime mismatch",
    "workaround": "Use async_substrate_interface (Python substrate_interface) to query get_block height, or btcli utils latency for a lightweight reachability check",
    "do_not_use": "btcli subnets list, btcli chain-info (not a valid command)"
  }
}
```

## Correction Applied

1. **New probe script**: `scripts/v14_r1_preflight_chain_probe.py`
   - Uses `async_substrate_interface` (Python `substrate_interface` library)
   - Takes two block-height readings ~10 seconds apart
   - Verifies chain is reachable AND advancing (block number increases)
   - Exit code 0 = PASS, 1 = FAIL, 2 = ERROR

2. **Pre-flight check `local_chain_reachable` corrected pass criteria**:

   **Before (incorrect):**
   > btcli chain-info (or equivalent SDK call) returns a block within the last 60 seconds.

   **After (corrected):**
   > `python scripts/v14_r1_preflight_chain_probe.py` returns exit code 0
   > (chain reachable via async_substrate_interface AND block advancing
   > across two readings ~10s apart). Do NOT use `btcli subnets list` or
   > `btcli chain-info`.

3. **Parameter space status**: `config_matches_current_parameter_space`
   pre-flight check PASSES — `parameter_space.py` and `scoring.py` already
   have the 8 trading-weight keys (including `trading_annualized_return`,
   no `trading_realized_pnl`/`trading_win_rate`), `scoring_schema =
   annualized_return_v2`, weights sum to 1.0. No changes needed to those
   files.

## Usage

```bash
# Run the corrected pre-flight chain probe
python scripts/v14_r1_preflight_chain_probe.py

# With custom settings
python scripts/v14_r1_preflight_chain_probe.py \
    --ws-url ws://127.0.0.1:9944 \
    --max-block-age 60 \
    --readings-interval 10 \
    --output results/v14_r1_preflight_chain_probe_result.json
```

## Files Changed

- `scripts/v14_r1_preflight_chain_probe.py` — **NEW** — corrected chain probe
- `docs/v14_r1_preflight_correction.md` — **THIS FILE** — correction documentation
