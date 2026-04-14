#!/usr/bin/env bash
set -euo pipefail
NETWORK_FLAG="${1:-ws://127.0.0.1:9945}"
wallets=(insignia-owner insignia-validator)
for i in $(seq 0 11); do
  wallets+=("insignia-miner-${i}")
done
for wallet in "${wallets[@]}"; do
  echo "=== ${wallet} ==="
  btcli wallet balance --wallet.name "$wallet" --network "$NETWORK_FLAG" || true
  echo
 done
