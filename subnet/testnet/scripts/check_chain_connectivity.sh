#!/usr/bin/env bash
set -euo pipefail
NETWORK_FLAG="${1:-ws://127.0.0.1:9945}"
btcli subnet list --network "$NETWORK_FLAG"
