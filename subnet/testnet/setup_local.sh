#!/usr/bin/env bash
###############################################################################
# Insignia Local Testnet Setup Script
#
# Automates the full local testnet deployment:
#   1. Start local subtensor via Docker
#   2. Install Python dependencies
#   3. Create wallets and fund from alice
#   4. Create subnet and register neurons
#   5. Run the emulator
#
# Usage:
#   cd subnet && bash testnet/setup_local.sh
#   cd subnet && bash testnet/setup_local.sh --skip-docker   # if chain is running
#   cd subnet && bash testnet/setup_local.sh --mode evolve   # evolutionary tuning
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBNET_DIR="$(dirname "$SCRIPT_DIR")"
SKIP_DOCKER=false
EMULATOR_MODE="single"
GENERATIONS=10
POPULATION=10
OUTPUT_DIR="testnet_results"

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-docker) SKIP_DOCKER=true; shift ;;
        --mode) EMULATOR_MODE="$2"; shift 2 ;;
        --generations) GENERATIONS="$2"; shift 2 ;;
        --population) POPULATION="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "================================================================"
echo "  Insignia Local Testnet Setup"
echo "  Mode: ${EMULATOR_MODE}"
echo "================================================================"

# Step 1: Start local subtensor
if [ "$SKIP_DOCKER" = false ]; then
    echo ""
    echo "[1/5] Starting local subtensor chain..."
    if command -v docker &> /dev/null; then
        if docker ps --format '{{.Names}}' | grep -q 'insignia-subtensor'; then
            echo "  Local subtensor already running"
        else
            cd "$SCRIPT_DIR"
            docker compose -f docker-compose.testnet.yml up -d subtensor
            echo "  Waiting for subtensor to be ready..."
            sleep 10

            for i in $(seq 1 30); do
                if curl -sf http://localhost:9933/health > /dev/null 2>&1; then
                    echo "  Subtensor ready at ws://127.0.0.1:9945"
                    break
                fi
                sleep 2
            done
        fi
    else
        echo "  Docker not available — skipping subtensor startup"
        echo "  Ensure subtensor is running at ws://127.0.0.1:9945"
    fi
else
    echo "[1/5] Skipping Docker (--skip-docker)"
fi

# Step 2: Install dependencies
echo ""
echo "[2/5] Checking Python dependencies..."
cd "$SUBNET_DIR"

if ! python3 -c "import numpy, pandas, sklearn, joblib" 2>/dev/null; then
    echo "  Installing dependencies..."
    pip install -q numpy pandas scikit-learn joblib pymoo pyyaml 2>/dev/null || \
        pip3 install -q numpy pandas scikit-learn joblib pymoo pyyaml
fi

if ! command -v btcli &> /dev/null; then
    echo "  btcli not found — installing bittensor-cli..."
    pip install -q bittensor-cli 2>/dev/null || \
        pip3 install -q bittensor-cli || \
        echo "  WARNING: Could not install btcli. Emulator will run in offline mode."
fi

echo "  Dependencies OK"

# Step 3: Run the emulator
echo ""
echo "[3/5] Starting emulator in '${EMULATOR_MODE}' mode..."

EMULATOR_ARGS=(
    --mode "$EMULATOR_MODE"
    --network local
    --output "$OUTPUT_DIR"
    --no-metrics
)

if [ "$EMULATOR_MODE" = "evolve" ] || [ "$EMULATOR_MODE" = "full" ]; then
    EMULATOR_ARGS+=(--generations "$GENERATIONS" --population "$POPULATION")
fi

cd "$SUBNET_DIR"
python3 -m testnet.run_emulator "${EMULATOR_ARGS[@]}"

echo ""
echo "================================================================"
echo "  Emulator complete!"
echo "  Results: ${OUTPUT_DIR}/"
echo "================================================================"

# Step 4: Optional — start monitoring stack
if [ "$SKIP_DOCKER" = false ] && command -v docker &> /dev/null; then
    echo ""
    read -p "Start Prometheus + Grafana monitoring? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "$SCRIPT_DIR"
        docker compose -f docker-compose.testnet.yml up -d prometheus grafana
        echo "  Prometheus: http://localhost:9090"
        echo "  Grafana:    http://localhost:3000 (admin/admin)"
    fi
fi

echo ""
echo "Done. See ${OUTPUT_DIR}/ for results."
