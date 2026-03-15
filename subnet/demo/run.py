"""Launch miner and validator servers.

Usage:
    uv run python -m demo.run
    uv run python demo/run.py
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import uvicorn

# Allow running from subnet/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def start_miner() -> None:
    uvicorn.run("demo.miner:app", host="0.0.0.0", port=8001, log_level="info")


def start_validator() -> None:
    uvicorn.run("demo.validator:app", host="0.0.0.0", port=8000, log_level="info")


def main() -> None:
    print("Starting Insignia subnet demo...")
    print("  Miner:     http://localhost:8001")
    print("  Validator:  http://localhost:8000")
    print("  Dashboard:  http://localhost:8000/dashboard")
    print()

    miner_thread = threading.Thread(target=start_miner, daemon=True)
    miner_thread.start()

    # Run validator in main thread (so Ctrl+C works)
    start_validator()


if __name__ == "__main__":
    main()
