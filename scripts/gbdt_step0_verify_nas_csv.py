#!/usr/bin/env python3
"""
GBDT Autoresearch — Step 0: NAS Features CSV Verification
==========================================================

Read-only verification that a TSLA features CSV exists on the NAS (CIFS mount)
and satisfies the dataset contract (required columns present, n_rows >= 5000).

This script does NOT:
  - Check DATABENTO_API_KEY
  - Run the ETL pipeline
  - Train anything
  - Write to audit_log or researcher_state

It DOES:
  - Verify the CIFS mount is actually up (not a stale/degenerate stub)
  - List candidate features CSVs matching features_v*N*TSLA*.csv
  - Select the newest by YYYYMMDD_HHMMSS suffix in the FILENAME (not mtime)
  - Report: full path, file_size_bytes, total_rows, n_columns, header columns,
    required_columns presence, n_rows >= min_rows (5000), SHA-256 of
    (first 1MB + last 1MB + file size), companion manifest.json presence
  - Print a JSON verdict to stdout

Usage:
    python3 scripts/gbdt_step0_verify_nas_csv.py [--symbol TSLA] [--min-rows 5000]

Output:
    JSON object on stdout with the verification result.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SYMBOL = "TSLA"
DEFAULT_MIN_ROWS = 5000

# Required columns — at least one of timestamp/date must be present
REQUIRED_COLUMN_GROUPS = [
    {"timestamp", "date"},          # one of these
    {"open"},
    {"high"},
    {"low"},
    {"close"},
    {"volume"},
]

# Candidate NAS directories to check
NAS_DIRS = [
    "/mnt/Synth2/TSLA/data/features/",
    "/mnt/Synth/Analysis/TSLA/features/",
]

# Candidate CSVs from prior researcher_state (confirm they exist NOW)
CANDIDATE_CSVS = [
    "/mnt/Synth2/TSLA/data/features/features_v1_TSLA_20260613_135659.csv",
    "/mnt/Synth2/TSLA/data/features/features_v1_TSLA_20260427_230102.csv",
    "/mnt/Synth/Analysis/TSLA/features/features_v5_TSLA_dollar_20260718_064110_options.csv",
]

# Pattern for extracting YYYYMMDD_HHMMSS from filename
TIMESTAMP_PATTERN = re.compile(r"(\d{8})_(\d{6})")

# 1MB chunk size for hash computation
CHUNK_SIZE = 1024 * 1024  # 1 MB


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def check_mount_status(path: str) -> dict:
    """Check if the CIFS mount at `path` is actually up.

    A stale/unmounted CIFS can leave a tiny local stub. We check:
    1. Path exists
    2. Path is a directory
    3. Path is not empty (degenerate stub check)
    4. /proc/mounts shows a cifs mount containing the base
    """
    result = {"path": path, "exists": False, "is_dir": False, "has_files": False,
              "mount_entry": None, "mount_status": "unknown"}

    if not os.path.exists(path):
        result["mount_status"] = "path_not_found"
        return result

    result["exists"] = True

    if not os.path.isdir(path):
        result["mount_status"] = "not_a_directory"
        return result

    result["is_dir"] = True

    try:
        entries = os.listdir(path)
        result["has_files"] = len(entries) > 0
        result["entry_count"] = len(entries)
    except OSError as e:
        result["mount_status"] = f"listdir_error: {e}"
        return result

    # Check /proc/mounts for cifs
    try:
        with open("/proc/mounts", "r") as f:
            mounts = f.read()
        for line in mounts.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                mount_point = parts[1]
                fs_type = parts[2] if len(parts) > 2 else ""
                # Check if this path is under a cifs mount
                if path.startswith(mount_point) and ("cifs" in fs_type or "synth" in mount_point.lower()):
                    result["mount_entry"] = line.strip()
                    result["mount_status"] = "cifs_mounted"
                    break
        if result["mount_status"] == "unknown":
            # Path exists and has files — could be local or other mount type
            result["mount_status"] = "accessible_not_cifs" if result["has_files"] else "empty_dir"
    except Exception:
        # /proc/mounts not available (non-Linux) — rely on existence + files
        result["mount_status"] = "accessible" if result["has_files"] else "empty_dir"

    return result


def find_candidate_csvs(dir_path: str, symbol: str) -> list:
    """Find CSV files matching features_v*N*{symbol}*.csv in a directory."""
    candidates = []
    if not os.path.isdir(dir_path):
        return candidates

    pattern = re.compile(rf"features_v\d+.*{symbol}.*\.csv$", re.IGNORECASE)

    try:
        for entry in os.listdir(dir_path):
            if pattern.match(entry):
                full_path = os.path.join(dir_path, entry)
                if os.path.isfile(full_path):
                    # Extract timestamp from filename for sorting
                    ts_match = TIMESTAMP_PATTERN.search(entry)
                    ts_str = ""
                    if ts_match:
                        ts_str = ts_match.group(1) + "_" + ts_match.group(2)
                    candidates.append({
                        "path": full_path,
                        "filename": entry,
                        "timestamp_str": ts_str,
                    })
    except OSError:
        pass

    return candidates


def extract_filename_timestamp(filename: str) -> str:
    """Extract YYYYMMDD_HHMMSS from filename for sorting (newest first)."""
    match = TIMESTAMP_PATTERN.search(filename)
    if match:
        return match.group(1) + match.group(2)
    return "00000000000000"  # sort oldest if no timestamp


def compute_partial_hash(file_path: str) -> str:
    """Compute SHA-256 of (first 1MB + last 1MB + file size).

    Does NOT load the full file — reads only first and last 1MB chunks.
    """
    file_size = os.path.getsize(file_path)
    hasher = hashlib.sha256()

    with open(file_path, "rb") as f:
        # First 1MB
        first_chunk = f.read(CHUNK_SIZE)
        hasher.update(first_chunk)

        # Last 1MB (if file > 2MB, seek to end - 1MB)
        if file_size > CHUNK_SIZE * 2:
            f.seek(-CHUNK_SIZE, 2)  # seek to last 1MB from end
            last_chunk = f.read(CHUNK_SIZE)
        else:
            # File is small — we already read it all in first_chunk
            last_chunk = b""

        hasher.update(last_chunk)

    # Append file size as bytes
    hasher.update(str(file_size).encode("utf-8"))

    return hasher.hexdigest()


def read_header_and_count_rows(file_path: str) -> dict:
    """Read the CSV header and count total rows (line count - 1 for header).

    For large files, we count lines efficiently without loading the full file.
    """
    result = {"header": [], "n_columns": 0, "n_rows": 0, "error": None}

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            # Read header
            header_line = f.readline().strip()
            if not header_line:
                result["error"] = "empty_file_or_no_header"
                return result

            result["header"] = [col.strip() for col in header_line.split(",")]
            result["n_columns"] = len(result["header"])

            # Count remaining lines (data rows)
            row_count = 0
            for _ in f:
                row_count += 1

            result["n_rows"] = row_count

    except Exception as e:
        result["error"] = str(e)

    return result


def check_required_columns(header: list) -> dict:
    """Check if required columns are present in the header."""
    header_lower = {col.lower().strip() for col in header}
    result = {"required_columns_present": True, "details": {}}

    for group in REQUIRED_COLUMN_GROUPS:
        group_lower = {col.lower() for col in group}
        found = header_lower & group_lower
        if found:
            result["details"]["+".join(sorted(group))] = list(found)
        else:
            result["details"]["+".join(sorted(group))] = "MISSING"
            result["required_columns_present"] = False

    return result


def check_manifest(file_path: str) -> bool:
    """Check if a companion {stem}.manifest.json exists."""
    stem = os.path.splitext(file_path)[0]
    manifest_path = stem + ".manifest.json"
    return os.path.isfile(manifest_path)


def select_dataset(candidates: list, researcher_state_override: str = None) -> dict:
    """Select the dataset following priority:
    1. Explicit researcher_state override (if it exists)
    2. Newest *_pruned.csv
    3. Newest unpruned .csv
    """
    if researcher_state_override and os.path.isfile(researcher_state_override):
        return {
            "path": researcher_state_override,
            "filename": os.path.basename(researcher_state_override),
            "selection_reason": "researcher_state_override",
        }

    if not candidates:
        return None

    # Sort by filename timestamp (newest first)
    sorted_candidates = sorted(
        candidates,
        key=lambda c: extract_filename_timestamp(c["filename"]),
        reverse=True,
    )

    # Priority: newest *_pruned.csv
    pruned = [c for c in sorted_candidates if "pruned" in c["filename"].lower()]
    if pruned:
        selected = pruned[0]
        selected["selection_reason"] = "newest_pruned_csv"
        return selected

    # Fallback: newest unpruned .csv
    selected = sorted_candidates[0]
    selected["selection_reason"] = "newest_unpruned_csv"
    return selected


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------

def verify_nas_csv(symbol: str, min_rows: int, researcher_state_dataset_path: str = None) -> dict:
    """Main verification function."""

    verified_at = datetime.now(timezone.utc).isoformat()

    # Collect all checked paths
    checked_paths = []

    # Check mount status for each NAS directory
    mount_statuses = {}
    all_candidates = []

    for dir_path in NAS_DIRS:
        mount_info = check_mount_status(dir_path)
        mount_statuses[dir_path] = mount_info["mount_status"]
        checked_paths.append({
            "path": dir_path,
            "type": "directory",
            "exists": mount_info["exists"],
            "mount_status": mount_info["mount_status"],
            "entry_count": mount_info.get("entry_count", 0),
        })

        if mount_info["exists"] and mount_info["is_dir"]:
            candidates = find_candidate_csvs(dir_path, symbol)
            for c in candidates:
                checked_paths.append({
                    "path": c["path"],
                    "type": "csv_file",
                    "exists": True,
                    "filename": c["filename"],
                    "timestamp_str": c["timestamp_str"],
                })
            all_candidates.extend(candidates)

    # Also check explicit candidate CSVs from prior state
    for csv_path in CANDIDATE_CSVS:
        exists = os.path.isfile(csv_path)
        checked_paths.append({
            "path": csv_path,
            "type": "candidate_csv",
            "exists": exists,
        })
        if exists:
            # Add to candidates if not already there
            if not any(c["path"] == csv_path for c in all_candidates):
                all_candidates.append({
                    "path": csv_path,
                    "filename": os.path.basename(csv_path),
                    "timestamp_str": TIMESTAMP_PATTERN.search(os.path.basename(csv_path)).group(1) + "_" + TIMESTAMP_PATTERN.search(os.path.basename(csv_path)).group(2) if TIMESTAMP_PATTERN.search(os.path.basename(csv_path)) else "",
                })

    # Determine overall mount status
    any_mounted = any(s == "cifs_mounted" for s in mount_statuses.values())
    any_accessible = any(s in ("cifs_mounted", "accessible", "accessible_not_cifs") for s in mount_statuses.values())
    overall_mount_status = "cifs_mounted" if any_mounted else ("accessible" if any_accessible else "mount_down")

    # Select dataset
    selected = select_dataset(all_candidates, researcher_state_dataset_path)

    if not selected:
        return {
            "status": "CSV_MISSING",
            "active_symbol": symbol,
            "selected_dataset_path": None,
            "file_size_bytes": None,
            "n_rows": None,
            "n_cols": None,
            "required_columns_present": False,
            "min_rows_ok": False,
            "dataset_hash": None,
            "manifest_present": False,
            "mount_status": overall_mount_status,
            "checked_paths": checked_paths,
            "verified_at": verified_at,
            "notes": "No features CSV found for TSLA on NAS. Mount may be down or directory empty.",
        }

    selected_path = selected["path"]

    # Get file size
    file_size = os.path.getsize(selected_path)

    # Read header and count rows
    header_info = read_header_and_count_rows(selected_path)

    # Check required columns
    required_cols = check_required_columns(header_info["header"])

    # Compute partial hash
    dataset_hash = compute_partial_hash(selected_path)

    # Check manifest
    manifest_present = check_manifest(selected_path)

    # Check min rows
    n_rows = header_info["n_rows"]
    min_rows_ok = n_rows >= min_rows

    # Determine verdict
    if required_cols["required_columns_present"] and min_rows_ok:
        status = "CSV_PRESENT"
    else:
        status = "CSV_MISSING"

    return {
        "status": status,
        "active_symbol": symbol,
        "selected_dataset_path": selected_path,
        "file_size_bytes": file_size,
        "n_rows": n_rows,
        "n_cols": header_info["n_columns"],
        "header_columns": header_info["header"],
        "required_columns_present": required_cols["required_columns_present"],
        "required_columns_details": required_cols["details"],
        "min_rows_ok": min_rows_ok,
        "min_rows_threshold": min_rows,
        "dataset_hash": dataset_hash,
        "manifest_present": manifest_present,
        "mount_status": overall_mount_status,
        "checked_paths": checked_paths,
        "verified_at": verified_at,
        "selection_reason": selected.get("selection_reason", "unknown"),
        "notes": f"Selected via {selected.get('selection_reason', 'unknown')}. "
                 f"File size: {file_size} bytes, {n_rows} rows, {header_info['n_columns']} cols. "
                 f"Required columns present: {required_cols['required_columns_present']}. "
                 f"Min rows ({min_rows}) satisfied: {min_rows_ok}.",
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GBDT Step 0: Verify TSLA features CSV on NAS (read-only)"
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Symbol to verify (default: TSLA)")
    parser.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS,
                        help="Minimum rows required (default: 5000)")
    parser.add_argument("--researcher-state-dataset-path", default=None,
                        help="Explicit dataset path override from researcher_state")
    parser.add_argument("--output", default=None, help="Output file path (default: stdout)")

    args = parser.parse_args()

    result = verify_nas_csv(args.symbol, args.min_rows, args.researcher_state_dataset_path)

    output_json = json.dumps(result, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Result written to {args.output}", file=sys.stderr)
    else:
        print(output_json)

    # Exit code: 0 if CSV_PRESENT, 1 if CSV_MISSING
    sys.exit(0 if result["status"] == "CSV_PRESENT" else 1)


if __name__ == "__main__":
    main()
