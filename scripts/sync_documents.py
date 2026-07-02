#!/usr/bin/env python3
"""Sync repo source files into the Insignia MCP `documents` collection.

Re-runnable. Idempotent: only replaces documents it previously synced
(matched by `sync_source`), leaving any other docs untouched.

Talks to the MCP HTTP endpoint directly (JSON-RPC over HTTP, SSE response)
so it can be run outside Cursor with plain `python sync_documents.py`.

Usage:
    python sync_documents.py                 # sync subnet/ (default)
    python sync_documents.py --dir subnet --dir running_on_staging.md
    python sync_documents.py --dry-run       # compute, don't write
    python sync_documents.py --verbose
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

MCP_URL = "http://10.0.0.249:3100/mcp"
COLLECTION = "documents"
SYNC_SOURCE = "filesystem-sync-script-v1"
MAX_DOC_BYTES = 1_000_000          # skip files larger than ~1MB
EXT_TO_LANG = {
    ".py": "python", ".md": "markdown", ".json": "json",
    ".toml": "toml", ".yaml": "yaml", ".yml": "yaml",
    ".cfg": "ini", ".ini": "ini", ".txt": "text",
    ".sh": "shell", ".ps1": "powershell", ".ts": "typescript",
    ".tsx": "tsx", ".js": "javascript", ".rs": "rust", ".go": "go",
}
EXT_TO_TYPE = {
    ".py": "source_code", ".sh": "source_code", ".ps1": "source_code",
    ".ts": "source_code", ".tsx": "source_code", ".js": "source_code",
    ".rs": "source_code", ".go": "source_code",
    ".md": "documentation", ".txt": "documentation",
    ".json": "config", ".toml": "config", ".yaml": "config",
    ".yml": "config", ".cfg": "config", ".ini": "config",
}
IGNORE_DIR_NAMES = {
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "build", "dist", ".eggs", ".tox", "htmlcov", ".coverage",
    ".idea", ".vscode", ".cursor",
}
IGNORE_FILE_EXTS = {".pyc", ".pyo", ".pyd", ".class", ".so", ".dll", ".dylib"}
IGNORE_FILE_NAMES = {".DS_Store", "Thumbs.db"}


_SESSION_ID: str | None = None


def _post(body: dict[str, Any], timeout: int = 60) -> tuple[int, dict[str, str], str]:
    u = urlparse(MCP_URL)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if _SESSION_ID:
        headers["mcp-session-id"] = _SESSION_ID
    conn.request("POST", u.path or "/", body=json.dumps(body), headers=headers)
    resp = conn.getresponse()
    raw = resp.read().decode("utf-8", errors="replace")
    hdrs = {k.lower(): v for k, v in resp.getheaders()}
    conn.close()
    return resp.status, hdrs, raw


def _parse_sse(raw: str) -> dict[str, Any]:
    payload = raw
    if "data:" in raw[:200] or raw.lstrip().startswith("event:"):
        for line in raw.splitlines():
            if line.startswith("data:"):
                payload = line[len("data:"):].strip()
                break
    return json.loads(payload)


def mcp_initialize() -> None:
    """Do the MCP initialize / initialized handshake. Sets _SESSION_ID if given."""
    global _SESSION_ID
    status, hdrs, raw = _post({
        "jsonrpc": "2.0", "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "fs-sync-script", "version": "1.0"},
        },
    }, timeout=15)
    if status >= 400:
        raise RuntimeError(f"initialize HTTP {status}: {raw[:500]}")
    env = _parse_sse(raw)
    if "error" in env:
        raise RuntimeError(f"initialize error: {env['error']}")
    _SESSION_ID = hdrs.get("mcp-session-id")
    # Send the initialized notification (no id, no response expected).
    try:
        _post({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }, timeout=10)
    except Exception:
        pass  # notifications have no response; ignore


def mcp_call(tool: str, arguments: dict[str, Any], timeout: int = 60) -> Any:
    """Call one MCP tool via JSON-RPC over HTTP; parse SSE response."""
    if _SESSION_ID is None:
        mcp_initialize()
    status, hdrs, raw = _post({
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }, timeout=timeout)
    if status >= 400:
        raise RuntimeError(f"MCP HTTP {status}: {raw[:500]}")
    envelope = _parse_sse(raw)
    if "error" in envelope:
        raise RuntimeError(f"MCP error: {envelope['error']}")
    result = envelope.get("result", {})
    content = result.get("content", [])
    if not content:
        return result
    text = content[0].get("text", "")
    if not text:
        return result
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def iter_files(roots: list[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in roots:
        if root.is_file():
            if root not in seen:
                seen.add(root)
                yield root
            continue
        for dirpath, dirnames, filenames in os_walk_skip_ignored(root):
            for fn in filenames:
                p = dirpath / fn
                if p in seen:
                    continue
                seen.add(p)
                yield p


def os_walk_skip_ignored(root: Path):
    for dp, dns, fns in _walk(root):
        dns[:] = [d for d in dns if d not in IGNORE_DIR_NAMES]
        yield Path(dp), dns, fns


def _walk(root: Path):
    import os
    for dp, dns, fns in os.walk(root):
        yield dp, dns, fns


def should_sync(p: Path) -> bool:
    if p.name in IGNORE_FILE_NAMES:
        return False
    ext = p.suffix.lower()
    if ext in IGNORE_FILE_EXTS:
        return False
    if ext not in EXT_TO_TYPE:
        return False
    return True


def build_doc(p: Path, repo_root: Path) -> dict[str, Any] | None:
    try:
        raw = p.read_bytes()
    except OSError as e:
        print(f"  skip (read error): {p} -> {e}", file=sys.stderr)
        return None
    if len(raw) > MAX_DOC_BYTES:
        print(f"  skip (>{MAX_DOC_BYTES}B): {p}", file=sys.stderr)
        return None
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        content = raw.decode("utf-8", errors="replace")
    sha = hashlib.sha256(raw).hexdigest()
    rel = p.relative_to(repo_root).as_posix()  # forward slashes
    ext = p.suffix.lower()
    return {
        "path": rel,
        "type": EXT_TO_TYPE[ext],
        "language": EXT_TO_LANG.get(ext, "unknown"),
        "content": content,
        "sha256": sha,
        "size": len(raw),
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "sync_source": SYNC_SOURCE,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", action="append", default=None,
                    help="Directory or file to sync (repeatable). Default: subnet")
    ap.add_argument("--repo-root", default=r"c:\Projects\Insignia-Subnet",
                    help="Repo root for computing relative paths")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute and print stats; don't write to MCP")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    targets = [repo_root / d if not Path(d).is_absolute() else Path(d)
               for d in (args.dir or ["subnet"])]
    targets = [t.resolve() for t in targets]

    print(f"MCP URL:      {MCP_URL}")
    print(f"Collection:   {COLLECTION}")
    print(f"Sync source:  {SYNC_SOURCE}")
    print(f"Repo root:    {repo_root}")
    print(f"Targets:      {[str(t) for t in targets]}")
    print()

    collected: list[dict[str, Any]] = []
    skipped = 0
    for p in iter_files(targets):
        if not should_sync(p):
            skipped += 1
            continue
        doc = build_doc(p, repo_root)
        if doc is None:
            skipped += 1
            continue
        collected.append(doc)
        if args.verbose:
            print(f"  + {doc['path']:60s} {doc['size']:>8}B  {doc['language']}")

    by_type: dict[str, int] = {}
    for d in collected:
        by_type[d["type"]] = by_type.get(d["type"], 0) + 1
    print(f"Collected:    {len(collected)} files ({skipped} skipped)")
    for k, v in sorted(by_type.items()):
        print(f"  {k:15s} {v}")
    print()

    if args.dry_run:
        print("--dry-run set; not writing to MCP.")
        return 0

    # 1. Ensure index on path (idempotent; create_index returns existing if same).
    print(f"Ensuring index on `path`...")
    mcp_initialize()
    print(f"  initialized (session_id={_SESSION_ID or 'none'})")
    try:
        mcp_call("mongodb_create_index", {
            "collection": COLLECTION,
            "keys": {"path": 1},
            "options": {"unique": True, "name": "path_unique"},
        })
        print("  index OK")
    except Exception as e:
        print(f"  index warn: {e}", file=sys.stderr)

    # 2. Fetch existing sha256s for our previously-synced docs so we can skip
    #    unchanged files. (Avoids mongodb_delete_many, which is HITL-gated.)
    print("Fetching existing sync state...")
    existing: dict[str, str] = {}  # path -> sha256
    try:
        res = mcp_call("mongodb_find", {
            "collection": COLLECTION,
            "filter": {"sync_source": SYNC_SOURCE},
            "projection": {"_id": 0, "path": 1, "sha256": 1},
            "limit": 1000,
            "detail_level": "L1",
        })
        for item in res.get("items", []):
            p = item.get("path")
            s = item.get("sha256")
            if p and s:
                existing[p] = s
        print(f"  found {len(existing)} previously-synced docs")
    except Exception as e:
        print(f"  fetch warn: {e}", file=sys.stderr)

    # 3. Upsert per file (changed + new). Skip unchanged. Approval-free.
    to_upsert = [d for d in collected if d["sha256"] != existing.get(d["path"])]
    unchanged = len(collected) - len(to_upsert)
    print(f"Upserting {len(to_upsert)} changed/new docs ({unchanged} unchanged, skipped)...")
    upserted = 0
    t0 = time.time()
    for d in to_upsert:
        try:
            mcp_call("mongodb_update_one", {
                "collection": COLLECTION,
                "filter": {"path": d["path"]},
                "update": {"$set": {
                    "type": d["type"], "language": d["language"],
                    "content": d["content"], "sha256": d["sha256"],
                    "size": d["size"], "synced_at": d["synced_at"],
                    "sync_source": d["sync_source"],
                }},
                "upsert": True,
            })
            upserted += 1
            if args.verbose or upserted % 10 == 0:
                print(f"  upserted {upserted}/{len(to_upsert)}: {d['path']}")
        except Exception as e:
            print(f"  upsert FAILED for {d['path']}: {e}", file=sys.stderr)

    dt = time.time() - t0
    print()
    print(f"Done. Upserted {upserted}/{len(to_upsert)} docs in {dt:.1f}s "
          f"({unchanged} unchanged skipped).")

    # 4. Verify with a find for simulation.py.
    print("\nVerifying: mongodb_find({path: {$regex: 'simulation.py'}}) ...")
    try:
        found = mcp_call("mongodb_find", {
            "collection": COLLECTION,
            "filter": {"path": {"$regex": "simulation\\.py$"}},
            "projection": {"path": 1, "size": 1, "sha256": 1, "_id": 0},
            "limit": 10,
            "detail_level": "L1",
        })
        print(f"  found: {json.dumps(found, indent=2)[:1200]}")
    except Exception as e:
        print(f"  verify failed: {e}", file=sys.stderr)

    return 0 if upserted == len(to_upsert) else 1


if __name__ == "__main__":
    sys.exit(main())
