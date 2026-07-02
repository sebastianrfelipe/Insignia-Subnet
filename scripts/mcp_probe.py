"""Probe the Insignia MCP HTTP endpoint to learn its wire protocol."""
import http.client, json, sys
from urllib.parse import urlparse

URL = "http://10.0.0.249:3100/mcp"
u = urlparse(URL)

def post(body, headers=None, timeout=15):
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if headers:
        h.update(headers)
    conn.request("POST", u.path or "/", body=json.dumps(body), headers=h)
    resp = conn.getresponse()
    raw = resp.read().decode("utf-8", errors="replace")
    conn.close()
    return resp.status, dict(resp.getheaders()), raw

def show(label, status, hdrs, raw, max=2000):
    print(f"\n=== {label} ===")
    print(f"status: {status}")
    print(f"content-type: {hdrs.get('content-type')}")
    print(f"raw ({len(raw)} chars):")
    print(raw[:max])
    if len(raw) > max:
        print(f"... ({len(raw)-max} more chars)")

# 1. Bare initialize
s, h, r = post({
    "jsonrpc": "2.0", "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "fs-sync-probe", "version": "0.1"}
    }
})
show("initialize (no session)", s, h, r)

# If there's a session header, capture it for the next call
session_id = h.get("mcp-session-id") or h.get("Mcp-Session-Id")

# 2. tools/list
hdrs = {}
if session_id:
    hdrs["mcp-session-id"] = session_id
s, h, r = post({
    "jsonrpc": "2.0", "id": 2,
    "method": "tools/list",
    "params": {}
}, headers=hdrs)
show("tools/list", s, h, r, max=3000)

# 3. tools/call mongodb_list_collections
s, h, r = post({
    "jsonrpc": "2.0", "id": 3,
    "method": "tools/call",
    "params": {"name": "mongodb_list_collections", "arguments": {}}
}, headers=hdrs)
show("tools/call mongodb_list_collections", s, h, r, max=3000)
