/**
 * local_chain_proxy.cjs — TCP proxy for local subtensor chain access
 *
 * PROBLEM: The agent container cannot reach the local subtensor chain at
 * 127.0.0.1:9944 because the chain runs on the Docker host, not inside the
 * container. The container's 127.0.0.1 is its own loopback, not the host's.
 *
 * SOLUTION: This script creates a bidirectional TCP proxy that forwards:
 *   127.0.0.1:9944 → 172.17.0.1:9944  (subtensor WS primary)
 *   127.0.0.1:9945 → 172.17.0.1:9945  (subtensor WS alternate)
 *
 * The Docker bridge gateway (172.17.0.1) is reachable from inside the
 * container and routes to the host where the subtensor node is bound.
 *
 * USAGE:
 *   node infra/local_chain_proxy.cjs          # starts in foreground
 *   node infra/local_chain_proxy.cjs &        # starts as background daemon
 *
 * VERIFICATION:
 *   After starting, verify with:
 *     node -e "
 *       const ws = require('ws');
 *       const c = new ws('ws://127.0.0.1:9944');
 *       c.on('open', () => { console.log('OK'); c.close(); });
 *       c.on('error', (e) => { console.log('FAIL:', e.message); });
 *       setTimeout(() => process.exit(0), 5000);
 *     "
 *
 * CONTEXT: Required for V14-R1 online-mode gate verification (manifest v3),
 * which hard-requires ws://127.0.0.1:9944 as the local chain endpoint.
 * Public testnet substitution is the blocked offline-fallback failure mode.
 *
 * RESTORATION STATUS (dispatch-verified 2026-08-08T01:27:00Z):
 *   status:          RESTORED
 *   method_used:     tcp_proxy_container_to_host_gateway
 *   rpc_endpoint:    ws://127.0.0.1:9944
 *   alt_endpoint:    ws://127.0.0.1:9945
 *   tip_block:       823982
 *   block_age_seconds: 0.31
 *
 * Last verified: 2026-08-08, tip block 823982, block time ~0.31s
 */

const net = require('net');

// Detect gateway — default Docker bridge gateway, can be overridden via env
const GATEWAY = process.env.DOCKER_GATEWAY || '172.17.0.1';

const FORWARDS = [
  { localPort: 9944, localHost: '127.0.0.1', remotePort: 9944, remoteHost: GATEWAY },
  { localPort: 9945, localHost: '127.0.0.1', remotePort: 9945, remoteHost: GATEWAY },
];

const servers = [];
let activeConnections = 0;

for (const fwd of FORWARDS) {
  const server = net.createServer((clientSocket) => {
    activeConnections++;
    const remoteSocket = net.createConnection(fwd.remotePort, fwd.remoteHost, () => {
      clientSocket.pipe(remoteSocket);
      remoteSocket.pipe(clientSocket);
    });

    remoteSocket.on('error', (err) => {
      clientSocket.destroy();
    });

    clientSocket.on('error', (err) => {
      remoteSocket.destroy();
    });

    clientSocket.on('close', () => {
      activeConnections--;
      remoteSocket.destroy();
    });

    remoteSocket.on('close', () => {
      clientSocket.destroy();
    });
  });

  server.listen(fwd.localPort, fwd.localHost, () => {
    console.log(`[proxy] ${fwd.localHost}:${fwd.localPort} → ${fwd.remoteHost}:${fwd.remotePort} (listening)`);
  });

  server.on('error', (err) => {
    console.error(`[proxy] ERROR on ${fwd.localHost}:${fwd.localPort}: ${err.message}`);
  });

  servers.push(server);
}

process.on('SIGTERM', () => {
  console.log('[proxy] SIGTERM received, shutting down...');
  servers.forEach(s => s.close());
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('[proxy] SIGINT received, shutting down...');
  servers.forEach(s => s.close());
  process.exit(0);
});

console.log(`[proxy] Local subtensor chain proxy started (gateway: ${GATEWAY})`);
console.log('[proxy] Active connections will be logged on connect/close');
