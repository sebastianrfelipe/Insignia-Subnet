"""
Prometheus Metrics Exporter

Instruments the simulation and optimization loop with metrics exposed
on an HTTP endpoint for Prometheus scraping. Grafana dashboards
visualize the tuning progress in real-time.

Metrics categories:
  - Simulation: per-miner scores, promotions, feedback
  - Attack detection: breach flags and severity per attack type
  - Optimizer: generation counter, fitness values, Pareto front
"""

from __future__ import annotations

import threading
import logging
from typing import Dict, List, Optional, Any
from http.server import HTTPServer, BaseHTTPRequestHandler

logger = logging.getLogger("metrics")

# In-memory metric storage (avoids hard dependency on prometheus_client
# while keeping the same exposition format for Prometheus scraping)


class MetricValue:
    __slots__ = ("name", "help_text", "type", "labels", "value")

    def __init__(self, name: str, help_text: str, mtype: str):
        self.name = name
        self.help_text = help_text
        self.type = mtype
        self.labels: Dict[str, Dict[str, float]] = {}

    def set(self, value: float, **labels):
        key = _label_key(labels)
        self.labels[key] = {"value": value, "labels": labels}

    def inc(self, amount: float = 1.0, **labels):
        key = _label_key(labels)
        if key not in self.labels:
            self.labels[key] = {"value": 0.0, "labels": labels}
        self.labels[key]["value"] += amount


def _label_key(labels: Dict[str, Any]) -> str:
    return ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))


class MetricsRegistry:
    """Thread-safe registry for all metrics."""

    def __init__(self):
        self._metrics: Dict[str, MetricValue] = {}
        self._lock = threading.Lock()

    def gauge(self, name: str, help_text: str = "") -> MetricValue:
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = MetricValue(name, help_text, "gauge")
            return self._metrics[name]

    def counter(self, name: str, help_text: str = "") -> MetricValue:
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = MetricValue(name, help_text, "counter")
            return self._metrics[name]

    def exposition(self) -> str:
        """Generate Prometheus text exposition format."""
        lines = []
        with self._lock:
            for metric in self._metrics.values():
                lines.append(f"# HELP {metric.name} {metric.help_text}")
                lines.append(f"# TYPE {metric.name} {metric.type}")
                for entry in metric.labels.values():
                    labels = entry["labels"]
                    value = entry["value"]
                    if labels:
                        label_str = "{" + ",".join(
                            f'{k}="{v}"' for k, v in sorted(labels.items())
                        ) + "}"
                    else:
                        label_str = ""
                    lines.append(f"{metric.name}{label_str} {value}")
        return "\n".join(lines) + "\n"


REGISTRY = MetricsRegistry()

# Pre-defined metrics
L1_SCORE = REGISTRY.gauge("insignia_l1_composite_score", "L1 miner composite score")
L1_WEIGHT = REGISTRY.gauge("insignia_l1_consensus_weight", "L1 miner consensus weight")
L2_SCORE = REGISTRY.gauge("insignia_l2_composite_score", "L2 strategy composite score")
L2_PNL = REGISTRY.gauge("insignia_l2_realized_pnl", "L2 strategy realized P&L")
L2_DRAWDOWN = REGISTRY.gauge("insignia_l2_max_drawdown", "L2 strategy max drawdown")
PROMOTION_COUNT = REGISTRY.gauge("insignia_promotion_active_count", "Active models in L2 pool")
FEEDBACK_ADJ = REGISTRY.gauge("insignia_feedback_adjustment", "Cross-layer feedback adjustment")

ATTACK_BREACH = REGISTRY.gauge("insignia_attack_breach", "Attack breach flag (0/1)")
ATTACK_SEVERITY = REGISTRY.gauge("insignia_attack_severity", "Attack severity score 0-1")
TOTAL_BREACHES = REGISTRY.gauge("insignia_total_breaches", "Total active attack breaches")

GEN_COUNTER = REGISTRY.counter("insignia_optimizer_generation", "Current evolutionary generation")
BEST_FITNESS = REGISTRY.gauge("insignia_best_fitness", "Best fitness value per objective")
PARETO_SIZE = REGISTRY.gauge("insignia_pareto_front_size", "Number of solutions on Pareto front")
POP_DIVERSITY = REGISTRY.gauge("insignia_population_diversity", "Population genetic diversity")


def export_simulation_metrics(
    sim_result: Any,
    breach_report: Any,
    generation: int = 0,
    individual: int = 0,
):
    """Export metrics from a simulation result and breach report."""
    gen_label = str(generation)
    ind_label = str(individual)

    for uid, score in sim_result.miner_scores.items():
        mtype = sim_result.miner_types.get(uid, "unknown")
        L1_SCORE.set(score, miner=uid, agent_type=mtype, generation=gen_label)

    for uid, score in sim_result.l2_scores.items():
        ltype = sim_result.l2_types.get(uid, "unknown")
        L2_SCORE.set(score, miner=uid, agent_type=ltype, generation=gen_label)

    if sim_result.promotion_summary:
        PROMOTION_COUNT.set(
            sim_result.promotion_summary.get("active_models", 0),
            generation=gen_label,
        )

    for mid, adj in sim_result.l1_feedback.items():
        FEEDBACK_ADJ.set(adj, model=mid, generation=gen_label)

    if breach_report:
        for b in breach_report.breaches:
            ATTACK_BREACH.set(float(b.breached), attack=b.attack_name, generation=gen_label)
            ATTACK_SEVERITY.set(b.severity, attack=b.attack_name, generation=gen_label)
        TOTAL_BREACHES.set(float(breach_report.n_breached), generation=gen_label)


def export_optimizer_metrics(
    generation: int,
    best_fitness: Dict[str, float],
    pareto_size: int,
    diversity: float,
):
    """Export optimizer state metrics."""
    GEN_COUNTER.set(float(generation))
    for obj_name, val in best_fitness.items():
        BEST_FITNESS.set(val, objective=obj_name)
    PARETO_SIZE.set(float(pareto_size))
    POP_DIVERSITY.set(diversity)


# ---------------------------------------------------------------------------
# HTTP Server for Prometheus scraping
# ---------------------------------------------------------------------------

class _MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            body = REGISTRY.exposition().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(200)
            body = b"Insignia Metrics Exporter. Visit /metrics for Prometheus data.\n"
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, format, *args):
        pass


_server: Optional[HTTPServer] = None
_server_thread: Optional[threading.Thread] = None


def start_metrics_server(port: int = 8000):
    """Start the metrics HTTP server in a background thread."""
    global _server, _server_thread
    if _server is not None:
        return

    _server = HTTPServer(("0.0.0.0", port), _MetricsHandler)
    _server_thread = threading.Thread(target=_server.serve_forever, daemon=True)
    _server_thread.start()
    logger.info("Metrics server started on :%d/metrics", port)


def stop_metrics_server():
    global _server, _server_thread
    if _server:
        _server.shutdown()
        _server = None
        _server_thread = None


if __name__ == "__main__":
    import time

    ATTACK_BREACH.set(1.0, attack="overfitting", generation="0")
    ATTACK_SEVERITY.set(0.35, attack="overfitting", generation="0")
    L1_SCORE.set(0.72, miner="honest_0", agent_type="honest", generation="0")
    L1_SCORE.set(0.65, miner="overfitter_0", agent_type="overfitter", generation="0")
    GEN_COUNTER.set(5.0)

    start_metrics_server(8000)
    print("Metrics server running on http://localhost:8000/metrics")
    print("Press Ctrl+C to stop")
    print("\nSample exposition:\n")
    print(REGISTRY.exposition())

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_metrics_server()
