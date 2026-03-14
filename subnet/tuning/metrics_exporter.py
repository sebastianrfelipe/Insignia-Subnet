"""
Prometheus Metrics Exporter

Instruments the Insignia subnet with metrics exposed on an HTTP endpoint
for Prometheus scraping. Two categories of metrics:

1. Bittensor Subnet Health (modeled after grafana.bittensor.church):
   - Metagraph neuron metrics: stake, rank, trust, consensus, incentive,
     dividends, emission, vtrust, updated blocks
   - Subnet hyperparameters: tempo, immunity_period, weights limits, bonds_moving_avg
   - Multi-mechanism awareness: per-mechid emission splits, weight matrices
   - Validator operational health: weight-setting recency, registration events

2. Insignia-Specific Metrics:
   - L1/L2 composite scores with per-metric breakdown (penalized_f1, etc.)
   - Cross-layer promotion and feedback
   - Attack detection breach flags and severity
   - Optimizer progress (generation, Pareto front, diversity)
"""

from __future__ import annotations

import threading
import logging
from typing import Dict, List, Optional, Any
from http.server import HTTPServer, BaseHTTPRequestHandler

logger = logging.getLogger("metrics")


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

# =========================================================================
# BITTENSOR SUBNET HEALTH METRICS (modeled after grafana.bittensor.church)
# =========================================================================

# --- Metagraph Neuron Metrics (per-UID) ---
BT_STAKE = REGISTRY.gauge("bt_neuron_stake_tao", "TAO staked on neuron")
BT_RANK = REGISTRY.gauge("bt_neuron_rank", "Neuron rank score")
BT_TRUST = REGISTRY.gauge("bt_neuron_trust", "Trust score assigned by other neurons")
BT_CONSENSUS = REGISTRY.gauge("bt_neuron_consensus", "Consensus score")
BT_INCENTIVE = REGISTRY.gauge("bt_neuron_incentive", "Incentive score (miner emission share)")
BT_DIVIDENDS = REGISTRY.gauge("bt_neuron_dividends", "Dividends earned")
BT_EMISSION = REGISTRY.gauge("bt_neuron_emission_rho", "Emission in rho per epoch")
BT_VTRUST = REGISTRY.gauge("bt_neuron_vtrust", "Validator trust score")
BT_UPDATED = REGISTRY.gauge("bt_neuron_updated_blocks", "Blocks since last weight update")
BT_ACTIVE = REGISTRY.gauge("bt_neuron_active", "Neuron active status (1=active, 0=inactive)")
BT_IS_VALIDATOR = REGISTRY.gauge("bt_neuron_is_validator", "Whether neuron is a validator (1/0)")
BT_IS_IMMUNITY = REGISTRY.gauge("bt_neuron_is_immune", "Whether neuron is in immunity period (1/0)")

# --- Subnet-Level Aggregate Metrics ---
BT_SUBNET_N = REGISTRY.gauge("bt_subnet_neuron_count", "Total neurons registered on subnet")
BT_SUBNET_VALIDATORS = REGISTRY.gauge("bt_subnet_validator_count", "Active validators")
BT_SUBNET_MINERS = REGISTRY.gauge("bt_subnet_miner_count", "Active miners")
BT_SUBNET_TOTAL_STAKE = REGISTRY.gauge("bt_subnet_total_stake_tao", "Total TAO staked on subnet")
BT_SUBNET_EMISSION_RATE = REGISTRY.gauge("bt_subnet_emission_rate", "Subnet emission rate per epoch")
BT_SUBNET_REGISTRATION_COST = REGISTRY.gauge("bt_subnet_registration_cost_tao", "Current cost to register on subnet")
BT_SUBNET_DIFFICULTY = REGISTRY.gauge("bt_subnet_difficulty", "Current registration difficulty")

# --- Subnet Hyperparameters ---
BT_HP_TEMPO = REGISTRY.gauge("bt_hyperparameter_tempo", "Epoch duration in blocks")
BT_HP_IMMUNITY = REGISTRY.gauge("bt_hyperparameter_immunity_period", "Immunity period in blocks")
BT_HP_MIN_WEIGHTS = REGISTRY.gauge("bt_hyperparameter_min_allowed_weights", "Min weights a validator must set")
BT_HP_MAX_WEIGHT = REGISTRY.gauge("bt_hyperparameter_max_weight_limit", "Max weight per UID")
BT_HP_BONDS_MA = REGISTRY.gauge("bt_hyperparameter_bonds_moving_avg", "Bonds EMA decay factor")
BT_HP_KAPPA = REGISTRY.gauge("bt_hyperparameter_kappa", "Consensus majority threshold")
BT_HP_MAX_REGS = REGISTRY.gauge("bt_hyperparameter_max_regs_per_block", "Max registrations per block")
BT_HP_SERVING_RATE = REGISTRY.gauge("bt_hyperparameter_serving_rate_limit", "Axon serving rate limit")
BT_HP_ADJUSTMENT_ALPHA = REGISTRY.gauge("bt_hyperparameter_adjustment_alpha", "Difficulty adjustment alpha")
BT_HP_COMMIT_REVEAL = REGISTRY.gauge("bt_hyperparameter_commit_reveal_enabled", "Commit-reveal weight setting (1/0)")
BT_HP_LIQUID_ALPHA = REGISTRY.gauge("bt_hyperparameter_liquid_alpha_enabled", "Liquid alpha enabled (1/0)")
BT_HP_ALPHA_HIGH = REGISTRY.gauge("bt_hyperparameter_alpha_high", "Liquid alpha upper bound")
BT_HP_ALPHA_LOW = REGISTRY.gauge("bt_hyperparameter_alpha_low", "Liquid alpha lower bound")

# --- Multi-Mechanism Metrics (Insignia uses 2 mechanisms: L1 + L2) ---
BT_MECH_COUNT = REGISTRY.gauge("bt_mechanism_count", "Number of incentive mechanisms on subnet")
BT_MECH_EMISSION_SPLIT = REGISTRY.gauge("bt_mechanism_emission_split", "Emission share per mechanism")
BT_MECH_INCENTIVE = REGISTRY.gauge("bt_mechanism_neuron_incentive", "Per-mechanism incentive score")
BT_MECH_RANK = REGISTRY.gauge("bt_mechanism_neuron_rank", "Per-mechanism rank score")
BT_MECH_CONSENSUS = REGISTRY.gauge("bt_mechanism_neuron_consensus", "Per-mechanism consensus score")
BT_MECH_DIVIDENDS = REGISTRY.gauge("bt_mechanism_neuron_dividends", "Per-mechanism dividends")

# --- Validator Operational Health ---
BT_VAL_WEIGHT_SETTING_LAG = REGISTRY.gauge("bt_validator_weight_setting_lag_blocks", "Blocks since last weight commit")
BT_VAL_WEIGHT_SETTING_OK = REGISTRY.gauge("bt_validator_weight_setting_healthy", "1 if weights set recently, 0 if stale")
BT_VAL_VTRUST_TREND = REGISTRY.gauge("bt_validator_vtrust_ema", "EMA of validator trust over time")

# --- Registration / Deregistration Events ---
BT_REGISTRATIONS = REGISTRY.counter("bt_registration_events_total", "Total registration events observed")
BT_DEREGISTRATIONS = REGISTRY.counter("bt_deregistration_events_total", "Total deregistration events observed")

# =========================================================================
# INSIGNIA-SPECIFIC METRICS
# =========================================================================

# --- L1 Scoring (per-miner, with metric breakdown) ---
L1_SCORE = REGISTRY.gauge("insignia_l1_composite_score", "L1 miner composite score")
L1_PENALIZED_F1 = REGISTRY.gauge("insignia_l1_penalized_f1", "L1 penalized F1 (mean - lambda*std)")
L1_MEAN_F1 = REGISTRY.gauge("insignia_l1_mean_f1", "L1 mean F1 across rolling windows")
L1_STD_F1 = REGISTRY.gauge("insignia_l1_std_f1", "L1 std F1 across rolling windows")
L1_PENALIZED_SHARPE = REGISTRY.gauge("insignia_l1_penalized_sharpe", "L1 penalized Sharpe (mean - lambda*std)")
L1_MEAN_SHARPE = REGISTRY.gauge("insignia_l1_mean_sharpe", "L1 mean Sharpe across rolling windows")
L1_STD_SHARPE = REGISTRY.gauge("insignia_l1_std_sharpe", "L1 std Sharpe across rolling windows")
L1_MAX_DD = REGISTRY.gauge("insignia_l1_max_drawdown", "L1 max drawdown score")
L1_GEN_GAP = REGISTRY.gauge("insignia_l1_generalization_gap", "L1 generalization gap |train_f1 - val_f1|")
L1_FEAT_EFF = REGISTRY.gauge("insignia_l1_feature_efficiency", "L1 feature efficiency score")
L1_LATENCY = REGISTRY.gauge("insignia_l1_latency_score", "L1 latency score")
L1_WEIGHT = REGISTRY.gauge("insignia_l1_consensus_weight", "L1 miner consensus weight")

# --- L2 Scoring ---
L2_SCORE = REGISTRY.gauge("insignia_l2_composite_score", "L2 strategy composite score")
L2_PNL = REGISTRY.gauge("insignia_l2_realized_pnl", "L2 strategy realized P&L")
L2_DRAWDOWN = REGISTRY.gauge("insignia_l2_max_drawdown", "L2 strategy max drawdown")
L2_OMEGA = REGISTRY.gauge("insignia_l2_omega_ratio", "L2 strategy Omega ratio")
L2_WIN_RATE = REGISTRY.gauge("insignia_l2_win_rate", "L2 strategy win rate")
L2_TRADE_COUNT = REGISTRY.gauge("insignia_l2_trade_count", "L2 total trades executed")

# --- Cross-Layer ---
PROMOTION_COUNT = REGISTRY.gauge("insignia_promotion_active_count", "Active models in L2 pool")
PROMOTION_TOTAL = REGISTRY.counter("insignia_promotion_events_total", "Total models ever promoted")
FEEDBACK_ADJ = REGISTRY.gauge("insignia_feedback_adjustment", "Cross-layer feedback adjustment")
MODEL_ATTRIBUTION_PNL = REGISTRY.gauge("insignia_model_attribution_pnl", "PnL attributed to L1 model from L2 usage")

# --- Attack Detection ---
ATTACK_BREACH = REGISTRY.gauge("insignia_attack_breach", "Attack breach flag (0/1)")
ATTACK_SEVERITY = REGISTRY.gauge("insignia_attack_severity", "Attack severity score 0-1")
TOTAL_BREACHES = REGISTRY.gauge("insignia_total_breaches", "Total active attack breaches")

# --- Optimizer ---
GEN_COUNTER = REGISTRY.counter("insignia_optimizer_generation", "Current evolutionary generation")
BEST_FITNESS = REGISTRY.gauge("insignia_best_fitness", "Best fitness value per objective")
PARETO_SIZE = REGISTRY.gauge("insignia_pareto_front_size", "Number of solutions on Pareto front")
POP_DIVERSITY = REGISTRY.gauge("insignia_population_diversity", "Population genetic diversity")


# =========================================================================
# Export Functions
# =========================================================================

def export_subnet_health(metagraph_data: Dict[str, Any]):
    """
    Export Bittensor metagraph and subnet health metrics.

    Call this from the validator's main loop after fetching the metagraph.
    metagraph_data should contain:
      - neurons: list of {uid, stake, rank, trust, consensus, incentive,
                          dividends, emission, vtrust, updated, active,
                          is_validator, is_immune, hotkey}
      - subnet: {n_neurons, n_validators, n_miners, total_stake,
                  emission_rate, registration_cost, difficulty}
      - hyperparameters: {tempo, immunity_period, min_allowed_weights, ...}
      - mechanisms: {count, splits: [{mechid, share}],
                     per_neuron: [{uid, mechid, incentive, rank, consensus, dividends}]}
    """
    netuid = str(metagraph_data.get("netuid", "0"))

    for neuron in metagraph_data.get("neurons", []):
        uid = str(neuron["uid"])
        role = "validator" if neuron.get("is_validator") else "miner"
        labels = dict(uid=uid, netuid=netuid, role=role)

        BT_STAKE.set(neuron.get("stake", 0), **labels)
        BT_RANK.set(neuron.get("rank", 0), **labels)
        BT_TRUST.set(neuron.get("trust", 0), **labels)
        BT_CONSENSUS.set(neuron.get("consensus", 0), **labels)
        BT_INCENTIVE.set(neuron.get("incentive", 0), **labels)
        BT_DIVIDENDS.set(neuron.get("dividends", 0), **labels)
        BT_EMISSION.set(neuron.get("emission", 0), **labels)
        BT_VTRUST.set(neuron.get("vtrust", 0), **labels)
        BT_UPDATED.set(neuron.get("updated", 0), **labels)
        BT_ACTIVE.set(float(neuron.get("active", 0)), **labels)
        BT_IS_VALIDATOR.set(float(neuron.get("is_validator", 0)), **labels)
        BT_IS_IMMUNITY.set(float(neuron.get("is_immune", 0)), **labels)

        if neuron.get("is_validator"):
            updated = neuron.get("updated", 0)
            BT_VAL_WEIGHT_SETTING_LAG.set(updated, uid=uid, netuid=netuid)
            BT_VAL_WEIGHT_SETTING_OK.set(1.0 if updated < 500 else 0.0, uid=uid, netuid=netuid)

    subnet = metagraph_data.get("subnet", {})
    BT_SUBNET_N.set(subnet.get("n_neurons", 0), netuid=netuid)
    BT_SUBNET_VALIDATORS.set(subnet.get("n_validators", 0), netuid=netuid)
    BT_SUBNET_MINERS.set(subnet.get("n_miners", 0), netuid=netuid)
    BT_SUBNET_TOTAL_STAKE.set(subnet.get("total_stake", 0), netuid=netuid)
    BT_SUBNET_EMISSION_RATE.set(subnet.get("emission_rate", 0), netuid=netuid)
    BT_SUBNET_REGISTRATION_COST.set(subnet.get("registration_cost", 0), netuid=netuid)
    BT_SUBNET_DIFFICULTY.set(subnet.get("difficulty", 0), netuid=netuid)

    hp = metagraph_data.get("hyperparameters", {})
    BT_HP_TEMPO.set(hp.get("tempo", 0), netuid=netuid)
    BT_HP_IMMUNITY.set(hp.get("immunity_period", 0), netuid=netuid)
    BT_HP_MIN_WEIGHTS.set(hp.get("min_allowed_weights", 0), netuid=netuid)
    BT_HP_MAX_WEIGHT.set(hp.get("max_weight_limit", 0), netuid=netuid)
    BT_HP_BONDS_MA.set(hp.get("bonds_moving_avg", 0), netuid=netuid)
    BT_HP_KAPPA.set(hp.get("kappa", 0), netuid=netuid)
    BT_HP_MAX_REGS.set(hp.get("max_regs_per_block", 0), netuid=netuid)
    BT_HP_SERVING_RATE.set(hp.get("serving_rate_limit", 0), netuid=netuid)
    BT_HP_ADJUSTMENT_ALPHA.set(hp.get("adjustment_alpha", 0), netuid=netuid)
    BT_HP_COMMIT_REVEAL.set(float(hp.get("commit_reveal_enabled", 0)), netuid=netuid)
    BT_HP_LIQUID_ALPHA.set(float(hp.get("liquid_alpha_enabled", 0)), netuid=netuid)
    BT_HP_ALPHA_HIGH.set(hp.get("alpha_high", 0), netuid=netuid)
    BT_HP_ALPHA_LOW.set(hp.get("alpha_low", 0), netuid=netuid)

    mechs = metagraph_data.get("mechanisms", {})
    BT_MECH_COUNT.set(mechs.get("count", 1), netuid=netuid)
    for split in mechs.get("splits", []):
        BT_MECH_EMISSION_SPLIT.set(split["share"], netuid=netuid, mechid=str(split["mechid"]))
    for entry in mechs.get("per_neuron", []):
        m_labels = dict(uid=str(entry["uid"]), netuid=netuid, mechid=str(entry["mechid"]))
        BT_MECH_INCENTIVE.set(entry.get("incentive", 0), **m_labels)
        BT_MECH_RANK.set(entry.get("rank", 0), **m_labels)
        BT_MECH_CONSENSUS.set(entry.get("consensus", 0), **m_labels)
        BT_MECH_DIVIDENDS.set(entry.get("dividends", 0), **m_labels)


def export_simulation_metrics(
    sim_result: Any,
    breach_report: Any,
    generation: int = 0,
    individual: int = 0,
):
    """Export metrics from a simulation result and breach report."""
    gen_label = str(generation)

    for uid, score in sim_result.miner_scores.items():
        mtype = sim_result.miner_types.get(uid, "unknown")
        L1_SCORE.set(score, miner=uid, agent_type=mtype, generation=gen_label)

    # Export per-metric L1 breakdown from last epoch
    if sim_result.l1_epoch_results:
        last_epoch = sim_result.l1_epoch_results[-1]
        for uid, r in last_epoch.get("results", {}).items():
            if not r.get("accepted"):
                continue
            mtype = sim_result.miner_types.get(uid, "unknown")
            raw = r.get("raw_metrics", {})
            labels = dict(miner=uid, agent_type=mtype, generation=gen_label)
            L1_PENALIZED_F1.set(raw.get("penalized_f1", 0), **labels)
            L1_MEAN_F1.set(raw.get("mean_f1", 0), **labels)
            L1_STD_F1.set(raw.get("std_f1", 0), **labels)
            L1_PENALIZED_SHARPE.set(raw.get("penalized_sharpe", 0), **labels)
            L1_MEAN_SHARPE.set(raw.get("mean_sharpe", 0), **labels)
            L1_STD_SHARPE.set(raw.get("std_sharpe", 0), **labels)
            L1_MAX_DD.set(raw.get("max_drawdown", 0), **labels)
            L1_GEN_GAP.set(raw.get("generalization_gap", 0), **labels)
            L1_FEAT_EFF.set(raw.get("feature_efficiency", 0), **labels)
            L1_LATENCY.set(raw.get("latency", 0), **labels)

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

    # Seed example metagraph data
    export_subnet_health({
        "netuid": 999,
        "neurons": [
            {"uid": 0, "stake": 1500.0, "rank": 0.85, "trust": 0.92, "consensus": 0.88,
             "incentive": 0.0, "dividends": 0.15, "emission": 0.003, "vtrust": 0.95,
             "updated": 120, "active": 1, "is_validator": True, "is_immune": False},
            {"uid": 1, "stake": 0.0, "rank": 0.72, "trust": 0.80, "consensus": 0.75,
             "incentive": 0.12, "dividends": 0.0, "emission": 0.002, "vtrust": 0.0,
             "updated": 50, "active": 1, "is_validator": False, "is_immune": False},
            {"uid": 2, "stake": 0.0, "rank": 0.65, "trust": 0.70, "consensus": 0.68,
             "incentive": 0.08, "dividends": 0.0, "emission": 0.001, "vtrust": 0.0,
             "updated": 80, "active": 1, "is_validator": False, "is_immune": True},
        ],
        "subnet": {"n_neurons": 3, "n_validators": 1, "n_miners": 2,
                    "total_stake": 1500.0, "emission_rate": 0.006},
        "hyperparameters": {"tempo": 360, "immunity_period": 7200,
                            "min_allowed_weights": 1, "max_weight_limit": 65535,
                            "bonds_moving_avg": 900000, "kappa": 32767,
                            "commit_reveal_enabled": 1, "liquid_alpha_enabled": 1,
                            "alpha_high": 0.9, "alpha_low": 0.7},
        "mechanisms": {
            "count": 2,
            "splits": [{"mechid": 0, "share": 0.6}, {"mechid": 1, "share": 0.4}],
            "per_neuron": [
                {"uid": 1, "mechid": 0, "incentive": 0.15, "rank": 0.72, "consensus": 0.75, "dividends": 0.0},
                {"uid": 1, "mechid": 1, "incentive": 0.08, "rank": 0.55, "consensus": 0.60, "dividends": 0.0},
            ],
        },
    })

    L1_SCORE.set(0.72, miner="honest_0", agent_type="honest", generation="0")
    ATTACK_BREACH.set(1.0, attack="overfitting", generation="0")
    GEN_COUNTER.set(5.0)

    start_metrics_server(8000)
    print("Metrics server running on http://localhost:8000/metrics")
    print("Press Ctrl+C to stop\n")
    print(REGISTRY.exposition()[:2000])

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_metrics_server()
