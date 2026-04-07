"""
FastAPI app for configuring and running distributed inference experiments.
Run: uvicorn webapp.app:app --port 9000 --reload
"""

import asyncio
import csv
import os
import socket
import signal
import sys
import yaml
import subprocess
from pathlib import Path
from datetime import datetime
from collections import Counter

import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.request_loader import _match_topics

app = FastAPI(title="Distributed Inference Experiment Runner")

# ── Pydantic models ──────────────────────────────────────────────

class ClusterModel(BaseModel):
    num_nodes: int = 4
    base_port: int = 8100

class NetworkModel(BaseModel):
    lan_delay_ms: float = 2.0
    delay_jitter_ms: float = 0.5

class NodeModel(BaseModel):
    max_tokens: int = 1000
    eviction_policy: str = "lru"

class LLMModel(BaseModel):
    prefill_cost_factor: float = 0.0001
    decode_cost_factor: float = 0.001
    max_output_tokens: int = 20

class RouterModel(BaseModel):
    policy: str = "cache_aware"

class BroadcastModel(BaseModel):
    trigger: str = "queue_length"
    queue_threshold: int = 3
    interval_s: float = 5.0

class RequestsModel(BaseModel):
    csv_path: str = "data/requests.csv"
    topics: list[str] = []
    rows_per_topic: int = 0
    shuffle: bool = True
    seed: int = 42
    node_assignment: str = "original"

class ExperimentConfig(BaseModel):
    cluster: ClusterModel = ClusterModel()
    network: NetworkModel = NetworkModel()
    node: NodeModel = NodeModel()
    llm: LLMModel = LLMModel()
    router: RouterModel = RouterModel()
    broadcast: BroadcastModel = BroadcastModel()
    requests: RequestsModel = RequestsModel()


class InferRequest(BaseModel):
    text: str
    node: int | None = None  # target node id, None = use node 0


# ── State ────────────────────────────────────────────────────────

_running_experiment: dict = {"process": None, "run_dir": None, "config_path": None}


def _get_available_topics() -> dict[str, int]:
    """Read CSV and return {subject: row_count}."""
    csv_path = ROOT / "data" / "requests.csv"
    counts: dict[str, int] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            topic = row["subject"]
            counts[topic] = counts.get(topic, 0) + 1
    return dict(sorted(counts.items()))


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def ui():
    """Serve the experiment config UI."""
    html_path = Path(__file__).parent / "index.html"
    return html_path.read_text()


@app.get("/api/topics")
async def list_topics():
    """List all available MMLU subjects with row counts."""
    return _get_available_topics()


@app.post("/api/topics/preview")
async def preview_topics(patterns: list[str]):
    """Preview which topics match the given regex patterns."""
    available = list(_get_available_topics().keys())
    matched, fractions = _match_topics(patterns, available)
    all_counts = _get_available_topics()
    return {t: all_counts[t] for t in matched}


@app.get("/api/config")
async def get_config():
    """Load current config from default.yaml."""
    config_path = ROOT / "config" / "default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@app.put("/api/config")
async def save_config(config: ExperimentConfig):
    """Save config to default.yaml."""
    config_path = ROOT / "config" / "default.yaml"
    data = config.model_dump()
    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return {"status": "saved", "path": str(config_path)}


@app.post("/api/experiment/start")
async def start_experiment(config: ExperimentConfig):
    """Save config and launch experiment as subprocess."""
    if _running_experiment["process"] is not None:
        poll = _running_experiment["process"].poll()
        if poll is None:
            raise HTTPException(400, "Experiment already running. Stop it first.")

    # Check if ports are already in use
    base_port = config.cluster.base_port
    num_nodes = config.cluster.num_nodes
    busy_ports = []
    for i in range(num_nodes):
        port = base_port + i
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) == 0:
                busy_ports.append(port)
    if busy_ports:
        raise HTTPException(
            409,
            f"Ports already in use: {busy_ports}. "
            f"Kill existing processes first (POST /api/experiment/kill-ports).",
        )

    # Write config to a timestamped file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = ROOT / "config" / f"run_{ts}.yaml"
    data = config.model_dump()
    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    # Also save as default
    default_path = ROOT / "config" / "default.yaml"
    with open(default_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    # Launch experiment
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.main", "--config", str(config_path)],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _running_experiment["process"] = proc
    _running_experiment["config_path"] = str(config_path)

    return {"status": "started", "pid": proc.pid, "config": str(config_path)}


@app.post("/api/experiment/kill-ports")
async def kill_ports(config: ExperimentConfig | None = None):
    """Kill any processes occupying the cluster ports."""
    # Read config for port range
    if config:
        base_port = config.cluster.base_port
        num_nodes = config.cluster.num_nodes
    else:
        config_path = ROOT / "config" / "default.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        base_port = cfg.get("cluster", {}).get("base_port", 8100)
        num_nodes = cfg.get("cluster", {}).get("num_nodes", 4)

    killed = []
    for i in range(num_nodes):
        port = base_port + i
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"], capture_output=True, text=True
        )
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            if pid:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    killed.append({"port": port, "pid": int(pid)})
                except (ProcessLookupError, ValueError):
                    pass

    # Also clear our tracked process
    _running_experiment["process"] = None

    return {"status": "killed", "killed": killed}


@app.post("/api/experiment/stop")
async def stop_experiment():
    """Stop a running experiment."""
    proc = _running_experiment["process"]
    if proc is None or proc.poll() is not None:
        _running_experiment["process"] = None
        return {"status": "not_running"}

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    _running_experiment["process"] = None
    return {"status": "stopped"}


@app.get("/api/experiment/status")
async def experiment_status():
    """Check if an experiment is running and get its output tail."""
    proc = _running_experiment["process"]
    if proc is None:
        return {"running": False}

    poll = proc.poll()
    if poll is not None:
        # Finished — read remaining output
        out = proc.stdout.read() if proc.stdout else ""
        _running_experiment["process"] = None
        return {"running": False, "exit_code": poll, "output": out[-5000:]}

    return {"running": True, "pid": proc.pid}


@app.get("/api/runs")
async def list_runs():
    """List all experiment runs with their result files."""
    runs_dir = ROOT / "runs"
    if not runs_dir.exists():
        return []

    runs = []
    for d in sorted(runs_dir.iterdir(), reverse=True):
        if d.is_dir() and d.name.startswith("run_"):
            results_csv = d / "results.csv"
            run_log = d / "run.log"
            info = {"name": d.name, "path": str(d)}
            if results_csv.exists():
                # Count rows and get summary
                with open(results_csv) as f:
                    reader = csv.DictReader(f, delimiter="|")
                    rows = list(reader)
                info["num_requests"] = len(rows)
                if rows:
                    hit_ratios = [float(r.get("cache_hit_ratio") or 0) for r in rows]
                    info["avg_cache_hit"] = round(sum(hit_ratios) / len(hit_ratios), 4)
                    times = [float(r.get("total_time_s") or 0) for r in rows]
                    info["avg_time_s"] = round(sum(times) / len(times), 4)
            if run_log.exists():
                info["has_log"] = True
            runs.append(info)
    return runs[:20]


@app.get("/api/runs/{run_name}/results")
async def get_run_results(run_name: str):
    """Get full results CSV for a run."""
    csv_path = ROOT / "runs" / run_name / "results.csv"
    if not csv_path.exists():
        raise HTTPException(404, f"Run {run_name} not found")
    with open(csv_path) as f:
        reader = csv.DictReader(f, delimiter="|")
        return list(reader)


@app.get("/api/runs/{run_name}/metrics")
async def get_run_metrics(run_name: str):
    """Get node metrics time-series for a run."""
    metrics_path = ROOT / "runs" / run_name / "metrics.csv"
    if not metrics_path.exists():
        raise HTTPException(404, f"Metrics for {run_name} not found")
    with open(metrics_path) as f:
        reader = csv.DictReader(f, delimiter="|")
        return list(reader)


@app.get("/dashboard/{run_name}", response_class=HTMLResponse)
async def dashboard(run_name: str):
    """Serve the dashboard UI for a specific run."""
    html_path = Path(__file__).parent / "dashboard.html"
    return html_path.read_text()


@app.get("/api/runs/{run_name}/config")
async def get_run_config(run_name: str):
    """Get the config YAML for a run."""
    config_path = ROOT / "runs" / run_name / "config.yaml"
    if not config_path.exists():
        raise HTTPException(404, f"Config for {run_name} not found")
    return {"yaml": config_path.read_text()}


@app.get("/api/runs/{run_name}/log")
async def get_run_log(run_name: str):
    """Get the last 200 lines of a run's log."""
    log_path = ROOT / "runs" / run_name / "run.log"
    if not log_path.exists():
        raise HTTPException(404, f"Log for {run_name} not found")
    lines = log_path.read_text().splitlines()
    return {"lines": lines[-200:]}


# ── Inference ────────────────────────────────────────────────────

@app.post("/api/infer")
async def infer(req: InferRequest):
    """Send a single inference request to a running cluster node. Returns JSON."""
    # Read current config for base_port
    config_path = ROOT / "config" / "default.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    base_port = cfg.get("cluster", {}).get("base_port", 8100)
    num_nodes = cfg.get("cluster", {}).get("num_nodes", 4)

    node_id = req.node if req.node is not None else 0
    if node_id < 0 or node_id >= num_nodes:
        raise HTTPException(400, f"node must be 0-{num_nodes - 1}")

    port = base_port + node_id
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://localhost:{port}/infer",
                json={"text": req.text},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                result = await resp.json()
                return result
    except aiohttp.ClientConnectorError:
        raise HTTPException(503, f"Node {node_id} not reachable on port {port}. Is the experiment running?")


@app.get("/api/nodes/status")
async def nodes_status():
    """Get status of all cluster nodes. Returns JSON array."""
    config_path = ROOT / "config" / "default.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    base_port = cfg.get("cluster", {}).get("base_port", 8100)
    num_nodes = cfg.get("cluster", {}).get("num_nodes", 4)

    statuses = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
        for i in range(num_nodes):
            port = base_port + i
            try:
                async with session.get(f"http://localhost:{port}/status") as resp:
                    data = await resp.json()
                    data["node_id"] = i
                    data["port"] = port
                    data["online"] = True
                    statuses.append(data)
            except Exception:
                statuses.append({"node_id": i, "port": port, "online": False})
    return statuses
