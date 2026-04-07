"""
Distributed inference with real SGLang servers.

Launches N SGLang HTTP servers (one per node), then sends requests
through the cache-aware router.

Usage:
  python -m src.sglang_main --config config/sglang.yaml
"""

import asyncio
import argparse
import csv
import logging
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
import aiohttp
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.request_loader import load_requests, RequestsConfig

_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
_run_dir = f"runs/run_{_timestamp}"
os.makedirs(_run_dir, exist_ok=True)
_log_path = f"{_run_dir}/run.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_path),
    ],
)
logger = logging.getLogger(__name__)


def load_sglang_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def launch_sglang_server(
    port: int,
    model_path: str,
    tp_size: int = 1,
    mem_fraction: float = 0.7,
    context_length: int = 2048,
    log_dir: str = ".",
) -> subprocess.Popen:
    """Launch a single SGLang server as a subprocess.

    On Mac (MPS), uses the local sglang source tree and disables CUDA graphs.
    """
    # Build the launch script inline so we can set PYTHONPATH for the sglang source
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sglang_src = os.path.join(project_root, "sglang", "python")

    launch_code = f"""
import sys, os
sys.path.insert(0, {sglang_src!r})
from sglang.srt.server_args import ServerArgs
from sglang.srt.entrypoints.http_server import launch_server

args = ServerArgs(
    model_path={model_path!r},
    port={port},
    host='127.0.0.1',
    tp_size={tp_size},
    mem_fraction_static={mem_fraction},
    context_length={context_length},
    disable_cuda_graph=True,
)
launch_server(args)
"""

    log_file = open(f"{log_dir}/sglang_node_{port}.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", launch_code],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONPATH": f"{sglang_src}:{os.environ.get('PYTHONPATH', '')}"},
    )
    logger.info(f"Launched SGLang server on port {port} (pid {proc.pid})")
    return proc


async def wait_for_server(port: int, timeout: int = 300) -> bool:
    """Wait for a SGLang server to become ready."""
    start = time.time()
    url = f"http://127.0.0.1:{port}/health"
    async with aiohttp.ClientSession() as session:
        while time.time() - start < timeout:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        logger.info(f"Server on port {port} is ready ({time.time()-start:.1f}s)")
                        return True
            except (aiohttp.ClientConnectorError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(2)
    logger.error(f"Server on port {port} failed to start within {timeout}s")
    return False


async def send_completion(
    session: aiohttp.ClientSession,
    port: int,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
) -> dict:
    """Send a completion request to a SGLang server (OpenAI-compatible API)."""
    url = f"http://127.0.0.1:{port}/v1/completions"
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
    }
    start = time.time()
    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            data = await resp.json()
            elapsed = time.time() - start

            # Parse SGLang response (OpenAI format)
            choice = data.get("choices", [{}])[0]
            usage = data.get("usage", {})

            return {
                "output_text": choice.get("text", ""),
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_time_s": round(elapsed, 4),
                "error": None,
            }
    except Exception as e:
        return {
            "output_text": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_time_s": round(time.time() - start, 4),
            "error": str(e),
        }


class PrefixTrie:
    """Lightweight trie rebuilt from a flat radix-tree snapshot.

    Each snapshot node has an `token_ids` edge. We reconstruct the tree
    structure so we can do proper greedy prefix matching against incoming
    prompts.
    """

    __slots__ = ("children",)

    def __init__(self):
        # token_id -> PrefixTrie
        self.children: dict[int, "PrefixTrie"] = {}

    def insert(self, token_ids: list[int]):
        node = self
        for tid in token_ids:
            if tid not in node.children:
                node.children[tid] = PrefixTrie()
            node = node.children[tid]

    def match_length(self, token_ids: list[int]) -> int:
        """Return how many leading tokens of `token_ids` exist in this trie."""
        node = self
        matched = 0
        for tid in token_ids:
            if tid not in node.children:
                break
            node = node.children[tid]
            matched += 1
        return matched


def build_trie_from_snapshot(snapshot_nodes: list[dict]) -> PrefixTrie:
    """Build a PrefixTrie from a flat snapshot.

    The snapshot is produced by RadixCache.snapshot() and contains nodes
    in DFS order. Each node's `token_ids` is the *edge* label from parent
    to that node. To reconstruct full paths we use `num_children` to track
    the DFS stack depth.

    However the snapshot doesn't carry explicit parent/child relationships
    beyond ordering. A simpler and correct approach: the snapshot carries
    every edge's token_ids. The full path from root to any node is the
    concatenation of edge token_ids along the path. Since we walk in DFS
    order and know `num_children`, we can reconstruct paths.
    """
    trie = PrefixTrie()
    if not snapshot_nodes:
        return trie

    # Reconstruct full paths via DFS using edge lengths + num_children.
    path_tokens: list[int] = []
    edge_lengths: list[int] = []
    children_remaining: list[int] = []

    for i, node in enumerate(snapshot_nodes):
        edge_tokens = node.get("token_ids", [])
        num_children = node.get("num_children", 0)

        if i == 0:
            # Root node — empty edge
            path_tokens = list(edge_tokens)
            edge_lengths = [len(edge_tokens)]
            children_remaining = [num_children]
            if path_tokens:
                trie.insert(path_tokens)
            continue

        # We are a child of the current top-of-stack parent.
        # Decrement parent's remaining children count.
        if children_remaining:
            children_remaining[-1] -= 1

        # Extend path with this edge
        path_tokens.extend(edge_tokens)
        edge_lengths.append(len(edge_tokens))
        children_remaining.append(num_children)

        # Insert this full path as a cached prefix
        if path_tokens:
            trie.insert(path_tokens)

        # Pop completed nodes (no remaining children)
        while children_remaining and children_remaining[-1] == 0:
            children_remaining.pop()
            if edge_lengths:
                pop_len = edge_lengths.pop()
                path_tokens = path_tokens[:-pop_len] if pop_len else path_tokens

    return trie


class TreeSnapshotCache:
    """Background-polled cache of radix tree snapshots from all nodes.

    Polls each SGLang server's /v1/cache/tree endpoint on a configurable
    interval and rebuilds prefix tries for fast routing lookups.
    """

    def __init__(self, ports: list[int], poll_interval: float = 2.0):
        self.ports = ports
        self.poll_interval = poll_interval
        self._tries: list[PrefixTrie] = [PrefixTrie() for _ in ports]
        self._raw_snapshots: list[list[dict]] = [[] for _ in ports]
        self._task: asyncio.Task | None = None
        self._session: aiohttp.ClientSession | None = None

    async def start(self):
        self._session = aiohttp.ClientSession()
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            f"Tree snapshot poller started (interval={self.poll_interval}s, "
            f"nodes={len(self.ports)})"
        )

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()

    async def _poll_loop(self):
        try:
            while True:
                await self._refresh_all()
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            pass

    async def _refresh_all(self):
        results = await asyncio.gather(
            *[self._fetch_one(i) for i in range(len(self.ports))],
            return_exceptions=True,
        )
        for i, result in enumerate(results):
            if isinstance(result, list):
                self._raw_snapshots[i] = result
                self._tries[i] = build_trie_from_snapshot(result)

    async def _fetch_one(self, idx: int) -> list[dict]:
        url = f"http://127.0.0.1:{self.ports[idx]}/v1/cache/tree"
        try:
            async with self._session.get(
                url, timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                data = await resp.json()
                snapshots = data.get("snapshots", [])
                if snapshots:
                    return snapshots[0].get("nodes", [])
        except Exception as e:
            logger.debug(f"Tree poll failed for port {self.ports[idx]}: {e}")
        return self._raw_snapshots[idx]  # keep old data on failure

    def route(self, token_ids: list[int]) -> tuple[int, int]:
        """Return (best_node_index, matched_prefix_length)."""
        best_node = 0
        best_len = 0
        for i, trie in enumerate(self._tries):
            match_len = trie.match_length(token_ids)
            if match_len > best_len:
                best_len = match_len
                best_node = i
        return best_node, best_len

    def get_snapshot_stats(self, node_idx: int) -> dict:
        """Return aggregate stats for a node's snapshot."""
        snap = self._raw_snapshots[node_idx]
        if not snap:
            return {"total_tokens": 0, "num_nodes": 0}
        total_kv = sum(n.get("num_kv_indices", 0) for n in snap)
        return {"total_tokens": total_kv, "num_nodes": len(snap)}


class MetricsPoller:
    """Periodically polls /server_info from each SGLang node and writes metrics.csv.

    Tracks per-node request/token counts from the router and combines with
    server-side cache stats for a complete picture.
    """

    def __init__(self, ports: list[int], run_dir: str, interval_s: float = 10.0):
        self.ports = ports
        self.interval_s = interval_s
        self._session: aiohttp.ClientSession | None = None
        self._task: asyncio.Task | None = None
        self._start_time = time.time()
        self._last_time = time.time()

        # Per-node cumulative counters (updated externally via record_request)
        self._node_reqs: dict[int, int] = {i: 0 for i in range(len(ports))}
        self._node_routed: dict[int, int] = {i: 0 for i in range(len(ports))}
        self._node_cache_hits: dict[int, int] = {i: 0 for i in range(len(ports))}
        self._node_input_tokens: dict[int, int] = {i: 0 for i in range(len(ports))}
        self._node_output_tokens: dict[int, int] = {i: 0 for i in range(len(ports))}
        self._prev_reqs: dict[int, int] = {i: 0 for i in range(len(ports))}
        self._prev_tokens: dict[int, int] = {i: 0 for i in range(len(ports))}

        csv_path = f"{run_dir}/metrics.csv"
        self._csv_file = open(csv_path, "w", newline="")
        self._writer = csv.DictWriter(
            self._csv_file,
            fieldnames=[
                "timestamp", "node_id", "queue_length", "running_requests",
                "total_requests", "total_routed", "total_cache_hits",
                "total_input_tokens", "total_output_tokens",
                "req_throughput", "token_throughput",
                "cache_used_tokens", "cache_max_tokens", "cache_utilization",
                "cache_entries", "radix_tree_tokens",
            ],
            delimiter="|",
        )
        self._writer.writeheader()
        self._csv_file.flush()

    def record_request(self, node_id: int, input_tokens: int, output_tokens: int,
                       cache_hit_tokens: int, was_routed: bool):
        """Called after each request completes to update per-node counters."""
        self._node_reqs[node_id] = self._node_reqs.get(node_id, 0) + 1
        self._node_input_tokens[node_id] = self._node_input_tokens.get(node_id, 0) + input_tokens
        self._node_output_tokens[node_id] = self._node_output_tokens.get(node_id, 0) + output_tokens
        if cache_hit_tokens > 0:
            self._node_cache_hits[node_id] = self._node_cache_hits.get(node_id, 0) + 1
        if was_routed:
            self._node_routed[node_id] = self._node_routed.get(node_id, 0) + 1

    async def start(self):
        self._session = aiohttp.ClientSession()
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        # Final collect before stopping
        await self._collect()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()
        self._csv_file.close()

    async def _loop(self):
        try:
            while True:
                await asyncio.sleep(self.interval_s)
                await self._collect()
        except asyncio.CancelledError:
            pass

    async def _collect(self):
        now = time.time()
        elapsed = max(now - self._last_time, 0.001)
        self._last_time = now

        for idx, port in enumerate(self.ports):
            try:
                info = await self._fetch_server_info(port)
            except Exception:
                continue

            mem = info.get("memory_usage", {})
            snap = info.get("radix_tree_snapshot", {})
            cache_used = snap.get("total_tokens", 0)
            cache_max = mem.get("token_capacity", 1)
            cache_util = round(cache_used / cache_max, 4) if cache_max > 0 else 0

            cur_reqs = self._node_reqs.get(idx, 0)
            cur_tokens = self._node_output_tokens.get(idx, 0)
            prev_reqs = self._prev_reqs.get(idx, 0)
            prev_tokens = self._prev_tokens.get(idx, 0)

            req_tp = round((cur_reqs - prev_reqs) / elapsed, 2)
            tok_tp = round((cur_tokens - prev_tokens) / elapsed, 2)
            self._prev_reqs[idx] = cur_reqs
            self._prev_tokens[idx] = cur_tokens

            row = {
                "timestamp": round(now, 2),
                "node_id": idx,
                "queue_length": 0,
                "running_requests": 0,
                "total_requests": cur_reqs,
                "total_routed": self._node_routed.get(idx, 0),
                "total_cache_hits": self._node_cache_hits.get(idx, 0),
                "total_input_tokens": self._node_input_tokens.get(idx, 0),
                "total_output_tokens": cur_tokens,
                "req_throughput": req_tp,
                "token_throughput": tok_tp,
                "cache_used_tokens": cache_used,
                "cache_max_tokens": cache_max,
                "cache_utilization": cache_util,
                "cache_entries": snap.get("num_nodes", 0),
                "radix_tree_tokens": cache_used,
            }
            self._writer.writerow(row)

        self._csv_file.flush()

    async def _fetch_server_info(self, port: int) -> dict:
        url = f"http://127.0.0.1:{port}/server_info"
        async with self._session.get(
            url, timeout=aiohttp.ClientTimeout(total=5)
        ) as resp:
            data = await resp.json()
            states = data.get("internal_states", [])
            if states:
                merged = dict(data)
                merged.update(states[0])
                return merged
            return data


class Tokenizer:
    """Wraps a HuggingFace tokenizer for converting prompts to token IDs.

    Uses the same tokenizer as the SGLang servers so that prefix matching
    against the radix tree works correctly.
    """

    def __init__(self, model_path: str):
        try:
            from transformers import AutoTokenizer
            logger.info(f"Loading tokenizer from {model_path}...")
            self._tok = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            logger.info("Tokenizer loaded.")
        except ImportError:
            logger.warning(
                "transformers not installed — falling back to hash tokenizer. "
                "Install with: pip install transformers"
            )
            self._tok = None

    def encode(self, text: str) -> list[int]:
        if self._tok is not None:
            return self._tok.encode(text)
        # Fallback: deterministic hash tokenizer
        return [hash(w) % 32000 for w in text.split()]


async def run_experiment(cfg: dict, no_launch: bool = False):
    """Run the distributed inference experiment with SGLang servers.

    Args:
        cfg: Experiment config dict.
        no_launch: If True, skip launching servers (connect to already-running ones).
    """
    import shutil
    config_path = cfg["_config_path"]
    shutil.copy2(config_path, f"{_run_dir}/config.yaml")

    cluster = cfg.get("cluster", {})
    sglang_cfg = cfg.get("sglang", {})
    router_cfg = cfg.get("router", {})
    req_cfg_raw = cfg.get("requests", {})

    num_nodes = cluster.get("num_nodes", 2)
    base_port = cluster.get("base_port", 8100)
    model_path = sglang_cfg.get("model_path", "Qwen/Qwen2.5-0.5B-Instruct")
    tp_size = sglang_cfg.get("tp_size", 1)
    mem_fraction = sglang_cfg.get("mem_fraction_static", 0.7)
    context_length = sglang_cfg.get("context_length", 2048)
    max_new_tokens = sglang_cfg.get("max_new_tokens", 64)
    temperature = sglang_cfg.get("temperature", 0.0)

    ports = [base_port + i for i in range(num_nodes)]

    procs = []
    if no_launch:
        print(f"\nConnecting to {num_nodes} existing servers on ports {ports}")
        # Quick health check
        ready = await asyncio.gather(*[wait_for_server(p, timeout=10) for p in ports])
        if not all(ready):
            print("ERROR: Not all servers reachable. Start them first:")
            print(f"  make run-mock-sglang  (or launch real SGLang servers)")
            return
        print(f"All {num_nodes} servers reachable!\n")
    else:
        print(f"\nLaunching {num_nodes} SGLang servers with model: {model_path}")
        print(f"Ports: {ports}")
        for port in ports:
            proc = launch_sglang_server(
                port=port,
                model_path=model_path,
                tp_size=tp_size,
                mem_fraction=mem_fraction,
                context_length=context_length,
                log_dir=_run_dir,
            )
            procs.append(proc)

        print("Waiting for servers to start (this may take 1-3 minutes)...")
        ready = await asyncio.gather(*[wait_for_server(p) for p in ports])
        if not all(ready):
            print("ERROR: Not all servers started. Check logs in", _run_dir)
            for proc in procs:
                proc.terminate()
            return
        print(f"All {num_nodes} servers ready!\n")

    # Load requests
    req_cfg = RequestsConfig(**req_cfg_raw)
    rows = load_requests(req_cfg, num_nodes)

    topic_counts = Counter(r["subject"] for r in rows)
    node_counts = Counter(r["target_node"] for r in rows)
    print(f"Loaded {len(rows)} requests across {len(topic_counts)} topics")
    for n in sorted(node_counts):
        print(f"  node {n}: {node_counts[n]} requests")
    print()

    requests_list = [(r["text"], r["target_node"], r["subject"]) for r in rows]

    routing_policy = router_cfg.get("policy", "round_robin")
    poll_interval = router_cfg.get("snapshot_poll_interval_s", 2.0)
    metrics_interval = router_cfg.get("metrics_interval_s", 20.0)
    print(f"Routing policy: {routing_policy}")

    # Initialize router — mirrors mock experiment's per-node Router
    # Each node keeps approx trees of all other nodes, polled at regular interval.
    # Routing = find max prefix match across all nodes' trees.
    from src.sglang_router import SGLangRouter

    tokenizer = None
    sglang_router = None
    if routing_policy == "cache_aware":
        tokenizer = Tokenizer(model_path)
        sglang_router = SGLangRouter(
            ports=ports,
            poll_interval=poll_interval,
            tokenizer=tokenizer,
        )
        await sglang_router.start()
        print(f"Cache-aware router started (polling trees every {poll_interval}s)")

    # Start metrics poller (writes metrics.csv for dashboard)
    metrics_poller = MetricsPoller(ports, _run_dir, interval_s=metrics_interval)
    await metrics_poller.start()

    # CSV setup — aligned with main.py fields so dashboard works for both
    csv_path = f"{_run_dir}/results.csv"
    fieldnames = [
        "request_id", "subject", "target_node", "processed_by",
        "routed_from", "routed_to", "input_tokens", "output_tokens",
        "cache_hit_tokens", "cache_hit_ratio", "inference_time_s",
        "total_time_s", "cache_utilization", "error",
    ]
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter="|")
    writer.writeheader()
    csv_file.flush()

    # Send requests
    print("--- Sending Requests ---")
    print(f"Results streaming to {csv_path}")
    rr_counter = 0  # round-robin counter
    for i, (text, target_node, subject) in enumerate(requests_list):
        default_node = target_node % num_nodes

        if routing_policy == "cache_aware" and sglang_router:
            # Full pipeline: tokenize -> check all nodes' trees -> forward
            result = await sglang_router.route_and_forward(
                prompt=text,
                default_node=default_node,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            chosen_node = result["processed_by"]
            prefix_match_len = result["cache_hit_tokens"]
            cache_hit_ratio = result["cache_hit_ratio"]
            cache_util = round(
                result["cache_utilization"] / context_length, 4
            ) if context_length > 0 else 0.0
        else:
            # Round-robin or random — no tree checking
            if routing_policy == "round_robin":
                chosen_node = rr_counter % num_nodes
                rr_counter += 1
            else:
                import random
                chosen_node = random.randint(0, num_nodes - 1)

            result = await send_completion(
                aiohttp.ClientSession(), ports[chosen_node], text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            prefix_match_len = 0
            cache_hit_ratio = 0.0
            cache_util = 0.0

        routed_from = default_node
        routed_to = chosen_node
        was_routed = routed_from != routed_to

        print(
            f"\nRequest {i+1}/{len(requests_list)}: "
            f"'{text[:50]}...' -> node {chosen_node} (port {ports[chosen_node]})"
            + (f" [prefix={prefix_match_len}]" if prefix_match_len else "")
            + (f" [routed {routed_from}->{routed_to}]" if was_routed else "")
        )
        print(f"  Tokens: {result.get('input_tokens', 0)} in, {result.get('output_tokens', 0)} out")
        print(f"  Cache hit: {prefix_match_len} tokens ({cache_hit_ratio:.0%})")
        print(f"  Time: {result.get('total_time_s', 0):.4f}s")
        if result.get("error"):
            print(f"  ERROR: {result['error']}")

        row = {
            "request_id": i,
            "subject": subject,
            "target_node": target_node,
            "processed_by": chosen_node,
            "routed_from": routed_from,
            "routed_to": routed_to,
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "cache_hit_tokens": prefix_match_len,
            "cache_hit_ratio": cache_hit_ratio,
            "inference_time_s": result.get("total_time_s", 0),
            "total_time_s": result.get("total_time_s", 0),
            "cache_utilization": cache_util,
            "error": result.get("error"),
        }
        writer.writerow(row)
        csv_file.flush()

        # Feed per-node stats to metrics poller
        metrics_poller.record_request(
            node_id=chosen_node,
            input_tokens=result.get("input_tokens", 0),
            output_tokens=result.get("output_tokens", 0),
            cache_hit_tokens=prefix_match_len,
            was_routed=was_routed,
        )

    # Print routing summary (mirrors mock's Visualizer.routing_stats)
    if sglang_router:
        stats = sglang_router.get_stats()
        print(f"\n--- Routing Summary ---")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Routed to different node: {stats['total_routed']} "
              f"({stats['route_ratio']:.0%})")
        print(f"  Approx tree sizes: {stats['approx_tree_sizes']}")
        await sglang_router.stop()

    await metrics_poller.stop()

    csv_file.close()
    print(f"\nResults saved to {csv_path}")

    # Summary
    print("\n" + "=" * 60)
    if procs:
        print("Experiment complete. Shutting down servers...")
        for proc in procs:
            proc.terminate()
        for proc in procs:
            proc.wait(timeout=10)
        print("All servers stopped.")
    else:
        print("Experiment complete. (Servers left running)")


async def main(config_path: str, no_launch: bool = False):
    cfg = load_sglang_config(config_path)
    cfg["_config_path"] = config_path

    try:
        await run_experiment(cfg, no_launch=no_launch)
    except KeyboardInterrupt:
        print("\nInterrupted. Cleaning up...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Inference with SGLang")
    parser.add_argument(
        "--config", default="config/sglang.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--no-launch", action="store_true",
        help="Don't launch servers; connect to already-running ones",
    )
    args = parser.parse_args()
    asyncio.run(main(args.config, no_launch=args.no_launch))
