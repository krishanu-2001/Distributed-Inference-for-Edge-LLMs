import argparse
import asyncio
import csv
import dataclasses
import logging
import os
import sys
from datetime import datetime

import aiohttp
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.node import InferenceNode
from src.network import NetworkSimulator
from src.request_loader import load_requests
from src.visualization import Visualizer

logger = logging.getLogger(__name__)


def setup_run_dir(run_dir: str | None = None) -> str:
    """Create the run dir (default runs/run_<timestamp>/) and log to it.
    Called from main, not on import."""
    run_dir = run_dir or f"runs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(run_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{run_dir}/run.log"),
        ],
    )
    return run_dir


async def send_request(port: int, text: str, max_output_tokens: int | None = None) -> dict:
    """Send an inference request to a node."""
    body = {"text": text}
    if max_output_tokens is not None:
        body["max_output_tokens"] = max_output_tokens
    async with aiohttp.ClientSession() as session:
        async with session.post(f"http://localhost:{port}/infer", json=body) as resp:
            return await resp.json()


async def get_status(port: int) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://localhost:{port}/status") as resp:
            return await resp.json()


async def get_tree(port: int) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://localhost:{port}/tree") as resp:
            return await resp.json()


async def run_demo(nodes: list[InferenceNode], config, run_dir: str, limit: int = 0):
    """Run demo requests to show cache-aware routing."""
    base_port = config.cluster.base_port

    print("\n" + "=" * 60)
    print("DISTRIBUTED LLM INFERENCE SIMULATION")
    print("=" * 60)
    print(f"Nodes: {config.cluster.num_nodes}")
    print(f"Ports: {base_port} - {base_port + config.cluster.num_nodes - 1}")
    print(f"Router policy: {config.router.policy}")
    print(f"KV cache per node: {config.node.max_tokens} tokens")
    print(f"Broadcast trigger: {config.broadcast.trigger}")
    print(f"Requests: {config.requests.path} "
          f"(node_assignment={config.requests.node_assignment})")
    print()

    # Load requests from CSV
    if not os.path.exists(config.requests.path):
        raise SystemExit(
            f"{config.requests.path} not found. Generate it with "
            f"`python -m src.abcd` (MMLU) or `python -m src.gen_sglang_gsp` (SGLang gsp)"
        )
    df = load_requests(config.requests, config.cluster.num_nodes, limit)
    print("--- Request Dataset ---")
    print(f"  Total requests: {len(df)}")
    print(f"  Unique prompts: {df['text'].nunique()}")
    print(f"  Requests per entry node: {df['target_node'].value_counts().sort_index().to_dict()}")
    print(f"  Avg prompt len: {df['text'].str.split().str.len().mean():.1f} words")
    print()

    out_lens = df["output_len"] if "output_len" in df else [None] * len(df)
    requests = list(zip(df["text"], df["target_node"], df["subject"], out_lens))

    # Open CSV for incremental writing
    csv_path = f"{run_dir}/results.csv"
    fieldnames = [
        "request_id", "subject", "text", "target_node", "processed_by",
        "routed_from", "routed_to", "input_tokens", "output_tokens",
        "cache_hit_tokens", "cache_hit_ratio", "expected_hit_tokens",
        "route_expected_tokens", "inference_time_s",
        "total_time_s", "cache_utilization", "uncached_tokens", "error",
    ]
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter="|")
    writer.writeheader()
    csv_file.flush()

    print("--- Sending Requests ---")
    print(f"Results streaming to {csv_path}")
    for i, (text, target_node, subject, out_len) in enumerate(requests):
        port = base_port + (target_node % config.cluster.num_nodes)
        print(f"\nRequest {i + 1}: '{text[:50]}...' -> Node {target_node}")
        result = await send_request(
            port, text, None if out_len is None else int(out_len)
        )
        print(f"  Processed by: Node {result.get('node_id', '?')}")
        print(f"  Cache hit: {result.get('cache_hit_tokens', 0)} tokens "
              f"({result.get('cache_hit_ratio', 0):.0%})")
        print(f"  Inference time: {result.get('inference_time_s', 0):.4f}s")
        print(f"  Total time: {result.get('total_time_s', 0):.4f}s")
        if "routed_from" in result:
            print(f"  Routed: Node {result['routed_from']} -> Node {result['routed_to']}")

        row = {
            "request_id": i,
            "subject": subject,
            "text": text[-100:],
            "target_node": target_node,
            "processed_by": result.get("node_id"),
            "routed_from": result.get("routed_from"),
            "routed_to": result.get("routed_to"),
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
            "cache_hit_tokens": result.get("cache_hit_tokens"),
            "cache_hit_ratio": result.get("cache_hit_ratio"),
            "expected_hit_tokens": result.get("expected_hit_tokens"),
            "route_expected_tokens": result.get("route_expected_tokens"),
            "inference_time_s": result.get("inference_time_s"),
            "total_time_s": result.get("total_time_s"),
            "cache_utilization": result.get("cache_utilization"),
            "uncached_tokens": result.get("uncached_tokens"),
            "error": result.get("error"),
        }
        writer.writerow(row)
        csv_file.flush()

    csv_file.close()
    print(f"\nResults saved to {csv_path}")

    # Show cluster state
    await asyncio.sleep(0.5)
    print("\n" + Visualizer.cluster_cache_view(nodes))

    print("\n--- Node Details ---")
    for node in nodes:
        print(Visualizer.node_status(node))
        print()

    print(Visualizer.routing_stats(nodes))


async def main(args):
    run_dir = setup_run_dir(args.run_dir)
    config = load_config(args.config)
    if args.requests:
        config.requests.path = args.requests
    if args.policy:
        config.router.policy = args.policy
    if args.node_assignment:
        config.requests.node_assignment = args.node_assignment
    with open(f"{run_dir}/config.yaml", "w") as f:  # effective config, incl. overrides
        yaml.safe_dump(dataclasses.asdict(config), f, sort_keys=False)
    network = NetworkSimulator(config.network.lan_delay_ms, config.network.delay_jitter_ms)

    all_node_ids = list(range(config.cluster.num_nodes))
    all_ports = {i: config.cluster.base_port + i for i in all_node_ids}

    nodes = []
    for i in all_node_ids:
        node = InferenceNode(
            node_id=i,
            port=all_ports[i],
            config=config,
            all_node_ids=all_node_ids,
            all_ports=all_ports,
            network=network,
        )
        nodes.append(node)

    # Start all nodes
    for node in nodes:
        await node.start()

    logger.info(f"All {config.cluster.num_nodes} nodes started.")

    try:
        await run_demo(nodes, config, run_dir, args.limit)
        if args.exit:
            return

        print("\n" + "=" * 60)
        print("Demo complete. Servers still running. Press Ctrl+C to stop.")
        print(f"Try: curl -X POST http://localhost:{config.cluster.base_port}/infer "
              f'-d \'{{"text": "your prompt here"}}\'')
        print("=" * 60)

        # Keep running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        for node in nodes:
            await node.stop()
        await network.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed LLM Inference Simulation")
    parser.add_argument(
        "--config", default="config/default.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--requests", help="Override requests.path from the config"
    )
    parser.add_argument(
        "--policy", choices=["cache_aware", "round_robin", "self"],
        help="Override router.policy from the config",
    )
    parser.add_argument(
        "--node-assignment",
        choices=["round_robin", "original", "hash_topic", "random", "weighted"],
        help="Override requests.node_assignment from the config",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Only send the first N requests (0 = all)"
    )
    parser.add_argument(
        "--run-dir", help="Output directory (default runs/run_<timestamp>)"
    )
    parser.add_argument(
        "--exit", action="store_true", help="Exit after the demo instead of serving"
    )
    args = parser.parse_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\nShutting down...")
