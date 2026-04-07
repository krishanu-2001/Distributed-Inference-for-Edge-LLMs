import asyncio
import argparse
import csv
import logging
import aiohttp
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.node import InferenceNode
from src.network import NetworkSimulator
from src.visualization import Visualizer
from src.request_loader import load_requests

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


async def send_request(port: int, text: str) -> dict:
    """Send an inference request to a node."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://localhost:{port}/infer", json={"text": text}
        ) as resp:
            return await resp.json()


async def get_status(port: int) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://localhost:{port}/status") as resp:
            return await resp.json()


async def get_tree(port: int) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://localhost:{port}/tree") as resp:
            return await resp.json()


async def run_demo(nodes: list[InferenceNode], config):
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
    print()

    # Load requests via config-driven loader
    req_cfg = config.requests
    rows = load_requests(req_cfg, config.cluster.num_nodes)

    # Print summary
    from collections import Counter
    topic_counts = Counter(r['subject'] for r in rows)
    node_counts = Counter(r['target_node'] for r in rows)
    print(f"Loaded {len(rows)} requests across {len(topic_counts)} topics")
    for n in sorted(node_counts):
        print(f"  node {n}: {node_counts[n]} requests")
    print()

    requests = [(r["text"], r["target_node"], r["subject"]) for r in rows]

    # Open CSV for incremental writing
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

    print("--- Sending Requests ---")
    print(f"Results streaming to {csv_path}")
    for i, (text, target_node, subject) in enumerate(requests):
        port = base_port + (target_node % config.cluster.num_nodes)
        print(f"\nRequest {i + 1}: '{text[:50]}...' -> Node {target_node}")
        result = await send_request(port, text)
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
            "target_node": target_node,
            "processed_by": result.get("node_id"),
            "routed_from": result.get("routed_from"),
            "routed_to": result.get("routed_to"),
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
            "cache_hit_tokens": result.get("cache_hit_tokens"),
            "cache_hit_ratio": result.get("cache_hit_ratio"),
            "inference_time_s": result.get("inference_time_s"),
            "total_time_s": result.get("total_time_s"),
            "cache_utilization": result.get("cache_utilization"),
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


async def main(config_path: str):
    config = load_config(config_path)

    # Save config to run directory
    import shutil
    shutil.copy2(config_path, f"{_run_dir}/config.yaml")
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
        node.start_metrics(_run_dir, interval_s=20.0)

    logger.info(f"All {config.cluster.num_nodes} nodes started.")

    try:
        await run_demo(nodes, config)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed LLM Inference Simulation")
    parser.add_argument(
        "--config", default="config/default.yaml", help="Path to config YAML"
    )
    args = parser.parse_args()
    asyncio.run(main(args.config))
