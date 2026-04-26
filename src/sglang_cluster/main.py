from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import signal
import socket
import sys
from datetime import datetime
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sglang_cluster.config import load_config
from src.sglang_cluster.network import NetworkSimulator
from src.sglang_cluster.node import InferenceNode


def resolve_run_dir(config) -> Path:
    if not config.run.name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "runs" / f"sglang_run_{timestamp}"

    run_name = str(config.run.name).strip()
    if not run_name:
        raise ValueError("run.name cannot be empty.")

    run_name_path = Path(run_name)
    if run_name_path.is_absolute() or len(run_name_path.parts) != 1:
        raise ValueError(
            "run.name must be a single directory name under runs/, not a path."
        )

    run_dir = PROJECT_ROOT / "runs" / run_name
    if run_dir.exists() and not config.run.overwrite_existing:
        raise FileExistsError(
            f"Run directory {run_dir} already exists. Choose another run.name "
            "or set run.overwrite_existing: true."
        )

    return run_dir


def configure_logging(run_dir: Path):
    log_path = run_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ],
    )


def ensure_ports_available(ports: dict[int, int], label: str):
    occupied = []
    for node_id, port in ports.items():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                occupied.append((node_id, port))

    if occupied:
        formatted = ", ".join(
            f"node {node_id}:127.0.0.1:{port}" for node_id, port in occupied
        )
        raise RuntimeError(
            f"{label} port(s) are already in use: {formatted}. "
            "Stop the stale process or change cluster.base_port / "
            "cluster.sglang_base_port in config/sglang.yaml."
        )


def resolve_requests_path(csv_path: str) -> Path | None:
    configured_path = Path(csv_path)
    if configured_path.is_absolute():
        return configured_path if configured_path.exists() else None

    candidates = [PROJECT_ROOT / configured_path]
    if len(configured_path.parts) == 1:
        candidates.append(PROJECT_ROOT / "datasets" / configured_path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


async def send_request(port: int, payload: dict) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://127.0.0.1:{port}/infer",
            json=payload,
        ) as response:
            return await response.json()


async def maybe_run_demo(run_dir: Path, config, base_port: int) -> bool:
    if not config.requests.csv_path:
        return False

    requests_path = resolve_requests_path(config.requests.csv_path)
    if requests_path is None:
        logging.warning(
            "Request file %s does not exist; skipping demo.",
            config.requests.csv_path,
        )
        return False

    results_path = run_dir / "results.csv"
    debug_path = run_dir / "debug.jsonl"
    result_fieldnames = [
        "request_id",
        "subject",
        "target_node",
        "ingress_node",
        "processed_by",
        "routed_from",
        "routed_to",
        "prefix_match_input_tokens",
        "expected_cache_hit_tokens",
        "expected_cache_hit_ratio",
        "sglang_prompt_tokens",
        "sglang_cache_hit_tokens",
        "sglang_cache_hit_ratio",
        "local_snapshot_match_after_process",
        "total_time_s",
        "generated_text",
        "error",
    ]
    with (
        requests_path.open() as source,
        results_path.open("w", newline="") as results_sink,
        debug_path.open("w") as debug_sink,
    ):
        reader = csv.DictReader(source)
        result_writer = csv.DictWriter(
            results_sink, fieldnames=result_fieldnames, delimiter="|"
        )
        result_writer.writeheader()
        results_sink.flush()
        debug_sink.flush()
        os.fsync(results_sink.fileno())
        os.fsync(debug_sink.fileno())

        for index, row in enumerate(reader):
            target_node = int(row.get("target_node", 0))
            ingress_node = target_node % config.cluster.num_nodes
            payload = {
                "text": row.get("text", ""),
                "max_tokens": (
                    int(row["max_tokens"])
                    if row.get("max_tokens")
                    else config.sglang.max_tokens
                ),
            }
            result = await send_request(base_port + ingress_node, payload)
            result_writer.writerow(
                {
                    "request_id": index,
                    "subject": row.get("subject"),
                    "target_node": target_node,
                    "ingress_node": ingress_node,
                    "processed_by": result.get("node_id"),
                    "routed_from": result.get("routed_from"),
                    "routed_to": result.get("routed_to"),
                    "prefix_match_input_tokens": result.get(
                        "prefix_match_input_tokens"
                    ),
                    "expected_cache_hit_tokens": result.get(
                        "expected_cache_hit_tokens"
                    ),
                    "expected_cache_hit_ratio": result.get(
                        "expected_cache_hit_ratio"
                    ),
                    "sglang_prompt_tokens": result.get("sglang_prompt_tokens"),
                    "sglang_cache_hit_tokens": result.get(
                        "sglang_cache_hit_tokens"
                    ),
                    "sglang_cache_hit_ratio": result.get(
                        "sglang_cache_hit_ratio"
                    ),
                    "local_snapshot_match_after_process": result.get(
                        "local_snapshot_match_after_process"
                    ),
                    "total_time_s": result.get("total_time_s"),
                    "generated_text": result.get("generated_text"),
                    "error": result.get("error"),
                }
            )
            routing_debug = result.get("routing_debug") or {}
            selected_node = routing_debug.get("selected_node_id")
            all_expected_matches = routing_debug.get(
                "all_expected_cache_hit_tokens"
            ) or {}
            debug_record = {
                "request_id": index,
                "subject": row.get("subject"),
                "target_node": target_node,
                "ingress_node": ingress_node,
                "selected_node": selected_node,
                "processed_by": result.get("node_id"),
                "routed_from": result.get("routed_from"),
                "routed_to": result.get("routed_to"),
                "prefix_match_input_tokens": result.get(
                    "prefix_match_input_tokens"
                ),
                "sglang_prompt_tokens": result.get("sglang_prompt_tokens"),
                "sglang_cache_hit_tokens": result.get(
                    "sglang_cache_hit_tokens"
                ),
                "sglang_cache_hit_ratio": result.get("sglang_cache_hit_ratio"),
                "per_node_expected_cache_hits": [
                    {
                        "node_id": int(candidate_node),
                        "expected_cache_hit_tokens": expected_tokens,
                        "was_selected": int(candidate_node) == selected_node,
                    }
                    for candidate_node, expected_tokens in sorted(
                        all_expected_matches.items(),
                        key=lambda item: int(item[0]),
                    )
                ],
                "local_snapshot_match_after_process": result.get(
                    "local_snapshot_match_after_process"
                ),
            }
            debug_sink.write(json.dumps(debug_record, ensure_ascii=True) + "\n")

            results_sink.flush()
            debug_sink.flush()
            os.fsync(results_sink.fileno())
            os.fsync(debug_sink.fileno())
            logging.info("Wrote result for request %s to %s", index, results_path)

    logging.info("Demo results written to %s", results_path)
    logging.info("Debug routing data written to %s", debug_path)
    return True


async def stop_nodes(nodes: list[InferenceNode]):
    for node in reversed(nodes):
        try:
            await node.stop()
        except Exception:
            logging.exception("Failed while stopping node %s", node.node_id)


async def main(config_path: str):
    config = load_config(config_path)

    run_dir = resolve_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(run_dir)

    tokenizer = AutoTokenizer.from_pretrained(config.sglang.model_path)
    network = NetworkSimulator(
        config.network.lan_delay_ms,
        config.network.delay_jitter_ms,
    )

    all_node_ids = list(range(config.cluster.num_nodes))
    all_ports = {
        node_id: config.cluster.base_port + node_id for node_id in all_node_ids
    }
    all_sglang_ports = {
        node_id: config.cluster.sglang_base_port + node_id
        for node_id in all_node_ids
    }
    ensure_ports_available(all_ports, "Inference node")
    ensure_ports_available(all_sglang_ports, "SGLang backend")

    nodes = [
        InferenceNode(
            node_id=node_id,
            port=all_ports[node_id],
            sglang_port=all_sglang_ports[node_id],
            config=config,
            all_node_ids=all_node_ids,
            all_ports=all_ports,
            network=network,
            tokenizer=tokenizer,
            project_root=PROJECT_ROOT,
        )
        for node_id in all_node_ids
    ]
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, RuntimeError):
            pass

    try:
        for node in nodes:
            await node.start()

        for node in nodes:
            await node.broadcast_snapshot()

        logging.info("Started %s SGLang-backed nodes.", config.cluster.num_nodes)

        processed_demo = await maybe_run_demo(run_dir, config, config.cluster.base_port)

        if processed_demo and not config.requests.keep_running_after_demo:
            logging.info("All configured requests processed; shutting down cluster.")
            return

        print()
        print("=" * 60)
        print("SGLANG-BACKED DISTRIBUTED INFERENCE CLUSTER")
        print("=" * 60)
        print(f"Node API ports: {all_ports}")
        print(f"SGLang ports:   {all_sglang_ports}")
        print(f"Router policy:  {config.router.policy}")
        print(
            "Example request:"
            f" curl -X POST http://127.0.0.1:{config.cluster.base_port}/infer"
            " -H 'Content-Type: application/json'"
            " -d '{\"text\": \"What is the capital of France?\"}'"
        )
        print("=" * 60)

        await stop_event.wait()
        print("\nShutdown requested...")
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await stop_nodes(nodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SGLang-backed distributed inference cluster"
    )
    parser.add_argument(
        "--config",
        default="config/sglang.yaml",
        help="Path to the SGLang cluster config file",
    )
    args = parser.parse_args()
    asyncio.run(main(args.config))
