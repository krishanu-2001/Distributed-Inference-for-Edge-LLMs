from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from aiohttp import web

from src.sglang_cluster.config import Config
from src.sglang_cluster.network import NetworkSimulator
from src.sglang_cluster.radix_snapshot import RadixSnapshot
from src.sglang_cluster.router import Router
from src.sglang_cluster.sglang_backend import SGLangBackend

logger = logging.getLogger(__name__)


class InferenceNode:
    """Peer-to-peer node backed by its own SGLang server."""

    PREFILL_PROXY_INTERCEPT_S = 0.10
    PREFILL_PROXY_LINEAR_COEFF_S = 2.4e-4
    PREFILL_PROXY_QUADRATIC_COEFF_S = 1.4e-7

    def __init__(
        self,
        node_id: int,
        port: int,
        sglang_port: int,
        config: Config,
        all_node_ids: list[int],
        all_ports: dict[int, int],
        network: NetworkSimulator,
        tokenizer: Any,
        project_root: Path,
    ):
        self.node_id = node_id
        self.port = port
        self.sglang_port = sglang_port
        self.config = config
        self.all_node_ids = all_node_ids
        self.all_ports = all_ports
        self.network = network
        self.tokenizer = tokenizer
        self.project_root = project_root

        self.router = Router(
            own_node_id=node_id,
            all_node_ids=all_node_ids,
            policy=config.router.policy,
        )
        self.backend = SGLangBackend(
            node_id=node_id,
            port=sglang_port,
            config=config.sglang,
            project_root=project_root,
            gpu_id=config.sglang.resolve_gpu_id(node_id),
        )

        self.incoming_queue: asyncio.Queue = asyncio.Queue()
        self.running_requests = 0
        self.request_log: list[dict[str, Any]] = []
        self.last_snapshot_at: float | None = None

        self.app = web.Application(
            client_max_size=int(
                self.config.router.sync_tree_max_payload_mb * 1024 * 1024
            )
        )
        self._setup_routes()
        self._runner = None
        self._site = None
        self._queue_task = None
        self._sync_task = None

    def _setup_routes(self):
        self.app.router.add_post("/infer", self.handle_infer)
        self.app.router.add_post("/internal_infer", self.handle_internal_infer)
        self.app.router.add_post("/sync_tree", self.handle_sync_tree)
        self.app.router.add_get("/status", self.handle_status)
        self.app.router.add_get("/tree", self.handle_tree)

    async def handle_infer(self, request: web.Request) -> web.Response:
        body = await request.json()
        future = asyncio.get_event_loop().create_future()
        await self.incoming_queue.put((body, future))
        result = await future
        return web.json_response(result)

    async def handle_internal_infer(self, request: web.Request) -> web.Response:
        setup_start = time.perf_counter()
        body = await request.json()
        payload = self._normalize_payload(body, allow_token_ids=True)
        token_ids = payload["token_ids"] or self._tokenize_for_prefix_match(
            payload["prompt_text"]
        )
        payload["token_ids"] = token_ids
        expected_matched_tokens = self.router.expected_match_for_node(
            self.node_id, token_ids
        )
        result = await self._process_local_request(
            payload,
            expected_matched_tokens,
            local_processing_setup_time_s=time.perf_counter() - setup_start,
        )
        return web.json_response(result)

    async def handle_sync_tree(self, request: web.Request) -> web.Response:
        body = await request.json()
        sender_id = int(body["node_id"])
        snapshot = RadixSnapshot.from_payload(body.get("snapshot"))
        self.router.update_snapshot(sender_id, snapshot)
        return web.json_response({"status": "ok", "node_id": self.node_id})

    async def handle_status(self, request: web.Request) -> web.Response:
        snapshot = self.router.snapshots[self.node_id]
        snapshot_age = None
        if self.last_snapshot_at is not None:
            snapshot_age = round(time.time() - self.last_snapshot_at, 3)

        return web.json_response(
            {
                "node_id": self.node_id,
                "port": self.port,
                "sglang_port": self.sglang_port,
                "queue_length": self.incoming_queue.qsize(),
                "running_requests": self.running_requests,
                "local_snapshot_nodes": snapshot.node_count,
                "local_snapshot_tokens": snapshot.total_tokens,
                "snapshot_age_s": snapshot_age,
                "router": self.router.get_stats(),
            }
        )

    async def handle_tree(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "node_id": self.node_id,
                "local_tree": self.router.snapshots[self.node_id].pretty_print(),
                "peer_trees": {
                    str(node_id): snapshot.pretty_print()
                    for node_id, snapshot in self.router.snapshots.items()
                },
            }
        )

    def _normalize_payload(
        self, body: dict[str, Any], allow_token_ids: bool = False
    ) -> dict[str, Any]:
        prompt_text = body.get("text")
        if prompt_text is not None:
            prompt_text = str(prompt_text)

        messages = body.get("messages")
        if messages is None:
            prompt_text = prompt_text or ""
            messages = [{"role": "user", "content": prompt_text}]

        if not isinstance(messages, list) or not messages:
            raise ValueError("Request must include non-empty 'messages' or 'text'.")

        if prompt_text is None:
            prompt_text = self._extract_prompt_text(messages)

        token_ids = body.get("token_ids") if allow_token_ids else None
        if token_ids is not None and not isinstance(token_ids, list):
            raise ValueError("'token_ids' must be a list when provided.")

        max_tokens = int(body.get("max_tokens", self.config.sglang.max_tokens))
        temperature = float(body.get("temperature", self.config.sglang.temperature))

        return {
            "messages": messages,
            "prompt_text": prompt_text,
            "token_ids": token_ids,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    def _extract_prompt_text(self, messages: list[dict[str, Any]]) -> str:
        user_contents = [
            self._message_content_to_text(message)
            for message in messages
            if message.get("role") == "user"
        ]
        if user_contents:
            return "\n".join(user_contents)

        return "\n".join(
            self._message_content_to_text(message) for message in messages
        )

    def _tokenize_for_prefix_match(self, prompt_text: str) -> list[int]:
        """Tokenize the original prompt exactly as it is sent to /generate."""
        token_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        return [int(token_id) for token_id in token_ids]

    def _message_content_to_text(self, message: dict[str, Any]) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=True)

    def _extract_generated_text(self, response: dict[str, Any]) -> str | None:
        text = response.get("text")
        if isinstance(text, str):
            return text
        return None

    def _round_time(self, value: float) -> float:
        return round(max(0.0, value), 4)

    def _compute_proxy_inference_time_s(self, uncached_prefill_tokens: int) -> float:
        return (
            self.PREFILL_PROXY_INTERCEPT_S
            + self.PREFILL_PROXY_LINEAR_COEFF_S * uncached_prefill_tokens
            + self.PREFILL_PROXY_QUADRATIC_COEFF_S
            * (uncached_prefill_tokens**2)
        )

    def _extract_sglang_cache_stats(self, response: dict[str, Any]) -> dict[str, Any]:
        meta_info = response.get("meta_info") or {}
        prompt_tokens = meta_info.get("prompt_tokens")
        cached_tokens = meta_info.get("cached_tokens")

        if prompt_tokens is not None:
            prompt_tokens = int(prompt_tokens)
        if cached_tokens is not None:
            cached_tokens = int(cached_tokens)

        uncached_prefill_tokens = None
        if prompt_tokens is not None:
            uncached_prefill_tokens = prompt_tokens
            if cached_tokens is not None:
                uncached_prefill_tokens = max(prompt_tokens - cached_tokens, 0)

        cache_hit_ratio = None

        if prompt_tokens is not None and cached_tokens is not None:
            cache_hit_ratio = (
                round(cached_tokens / prompt_tokens, 4) if prompt_tokens else 0.0
            )

        return {
            "sglang_prompt_tokens": prompt_tokens,
            "sglang_cache_hit_tokens": cached_tokens,
            "sglang_uncached_prefill_tokens": uncached_prefill_tokens,
            "sglang_cache_hit_ratio": cache_hit_ratio,
        }

    def _finalize_reported_timing(
        self,
        result: dict[str, Any],
        *,
        communication_time_s: float = 0.0,
        routing_processing_time_s: float = 0.0,
    ) -> None:
        result["communication_time_s"] = self._round_time(
            float(result.get("communication_time_s", 0.0)) + communication_time_s
        )
        result["routing_processing_time_s"] = self._round_time(
            float(result.get("routing_processing_time_s", 0.0))
            + routing_processing_time_s
        )
        result["local_processing_time_s"] = self._round_time(
            float(result.get("local_processing_time_s", 0.0))
        )
        result["other_processing_time_s"] = self._round_time(
            result["routing_processing_time_s"] + result["local_processing_time_s"]
        )
        result["inference_time_s"] = self._round_time(
            float(result.get("inference_time_s", 0.0))
        )
        result["actual_sglang_inference_time_s"] = self._round_time(
            float(result.get("actual_sglang_inference_time_s", 0.0))
        )
        result["total_time_s"] = self._round_time(
            result["inference_time_s"]
            + result["communication_time_s"]
            + result["other_processing_time_s"]
        )

    async def _process_local_request(
        self,
        payload: dict[str, Any],
        expected_matched_tokens: int,
        local_processing_setup_time_s: float = 0.0,
    ) -> dict[str, Any]:
        self.running_requests += 1
        self.router.update_load(self.node_id, 1)

        try:
            sglang_inference_start = time.perf_counter()
            response = await self.backend.generate(
                text=payload["prompt_text"],
                max_new_tokens=payload["max_tokens"],
                temperature=payload["temperature"],
            )
            actual_sglang_inference_time_s = (
                time.perf_counter() - sglang_inference_start
            )
            token_ids = payload["token_ids"]
            post_process_start = time.perf_counter()
            local_snapshot_match_after_process = (
                await self._refresh_local_snapshot_until_visible(token_ids)
            )
            post_inference_processing_time_s = (
                time.perf_counter() - post_process_start
            )
            sglang_cache_stats = self._extract_sglang_cache_stats(response)
            uncached_prefill_tokens = (
                sglang_cache_stats.get("sglang_uncached_prefill_tokens") or 0
            )
            proxy_inference_time_s = self._compute_proxy_inference_time_s(
                int(uncached_prefill_tokens)
            )
            inference_time_s = (
                proxy_inference_time_s
                if self.config.sglang.use_proxy_inference_time
                else actual_sglang_inference_time_s
            )
            local_processing_time_s = (
                local_processing_setup_time_s + post_inference_processing_time_s
            )

            result = {
                "node_id": self.node_id,
                "port": self.port,
                "sglang_port": self.sglang_port,
                "prefix_match_input_tokens": len(token_ids),
                "expected_cache_hit_tokens": expected_matched_tokens,
                "expected_cache_hit_ratio": (
                    round(expected_matched_tokens / len(token_ids), 4)
                    if token_ids else 0.0
                ),
                **sglang_cache_stats,
                "inference_time_s": self._round_time(inference_time_s),
                "actual_sglang_inference_time_s": self._round_time(
                    actual_sglang_inference_time_s
                ),
                "communication_time_s": 0.0,
                "routing_processing_time_s": 0.0,
                "local_processing_time_s": self._round_time(
                    local_processing_time_s
                ),
                "other_processing_time_s": 0.0,
                "total_time_s": 0.0,
                "meta_info": response.get("meta_info"),
                "generated_text": self._extract_generated_text(response),
                "local_snapshot_match_after_process": (
                    local_snapshot_match_after_process
                ),
                "response": response,
            }
            self._finalize_reported_timing(result)
            self.request_log.append(result)
            return result
        finally:
            self.running_requests -= 1
            self.router.update_load(self.node_id, -1)

    async def _queue_loop(self):
        while True:
            body, future = await self.incoming_queue.get()

            try:
                routing_start = time.perf_counter()
                payload = self._normalize_payload(body, allow_token_ids=False)
                token_ids = self._tokenize_for_prefix_match(payload["prompt_text"])
                payload["token_ids"] = token_ids

                # Peer snapshots are intentionally lazy estimates, but the ingress node
                # should always route using a fresh view of its own SGLang radix tree.
                await self.refresh_local_snapshot(broadcast=False)
                if not self.config.router.routing_enabled:
                    expected_matched_tokens = self.router.expected_match_for_node(
                        self.node_id, token_ids
                    )
                    routing_processing_time_s = time.perf_counter() - routing_start
                    result = await self._process_local_request(
                        payload,
                        expected_matched_tokens=expected_matched_tokens,
                    )
                    selected_node_id = self.node_id
                    all_expected_matches = {self.node_id: expected_matched_tokens}
                    self._finalize_reported_timing(
                        result,
                        routing_processing_time_s=routing_processing_time_s,
                    )
                else:
                    decision = self.router.route(token_ids)
                    selected_node_id = decision.node_id
                    all_expected_matches = decision.all_expected_matches
                    if decision.node_id == self.node_id:
                        routing_processing_time_s = time.perf_counter() - routing_start
                        result = await self._process_local_request(
                            payload,
                            expected_matched_tokens=decision.expected_matched_tokens,
                        )
                        self._finalize_reported_timing(
                            result,
                            routing_processing_time_s=routing_processing_time_s,
                        )
                    else:
                        routing_processing_time_s = time.perf_counter() - routing_start
                        logger.info(
                            "Node %s routing request to node %s "
                            "(expected matched tokens=%s)",
                            self.node_id,
                            decision.node_id,
                            decision.expected_matched_tokens,
                        )
                        result, communication_time_s = await self.network.forward_request(
                            self.all_ports[decision.node_id],
                            payload,
                        )
                        result["routed_from"] = self.node_id
                        result["routed_to"] = decision.node_id
                        self._finalize_reported_timing(
                            result,
                            communication_time_s=communication_time_s,
                            routing_processing_time_s=routing_processing_time_s,
                        )

                result["routing_debug"] = {
                    "ingress_node_id": self.node_id,
                    "routing_enabled": self.config.router.routing_enabled,
                    "selected_node_id": selected_node_id,
                    "all_expected_cache_hit_tokens": all_expected_matches,
                }
                future.set_result(result)
            except Exception as exc:
                future.set_result({"error": str(exc), "node_id": self.node_id})

    async def refresh_local_snapshot(self, broadcast: bool) -> RadixSnapshot:
        payload = await self.backend.get_radix_tree()
        snapshot = RadixSnapshot.from_payload(payload)
        self.router.update_snapshot(self.node_id, snapshot)
        self.last_snapshot_at = time.time()

        if broadcast:
            await self.broadcast_snapshot(payload)

        return snapshot

    async def _refresh_local_snapshot_until_visible(self, token_ids: list[int]) -> int:
        deadline = time.monotonic() + self.config.router.local_snapshot_settle_timeout_s
        poll_s = self.config.router.local_snapshot_settle_poll_s
        best_match = 0

        while True:
            snapshot = await self.refresh_local_snapshot(broadcast=False)
            best_match = snapshot.match_prefix(token_ids)
            if best_match > 0 or time.monotonic() >= deadline:
                break
            await asyncio.sleep(poll_s)

        if best_match == 0 and token_ids:
            logger.warning(
                "Node %s local SGLang radix tree did not expose the processed "
                "request within %.3fs.",
                self.node_id,
                self.config.router.local_snapshot_settle_timeout_s,
            )

        return best_match

    async def broadcast_snapshot(self, payload: Any | None = None):
        if payload is None:
            payload = self.router.snapshots[self.node_id].raw_payload

        data = {
            "node_id": self.node_id,
            "snapshot": payload,
            "captured_at": self.last_snapshot_at,
        }
        all_ports = [self.all_ports[node_id] for node_id in self.all_node_ids]
        await self.network.broadcast(self.port, all_ports, "sync_tree", data)

    async def _sync_loop(self):
        interval_s = self.config.router.snapshot_poll_interval_s
        while True:
            try:
                await self.refresh_local_snapshot(broadcast=True)
            except Exception as exc:
                logger.warning(
                    "Node %s failed to refresh radix snapshot: %s",
                    self.node_id,
                    exc,
                )
            await asyncio.sleep(interval_s)

    async def start(self):
        try:
            await self.backend.start()

            self._runner = web.AppRunner(self.app)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, "127.0.0.1", self.port)
            await self._site.start()

            await self.refresh_local_snapshot(broadcast=False)
            self._queue_task = asyncio.create_task(self._queue_loop())
            self._sync_task = asyncio.create_task(self._sync_loop())
            logger.info(
                "Node %s started on port %s with SGLang on port %s",
                self.node_id,
                self.port,
                self.sglang_port,
            )
        except Exception:
            await self.stop()
            raise

    async def stop(self):
        if self._queue_task:
            self._queue_task.cancel()
            try:
                await self._queue_task
            except asyncio.CancelledError:
                pass
            self._queue_task = None

        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None

        if self._runner:
            await self._runner.cleanup()
            self._runner = None
            self._site = None

        await self.backend.stop()
        logger.info("Node %s stopped", self.node_id)
