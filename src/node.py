import asyncio
import json
import random
import time
import logging
from aiohttp import web

from src.radix_tree import RadixTree
from src.kv_cache import KVCacheManager
from src.router import Router, mock_tokenize
from src.network import NetworkSimulator
from src.config import Config

logger = logging.getLogger(__name__)


class MockLLM:
    """Simulates LLM inference with O(n^2) prefill + O(n) decode timing."""

    def __init__(self, prefill_cost_factor: float, decode_cost_factor: float):
        self.prefill_cost = prefill_cost_factor
        self.decode_cost = decode_cost_factor

    async def generate(
        self, input_tokens: list[int], cached_prefix_len: int, max_output_tokens: int
    ) -> tuple[list[int], float]:
        """
        Simulate inference.
        - Prefill: O(n^2) on uncached tokens
        - Decode: O(n) per output token (n = current sequence length)
        Returns (output_token_ids, elapsed_seconds).
        """
        new_tokens = len(input_tokens) - cached_prefix_len # 9  
        prefill_time = self.prefill_cost * (new_tokens ** 2) # LP^2D

        seq_len = len(input_tokens)
        decode_time = 0.0
        for i in range(max_output_tokens):
            decode_time += self.decode_cost * (seq_len + i) # SUM i_0_n(L((P+i)D + D^2))

        total_time = prefill_time + decode_time
        await asyncio.sleep(total_time)

        output_tokens = [random.randint(0, 31999) for _ in range(max_output_tokens)]
        return output_tokens, total_time


class InferenceNode:
    """A single node in the distributed inference cluster."""

    def __init__(
        self,
        node_id: int,
        port: int,
        config: Config,
        all_node_ids: list[int],
        all_ports: dict[int, int],
        network: NetworkSimulator,
    ):
        self.node_id = node_id
        self.port = port
        self.config = config
        self.all_node_ids = all_node_ids
        self.all_ports = all_ports  # node_id -> port

        self.radix_tree = RadixTree()
        self.kv_cache = KVCacheManager(config.node.max_tokens)
        self.llm = MockLLM(config.llm.prefill_cost_factor, config.llm.decode_cost_factor)
        self.router = Router(
            own_node_id=node_id,
            all_node_ids=all_node_ids,
            policy=config.router.policy,
        )
        self.network = network

        self.incoming_queue: asyncio.Queue = asyncio.Queue()
        self.running_requests: int = 0

        self.app = web.Application()
        self._setup_routes()
        self._runner = None
        self._queue_task = None
        self._broadcast_task = None
        self.request_log: list[dict] = []

    def _setup_routes(self):
        self.app.router.add_post("/infer", self.handle_infer)
        self.app.router.add_post("/internal_infer", self.handle_internal_infer)
        self.app.router.add_get("/status", self.handle_status)
        self.app.router.add_get("/tree", self.handle_tree)
        self.app.router.add_post("/sync_tree", self.handle_sync_tree)

    # --- HTTP Handlers ---

    async def handle_infer(self, request: web.Request) -> web.Response:
        """External request: enqueue and return response when done."""
        body = await request.json()
        text = body.get("text", "")
        future = asyncio.get_event_loop().create_future()
        await self.incoming_queue.put((text, future))
        result = await future
        return web.json_response(result)

    async def handle_internal_infer(self, request: web.Request) -> web.Response:
        """Internal forwarded request: process directly (no re-routing)."""
        body = await request.json()
        token_ids = body["token_ids"]
        result = await self._process_request(token_ids)
        return web.json_response(result)

    async def handle_status(self, request: web.Request) -> web.Response:
        cache_stats = self.kv_cache.stats()
        return web.json_response({
            "node_id": self.node_id,
            "port": self.port,
            "queue_length": self.incoming_queue.qsize(),
            "running_requests": self.running_requests,
            "cache": cache_stats,
            "radix_tree_tokens": self.radix_tree.total_tokens,
            "router": self.router.get_stats(),
        })

    async def handle_tree(self, request: web.Request) -> web.Response:
        return web.json_response({
            "node_id": self.node_id,
            "actual_tree": self.radix_tree.pretty_print(),
            "approx_trees": {
                str(nid): self.router.approx_trees[nid].pretty_print()
                for nid in self.all_node_ids
            },
        })

    async def handle_sync_tree(self, request: web.Request) -> web.Response:
        """Receive broadcast: update approximate tree for the sending node."""
        body = await request.json()
        sender_id = body["node_id"]
        sequences = body["sequences"]
        self.router.replace_approx_tree(sender_id, sequences)
        return web.json_response({"status": "ok"})

    # --- Core Processing ---

    async def _process_request(self, token_ids: list[int]) -> dict:
        """Process an inference request on this node."""
        self.running_requests += 1
        self.router.update_load(self.node_id, 1)
        start_time = time.time()

        try:
            # 1. Match prefix in actual radix tree
            match_len, matched_nodes = self.radix_tree.match_prefix(token_ids)
            self.radix_tree.inc_ref(matched_nodes)

            # 2. Ensure enough memory for new tokens
            new_token_count = len(token_ids) - match_len + self.config.llm.max_output_tokens
            evicted_total = 0
            while self.kv_cache.free_tokens < new_token_count:
                result = self.radix_tree.evict_lru()
                if result == 0 or result is None:
                    break
                freed_tokens, cache_id = result
                if cache_id:
                    self.kv_cache.free(cache_id)
                evicted_total += freed_tokens
            if evicted_total > 0:
                logger.info(
                    f"Node {self.node_id}: evicted {evicted_total} tokens "
                    f"(needed {new_token_count}, free {self.kv_cache.free_tokens})"
                )

            # 3. Run mock LLM inference
            output_tokens, inference_time = await self.llm.generate(
                token_ids, match_len, self.config.llm.max_output_tokens
            )

            # 4. Allocate KV cache for new tokens
            full_sequence = token_ids + output_tokens
            new_tokens_to_cache = len(full_sequence) - match_len
            entry = self.kv_cache.allocate(new_tokens_to_cache)
            cache_id = entry.cache_id if entry else None

            # 5. Insert full sequence into actual radix tree
            self.radix_tree.insert(full_sequence, kv_cache_id=cache_id)

            # 6. Dec ref on matched nodes
            self.radix_tree.dec_ref(matched_nodes)

            elapsed = time.time() - start_time
            result = {
                "node_id": self.node_id,
                "input_tokens": len(token_ids),
                "output_tokens": len(output_tokens),
                "cache_hit_tokens": match_len,
                "cache_hit_ratio": round(match_len / len(token_ids), 4) if token_ids else 0,
                "inference_time_s": round(inference_time, 4),
                "total_time_s": round(elapsed, 4),
                "cache_utilization": round(self.kv_cache.utilization, 4),
            }
            self.request_log.append(result)
            return result

        finally:
            self.running_requests -= 1
            self.router.update_load(self.node_id, -1)

    # --- Queue Processing Loop ---

    async def _queue_loop(self):
        """Process requests from incoming queue one-by-one."""
        while True:
            text, future = await self.incoming_queue.get()
            try:
                token_ids = mock_tokenize(text)

                # Route decision
                best_node = self.router.route(token_ids)

                if best_node == self.node_id:
                    result = await self._process_request(token_ids)
                else:
                    # Forward to another node
                    # TODO: redirecting the redirects. Need to send it to the router.
                    # TODO: parallelize update the approx tree and request forwarding & error handling.
                    logger.info(
                        f"Node {self.node_id}: routing to Node {best_node}"
                    )
                    result = await self.network.forward_request(
                        self.all_ports[best_node], token_ids
                    )
                    self.router.update_approx_tree(best_node, token_ids)
                    result["routed_from"] = self.node_id
                    result["routed_to"] = best_node

                future.set_result(result)
            except Exception as e:
                future.set_result({"error": str(e), "node_id": self.node_id})

            # Check broadcast trigger after processing
            await self._maybe_broadcast()

    # --- Broadcast Sync ---

    async def _maybe_broadcast(self):
        """Broadcast actual tree state if trigger condition is met."""
        trigger = self.config.broadcast.trigger
        if trigger == "queue_length":
            if self.incoming_queue.qsize() < self.config.broadcast.queue_threshold:
                await self._do_broadcast()

    async def _broadcast_timer_loop(self):
        """Timer-based broadcast loop."""
        while True:
            await asyncio.sleep(self.config.broadcast.interval_s)
            await self._do_broadcast()

    async def _do_broadcast(self):
        """Send actual tree metadata to all other nodes."""
        sequences = self.radix_tree.get_all_sequences()
        if not sequences:
            return

        data = {"node_id": self.node_id, "sequences": sequences}
        all_ports = [self.all_ports[nid] for nid in self.all_node_ids]
        await self.network.broadcast(self.port, all_ports, "sync_tree", data)

    # --- Lifecycle ---

    async def start(self):
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "localhost", self.port)
        await site.start()
        logger.info(f"Node {self.node_id} started on port {self.port}")

        # Start queue processing loop
        self._queue_task = asyncio.create_task(self._queue_loop())

        # Start broadcast timer if configured
        if self.config.broadcast.trigger == "timer":
            self._broadcast_task = asyncio.create_task(self._broadcast_timer_loop())

    async def stop(self):
        if self._queue_task:
            self._queue_task.cancel()
        if self._broadcast_task:
            self._broadcast_task.cancel()
        if self._runner:
            await self._runner.cleanup()
        logger.info(f"Node {self.node_id} stopped")
