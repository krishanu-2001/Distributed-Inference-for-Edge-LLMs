import asyncio
import json
import random
import time
import logging
import aiohttp
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
    ) -> tuple[list[int], float, int]:
        """
        Simulate inference.
        - Prefill: O(n^2) on uncached tokens
        - Decode: O(n) per output token (n = current sequence length)
        Returns (output_token_ids, elapsed_seconds, cached_tokens). The mock's
        cache is exactly the node's radix tree, so cached_tokens = cached_prefix_len.
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
        return output_tokens, total_time, cached_prefix_len

    async def close(self):
        pass


class SGLangLLM:
    """Real inference against an SGLang server (one server per node).

    Sends token ids directly (`input_ids`) so SGLang's RadixAttention cache sees
    exactly the sequences our routers reason about. Greedy decoding with
    ignore_eos gives a fixed output length per request. SGLang's own
    `meta_info.cached_tokens` is returned as the ground-truth cache hit.
    """

    def __init__(self, url: str, timeout_s: float):
        self.url = url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def generate(
        self, input_tokens: list[int], cached_prefix_len: int, max_output_tokens: int
    ) -> tuple[list[int], float, int | None]:
        payload = {
            "input_ids": input_tokens,
            "sampling_params": {
                "max_new_tokens": max_output_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
            },
        }
        t0 = time.perf_counter()
        async with self._get_session().post(f"{self.url}/generate", json=payload) as resp:
            if resp.status != 200:
                raise RuntimeError(f"SGLang {resp.status}: {(await resp.text())[:300]}")
            data = await resp.json()
        elapsed = time.perf_counter() - t0
        if isinstance(data, list):
            data = data[0]
        meta = data.get("meta_info", {})
        output_tokens = data.get("output_ids")
        if not output_tokens:
            n = meta.get("completion_tokens", max_output_tokens)
            output_tokens = [random.randint(0, 31999) for _ in range(n)]
        return output_tokens, elapsed, meta.get("cached_tokens")

    async def close(self):
        if self._session is not None and not self._session.closed:
            await self._session.close()


def make_llm(config: Config, node_id: int):
    llm = config.llm
    if llm.backend == "mock":
        return MockLLM(llm.prefill_cost_factor, llm.decode_cost_factor)
    if llm.backend == "sglang":
        return SGLangLLM(
            f"http://{llm.sglang_host}:{llm.sglang_base_port + node_id}",
            llm.request_timeout_s,
        )
    raise ValueError(f"unknown llm.backend {llm.backend!r} (mock | sglang)")


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

        self.kv_cache = KVCacheManager(config.node.max_tokens)
        self.radix_tree = RadixTree(kv_cache=self.kv_cache)
        self.llm = make_llm(config, node_id)
        self.router = Router(
            own_node_id=node_id,
            all_node_ids=all_node_ids,
            policy=config.router.policy,
            local_tree=self.radix_tree,
        )
        self.network = network

        self.incoming_queue: asyncio.Queue = asyncio.Queue()
        self.running_requests: int = 0

        self.app = web.Application()
        self._setup_routes()
        self._runner = None
        self._queue_task = None
        self._broadcast_task = None
        self._inflight_broadcast: asyncio.Task | None = None
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
        max_out = body.get("max_output_tokens")
        future = asyncio.get_running_loop().create_future()
        await self.incoming_queue.put((text, max_out, future))
        result = await future
        return web.json_response(result)

    async def handle_internal_infer(self, request: web.Request) -> web.Response:
        """Internal forwarded request: process directly (no re-routing)."""
        body = await request.json()
        token_ids = body["token_ids"]
        result = await self._process_request(token_ids, body.get("max_output_tokens"))
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
                str(nid): tree.pretty_print()
                for nid, tree in self.router.approx_trees.items()
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

    def _ensure_free(self, needed: int) -> int:
        """Evict LRU leaves until `needed` tokens are free (or nothing is evictable).
        Returns tokens evicted."""
        evicted = 0
        while self.kv_cache.free_tokens < needed:
            freed = self.radix_tree.evict_lru()
            if freed == 0:
                break
            evicted += freed
        return evicted

    def _cache_sequence(self, sequence: list[int]) -> int:
        """Insert `sequence` into the local tree, evicting as needed.

        If memory is still short (everything left is pinned by in-flight
        requests), only the prefix that fits is cached and a warning is logged.
        Returns the number of tokens that could not be cached.
        """
        match_len, _ = self.radix_tree.match_prefix(sequence, touch=False)
        needed = len(sequence) - match_len
        self._ensure_free(needed)
        dropped = max(0, needed - self.kv_cache.free_tokens)
        if dropped:
            logger.warning(
                f"Node {self.node_id}: KV cache full, caching only "
                f"{needed - dropped}/{needed} new tokens"
            )
            sequence = sequence[: len(sequence) - dropped]
        self.radix_tree.insert(sequence)
        return dropped

    async def _process_request(
        self, token_ids: list[int], max_output_tokens: int | None = None
    ) -> dict:
        """Process an inference request on this node.

        max_output_tokens: per-request output length (e.g. from SGLang's
        benchmark); defaults to config llm.max_output_tokens."""
        if max_output_tokens is None:
            max_output_tokens = self.config.llm.max_output_tokens
        self.running_requests += 1
        self.router.update_load(self.node_id, 1)
        start_time = time.time()
        matched_nodes = []

        try:
            # 1. Match prefix in actual radix tree; pin matched nodes
            match_len, matched_nodes = self.radix_tree.match_prefix(token_ids)
            self.radix_tree.inc_ref(matched_nodes)

            # 2. Reserve memory for new prompt + output tokens
            new_token_count = (
                len(token_ids) - match_len + max_output_tokens
            )
            evicted = self._ensure_free(new_token_count)
            if evicted:
                logger.info(
                    f"Node {self.node_id}: evicted {evicted} tokens "
                    f"(needed {new_token_count}, free {self.kv_cache.free_tokens})"
                )

            # 3. Run mock LLM inference
            output_tokens, inference_time, backend_cached = await self.llm.generate(
                token_ids, match_len, max_output_tokens
            )
            # Ground truth from the backend if it reports it (SGLang); for the
            # mock, the local tree *is* the cache.
            actual_hit = match_len if backend_cached is None else backend_cached

            # 4. Cache the full sequence (allocates KV per new tree node)
            uncached = self._cache_sequence(token_ids + output_tokens)

            elapsed = time.time() - start_time
            result = {
                "node_id": self.node_id,
                "input_tokens": len(token_ids),
                "output_tokens": len(output_tokens),
                "cache_hit_tokens": actual_hit,
                "cache_hit_ratio": round(actual_hit / len(token_ids), 4) if token_ids else 0,
                "expected_hit_tokens": match_len,  # this node's tree estimate
                "inference_time_s": round(inference_time, 4),
                "total_time_s": round(elapsed, 4),
                "cache_utilization": round(self.kv_cache.utilization, 4),
                "uncached_tokens": uncached,
            }
            self.request_log.append(result)
            return result

        finally:
            self.radix_tree.dec_ref(matched_nodes)
            self.running_requests -= 1
            self.router.update_load(self.node_id, -1)

    # --- Queue Processing Loop ---

    async def _queue_loop(self):
        """Process requests from incoming queue one-by-one."""
        while True:
            text, max_out, future = await self.incoming_queue.get()
            try:
                token_ids = mock_tokenize(text)

                # Route decision, plus the hit the router expects at that node
                best_node = self.router.route(token_ids)
                route_expected = self.router.tree_for(best_node).match_prefix(
                    token_ids, touch=False
                )[0]

                if best_node == self.node_id:
                    result = await self._process_request(token_ids, max_out)
                else:
                    # Forward to another node
                    # TODO: redirecting the redirects. Need to send it to the router.
                    # TODO: parallelize update the approx tree and request forwarding & error handling.
                    logger.info(
                        f"Node {self.node_id}: routing to Node {best_node}"
                    )
                    result = await self.network.forward_request(
                        self.all_ports[best_node], token_ids, max_out
                    )
                    # Only update the estimate if the peer actually served it
                    if "error" not in result:
                        self.router.update_approx_tree(best_node, token_ids)
                    result["routed_from"] = self.node_id
                    result["routed_to"] = best_node

                result["route_expected_tokens"] = route_expected
                future.set_result(result)
            except Exception as e:
                future.set_result({"error": str(e), "node_id": self.node_id})

            # Check broadcast trigger after processing (non-blocking)
            self._maybe_broadcast()

    # --- Broadcast Sync ---

    def _maybe_broadcast(self):
        """Start a background broadcast if the trigger condition is met.

        Runs as a task so the request loop never waits on anti-entropy; skipped
        if the previous broadcast is still in flight.
        """
        trigger = self.config.broadcast.trigger
        if trigger != "queue_length":
            return
        if self.incoming_queue.qsize() >= self.config.broadcast.queue_threshold:
            return
        if self._inflight_broadcast and not self._inflight_broadcast.done():
            return
        self._inflight_broadcast = asyncio.create_task(self._do_broadcast())

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
        if self._inflight_broadcast:
            self._inflight_broadcast.cancel()
        await self.llm.close()
        if self._runner:
            await self._runner.cleanup()
        logger.info(f"Node {self.node_id} stopped")
