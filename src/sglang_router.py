"""
External cache-aware router for SGLang servers.

Mirrors the mock experiment's architecture:
  - Maintains an approximate radix tree for each SGLang node
  - Polls each node's /v1/cache/tree at a regular interval
  - Routes incoming requests to the node with the best prefix match
  - Forwards the request to that node's /v1/completions endpoint

This is equivalent to each mock InferenceNode having a Router with
approx_trees for all nodes — but implemented as an external proxy
since we can't inject code into SGLang servers.

Usage:
  router = SGLangRouter(ports=[8100, 8101], poll_interval=2.0)
  await router.start()
  node_idx, match_len = router.route(token_ids)
  result = await router.forward(node_idx, prompt, max_tokens=64)
  await router.stop()
"""

import asyncio
import logging
import time

import aiohttp

from src.sglang_main import (
    PrefixTrie,
    TreeSnapshotCache,
    build_trie_from_snapshot,
)

logger = logging.getLogger(__name__)


class SGLangRouter:
    """External cache-aware router that mirrors the mock experiment's pattern.

    Like the mock Router, it keeps an approximate tree per node and routes
    based on longest prefix match. The key difference:
    - Mock: each node polls others via /sync_tree broadcast
    - SGLang: this router polls all nodes via /v1/cache/tree

    The routing logic is identical: find the node whose radix tree has the
    longest prefix match for the incoming request's token IDs.
    """

    def __init__(
        self,
        ports: list[int],
        poll_interval: float = 2.0,
        tokenizer=None,
    ):
        self.ports = ports
        self.num_nodes = len(ports)
        self.poll_interval = poll_interval
        self.tokenizer = tokenizer

        # Per-node state — mirrors Router.approx_trees from mock
        self._snapshot_cache = TreeSnapshotCache(ports, poll_interval=poll_interval)
        self._session: aiohttp.ClientSession | None = None

        # Stats
        self.total_requests = 0
        self.total_routed = 0  # requests sent to a different node than default
        self.route_decisions: list[dict] = []

    async def start(self):
        """Start the router and begin polling node trees."""
        self._session = aiohttp.ClientSession()
        await self._snapshot_cache.start()
        # Wait for first poll cycle
        await asyncio.sleep(self.poll_interval + 0.5)
        logger.info(
            f"SGLangRouter started: {self.num_nodes} nodes, "
            f"poll interval={self.poll_interval}s"
        )

    async def stop(self):
        """Stop polling and clean up."""
        await self._snapshot_cache.stop()
        if self._session:
            await self._session.close()

    def route(self, token_ids: list[int], default_node: int = 0) -> tuple[int, int]:
        """Route based on longest prefix match across all nodes' trees.

        Returns (best_node_index, prefix_match_length).
        Falls back to default_node if no prefix match found.
        """
        best_node, match_len = self._snapshot_cache.route(token_ids)

        # If no match anywhere, use the default node (like mock's own_node_id)
        if match_len == 0:
            best_node = default_node

        return best_node, match_len

    async def forward_completion(
        self,
        node_idx: int,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.0,
    ) -> dict:
        """Forward a completion request to a specific node."""
        port = self.ports[node_idx]
        url = f"http://127.0.0.1:{port}/v1/completions"
        payload = {
            "model": "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        start = time.time()
        try:
            async with self._session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                data = await resp.json()
                elapsed = time.time() - start

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

    async def route_and_forward(
        self,
        prompt: str,
        default_node: int = 0,
        max_tokens: int = 64,
        temperature: float = 0.0,
    ) -> dict:
        """Full routing pipeline: tokenize -> route -> forward -> return result.

        This is the equivalent of InferenceNode._queue_loop() in the mock:
        it tokenizes, checks all approximate trees, and forwards to the best node.
        """
        # Tokenize
        if self.tokenizer:
            token_ids = self.tokenizer.encode(prompt)
        else:
            # Fallback hash tokenizer (same as mock_tokenize)
            token_ids = [hash(w) % 32000 for w in prompt.split()]

        # Route — mirrors Router._cache_aware()
        chosen_node, match_len = self.route(token_ids, default_node=default_node)

        was_routed = chosen_node != default_node
        if was_routed:
            self.total_routed += 1

        # Forward to chosen node
        result = await self.forward_completion(
            chosen_node, prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Compute cache hit ratio
        input_tokens = result["input_tokens"]
        cache_hit_ratio = round(
            match_len / input_tokens, 4
        ) if input_tokens > 0 and match_len > 0 else 0.0

        # Get cache utilization from snapshot
        stats = self._snapshot_cache.get_snapshot_stats(chosen_node)

        self.total_requests += 1

        # Enrich result with routing info
        result.update({
            "processed_by": chosen_node,
            "routed_from": default_node,
            "routed_to": chosen_node,
            "cache_hit_tokens": match_len,
            "cache_hit_ratio": cache_hit_ratio,
            "cache_utilization": stats["total_tokens"],
        })

        decision = {
            "request_id": self.total_requests - 1,
            "default_node": default_node,
            "chosen_node": chosen_node,
            "match_len": match_len,
            "was_routed": was_routed,
        }
        self.route_decisions.append(decision)

        return result

    def get_stats(self) -> dict:
        """Return routing statistics — mirrors Router.get_stats()."""
        return {
            "total_requests": self.total_requests,
            "total_routed": self.total_routed,
            "route_ratio": round(
                self.total_routed / self.total_requests, 4
            ) if self.total_requests > 0 else 0.0,
            "approx_tree_sizes": {
                i: self._snapshot_cache.get_snapshot_stats(i)["total_tokens"]
                for i in range(self.num_nodes)
            },
        }
