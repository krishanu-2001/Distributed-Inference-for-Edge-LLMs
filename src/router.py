import zlib

from src.radix_tree import RadixTree


def mock_tokenize(text: str) -> list[int]:
    """Deterministic tokenizer: split on whitespace, hash each word to [0, 32000).

    Uses crc32 rather than hash(), which is randomized per process
    (PYTHONHASHSEED) and would make token ids differ between runs.
    """
    return [zlib.crc32(w.encode()) % 32000 for w in text.split()]


class Router:
    """Per-node cache-aware router: exact local tree + approximate trees for peers."""

    def __init__(
        self,
        own_node_id: int,
        all_node_ids: list[int],
        policy: str = "cache_aware",
        local_tree: RadixTree | None = None,
    ):
        self.own_node_id = own_node_id
        # The node's exact local tree; used instead of an estimate for itself.
        self.local_tree = local_tree
        self.all_node_ids = list(all_node_ids)
        self.policy = policy

        # Approximate radix tree per peer (metadata only)
        self.approx_trees: dict[int, RadixTree] = {
            nid: RadixTree() for nid in all_node_ids if nid != own_node_id
        }
        # Tracked load per node (running request count)
        self.node_loads: dict[int, int] = {nid: 0 for nid in all_node_ids}
        self._rr_index = 0

    # TODO: Use routing from LLM-d for this.
    def route(self, token_ids: list[int]) -> int:
        """Decide which node should handle this request. Returns node_id."""
        if self.policy == "round_robin":
            return self._round_robin()
        elif self.policy == "self":
            return self.own_node_id
        else:
            return self._cache_aware(token_ids)

    def tree_for(self, node_id: int) -> RadixTree:
        """Exact tree for this node, estimate for peers."""
        if node_id == self.own_node_id and self.local_tree is not None:
            return self.local_tree
        return self.approx_trees.setdefault(node_id, RadixTree())

    def match_lengths(self, token_ids: list[int]) -> dict[int, int]:
        """Prefix match length per node (read-only: does not touch LRU order)."""
        return {
            nid: self.tree_for(nid).match_prefix(token_ids, touch=False)[0]
            for nid in self.all_node_ids
        }

    def _cache_aware(self, token_ids: list[int]) -> int:
        """Cache-aware routing: forward only if a peer has a strictly longer match.

        Ties (including no match anywhere) stay local, which avoids a pointless
        network round trip.
        """
        matches = self.match_lengths(token_ids)
        best_node = self.own_node_id
        for nid in self.all_node_ids:
            if matches[nid] > matches[best_node]:
                best_node = nid
        return best_node

    def _round_robin(self) -> int:
        nid = self.all_node_ids[self._rr_index % len(self.all_node_ids)]
        self._rr_index += 1
        return nid

    def update_load(self, node_id: int, delta: int):
        self.node_loads[node_id] = max(0, self.node_loads[node_id] + delta)

    def update_approx_tree(self, node_id: int, token_ids: list[int]):
        """Insert tokens into a peer's approximate tree (lazy update after routing)."""
        if node_id == self.own_node_id:
            return
        self.tree_for(node_id).insert(token_ids)

    # TODO: Use optimized way for updation and deletion of the radix tree.
    def replace_approx_tree(self, node_id: int, sequences: list[list[int]]):
        """Replace approximate tree for a peer with fresh data from broadcast."""
        if node_id == self.own_node_id:
            return
        new_tree = RadixTree()
        for seq in sequences:
            new_tree.insert(seq)
        self.approx_trees[node_id] = new_tree

    def get_stats(self) -> dict:
        return {
            "policy": self.policy,
            "node_loads": dict(self.node_loads),
            "approx_tree_sizes": {
                nid: tree.total_tokens
                for nid, tree in self.approx_trees.items()
            },
        }
