import random
from src.radix_tree import RadixTree


def mock_tokenize(text: str) -> list[int]:
    """Deterministic tokenizer: split on whitespace, hash each word to [0, 32000)."""
    return [hash(w) % 32000 for w in text.split()]


class Router:
    """Per-node cache-aware router with approximate radix trees for all nodes."""

    def __init__(
        self,
        own_node_id: int,
        all_node_ids: list[int],
        policy: str = "cache_aware",
    ):
        self.own_node_id = own_node_id
        self.all_node_ids = list(all_node_ids)
        self.policy = policy

        # Approximate radix tree per node (metadata only)
        self.approx_trees: dict[int, RadixTree] = {
            nid: RadixTree() for nid in all_node_ids
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

    def _cache_aware(self, token_ids: list[int]) -> int:
        """Cache-aware routing: longest prefix match with load balancing."""
        matches = {}
        for nid in self.all_node_ids:
            match_len, _ = self.approx_trees[nid].match_prefix(token_ids)
            matches[nid] = match_len

        # Find best cache match
        best_node = max(matches, key=matches.get)

        # No cache hit anywhere — process locally
        if matches[best_node] == 0:
            return self.own_node_id

        return best_node

    def _round_robin(self) -> int:
        nid = self.all_node_ids[self._rr_index % len(self.all_node_ids)]
        self._rr_index += 1
        return nid

    def update_load(self, node_id: int, delta: int):
        self.node_loads[node_id] = max(0, self.node_loads[node_id] + delta)

    def update_approx_tree(self, node_id: int, token_ids: list[int]):
        """Insert tokens into a node's approximate tree (lazy update after routing)."""
        self.approx_trees[node_id].insert(token_ids)

    # TODO: Use optimized way for updation and deletion of the radix tree.
    def replace_approx_tree(self, node_id: int, sequences: list[list[int]]):
        """Replace approximate tree for a node with fresh data from broadcast."""
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
