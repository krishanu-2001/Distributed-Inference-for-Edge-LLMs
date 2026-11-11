import time
from typing import Optional


class RadixNode:
    __slots__ = [
        "token_ids", "children", "ref_count",
        "last_access", "kv_cache_id", "parent",
    ]

    def __init__(self, token_ids: tuple = (), parent: Optional["RadixNode"] = None):
        self.token_ids: tuple = token_ids
        self.children: dict[int, "RadixNode"] = {}
        self.ref_count: int = 0
        self.last_access: float = time.time()
        self.kv_cache_id: Optional[str] = None
        self.parent: Optional["RadixNode"] = parent

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class RadixTree:
    """Prefix trie over token ids.

    If `kv_cache` is given (a node's *actual* tree), every node owns one KV
    entry covering exactly its edge tokens: inserts allocate, edge splits split
    the entry, evictions free it. Approximate (peer-estimate) trees pass None.
    """

    def __init__(self, kv_cache=None):
        self.root = RadixNode()
        self.total_tokens: int = 0
        self.kv_cache = kv_cache

    def match_prefix(
        self, token_ids: list[int], touch: bool = True
    ) -> tuple[int, list[RadixNode]]:
        """Find longest matching prefix. Returns (match_length, fully matched nodes).

        match_length includes a partial match into the last edge. Set
        touch=False for read-only probes (e.g. routing) so LRU order is unchanged.
        """
        node = self.root
        matched = 0
        pos = 0
        matched_nodes = []
        now = time.time()

        while pos < len(token_ids):
            child = node.children.get(token_ids[pos])
            if child is None:
                break

            match_len = self._common_len(child.token_ids, token_ids, pos)
            matched += match_len
            pos += match_len

            if match_len < len(child.token_ids):
                # Partial edge match: can't descend further
                break

            if touch:
                child.last_access = now
            matched_nodes.append(child)
            node = child

        return matched, matched_nodes

    @staticmethod
    def _common_len(edge: tuple, token_ids: list[int], pos: int) -> int:
        n = min(len(edge), len(token_ids) - pos)
        i = 0
        while i < n and edge[i] == token_ids[pos + i]:
            i += 1
        return i

    def _new_node(self, token_ids: tuple, parent: RadixNode) -> RadixNode:
        node = RadixNode(token_ids=token_ids, parent=parent)
        if self.kv_cache is not None:
            entry = self.kv_cache.allocate(len(token_ids))
            if entry is None:
                raise MemoryError(
                    f"KV cache full: need {len(token_ids)}, "
                    f"free {self.kv_cache.free_tokens}"
                )
            node.kv_cache_id = entry.cache_id
        parent.children[token_ids[0]] = node
        self.total_tokens += len(token_ids)
        return node

    def insert(self, token_ids: list[int]) -> int:
        """Insert token sequence, splitting edges as needed. Returns tokens added.

        With a kv_cache attached, the caller must ensure
        len(token_ids) - match_prefix(token_ids)[0] tokens are free first.
        """
        node = self.root
        pos = 0
        now = time.time()

        while pos < len(token_ids):
            child = node.children.get(token_ids[pos])
            if child is None:
                self._new_node(tuple(token_ids[pos:]), node).last_access = now
                return len(token_ids) - pos

            edge = child.token_ids
            match_len = self._common_len(edge, token_ids, pos)

            if match_len < len(edge):
                # Partial match: split edge into mid (shared part) + child (rest)
                mid = RadixNode(token_ids=edge[:match_len], parent=node)
                mid.last_access = now
                if self.kv_cache is not None and child.kv_cache_id is not None:
                    mid.kv_cache_id = self.kv_cache.split(
                        child.kv_cache_id, match_len
                    ).cache_id
                # No ref_count copy needed: mid always has `child` below it,
                # so it can't become an evictable leaf while child is pinned.
                child.token_ids = edge[match_len:]
                child.parent = mid
                mid.children[child.token_ids[0]] = child
                node.children[mid.token_ids[0]] = mid
                node = mid
            else:
                child.last_access = now
                node = child
            pos += match_len

        return 0

    def evict_lru(self) -> int:
        """Evict the LRU leaf with ref_count == 0. Returns tokens freed (0 if none).

        Nodes are never merged after eviction: merging would orphan nodes held
        by in-flight requests and complicate per-node KV ownership.
        """
        leaves = []
        self._collect_leaves(self.root, leaves)
        evictable = [l for l in leaves if l.ref_count == 0]
        if not evictable:
            return 0

        victim = min(evictable, key=lambda n: n.last_access)
        del victim.parent.children[victim.token_ids[0]]
        freed = len(victim.token_ids)
        if self.kv_cache is not None and victim.kv_cache_id is not None:
            self.kv_cache.free(victim.kv_cache_id)
        self.total_tokens -= freed
        return freed

    def _collect_leaves(self, node: RadixNode, leaves: list):
        if node.is_leaf and node is not self.root:
            leaves.append(node)
        for child in node.children.values():
            self._collect_leaves(child, leaves)

    def inc_ref(self, nodes: list[RadixNode]):
        for n in nodes:
            n.ref_count += 1

    def dec_ref(self, nodes: list[RadixNode]):
        for n in nodes:
            n.ref_count = max(0, n.ref_count - 1)

    def get_all_sequences(self) -> list[list[int]]:
        """Return all token sequences stored in the tree (for broadcast sync)."""
        sequences = []
        self._collect_sequences(self.root, [], sequences)
        return sequences

    def _collect_sequences(self, node: RadixNode, prefix: list[int], out: list):
        if node is not self.root:
            prefix = prefix + list(node.token_ids)
        if node.is_leaf and node is not self.root:
            out.append(prefix)
        for child in node.children.values():
            self._collect_sequences(child, prefix, out)

    def pretty_print(self) -> str:
        lines = []
        self._pp(self.root, lines, "", True)
        return "\n".join(lines)

    def _pp(self, node: RadixNode, lines: list, prefix: str, is_last: bool):
        if node is self.root:
            lines.append(f"ROOT (total_tokens={self.total_tokens})")
        else:
            connector = "└── " if is_last else "├── "
            tokens_str = list(node.token_ids)
            extras = []
            if node.kv_cache_id:
                extras.append(f"cache={node.kv_cache_id[:8]}")
            if node.ref_count > 0:
                extras.append(f"ref={node.ref_count}")
            extra = f" ({', '.join(extras)})" if extras else ""
            lines.append(f"{prefix}{connector}{tokens_str}{extra}")

        children = list(node.children.values())
        for i, child in enumerate(children):
            is_last_child = i == len(children) - 1
            if node is self.root:
                child_prefix = ""
            else:
                child_prefix = prefix + ("    " if is_last else "│   ")
            self._pp(child, lines, child_prefix, is_last_child)
