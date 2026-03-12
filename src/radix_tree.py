import time
from typing import Optional


class RadixNode:
    __slots__ = [
        "token_ids", "children", "ref_count",
        "last_access", "kv_cache_id", "parent",
    ]

    def __init__(self, token_ids: tuple = (), parent: "RadixNode | None" = None):
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
    def __init__(self):
        self.root = RadixNode()
        self.total_tokens: int = 0

    def match_prefix(self, token_ids: list[int]) -> tuple[int, list[RadixNode]]:
        """Find longest matching prefix. Returns (match_length, list of matched nodes)."""
        node = self.root
        matched = 0
        pos = 0
        matched_nodes = []

        while pos < len(token_ids):
            first_token = token_ids[pos]
            if first_token not in node.children:
                break

            child = node.children[first_token]
            edge = child.token_ids
            edge_len = len(edge)
            remaining = len(token_ids) - pos

            match_len = 0
            for i in range(min(edge_len, remaining)):
                if edge[i] != token_ids[pos + i]:
                    break
                match_len += 1

            if match_len == 0:
                break

            matched += match_len
            pos += match_len

            if match_len < edge_len:
                # Partial edge match — can't descend further
                break

            # Full edge match
            child.last_access = time.time()
            matched_nodes.append(child)
            node = child

        return matched, matched_nodes

    def insert(self, token_ids: list[int], kv_cache_id: str | None = None):
        """Insert token sequence into tree, splitting edges as needed."""
        if not token_ids:
            return

        node = self.root
        pos = 0

        while pos < len(token_ids):
            first_token = token_ids[pos]

            if first_token not in node.children:
                # Create new leaf
                new_node = RadixNode(
                    token_ids=tuple(token_ids[pos:]),
                    parent=node,
                )
                new_node.kv_cache_id = kv_cache_id
                new_node.last_access = time.time()
                node.children[first_token] = new_node
                self.total_tokens += len(new_node.token_ids)
                return

            child = node.children[first_token]
            edge = child.token_ids
            edge_len = len(edge)
            remaining = len(token_ids) - pos

            match_len = 0
            for i in range(min(edge_len, remaining)):
                if edge[i] != token_ids[pos + i]:
                    break
                match_len += 1

            if match_len == 0:
                # Shouldn't happen since first_token matched
                break

            if match_len < edge_len:
                # Partial match — split the edge
                mid_node = RadixNode(
                    token_ids=edge[:match_len],
                    parent=node,
                )
                mid_node.last_access = time.time()

                # Move original child under mid_node
                child.token_ids = edge[match_len:]
                child.parent = mid_node
                mid_node.children[child.token_ids[0]] = child

                # Replace in parent
                node.children[first_token] = mid_node

                # Add remaining tokens as new branch
                pos += match_len
                if pos < len(token_ids):
                    new_node = RadixNode(
                        token_ids=tuple(token_ids[pos:]),
                        parent=mid_node,
                    )
                    new_node.kv_cache_id = kv_cache_id
                    new_node.last_access = time.time()
                    mid_node.children[token_ids[pos]] = new_node
                    self.total_tokens += len(new_node.token_ids)

                return

            # Full edge match — descend
            pos += match_len
            child.last_access = time.time()
            node = child

        # All tokens already in tree — nothing to add

    def evict_lru(self) -> int:
        """Evict LRU leaf with ref_count == 0. Returns tokens freed."""
        leaves = []
        self._collect_leaves(self.root, leaves)

        evictable = [l for l in leaves if l.ref_count == 0]
        if not evictable:
            return 0

        victim = min(evictable, key=lambda n: n.last_access)
        freed = len(victim.token_ids)
        cache_id = victim.kv_cache_id

        # Remove from parent
        parent = victim.parent
        if parent is not None:
            key = victim.token_ids[0]
            if key in parent.children:
                del parent.children[key]

            # Merge parent with its only child if parent is not root
            if (
                len(parent.children) == 1
                and parent is not self.root
                and parent.ref_count == 0
            ):
                only_child = list(parent.children.values())[0]
                parent.token_ids = parent.token_ids + only_child.token_ids
                parent.kv_cache_id = only_child.kv_cache_id
                parent.children = only_child.children
                parent.last_access = only_child.last_access
                parent.ref_count = only_child.ref_count
                for c in parent.children.values():
                    c.parent = parent

        self.total_tokens -= freed
        return freed, cache_id

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
