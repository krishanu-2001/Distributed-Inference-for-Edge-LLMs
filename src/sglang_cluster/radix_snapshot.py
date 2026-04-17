from __future__ import annotations

from copy import deepcopy
from typing import Any


def _normalize_entries(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise TypeError(f"Unsupported radix payload type: {type(payload)!r}")


def _count_tokens(node: dict[str, Any]) -> int:
    total = len(node.get("segment_token_ids", []))
    for child in node.get("children", []):
        total += _count_tokens(child)
    return total


def _count_nodes(node: dict[str, Any]) -> int:
    total = 1
    for child in node.get("children", []):
        total += _count_nodes(child)
    return total


class RadixSnapshot:
    """Lightweight parsed view of SGLang's radix-tree debug payload."""

    def __init__(self, roots: list[dict[str, Any]], raw_payload: Any):
        self.roots = roots
        self.raw_payload = raw_payload
        self.node_count = sum(_count_nodes(root) for root in roots)
        self.total_tokens = sum(_count_tokens(root) for root in roots)

    @classmethod
    def empty(cls) -> "RadixSnapshot":
        return cls(roots=[], raw_payload=[])

    @classmethod
    def from_payload(cls, payload: Any) -> "RadixSnapshot":
        entries = _normalize_entries(payload)
        roots: list[dict[str, Any]] = []

        for entry in entries:
            nodes = []
            for raw_node in entry.get("nodes", []):
                node = deepcopy(raw_node)
                node["children"] = []
                node["prefix_token_ids"] = list(node.get("prefix_token_ids", []))
                node["segment_token_ids"] = list(node.get("segment_token_ids", []))
                nodes.append(node)

            if not nodes:
                continue

            nodes_sorted = sorted(
                nodes,
                key=lambda item: (item.get("prefix_length", 0), item.get("node_id", -1)),
            )

            root = None
            for node in nodes_sorted:
                if node.get("is_root"):
                    root = node
                    continue

                node_prefix = node.get("prefix_token_ids", [])
                parent = None
                parent_len = -1

                for candidate in nodes_sorted:
                    if candidate.get("node_id") == node.get("node_id"):
                        continue

                    candidate_prefix = candidate.get("prefix_token_ids", [])
                    if len(candidate_prefix) >= len(node_prefix):
                        continue

                    if node_prefix[: len(candidate_prefix)] == candidate_prefix:
                        if len(candidate_prefix) > parent_len:
                            parent = candidate
                            parent_len = len(candidate_prefix)

                if parent is not None:
                    parent["children"].append(node)

            if root is not None:
                roots.append(root)

        if not roots:
            return cls.empty()

        return cls(roots=roots, raw_payload=payload)

    @property
    def is_empty(self) -> bool:
        return not self.roots

    def match_prefix(self, token_ids: list[int]) -> int:
        best = 0
        for root in self.roots:
            best = max(best, self._match_in_tree(token_ids, root))
        return best

    def _match_in_tree(self, token_ids: list[int], root: dict[str, Any]) -> int:
        node = root
        matched = 0

        while True:
            best_child = None
            best_match_len = 0

            for child in node.get("children", []):
                segment = child.get("segment_token_ids", [])
                start = matched
                max_cmp = min(len(segment), len(token_ids) - start)

                cur_match_len = 0
                while (
                    cur_match_len < max_cmp
                    and segment[cur_match_len] == token_ids[start + cur_match_len]
                ):
                    cur_match_len += 1

                if cur_match_len > best_match_len:
                    best_child = child
                    best_match_len = cur_match_len

            if best_child is None or best_match_len == 0:
                break

            matched += best_match_len
            if best_match_len < len(best_child.get("segment_token_ids", [])):
                break

            node = best_child

        return matched

    def pretty_print(self) -> str:
        if not self.roots:
            return "<empty>"

        lines: list[str] = []
        for index, root in enumerate(self.roots):
            lines.append(f"Tree {index}")
            lines.extend(self._render_node(root, indent="", is_last=True))
        return "\n".join(lines)

    def _render_node(
        self, node: dict[str, Any], indent: str, is_last: bool
    ) -> list[str]:
        branch = "`-- " if is_last else "|-- "
        lines = [
            (
                f"{indent}{branch}"
                f"id={node.get('node_id')} "
                f"prefix_len={node.get('prefix_length', 0)} "
                f"seg={node.get('segment_token_ids', [])}"
            )
        ]

        children = node.get("children", [])
        next_indent = indent + ("    " if is_last else "|   ")
        for index, child in enumerate(children):
            lines.extend(
                self._render_node(
                    child,
                    indent=next_indent,
                    is_last=index == len(children) - 1,
                )
            )
        return lines
