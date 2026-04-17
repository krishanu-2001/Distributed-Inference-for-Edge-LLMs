from __future__ import annotations

from dataclasses import dataclass

from src.sglang_cluster.radix_snapshot import RadixSnapshot


@dataclass
class RouteDecision:
    node_id: int
    expected_matched_tokens: int
    all_expected_matches: dict[int, int]


class Router:
    """Prefix-aware router that uses the latest radix snapshot for each node."""

    def __init__(
        self,
        own_node_id: int,
        all_node_ids: list[int],
        policy: str = "cache_aware",
    ):
        self.own_node_id = own_node_id
        self.all_node_ids = list(all_node_ids)
        self.policy = policy
        self.snapshots: dict[int, RadixSnapshot] = {
            node_id: RadixSnapshot.empty() for node_id in all_node_ids
        }
        self.node_loads: dict[int, int] = {node_id: 0 for node_id in all_node_ids}
        self._rr_index = 0

    def route(self, token_ids: list[int]) -> RouteDecision:
        if self.policy == "round_robin":
            node_id = self._round_robin()
            matches = self._collect_matches(token_ids)
            return RouteDecision(
                node_id=node_id,
                expected_matched_tokens=matches.get(node_id, 0),
                all_expected_matches=matches,
            )

        if self.policy == "self":
            matches = self._collect_matches(token_ids)
            return RouteDecision(
                node_id=self.own_node_id,
                expected_matched_tokens=matches.get(self.own_node_id, 0),
                all_expected_matches=matches,
            )

        return self._cache_aware(token_ids)

    def _cache_aware(self, token_ids: list[int]) -> RouteDecision:
        matches = self._collect_matches(token_ids)
        best_match = max(matches.values(), default=0)

        if best_match == 0:
            return RouteDecision(
                node_id=self.own_node_id,
                expected_matched_tokens=0,
                all_expected_matches=matches,
            )

        candidates = [
            node_id for node_id, matched_tokens in matches.items()
            if matched_tokens == best_match
        ]
        if self.own_node_id in candidates:
            return RouteDecision(
                node_id=self.own_node_id,
                expected_matched_tokens=best_match,
                all_expected_matches=matches,
            )

        best_node = min(
            candidates,
            key=lambda node_id: (
                self.node_loads[node_id],
                node_id,
            ),
        )
        return RouteDecision(
            node_id=best_node,
            expected_matched_tokens=best_match,
            all_expected_matches=matches,
        )

    def _round_robin(self) -> int:
        node_id = self.all_node_ids[self._rr_index % len(self.all_node_ids)]
        self._rr_index += 1
        return node_id

    def _collect_matches(self, token_ids: list[int]) -> dict[int, int]:
        return {
            node_id: snapshot.match_prefix(token_ids)
            for node_id, snapshot in self.snapshots.items()
        }

    def update_snapshot(self, node_id: int, snapshot: RadixSnapshot):
        self.snapshots[node_id] = snapshot

    def expected_match_for_node(self, node_id: int, token_ids: list[int]) -> int:
        return self.snapshots[node_id].match_prefix(token_ids)

    def update_load(self, node_id: int, delta: int):
        self.node_loads[node_id] = max(0, self.node_loads[node_id] + delta)

    def get_stats(self) -> dict:
        return {
            "policy": self.policy,
            "node_loads": dict(self.node_loads),
            "snapshot_tokens": {
                node_id: snapshot.total_tokens
                for node_id, snapshot in self.snapshots.items()
            },
            "snapshot_nodes": {
                node_id: snapshot.node_count
                for node_id, snapshot in self.snapshots.items()
            },
        }
