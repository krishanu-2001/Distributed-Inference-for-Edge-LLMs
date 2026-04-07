from src.radix_tree import RadixTree


class Visualizer:

    @staticmethod
    def radix_tree(tree: RadixTree, label: str = "") -> str:
        header = f"=== Radix Tree: {label} ===" if label else "=== Radix Tree ==="
        return f"{header}\n{tree.pretty_print()}"

    @staticmethod
    def cache_bar(node_id: int, used: int, capacity: int, width: int = 40) -> str:
        if capacity == 0:
            ratio = 0
        else:
            ratio = used / capacity
        filled = int(ratio * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"Node {node_id}: [{bar}] {used}/{capacity} tokens ({ratio:.0%})"

    @staticmethod
    def cluster_cache_view(nodes) -> str:
        """Show cache utilization across all nodes."""
        lines = ["=== Cluster Cache Utilization ==="]
        for node in nodes:
            stats = node.kv_cache.stats()
            lines.append(
                Visualizer.cache_bar(
                    node.node_id, stats["used_tokens"], stats["max_tokens"]
                )
            )
        return "\n".join(lines)

    @staticmethod
    def node_status(node) -> str:
        """Detailed status of a single node."""
        cache_stats = node.kv_cache.stats()
        lines = [
            f"=== Node {node.node_id} (port {node.port}) ===",
            f"  Queue length: {node.incoming_queue.qsize()}",
            f"  Running requests: {node.running_requests}",
            f"  Cache: {cache_stats['used_tokens']}/{cache_stats['max_tokens']} tokens "
            f"({cache_stats['utilization']:.0%})",
            f"  Cache entries: {cache_stats['num_entries']}",
            f"  Radix tree tokens: {node.radix_tree.total_tokens}",
            f"  Actual tree:",
        ]
        for line in node.radix_tree.pretty_print().split("\n"):
            lines.append(f"    {line}")
        return "\n".join(lines)

    @staticmethod
    def routing_stats(nodes) -> str:
        """Show routing statistics from all nodes' routers."""
        lines = ["=== Routing Statistics ==="]
        for node in nodes:
            stats = node.router.get_stats()
            lines.append(f"Node {node.node_id} router view:")
            lines.append(f"  Policy: {stats['policy']}")
            lines.append(f"  Observed loads: {stats['node_loads']}")
            lines.append(f"  Approx tree sizes: {stats['approx_tree_sizes']}")
        return "\n".join(lines)

    @staticmethod
    def compare_trees(
        actual: RadixTree, approximate: RadixTree, node_id: int
    ) -> str:
        """Side-by-side comparison of actual vs approximate tree for a node."""
        lines = [f"=== Tree Comparison for Node {node_id} ==="]
        lines.append(f"Actual tree ({actual.total_tokens} tokens):")
        for line in actual.pretty_print().split("\n"):
            lines.append(f"  {line}")
        lines.append(f"Approximate tree ({approximate.total_tokens} tokens):")
        for line in approximate.pretty_print().split("\n"):
            lines.append(f"  {line}")
        return "\n".join(lines)
