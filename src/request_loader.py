"""
Request loader: filters topics (with regex), subsets rows,
shuffles across topics, and assigns node IDs.
"""

import csv
import hashlib
import random
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RequestsConfig:
    csv_path: str = "data/requests.csv"
    topics: list[str] = field(default_factory=list)  # regex patterns; empty = all
    rows_per_topic: int = 0  # 0 = all rows
    shuffle: bool = True
    seed: int = 42
    node_assignment: str = "original"  # original | round_robin | hash_topic | random


def _parse_topic_entry(entry: str) -> tuple[str, Optional[float]]:
    """
    Parse a topic entry which may include a fraction.
    Formats:
      "philosophy"         -> ("philosophy", None)  # None means all rows
      "philosophy:0.2"     -> ("philosophy", 0.2)
      "philosophy:0"       -> ("philosophy", 0.0)   # explicitly 0 = skip
      "high_school_.*:0.5" -> ("high_school_.*", 0.5)
    """
    if ":" in entry:
        parts = entry.rsplit(":", 1)
        try:
            frac = float(parts[1])
            return parts[0], frac
        except ValueError:
            pass
    return entry, None


def _match_topics(patterns: list[str], available: list[str]) -> tuple[list[str], dict[str, float]]:
    """
    Return available topics that match any of the regex patterns,
    plus per-topic fraction overrides.

    Entries are processed in order. "*" matches all topics not already
    matched by previous entries:
        - "philosophy:0.2"       # 20% of philosophy
        - "machine_learning:0.2" # 20% of machine_learning
        - "*:0.8"                # 80% of every remaining topic
    """
    if not patterns:
        return sorted(available), {}

    matched_order: list[str] = []  # preserve insertion order
    matched_set: set[str] = set()
    fractions: dict[str, float] = {}

    for entry in patterns:
        pattern, frac = _parse_topic_entry(entry)

        # "*" = wildcard for all topics not yet matched
        if pattern == "*":
            for topic in sorted(available):
                if topic not in matched_set:
                    matched_set.add(topic)
                    matched_order.append(topic)
                    if frac is not None:
                        fractions[topic] = frac
        else:
            regex = re.compile(pattern, re.IGNORECASE)
            for topic in available:
                if regex.search(topic) and topic not in matched_set:
                    matched_set.add(topic)
                    matched_order.append(topic)
                    if frac is not None:
                        fractions[topic] = frac

    return sorted(matched_order), fractions


def _assign_nodes(
    rows: list[dict], num_nodes: int, strategy: str, rng: random.Random
) -> list[dict]:
    """Assign target_node to each row based on strategy."""
    if strategy == "original":
        return rows

    for i, row in enumerate(rows):
        if strategy == "round_robin":
            row["target_node"] = i % num_nodes
        elif strategy == "hash_topic":
            h = int(hashlib.md5(row["subject"].encode()).hexdigest(), 16)
            row["target_node"] = h % num_nodes
        elif strategy == "random":
            row["target_node"] = rng.randint(0, num_nodes - 1)
        else:
            raise ValueError(f"Unknown node_assignment strategy: {strategy}")

    return rows


def load_requests(requests_cfg: RequestsConfig, num_nodes: int) -> list[dict]:
    """
    Load and prepare requests from CSV according to config.

    Returns list of dicts with keys: text, target_node, subject
    """
    # Read all rows
    all_rows: dict[str, list[dict]] = {}  # topic -> rows
    with open(requests_cfg.csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            topic = row["subject"]
            all_rows.setdefault(topic, []).append(row)

    available_topics = list(all_rows.keys())

    # Match topics via regex patterns
    selected_topics, fractions = _match_topics(requests_cfg.topics, available_topics)

    if not selected_topics:
        raise ValueError(
            f"No topics matched patterns {requests_cfg.topics}. "
            f"Available: {sorted(available_topics)}"
        )

    # Compute how many rows to take per topic
    def _rows_to_take(topic: str, total: int) -> int:
        # Per-topic fraction override (e.g. "philosophy:0.2", "*:0" = skip)
        if topic in fractions:
            return int(fractions[topic] * total)
        # Default: all rows
        return total

    # Filter out topics with 0 rows to take
    selected_topics = [t for t in selected_topics if _rows_to_take(t, len(all_rows[t])) > 0]

    # Log expanded config
    logger.info("--- Request Config ---")
    logger.info(f"  CSV: {requests_cfg.csv_path}")
    logger.info(f"  Topic patterns: {requests_cfg.topics or ['* (all)']}")
    logger.info(f"  Matched topics ({len(selected_topics)}):")
    for t in selected_topics:
        count = len(all_rows[t])
        take = _rows_to_take(t, count)
        frac_str = f" ({fractions[t]:.0%})" if t in fractions else ""
        logger.info(f"    - {t}: {take}/{count} rows{frac_str}")
    logger.info(f"  Shuffle: {requests_cfg.shuffle}")
    logger.info(f"  Seed: {requests_cfg.seed}")
    logger.info(f"  Node assignment: {requests_cfg.node_assignment}")

    rng = random.Random(requests_cfg.seed)

    # Subset rows per topic
    result = []
    for topic in selected_topics:
        rows = all_rows[topic]
        take = _rows_to_take(topic, len(rows))
        if take < len(rows):
            sampled = rng.sample(rows, take)
        else:
            sampled = list(rows)
        result.extend(sampled)

    # Shuffle across topics
    if requests_cfg.shuffle:
        rng.shuffle(result)

    # Coerce target_node to int (CSV reads as string)
    for row in result:
        row["target_node"] = int(row["target_node"])

    # Assign nodes
    result = _assign_nodes(result, num_nodes, requests_cfg.node_assignment, rng)

    logger.info(f"  Total requests: {len(result)}")
    return result
