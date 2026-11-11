"""Load a requests CSV and decide which node each request enters the cluster at.

The entry node is where the request arrives (its local router then decides
whether to forward it). Strategies (config `requests.node_assignment`):

  round_robin  cycle through nodes in request order (default; paper setup)
  original     keep the CSV's target_node column
  hash_topic   all requests of a subject/prefix group enter at the same node
  random       uniform random node (seeded)
  weighted     random node with probability proportional to `node_weights`
"""

import random
import zlib

import pandas as pd

from src.config import RequestsConfig

STRATEGIES = ("round_robin", "original", "hash_topic", "random", "weighted")


def assign_nodes(df: pd.DataFrame, num_nodes: int, cfg: RequestsConfig) -> pd.Series:
    strategy = cfg.node_assignment
    n = len(df)
    if strategy == "round_robin":
        return pd.Series([i % num_nodes for i in range(n)], index=df.index)
    if strategy == "original":
        if "target_node" not in df:
            raise ValueError("node_assignment 'original' needs a target_node column")
        return df["target_node"].astype(int) % num_nodes
    if strategy == "hash_topic":
        return df["subject"].map(lambda s: zlib.crc32(str(s).encode()) % num_nodes)
    rng = random.Random(cfg.seed)
    if strategy == "random":
        return pd.Series([rng.randrange(num_nodes) for _ in range(n)], index=df.index)
    if strategy == "weighted":
        weights = cfg.node_weights
        if not weights or len(weights) != num_nodes or min(weights) < 0 or sum(weights) <= 0:
            raise ValueError(
                f"node_assignment 'weighted' needs node_weights: one non-negative "
                f"weight per node ({num_nodes}), got {weights}"
            )
        return pd.Series(
            rng.choices(range(num_nodes), weights=weights, k=n), index=df.index
        )
    raise ValueError(f"unknown node_assignment {strategy!r}; use one of {STRATEGIES}")


def load_requests(cfg: RequestsConfig, num_nodes: int, limit: int = 0) -> pd.DataFrame:
    """Read cfg.path, apply limit, assign entry nodes. Adds a `subject` column
    if missing and keeps `output_len` (per-request output tokens) if present."""
    df = pd.read_csv(cfg.path)
    if "subject" not in df:
        df["subject"] = "all"
    if limit > 0:
        df = df.head(limit)
    df = df.reset_index(drop=True)
    df["target_node"] = assign_nodes(df, num_nodes, cfg).astype(int)
    return df
