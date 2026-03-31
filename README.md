# Distributed LLM Inference Simulator

Course project for **CS 380D - Distributed Systems**, UT Austin. Simulates distributed LLM inference with cache-aware routing inspired by [SGLang](https://github.com/sgl-project/sglang) and [LLM-D](https://github.com/llm-d/llm-d).

## Introduction

LLM serving wastes compute recomputing KV caches for shared prefixes. This simulator runs a cluster of inference nodes, each with a **radix tree** of cached prefixes and a **router** that forwards requests to the node with the longest cache match.

**Request flow** (see [`CS 380D - Distributed System (3).pdf`](CS%20380D%20-%20Distributed%20System%20(3).pdf) for architecture diagram):

1. Query arrives at any node's router
2. Router matches token prefix against **approximate radix trees** (one per node) to find best cache hit
3. Request routed to selected node over simulated LAN
4. Target node matches against its **actual radix tree**, reuses cached KV, runs inference on new tokens only
5. Response returned; nodes periodically **broadcast** trie state to keep approximate trees in sync

## Setup

```bash
conda create -n dist python=3.11
conda activate dist
pip install aiohttp pyyaml pandas
```

## Running

```bash
python -m src.main --config config/default.yaml
```

Starts 4 nodes on ports 8100-8103, loads requests from MMLU dataset, prints expanded config + cache hit metrics per request, saves results to `runs/run_<timestamp>/results.csv`.

Send manual requests while running:

```bash
curl -X POST http://localhost:8100/infer -H "Content-Type: application/json" -d '{"text": "your prompt"}'
curl http://localhost:8100/status
```

## Configuration

Everything is in one YAML file (`config/default.yaml`):

```yaml
cluster:
  num_nodes: 4
  base_port: 8100

node:
  max_tokens: 1000           # KV cache capacity per node
  eviction_policy: "lru"

router:
  policy: "cache_aware"      # cache_aware | round_robin | self

requests:
  csv_path: "data/requests.csv"
  topics:                    # regex patterns (empty = all 57 MMLU subjects)
    - "high_school_.*"
    - "philosophy"
    - "machine_learning"
  rows_per_topic: 50         # 0 = all rows
  shuffle: true
  seed: 42
  node_assignment: "round_robin"  # original | round_robin | hash_topic | random
```

Topics accept regex — `"high_school_.*"` expands to all 14 high school subjects. At startup the system prints matched topics with row counts and node distribution.

**Node assignment strategies:**

| Strategy | Behavior |
|---|---|
| `original` | Keep `target_node` from CSV |
| `round_robin` | Cycle across nodes evenly |
| `hash_topic` | Same topic always maps to same node (maximizes prefix sharing) |
| `random` | Uniform random |

## Experiments

**Routing vs no routing** — compare `router.policy: "cache_aware"` vs `"self"`

**Topic locality** — compare `node_assignment: "hash_topic"` (high prefix sharing) vs `"round_robin"` (spread across nodes)

**Cache pressure** — vary `node.max_tokens` (e.g. 500 vs 10000) to test LRU eviction under different memory constraints

**Topic subsets** — use regex to select domains, e.g. `[".*law.*"]` or `[".*math.*", ".*physics.*", "machine_learning"]`

Each run outputs to `runs/run_<timestamp>/`:
- `results.csv` — per-request: cache_hit_ratio, inference_time, routing decisions, node assignments
- `run.log` — full execution log

## Technical Details

| Component | File | Role |
|---|---|---|
| Entry point | `src/main.py` | Starts cluster, runs experiment |
| Node | `src/node.py` | HTTP server, queue, cache management, routing |
| Router | `src/router.py` | Cache-aware routing with approximate radix trees per node |
| Radix tree | `src/radix_tree.py` | Prefix trie with LRU eviction and reference counting |
| KV cache | `src/kv_cache.py` | Memory allocation tracker |
| Network | `src/network.py` | LAN delay + jitter simulation |
| Request loader | `src/request_loader.py` | Topic filtering (regex), subsetting, shuffling, node assignment |
| Config | `src/config.py` | Dataclasses loaded from YAML |

LLM inference is mocked with realistic timing: prefill scales O(n^2) on new tokens, decode scales O(n) per token. Network delay defaults to ~2ms + 0.5ms jitter (LAN).

## References

- **CS 380D** - Distributed Systems, UT Austin
- Zheng et al., "Efficiently Programming Large Language Models using SGLang" (2023) — [arXiv:2312.07104](https://arxiv.org/pdf/2312.07104)
- [SGLang](https://github.com/sgl-project/sglang) — RadixAttention and cache-aware scheduling
- [LLM-D](https://github.com/llm-d/llm-d) — Distributed LLM inference gateway
- MMLU (Hendrycks et al.) — Massive Multitask Language Understanding benchmark
