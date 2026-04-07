# Distributed LLM Inference Simulator

CS 380D project (UT Austin). Distributed LLM inference with cache-aware routing, inspired by [SGLang](https://github.com/sgl-project/sglang) and [LLM-D](https://github.com/llm-d/llm-d).

A cluster of nodes each holds a **radix tree** of cached prefixes. The router forwards each request to the node with the longest cache match, so only new tokens are recomputed. Approximate trees are kept in sync via periodic broadcasts (mock) or polling (real SGLang).

See [`CS 380D - Distributed System (3).pdf`](CS%20380D%20-%20Distributed%20System%20(3).pdf) for the architecture diagram.

## Setup

```bash
conda create -n dist python=3.11 && conda activate dist
pip install aiohttp pyyaml pandas
```

## Mock simulator

```bash
python -m src.main --config config/default.yaml
```

Starts 4 nodes on ports 8100-8103, loads MMLU requests, writes results to `runs/run_<timestamp>/results.csv`.

Manual requests:

```bash
curl -X POST http://localhost:8100/infer -H "Content-Type: application/json" -d '{"text": "your prompt"}'
curl http://localhost:8100/status
```

## Real SGLang servers

A vendored SGLang tree (`sglang/`) is patched to expose its radix tree via `GET /v1/cache/tree`, so an external router can match prefixes across nodes without touching the inference loop.

```bash
pip install transformers torch
pip install -e sglang/python
python -m src.sglang_main --config config/sglang.yaml
```

This launches N SGLang servers as subprocesses, waits for `/health`, starts the cache-aware router, streams requests, and shuts the servers down at the end. Add `--no-launch` to attach to already-running servers.

For each request, `SGLangRouter` (`src/sglang_router.py`) polls every node's `/v1/cache/tree`, rebuilds an approximate `PrefixTrie` per node, picks the node with the longest prefix match, and forwards via `POST /v1/completions`. Outputs match the mock run schema and add `metrics.csv` for the dashboard.

## Configuration (`config/default.yaml`)

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
  topics:                    # regex (empty = all 57 MMLU subjects)
    - "high_school_.*"
    - "philosophy"
  rows_per_topic: 50
  shuffle: true
  seed: 42
  node_assignment: "round_robin"  # original | round_robin | hash_topic | random
```

`node_assignment`: `original` keeps CSV target; `round_robin` cycles; `hash_topic` pins each topic to one node (max prefix sharing); `random` is uniform.

## Experiments

- **Routing on/off**: `router.policy: cache_aware` vs `self`
- **Topic locality**: `node_assignment: hash_topic` vs `round_robin`
- **Cache pressure**: vary `node.max_tokens` (e.g. 500 vs 10000)
- **Topic subsets**: regex like `[".*law.*"]` or `[".*math.*", "machine_learning"]`

Each run writes `runs/run_<timestamp>/{results.csv, run.log}`.

## Files

| File | Role |
|---|---|
| `src/main.py` | Mock cluster entry point |
| `src/node.py` | HTTP server, queue, cache, routing |
| `src/router.py` | Cache-aware routing over approximate trees |
| `src/radix_tree.py` | Prefix trie + LRU + refcounts |
| `src/kv_cache.py` | Memory allocation tracker |
| `src/network.py` | LAN delay + jitter |
| `src/request_loader.py` | Topic filtering, shuffling, node assignment |
| `src/sglang_main.py` | Launches real SGLang servers, runs experiment |
| `src/sglang_router.py` | External cache-aware proxy (polls `/v1/cache/tree`) |
| `sglang/python/sglang/srt/...` | Patches: `RadixCache.snapshot()` + `/v1/cache/tree` |

Mock inference timing: prefill O(n²) on new tokens, decode O(n) per token. Network: ~2ms + 0.5ms jitter.

## References

- Zheng et al., "Efficiently Programming Large Language Models using SGLang" (2023) — [arXiv:2312.07104](https://arxiv.org/pdf/2312.07104)
- [SGLang](https://github.com/sgl-project/sglang) — RadixAttention and cache-aware scheduling
- [LLM-D](https://github.com/llm-d/llm-d) — Distributed LLM inference gateway
- MMLU (Hendrycks et al.)
