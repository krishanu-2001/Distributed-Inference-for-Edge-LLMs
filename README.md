# Distributed LLM Inference Simulator

Course project for **CS 380D - Distributed Systems**, UT Austin. Simulates distributed LLM inference with cache-aware routing inspired by [SGLang](https://github.com/sgl-project/sglang) and [LLM-D](https://github.com/llm-d/llm-d).

## Introduction

LLM serving wastes compute recomputing KV caches for shared prefixes. This simulator runs a cluster of inference nodes, each with a **radix tree** of cached prefixes and a **router** that forwards requests to the node with the longest cache match.

**Request flow** (see [`docs/CS 380D - Distributed System (1).pdf`](docs/CS%20380D%20-%20Distributed%20System%20(1).pdf) for architecture diagram):

1. Query arrives at any node's router
2. Router matches the token prefix against its own **exact radix tree** and **approximate radix trees** of each peer
3. Request is forwarded only if a peer has a strictly longer match (ties stay local); otherwise it runs locally
4. Target node matches against its **actual radix tree**, reuses cached KV, runs inference on new tokens only
5. Response returned; nodes periodically **broadcast** trie state to keep approximate trees in sync

## Setup

```bash
python3.12 -m venv .venv          # Python 3.10+ required
source .venv/bin/activate
pip install aiohttp pyyaml pandas pyarrow pytest

# Build data/requests.csv (5-shot MMLU prompts, 13,757 requests) from src/MMLU.parquet
python -m src.abcd

# Optional: SGLang's "generated-shared-prefix" benchmark workload (1,024 requests)
python -m src.gen_sglang_gsp --out data/sglang_gsp.csv
python -m src.gen_sglang_gsp --group-distribution zipf --zipf-alpha 1.1 --out data/sglang_gsp_zipf.csv
```

## Running

```bash
python -m src.main --config config/default.yaml
python -m src.main --config config/default.yaml --limit 200 --exit   # quick run, then exit

# SGLang shared-prefix benchmark, routed vs. no routing, then compare
python -m src.main --config config/sglang_gsp.yaml --exit
python -m src.main --config config/sglang_gsp.yaml --exit --policy self
python -m src.summarize runs/run_<A> runs/run_<B>
```

CLI overrides: `--requests PATH`, `--policy {cache_aware,round_robin,self}`, `--node-assignment {...}`, `--limit N`, `--exit`. Each run saves the effective config to `runs/run_<timestamp>/config.yaml`.

Starts 4 nodes on ports 8100-8103, sends each request in `requests.path` to the entry node chosen by `requests.node_assignment`, prints cache hit metrics per request, and saves results to `runs/run_<timestamp>/results.csv`. Without `--exit`, the servers keep running afterwards for manual requests.

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

network:
  lan_delay_ms: 2.0          # one-way delay per message
  delay_jitter_ms: 0.5

node:
  max_tokens: 1000           # KV cache capacity per node (tokens)
  eviction_policy: "lru"

llm:
  prefill_cost_factor: 0.0001
  decode_cost_factor: 0.001
  max_output_tokens: 20

router:
  policy: "cache_aware"      # cache_aware | round_robin | self

broadcast:
  trigger: "queue_length"    # queue_length | timer
  queue_threshold: 3         # queue_length: broadcast after a request when queue < threshold
  interval_s: 5.0            # timer: broadcast period

requests:
  path: "data/requests.csv"
  node_assignment: "round_robin"   # round_robin | original | hash_topic | random | weighted
  node_weights: [0.4, 0.3, 0.2, 0.1]  # only for weighted
  seed: 42
```

Broadcasts run in the background and never block request processing.

**Node assignment** decides which node each request *enters* at; that node's router may then forward it:

| Strategy | Behavior |
|---|---|
| `round_robin` | Cycle across nodes in request order (default) |
| `original` | Keep `target_node` from the CSV |
| `hash_topic` | All requests of a subject / prefix group enter at the same node |
| `random` | Uniform random (seeded) |
| `weighted` | Random with probability ∝ `node_weights` |

## Experiments

**Routing vs no routing** — compare `router.policy: "cache_aware"` vs `"self"`

**Cache pressure** — vary `node.max_tokens` (e.g. 500 vs 10000) to test LRU eviction under different memory constraints

**SGLang shared-prefix benchmark** — `config/sglang_gsp.yaml` runs SGLang's `generated-shared-prefix` workload (64 groups × 16 prompts, 2,048-token shared system prompt, 128-token question, 256 output tokens; uniform or Zipf group popularity) through the mock LLM. Prompts are random words, one word per token.

**Anti-entropy frequency** — vary `broadcast.interval_s` with `trigger: "timer"`

Each run outputs to `runs/run_<timestamp>/`:
- `results.csv` — per-request (`|`-delimited): cache_hit_ratio, inference_time, routing decisions, node assignments, uncached_tokens, error
- `run.log` — full execution log

### Real SGLang servers on TACC (Lonestar6)

`scripts/tacc/run_experiments.slurm` starts 4 SGLang servers (Qwen2.5-0.5B-Instruct) on one GPU node, caps each KV cache with `--max-total-tokens` to force eviction, and runs `cache_aware` vs `self` on the shared-prefix (uniform, Zipf) and MMLU workloads (`config/tacc/*.yaml`, `llm.backend: sglang`). Caches are flushed between runs and a comparison table is written to `runs/tacc/job_<id>/summary.txt`.

```bash
# one time, on a login node
bash scripts/tacc/setup_sglang.sh        # venv in $SCRATCH/llm/venv-sglang + model download
python -m src.gen_sglang_gsp --num-groups 128 --prompts-per-group 32 --output-len 32 --out data/sglang_gsp_big.csv
python -m src.gen_sglang_gsp --num-groups 128 --prompts-per-group 32 --output-len 32 \
    --group-distribution zipf --zipf-alpha 1.1 --out data/sglang_gsp_big_zipf.csv
python -m src.abcd
mkdir -p runs/tacc

# Recommended: gpu-a100-dev (2 h limit). The whole matrix takes ~1.5 h, and dev
# usually starts within minutes, while gpu-a100 can queue for 15+ hours.
JOB_SECONDS=7200 sbatch -p gpu-a100-dev -t 02:00:00 scripts/tacc/run_experiments.slurm

# Quick end-to-end check (30 requests per run)
SMOKE=1 JOB_SECONDS=1800 sbatch -p gpu-a100-dev -t 00:30:00 scripts/tacc/run_experiments.slurm
```

Notes: the job uses allocation `IRI26037` (edit `#SBATCH -A`). Keep `JOB_SECONDS` equal to `-t`, since runs that won't fit in the remaining time are skipped. SGLang JIT-compiles C++20 kernels at server start, so the script puts gcc 13.2 first on `PATH`; the default gcc 9.4 fails with `fatal error: concepts`. The first server start takes about 10 minutes while kernels compile.

## Technical Details

| Component | File | Role |
|---|---|---|
| Entry point | `src/main.py` | Starts cluster, runs experiment |
| Node | `src/node.py` | HTTP server, queue, cache management, routing |
| Router | `src/router.py` | Cache-aware routing with approximate radix trees per node |
| Radix tree | `src/radix_tree.py` | Prefix trie with LRU eviction and reference counting; each node owns the KV entry for its edge tokens |
| KV cache | `src/kv_cache.py` | Token-granular memory allocation tracker |
| Network | `src/network.py` | LAN delay + jitter simulation |
| Request builders | `src/abcd.py`, `src/gen_sglang_gsp.py` | 5-shot MMLU prompts; SGLang generated-shared-prefix workload |
| Request loader | `src/request_loader.py` | Reads the requests CSV and assigns entry nodes |
| Summary | `src/summarize.py` | Comparison table across run directories |
| Config | `src/config.py` | Dataclasses loaded from YAML |

Run the tests with `python -m pytest tests`.

LLM inference is mocked with realistic timing: prefill scales O(n^2) on new tokens, decode scales O(n) per token. Network delay defaults to ~2ms + 0.5ms jitter (LAN).

## References

- **CS 380D** - Distributed Systems, UT Austin
- Zheng et al., "Efficiently Programming Large Language Models using SGLang" (2023) — [arXiv:2312.07104](https://arxiv.org/pdf/2312.07104)
- [SGLang](https://github.com/sgl-project/sglang) — RadixAttention and cache-aware scheduling
- [LLM-D](https://github.com/llm-d/llm-d) — Distributed LLM inference gateway
- MMLU (Hendrycks et al.) — Massive Multitask Language Understanding benchmark
