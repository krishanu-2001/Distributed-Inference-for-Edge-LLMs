Context
                                                                                                     
CS 380D project simulating a distributed LLM inference system inspired by SGLang. The system demonstrates
cache-aware routing, radix tree-based prefix caching, and distributed node coordination with realistic
timing models.

File Structure

config/
default.yaml          # All configurable parameters
src/
__init__.py
config.py             # YAML config loading with dataclasses
radix_tree.py         # Radix tree (compressed trie) - core data structure
kv_cache.py           # Memory-bounded KV cache manager
node.py               # Node server (aiohttp) + MockLLM + per-node Router
router.py             # Per-node router with approximate trees + broadcast sync
network.py            # Network delay simulation
visualization.py      # ASCII tree viz, cache heatmaps, debug tools
main.py               # Entry point: launches cluster + demo

Architecture

Request Flow

1. Client POSTs to any node
2. Request enters node's incoming queue (async queue)
3. Node ingests requests from queue one-by-one with configurable interval (incoming_time, default 100ms)
4. For each dequeued request: tokenize text, consult LOCAL router (approximate radix trees for ALL nodes)
5. Router picks best node: longest cache match (with load balancing)
6. If routed locally: process directly. If remote: forward via network (with simulated LAN delay)
7. Processing node: match prefix on actual radix tree, run mock LLM (O(n²) prefill for uncached + O(n)
decode), store KV cache, insert into radix tree
8. The routing node updates its OWN approximate tree for the target worker

Node Incoming Queue

Each node has an async incoming queue. Requests are enqueued immediately and processed one-by-one:
- incoming_time (default 100ms): interval between processing successive requests from the queue
- This simulates realistic request ingestion pacing

Approximate Tree Synchronization (Broadcast)

Each node maintains approximate radix trees for every OTHER node (metadata only, no KV tensors). Updated
in two ways:
- Lazy update: When a node routes a request to worker X, it immediately inserts the request tokens into
its local approximate tree for X
- Broadcast sync: Nodes broadcast their actual radix tree metadata to all other nodes. Trigger is
configurable:
- "queue_length" mode: broadcast when incoming queue length < threshold (default 3) — sync when node is
not busy
- "timer" mode: broadcast on a fixed interval (default 5s)
- Config params: broadcast_trigger, broadcast_queue_threshold (default 3), broadcast_interval_s (default
5)
- All params configurable in config/default.yaml

Different nodes may have divergent views of the cluster's cache state between broadcasts — a realistic
property of distributed systems.

Key Design Decisions

- Per-node routing: Each node has its own Router instance with its own set of approximate trees. Routing
decisions are local.
- Same RadixTree class for actual (with KV cache refs) and approximate (metadata-only) trees
- aiohttp for HTTP servers per the spec's "separate server on different sockets" requirement
- dataclasses for config (avoid pydantic dependency - keep deps minimal)
- Mock tokenizer: deterministic hash-based (word → hash % 32000) so identical prefixes always match

Implementation Phases

Phase 1: Core Data Structures

1. config/default.yaml + src/config.py
2. src/radix_tree.py - insert (with edge splitting), match_prefix, evict_lru, pretty_print
3. src/kv_cache.py - allocate, free, utilization tracking

Phase 2: LLM + Router

4. MockLLM in src/node.py - timing: prefill = factor * uncached_tokens², decode = factor * output_tokens
5. src/router.py - Per-node router with:
- cache_aware (longest match + load balance + threshold), round_robin, random policies
- Approximate radix trees for each other node
- Broadcast sync: nodes send tree metadata to all others (trigger: utilization < threshold OR timer)

Phase 3: Networking + Servers

6. src/network.py - delay simulation + HTTP forwarding
7. src/node.py - full aiohttp server with endpoints: /infer, /internal_infer, /status, /tree
8. src/main.py - launches all nodes, runs demo requests

Phase 4: Visualization

9. src/visualization.py - ASCII radix trees, cache memory bars, routing stats, actual-vs-approximate tree
comparison

Dependencies

- aiohttp (async HTTP server/client)
- pyyaml (config loading)
- Standard library only otherwise

Verification

1. Start cluster with 4 nodes
2. Send requests with shared prefixes → verify cache hits on correct nodes
3. Check radix tree visualization shows expected sharing structure
4. Verify routing decisions prefer cache-matched nodes
5. Verify eviction when memory fills up
6. Check timing model produces realistic delays
