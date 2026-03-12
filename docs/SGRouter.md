The SGLang paper itself describes the foundational cache-aware scheduling algorithm. Let me also grab the current router implementation details since it's evolved significantly.Now I have the full picture. Let me synthesize this with the paper's original algorithm.

SGLang's routing operates at two distinct levels — **intra-instance scheduling** (the original paper's cache-aware scheduling within a single RadixAttention engine) and **inter-instance routing** (the `sglang_router` that distributes across multiple engine instances). Here's how both work:
---

## Level 1: Intra-instance scheduling (from the paper you uploaded)

This is the Algorithm 1 in Appendix A.2 of the SGLang paper. When multiple requests are queued within a single engine instance, the scheduler decides **which requests to run next** to maximize RadixAttention cache hits.

**The algorithm (longest-shared-prefix-first):**

1. For every waiting request, do a `match_prefix()` against the radix tree to find how many tokens are already cached.
2. **Sort requests by matched prefix length** (longest first), not FCFS.
3. Greedily select requests into the next batch, checking that enough memory exists (evictable cache + free pool).
4. As each request is selected, increment its prefix node's reference counter (preventing eviction of shared prefixes).
5. After a batch finishes, insert completed request KV caches back into the radix tree and decrement reference counters.

**Why this works:** The paper proves (Theorem 3.1) that for a batch of requests, visiting the radix tree in **depth-first search order** achieves the optimal cache hit rate — and longest-shared-prefix-first is equivalent to DFS order. In the online case, DFS gets disrupted by new arrivals, but the schedule still approximates DFS behavior on the augmented tree.

**The single factor here is matched prefix length.** The tradeoff is throughput vs. fairness — greedy cache-aware scheduling can starve requests that don't share prefixes with the current hot path.

---

## Level 2: Inter-instance routing (the `sglang_router`)

This is the Rust-based router that sits in front of multiple independent SGLang instances (data-parallel workers). It has **four routing policies**, each using different factors:

### Policy 1: `cache_aware` (default, the interesting one)

This is the flagship policy. The router maintains an **approximate radix tree** per worker, tracking each worker's cache state, with three key parameters: `cache_threshold` (default 0.5), `balance_abs_threshold` (default 32), and `balance_rel_threshold` (default 1.0001).

**How it works:**

1. **Tokenize the incoming request** (the router runs its own tokenizer in-process).
2. **Match against each worker's approximate radix tree** to compute the prefix hit length for every worker.
3. **Pick the worker with the longest cache match** — but only if it passes a load-balance check.
4. **Load-balance check:** Even if Worker A has a longer cache match, if it's overloaded relative to Worker B, route to B instead. Specifically, if `(load_A - load_B) > balance_abs_threshold` AND `(load_A / load_B) > balance_rel_threshold`, bypass the cache match and route to the less-loaded worker.
5. **Cache threshold:** If no worker has a cache match exceeding `cache_threshold` fraction of the request's tokens, fall back to **shortest-queue routing** instead.
6. **Update the approximate tree:** After routing, insert the request's tokens into the chosen worker's tree (lazily, without waiting for actual computation to confirm caching).

So the factors for `cache_aware` are:

- **Prefix cache match length** per worker (primary)
- **Worker load** — running request count (secondary, for balance)
- **Cache threshold** — minimum match quality to bother with cache affinity (fallback trigger)

The approximate radix tree is maintained entirely in the router's memory and is **not synchronized** with workers' actual radix trees — it's a prediction of what each worker has cached based on what the router sent there. This is intentionally lossy: with multiple router replicas, the radix trees are not synchronized across replicas, so cache efficiency degrades under multi-replica router deployments.

You can tune this with `--balance-*` and `--cache-threshold` flags, and `--max-tree-size` (default 2²⁴) controls how large the approximate tree can grow before eviction.

### Policy 2: `power_of_two`

The "power of two choices" algorithm from load-balancing theory. Randomly sample two workers, pick the one with the shorter queue. No cache awareness at all — purely load-based. This is commonly used as the decode policy in PD-disaggregated setups since decode workers don't benefit from prefix caching (they already have the KV cache from prefill).

### Policy 3: `round_robin`

Stateless, deterministic cycling through workers. No cache or load awareness.

### Policy 4: `random`

Uniform random selection. No factors considered.

---

## Level 3: PD-disaggregated routing

In prefill-decode disaggregation mode, the router makes **two routing decisions per request:**

You can set different policies per phase: `--prefill-policy cache_aware --decode-policy power_of_two`.

1. **Prefill routing:** Uses `cache_aware` to find the prefill worker with the most cached prefix (since prefill is compute-heavy and benefits enormously from skipping cached tokens).
2. **Decode routing:** Uses `power_of_two` to find the least-loaded decode worker (since decode is memory-bound and doesn't benefit from prefix affinity — each request has unique generated tokens).

The router then coordinates the handoff: prefill worker computes and sends KV cache to the selected decode worker via Mooncake/NIXL RDMA, and the router merges metadata and streams results back.

---

## Additional routing factors in the broader system

Beyond the core policies, the router's resilience layer adds several additional factors that modulate routing decisions: per-worker **circuit breakers** (stop routing to unhealthy workers), **rate limiting** (token-bucket per tenant), **retry logic** with jitter and exponential backoff, and a **health checker** that continuously probes workers and updates readiness.

The roadmap includes model-aware selection (choosing based on priority, cost, and input complexity) that runs *before* worker selection and composes with all existing policies.

---

## Summary: all factors in one place

| Factor | Where it's used | How |
|---|---|---|
| **Prefix cache match length** | Intra-instance scheduler + cache_aware router | Longest match gets priority |
| **Worker queue depth / running load** | cache_aware (balance check), power_of_two | Prevents hot-spotting |
| **Cache threshold** | cache_aware router | Falls back to shortest-queue if match too short |
| **Worker health / circuit breaker state** | All policies (resilience layer) | Excludes unhealthy workers |
| **Rate limits** | Resilience layer | Per-tenant/global throttling |
| **Prefill vs. decode phase** | PD router | Different policies per phase |
| **Approximate tree size** | cache_aware router | Evicts old entries from router's prediction tree |

The core insight connecting this to your 380D work: the cache-aware scheduling is essentially a **locality-optimized scheduling problem** where the "locality" is KV cache prefix sharing. The DFS-optimal proof in the paper maps directly to the kind of analysis you'd see in I/O scheduling or page replacement — it's the distributed systems equivalent of "minimize cache misses by exploiting temporal locality in access patterns."