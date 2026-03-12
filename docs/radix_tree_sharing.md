The core problem
The router needs to decide which worker has the best cache match for an incoming request. But the router is a separate process — it doesn't have access to each worker's actual GPU-resident radix tree. So it maintains a prediction of what each worker has cached, based purely on what the router has previously routed there.
How the approximate tree is built
The router maintains one radix tree per worker in its own memory (CPU-side, in Rust). These trees store only token sequences — no actual KV cache tensors. They're lightweight metadata structures.
When a request arrives and gets routed to Worker 2, the router does:

Tokenize the request into token IDs.
Insert those token IDs into Worker 2's approximate tree.

That's it. The router assumes that if it sent a request with tokens [A, B, C, D, E] to Worker 2, then Worker 2 now has the KV cache for prefix [A, B, C, D, E] in its RadixAttention tree. This is a prediction — the router never confirms with the worker whether the cache actually still exists.

How routing decisions use the tree
When the next request arrives with tokens [A, B, C, D, E, F, G]:

The router runs match_prefix() against each worker's approximate tree.
Worker 2's tree returns a match of length 5 (tokens [A, B, C, D, E]).
Worker 1's tree returns a match of length 0 (never saw this prefix).
Worker 2 wins the cache affinity check (subject to load balancing).
The router inserts [A, B, C, D, E, F, G] into Worker 2's tree.

The lazy update mechanism
The update from router to tree is synchronous with routing but asynchronous with computation. When the router routes request R to Worker 2:
1. Route decision made → immediately insert R's tokens into Worker 2's tree
2. R is sent to Worker 2 via HTTP/gRPC
3. Worker 2 does prefix matching against its ACTUAL radix tree
4. Worker 2 computes prefill, inserts into its ACTUAL tree
Steps 3–4 happen independently. The router never polls the worker to reconcile. There's no feedback channel from worker → router about actual cache state. This is a deliberate design choice — it avoids synchronization overhead entirely and keeps the router stateless with respect to worker internals.

Each time a generation happens. Both the routers meta-tree and the nodes tree gets updated.