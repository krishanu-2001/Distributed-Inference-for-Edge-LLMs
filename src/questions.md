1. When does a request get routed to a different node?
2. What all things does the router layer check to do the above.


3 nodes
R1, 0
R2, 0
R3, 0

Is there a delay in the R1's

Currently we are sending one by one R1, R2 etc not in parallel.



Request 1: 'The quick brown fox jumps over the lazy dog...' -> Node 0
2026-03-07 22:46:56,775 [INFO] ::1 [07/Mar/2026:22:46:56 -0600] "POST /infer HTTP/1.1" 200 342 "-" "Python/3.12 aiohttp/3.13.3"
  Processed by: Node 0
  Cache hit: 0 tokens (0%)
  Inference time: 0.3781s

Request 2: 'The quick brown fox runs through the forest...' -> Node 1
2026-03-07 22:46:56,780 [INFO] Node 1: routing to Node 0
2026-03-07 22:46:56,884 [INFO] ::1 [07/Mar/2026:22:46:56 -0600] "POST /sync_tree HTTP/1.1" 200 175 "-" "Python/3.12 aiohttp/3.13.3"
2026-03-07 22:46:56,884 [INFO] ::1 [07/Mar/2026:22:46:56 -0600] "POST /sync_tree HTTP/1.1" 200 175 "-" "Python/3.12 aiohttp/3.13.3"
2026-03-07 22:46:56,884 [INFO] ::1 [07/Mar/2026:22:46:56 -0600] "POST /sync_tree HTTP/1.1" 200 175 "-" "Python/3.12 aiohttp/3.13.3"
2026-03-07 22:46:57,138 [INFO] ::1 [07/Mar/2026:22:46:56 -0600] "POST /internal_infer HTTP/1.1" 200 342 "-" "Python/3.12 aiohttp/3.13.3"
2026-03-07 22:46:57,141 [INFO] ::1 [07/Mar/2026:22:46:56 -0600] "POST /infer HTTP/1.1" 200 376 "-" "Python/3.12 aiohttp/3.13.3"
  Processed by: Node 0
  Cache hit: 4 tokens (50%)
  Inference time: 0.3516s
  Routed: Node 1 -> Node 0

  Is the delta: 0.200 sec


Change the inference_time function?