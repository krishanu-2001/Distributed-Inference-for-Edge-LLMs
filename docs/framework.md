### Task
We are trying to simulate an distributed LLM-inference framework inspired by SGLang. Where one request can come to any one of the sockets and each of the nodes have a router that transfers the query to another node based on some conditions. 

### Requirements
1. Create n - nodes connected with each other in a network.
Each node starts as a separate server with node_id on different sockets.
Network has a prefixed delay between each node to simulate actual distributed environment.
Assume the nodes are closeby in a LAN network. 
Each nodes need to have a fixed GPU memory which will be used by KV cache. 

2. Each node hosts a LLM instance. For now mock the functioning of the LLM that take in text and return text. It also needs to save the KV cache. The delay of LLM inference scales as a function of text which is O (n^2 = prefil) + O(n = per decode token) (do research and get a correct estimation).
3. The KV cache needs to be stored in the some array format in memory of node.
Now that we are just mocking the system. You can keep some random identifiers in place of the KV Caches. 
4. The Prefixes needs to be stored as radix-tree for a node. Inspired from SGLang: https://github.com/sgl-project/sglang       

### helper functions. 
Each node hitting the other node and document them how to use them. 
For cache visualization for different nodes.
Create functions for debugging radix trees and visualize it.
Create a config file to edit the following parameters
1. Number of nodes
2. Memory Limit
3. Different Delays

### References
https://github.com/sgl-project/sglang
https://arxiv.org/pdf/2312.07104
https://github.com/llm-d/llm-d