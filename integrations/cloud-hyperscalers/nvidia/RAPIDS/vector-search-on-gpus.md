# Vector Search on GPUs

Up until CAGRA, most Approximate Nearest Neighbor Search algorithms tested on GPUs were originally designed for CPUs. CAGRA is built from the ground up to optimize the massive parallelism and memory bandwidth offered by modern GPUs:

- Massive Parallelism: GPUs have thousands of smaller cores designed to perform many simple tasks simultaneously, which suits highly parallel tasks like vector search. Instead of sequentially searching through the data, a GPU can examine many graph nodes at once, greatly speeding up the search.

- Memory Bandwidth: GPUs have much higher memory bandwidth than CPUs. While a typical CPU might have memory bandwidth in the range of 50-100 GB/s, modern GPUs can have memory bandwidths exceeding 500 GB/s. For vector searches in large datasets, this means the GPU can load and process data much faster. In CAGRA, this translates to more nodes being checked in parallel, enabling faster searches.

## Applications of CAGRA

### Query Speed

A user searches for “wireless headphones” and finds the result in 5 ms versus 1 ms. This is especially pronounced in Compound AI Systems that leverage query writers and perform many searches to complete a task. For example, a system that searches in 1 ms vs. 5 ms is 20 ms faster when the system makes 5 search requests. Studies have shown the link between response time and revenue for products such as e-Commerce stores. 

This further helps with server load. If your system can handle 10,000 searches per second (QPS) with HNSW but 77,000 QPS with CAGRA, you can serve more users with the same hardware or reduce the required hardware for the same load.

Even faster query speed can be achieved when batching requests to CAGRA. Continuous batching for online latency refers to creating some kind of adaptive strategy with query buffers and time windows to group requests. Some applications more naturally lend themselve to offline computation to leverage batching, such as Recommendation and Generative Feedback Loops.

### Time to build the Vector Index: 

Also referred to as TTI (Time-to-Index), across experimental results CAGRA achieves 2.2-27 times faster building time than HNSW on CPUs. We see three main benefits of this:

Faster Re-Indexing: In environments where data changes frequently, such as product catalogs or news articles, you need to rebuild the graph often. Faster building helps you achieve more up-to-date search results faster.
Save Computational Resources: Further, faster graph building leads to less computational time and resources spent on computing indexes. In our Discussion section, we further explain advances in GPU computing from our friends at Modal and Foundry, to give a couple of examples.
Hyperparameter Tuning: Weaviate’s ANN Benchmarks illustrate what we have found customizing HNSW parameters such as ef, efConstruction, and maxNeighbors to help Weaviate users find the pareto optimal point of memory and query speed. Building the graphs faster further help us test many configurations of CAGRA.
Core Basics of CAGRA

### Quick Differences

- Fixed Out Degree: Every node in the graph has the same number of edges (often set to 32). This is because GPUs perform the best when all threads in a warp (32 threads that execute the same instruction) are doing similar amounts of work. With a fixed out-degree, every thread processes the same number of edges per node, reducing the variance in workload across threads. This is then used in traversal because all threads can explore nodes in lockstep, minimizing synchronization overhead to maximize throughput. The term “lockstep” is used to reference the lack of warp divergence because all threads in a warp are doing the exact same work at the same time. This facilitates GPU throughput optimization.

- No Hierarchy: HNSW graphs use a multi-layer structure to achieve a “coarse-to-fine” / “zoom-out / zoom-in” approach to nearest neighbor routing. On GPUs, handling multiple levels of hierarchy can be inefficient because you have many threads that need to be synchronized. Thus, CAGRA utilizes a flat, fixed-degree to more efficiently parallelize the search with GPU threds.

## Core Innovations

### Reordering and Reverse Edge Addition

The initial CAGRA graph is built with NN-Descent. NN-Descent is an iterative optimization algorithm, swapping neighbors with neighbors, round by round. CAGRA then reorders neihgbors by how well they contribute to “2-hop connectivity”. Note, this is similar to the mechanism used in ACORN to speed up Filtered Vector Search. After re-ordering, edges are pruned to maintain the fixed out-degree. All neighbor connections are then made bi-directional with reverse edge addition.

## GPU Optimization

- GPU Warps: A warp is a group of 32 threads that execut ethe same instruction at the same time on a GPU. By grouping threads into warps, the GPU can execute many distance computations or graph traversals in parallel. If all threads in a warp do similar work (like processing nodes with a fixed out-degree), it maximizes hardware utilization.

- Forgettable Hash Table Management: Whereas HNSW implementations typically use a static visited list, CAGRA deploys a forgettable hash table to save memory. This may reduce in some redundant neighbor explorations, but this tradeoff is minimal thanks to the massive parallelization of node exploration.

## Discussion

### GPU Computing Infrastructure and Weaviate

The cost benefit potential of GPU computing is further being pioneered by our partners at Modal and Foundry, to give a couple of examples. Paraphrasing from Jared Quincy Davis on the No Priors podcast, GPU computing is like a parking lot business. Say Erik pays for a $100 premium membership in a fancy downtown lot and Connor pays for a $10 standard membership. Connor is able to park in Erik’s fancy spot next to the elevator while Erik is away. However, when Erik returns to the parking lot to park his car, Connor’s car must be moved to a spot in the standard lot zone. This transfer of “car” or “GPU-enabled runtime state” from spot tier to spot tier has been tremendously difficult until recent innovations from Modal and Foundry.
