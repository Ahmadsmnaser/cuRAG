import time
import numpy as np
import curag


def benchmark(num_vectors, dim, k, num_queries):
    rng = np.random.default_rng(42)

    corpus = rng.normal(size=(num_vectors, dim)).astype(np.float32)
    queries = rng.normal(size=(num_queries, dim)).astype(np.float32)

    index = curag.Index(dim)

    t0 = time.perf_counter()
    index.build(corpus)
    t1 = time.perf_counter()

    single_query = queries[0]

    # warmup
    index.search(single_query, k)
    index.search_batch(queries, k)

    t2 = time.perf_counter()
    for q in queries:
        index.search(q, k)
    t3 = time.perf_counter()

    t4 = time.perf_counter()
    index.search_batch(queries, k)
    t5 = time.perf_counter()

    build_ms = (t1 - t0) * 1000.0
    repeated_ms = (t3 - t2) * 1000.0
    batch_ms = (t5 - t4) * 1000.0

    print(f"N={num_vectors} dim={dim} K={k} queries={num_queries}")
    print(f"  build:           {build_ms:9.3f} ms")
    print(f"  repeated search: {repeated_ms:9.3f} ms  latency={repeated_ms / num_queries:8.3f} ms/query")
    print(f"  search_batch:    {batch_ms:9.3f} ms  latency={batch_ms / num_queries:8.3f} ms/query")
    print(f"  speedup:         {repeated_ms / batch_ms:9.3f}x")
    print()


if __name__ == "__main__":
    benchmark(num_vectors=10_000, dim=768, k=10, num_queries=100)
    benchmark(num_vectors=100_000, dim=768, k=10, num_queries=100)
