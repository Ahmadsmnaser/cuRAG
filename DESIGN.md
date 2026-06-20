# cuRAG Design Notes

## Phase 1 Baseline

Phase 1 establishes a correct flat cosine-similarity baseline before top-K
selection or indexing abstractions are introduced.

### Data Layout

Corpus vectors use row-major storage. For corpus vector `i` and component `j`,
the value is stored at `corpus[i * dim + j]`. This keeps the initial
implementation simple and matches the layout commonly produced by C++ and
Python arrays.

### Cosine Similarity

The cosine-similarity kernel assumes that the query and corpus vectors are
already L2-normalized. Cosine similarity is therefore computed as a dot
product.

Each CUDA thread computes the score for one corpus vector. Threads in a block
cooperatively copy the query into shared memory before reading corpus rows from
global memory. The baseline block size is `256` threads. It is intentionally
fixed until profiling gives a reason to change it.

### L2 Normalization

The normalization kernel launches one CUDA block per vector. Threads within a
block compute partial sums, reduce them in shared memory, and then normalize
the vector in place. Zero vectors remain zero.

Corpus vectors should be normalized once when an index is built. Query vectors
are normalized before searching. This amortizes corpus normalization across
searches.

### Scope And Limits

- Phase 1 uses `float32` only.
- The current maximum vector dimension is `1024`.
- Search handles one query against many corpus vectors.
- Top-K selection is deferred to Phase 2.
- The current API accepts device pointers. Host-to-device transfer is kept
  outside the kernels so transfer and compute costs can be measured separately.

## Benchmark Environment

The current numbers are local development baselines. They are useful for
tracking implementation changes, but they are not final comparisons against
FAISS, PyTorch, or other production libraries.

| Component | Value |
| --- | --- |
| GPU | NVIDIA GeForce MX250 |
| CUDA version | TODO |
| OS | TODO |
| Compiler | TODO |
| Build type | TODO |

### Baseline Benchmark

`benchmark_cosine_similarity` measures only cosine-similarity kernel execution
using CUDA events. It performs warmup launches before timing and excludes CUDA
runtime initialization, memory allocation, transfers, and normalization.

Run it with:

```bash
cmake -S . -B build-linux
cmake --build build-linux
./build-linux/benchmark_cosine_similarity
```

Initial local baseline:

| Corpus vectors | Dimension | Kernel time | Throughput |
| ---: | ---: | ---: | ---: |
| 10,000 | 768 | 2.741 ms | 3,648,382 vectors/sec |
| 100,000 | 768 | 26.169 ms | 3,821,377 vectors/sec |

These numbers are an initial development-machine snapshot, not a comparison
against FAISS or PyTorch. Hardware details and comparative benchmarks will be
recorded with the benchmark suite.

For an equivalent naive PyTorch CUDA baseline, run:

```bash
python3 benchmarks/benchmark_pytorch.py
```

The PyTorch comparison script uses `torch.mv`, the same `10K` and `100K`
corpus sizes, dimension `768`, warmup launches, and CUDA-event timing. PyTorch
is not currently installed in the local development environment, so comparison
numbers have not been recorded yet.

## Phase 2 - Top-K Selection

Phase 2 adds GPU top-K selection after cosine similarity.

The first implementation used a correctness-first baseline: one thread scanned
all scores and maintained a sorted top-K buffer. This established the API,
output format, and tests before introducing parallel work.

The current implementation uses block-wise bitonic selection. Each block loads
up to `256` scores into shared memory and bitonic-sorts them in descending
order. Each block emits its best `min(K, 256)` candidates. A second kernel
merges those candidates into the final top-K result.

Top-K returns both values and indices. The indices are required by the RAG
layer to map retrieved vectors back to documents.

### Why Top-K Instead Of Full Sort?

Sorting all `N` similarity scores is wasteful when retrieval only needs the
best `K`. In a RAG workload, `K` is usually much smaller than `N`: for example,
the system may retrieve `K = 10` vectors from a corpus of `100,000`.

The current design therefore generates local candidates in each block and
merges those candidates globally instead of sorting the complete score array.

### Why Not CPU Top-K?

Performing Top-K selection on the CPU would require copying all `N` similarity
scores from GPU memory to CPU memory. Keeping selection on the GPU allows the
retrieval path to copy back only `K` values and `K` vector indices.

This matches the goal of building a GPU retrieval backend and avoids making a
full score-array transfer part of every query.

### Why Not Thrust or CUB?

Thrust and CUB provide optimized primitives that could sort or select results
more efficiently. cuRAG intentionally avoids those primitives in Phase 2
because the goal is to understand and implement the internals manually.

The learning target is local selection, shared-memory sorting, candidate
generation, and merge behavior. Production systems may use optimized
primitives, but this project prioritizes learning and analysis.

### Deterministic Tie-Breaking

When two scores are equal, the smaller vector index is ranked first. This keeps
tests deterministic and avoids unstable output ordering when similarity scores
tie.

### Current Limits

The block-local shared-memory usage is now fixed: `256` values and `256`
indices per block. This removes the previous shared-memory bottleneck that
prevented larger `K` values such as `K = 100` on the local GPU.

The final merge stage currently runs on a single CUDA thread and maintains a
sorted insertion buffer of up to `K = 1024` results. This is correct but
intentionally not the final optimized implementation. Its cost grows quickly
as `K` increases.

### Top-K Baseline Benchmark

`benchmark_topk` measures calls to `topk()` using CUDA events. It performs five
warmup runs and averages twenty measured runs. Input allocation and
host-to-device transfer occur before timing. The current `topk()` API allocates
and frees temporary partial-result buffers internally, so that overhead is
included in these measurements.

Run it with:

```bash
cmake -S . -B build-linux
cmake --build build-linux --target benchmark_topk
./build-linux/benchmark_topk
```

Initial block-wise insertion baseline for `K = 10`:

| Input scores | K | Top-K time |
| ---: | ---: | ---: |
| 10,000 | 10 | 1.40022 ms |
| 100,000 | 10 | 9.70957 ms |

Block-wise bitonic local-selection baseline:

| Input scores | K | Top-K time |
| ---: | ---: | ---: |
| 10,000 | 10 | 0.389888 ms |
| 100,000 | 10 | 2.23401 ms |
| 10,000 | 100 | 8.28913 ms |
| 100,000 | 100 | 23.6704 ms |

For `K = 10`, the bitonic local-selection redesign reduced measured top-K
latency by approximately `72%` at `10,000` scores and `77%` at `100,000`
scores.

### Integrated Search Baseline

`benchmark_search_pipeline` measures the complete GPU search path:

```text
query normalization -> cosine similarity -> top-K selection
```

Corpus normalization is excluded because corpus vectors are normalized once
when an index is built. Host-to-device transfer is also excluded. The query is
copied back to the GPU before each timed iteration because normalization
modifies it in place. As with the standalone top-K benchmark, top-K timing
includes its internal temporary partial-result allocation and free.

Run it with:

```bash
cmake -S . -B build-linux
cmake --build build-linux --target benchmark_search_pipeline
./build-linux/benchmark_search_pipeline
```

Initial local baseline for dimension `768` and `K = 10`:

| Corpus vectors | Query normalization | Cosine similarity | Top-K | Total search |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 0.041 ms | 2.756 ms | 0.384 ms | 3.275 ms |
| 100,000 | 0.042 ms | 26.134 ms | 1.846 ms | 28.102 ms |

At `100,000` vectors, cosine similarity dominates search latency. Top-K
accounts for approximately `7%` of the measured total latency for `K = 10`.
The serial merge remains a meaningful optimization target for larger values of
`K`.

## Phase 3 - Index Ownership And Memory Reuse

Phase 3 introduces a synchronous C++ `Index` abstraction. The index owns the
normalized GPU corpus and reuses GPU buffers for the query, similarity scores,
and final Top-K outputs across searches.

Corpus upload and normalization happen once during `Index::build()`. A call to
`Index::search()` performs:

```text
query host-to-device copy
-> query normalization
-> cosine similarity
-> Top-K selection
-> K values and K indices copied back to the host
```

The single-query API remains synchronous. Batch search has progressed from a
sequential API wrapper to a partially batched GPU path. Query transfer,
normalization, and cosine similarity are batched; Top-K selection still loops
over queries on the host and launches the existing single-query Top-K path.
CUDA streams and a fully batched Top-K implementation remain Phase 3 work.

### Synchronous Index Search Baseline

`benchmark_index_search` measures the public `Index::search()` API using host
wall-clock timing. It builds and normalizes the corpus before timing, performs
five warmup queries, and averages `100` measured queries.

Unlike the kernel-only benchmarks, these measurements include query transfer,
kernel execution, synchronization, final Top-K result transfer, and host result
construction. Top-K still allocates and frees its partial-result buffers
internally on every search.

Run it with:

```bash
cmake -S . -B build-linux
cmake --build build-linux --target benchmark_index_search
./build-linux/benchmark_index_search
```

Initial synchronous `Index::search()` baseline for dimension `768` and
`K = 10`:

| Corpus vectors | Queries | Average latency | Throughput |
| ---: | ---: | ---: | ---: |
| 10,000 | 100 | 3.383 ms/query | 295.611 queries/sec |
| 100,000 | 100 | 28.491 ms/query | 35.099 queries/sec |

These are local development baselines for tracking Phase 3 changes. They are
not final throughput claims or comparisons against production libraries.

### Batch Search Version 1 Baseline

The first `Index::search_batch()` implementation is a correctness-oriented API
baseline. It loops over the input queries, calls synchronous `Index::search()`
for each query, and packs the individual results into one `BatchSearchResult`.
It does not yet batch query transfers, execute multi-query kernels, overlap
work with CUDA streams, or reduce the number of per-query synchronizations.

`benchmark_index_batch` compares this first batch implementation with an
explicit loop of repeated `Index::search()` calls over the same `100` queries.
Both paths use host wall-clock timing and include the complete synchronous
search cost.

Run it with:

```bash
cmake -S . -B build-linux
cmake --build build-linux --target benchmark_index_batch
./build-linux/benchmark_index_batch
```

Initial local baseline for dimension `768` and `K = 10`:

| Corpus vectors | Method | Total time | Latency | Throughput | Relative speedup |
| ---: | --- | ---: | ---: | ---: | ---: |
| 10,000 | Repeated `search()` | 528.842 ms | 5.288 ms/query | 189.093 queries/sec | 1.000x |
| 10,000 | `search_batch()` v1 | 503.015 ms | 5.030 ms/query | 198.801 queries/sec | 1.051x |
| 100,000 | Repeated `search()` | 2,897.335 ms | 28.973 ms/query | 34.514 queries/sec | 1.000x |
| 100,000 | `search_batch()` v1 | 3,027.159 ms | 30.272 ms/query | 33.034 queries/sec | 0.957x |

The two methods have effectively equivalent performance because they execute
the same per-query search path. The small differences are normal run-to-run
variation and host-side result-packing overhead, not evidence of GPU batching.
These measurements establish the baseline that a future stream-based or
multi-query kernel implementation must improve.

### Batch Search Version 2

Version 2 uploads all queries in one transfer, normalizes them in one kernel
launch, and computes the complete `[num_queries, num_vectors]` score matrix
with a two-dimensional CUDA grid. It then copies only the final
`num_queries * K` values and indices back to the host.

The Top-K stage is not fully batched yet. `batched_topk()` loops over queries on
the host and invokes the existing GPU Top-K implementation once per score row.
The current speedup therefore comes primarily from batched query transfer,
normalization, and cosine-similarity execution rather than concurrent Top-K
selection.

Version 2 results for `100` queries, dimension `768`, and `K = 10`:

| Corpus vectors | Method | Total time | Latency | Throughput | Relative speedup |
| ---: | --- | ---: | ---: | ---: | ---: |
| 10,000 | Repeated `search()` | 386.687 ms | 3.867 ms/query | 258.607 queries/sec | 1.000x |
| 10,000 | `search_batch()` v2 | 224.063 ms | 2.241 ms/query | 446.304 queries/sec | 1.726x |
| 100,000 | Repeated `search()` | 2,820.624 ms | 28.206 ms/query | 35.453 queries/sec | 1.000x |
| 100,000 | `search_batch()` v2 | 1,771.905 ms | 17.719 ms/query | 56.436 queries/sec | 1.592x |

Compared with repeated synchronous search in the same benchmark run, version
2 improved throughput by approximately `73%` for `10,000` corpus vectors and
`59%` for `100,000` corpus vectors. These are local development measurements,
not production throughput claims. The next optimization target is replacing
the host loop in `batched_topk()` with a genuinely batched GPU selection path.

### Index Serialization

`Index::save()` writes the normalized corpus to disk in a simple binary format:

```text
magic: "CURAGIDN"
version: uint32
dim: int32
num_vectors: int32
corpus: float32[num_vectors * dim]
```

The stored corpus is already L2-normalized because normalization happens during
`Index::build()`. `Index::load()` validates the magic bytes, version,
dimension, and vector count before copying the corpus back to GPU memory.

This format is intentionally minimal for Phase 3. It does not include metadata
such as document IDs, embedding model information, endianness markers, or
checksums.

## Phase 4 - Python Bindings

The Python extension is implemented with pybind11 and packaged with
scikit-build-core. An editable installation builds the CUDA-backed extension
module and exposes the C++ `Index` class as `curag.Index`.

The current Python API accepts NumPy arrays with these logical shapes:

```text
Index.build(corpus)          corpus:  [num_vectors, dim]
Index.search(query, k)       query:   [dim]
Index.search_batch(q, k)     queries: [num_queries, dim]
```

Single and batch searches return dictionaries containing `values` and
`indices`. Batch results also include `num_queries` and `k`. Index `save()` and
the static `Index.load()` method are exposed directly.

### Python Binding Tests

The Python test suite covers index construction, single-query search, batch
search, save/load round trips, invalid array rank, and dimension mismatch.

Run it with:

```bash
python3 -m pytest python_tests/test_python_bindings.py -q
```

Current result:

```text
5 passed
```

### Python Binding Benchmark

`benchmark_python_bindings.py` measures the public Python API with
`time.perf_counter()`. Corpus and query generation occur before timing. Build
time includes corpus host-to-device transfer, GPU normalization, and
synchronization. Search timings include Python-to-C++ binding overhead and the
complete synchronous search path.

Run it with:

```bash
python3 benchmarks/benchmark_python_bindings.py
```

Initial local results for dimension `768`, `K = 10`, and `100` queries:

| Corpus vectors | Build time | Method | Total search time | Latency | Relative speedup |
| ---: | ---: | --- | ---: | ---: | ---: |
| 10,000 | 2,446.697 ms | Repeated `search()` | 319.269 ms | 3.193 ms/query | 1.000x |
| 10,000 | 2,446.697 ms | `search_batch()` | 181.258 ms | 1.813 ms/query | 1.761x |
| 100,000 | 155.565 ms | Repeated `search()` | 2,843.003 ms | 28.430 ms/query | 1.000x |
| 100,000 | 155.565 ms | `search_batch()` | 1,760.193 ms | 17.602 ms/query | 1.615x |

The first `10,000`-vector build is the first CUDA operation in the benchmark
process and likely includes CUDA context initialization. Its build time should
not be compared directly with the later `100,000`-vector build as a corpus-size
scaling result. A future benchmark should perform an explicit CUDA warmup and
report multiple build iterations.

The Python batch speedups are consistent with the C++ batch benchmark and come
from batched query transfer, normalization, and cosine-similarity execution.
Top-K still launches the single-query Top-K path once per query.

### Current Binding Limits

- CUDA calls execute while the Python GIL is held; GIL release remains to be
  added around long-running build and search operations.
- NumPy arguments use pybind11 `forcecast`, so incompatible dtypes and
  non-contiguous inputs may be copied into temporary contiguous `float32`
  arrays instead of being rejected.
- Search outputs are currently converted to Python lists inside dictionaries,
  rather than returned as shaped NumPy arrays.
- PyTorch tensor interoperability and zero-copy device input are not yet
  implemented.
