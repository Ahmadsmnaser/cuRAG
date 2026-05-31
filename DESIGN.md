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

### Current Limits

The block-local shared-memory usage is now fixed: `256` values and `256`
indices per block. This removes the previous `K <= 24` shared-memory limit on
the local NVIDIA GeForce MX250.

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
