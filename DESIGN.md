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

## Phase 2 — Top-K Selection

Phase 2 adds GPU top-K selection after cosine similarity.

The initial implementation uses a correctness-first baseline: it scans all scores and maintains a sorted top-K buffer. This is intentionally slow, but it establishes the API, output format, and tests before introducing a parallel implementation.

The optimized design will use block-wise top-K selection. Each CUDA block computes a local top-K candidate list, then a second merge stage produces the final global top-K.

Top-K returns both values and indices. The indices are required by the RAG layer to map retrieved vectors back to documents.