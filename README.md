# cuRAG

cuRAG is a CUDA vector search library built from scratch for learning and experimenting with retrieval backends used in RAG systems.

The goal of this project is not to replace FAISS, but to understand and implement the core GPU systems behind vector retrieval:

- L2 normalization
- Cosine similarity
- Top-k selection
- Reusable GPU memory management
- Index build/search APIs
- Batched query search
- Python bindings
- Minimal RAG-style retrieval demo

## Status

Current features:

- CUDA cosine similarity kernel
- CUDA L2 normalization kernel
- GPU top-k selection
- `curag::Index` C++ API
- RAII GPU memory management with `DeviceBuffer`
- Index save/load
- Batched GPU search
- Python bindings using pybind11
- Python tests and benchmark
- Minimal retrieval demo in Python

## Architecture

```text
documents / vectors
        ↓
L2 normalization
        ↓
cuRAG Index
        ↓
cosine similarity
        ↓
top-k selection
        ↓
retrieved document indices
```

For batched search:

```text
queries[Q, dim]
        ↓
batched normalization
        ↓
batched cosine similarity
        ↓
batched top-k
        ↓
values[Q, K], indices[Q, K]
```

## Repository Layout

```text
include/curag/       Public C++ headers
kernels/             CUDA kernels
src/                 C++/CUDA library implementation
tests/               C++ correctness tests
benchmarks/          C++ and Python benchmarks
bindings/            pybind11 Python bindings
python_tests/        Python binding tests
examples/            Minimal RAG retrieval demo
```

## Build C++/CUDA

```bash
cmake -S . -B build-linux
cmake --build build-linux
```

## Run C++ Tests

```bash
ctest --test-dir build-linux --output-on-failure
```

Expected result:

```text
100% tests passed
```

## Python Installation

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install in editable mode:

```bash
python3 -m pip install -e .
```

Test import:

```bash
python3 -c "import curag; print(curag.__doc__)"
```

Expected output:

```text
cuRAG CUDA vector search bindings
```

## Python Usage

```python
import numpy as np
import curag

corpus = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.8, 0.6],
    ],
    dtype=np.float32,
)

query = np.array([1.0, 0.0], dtype=np.float32)

index = curag.Index(2)
index.build(corpus)

result = index.search(query, k=2)

print(result["values"])
print(result["indices"])
```

Example output:

```text
[1.0, 0.800000011920929]
[0, 3]
```

## Batched Search from Python

```python
import numpy as np
import curag

corpus = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.8, 0.6],
        [0.6, 0.8],
    ],
    dtype=np.float32,
)

queries = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
    ],
    dtype=np.float32,
)

index = curag.Index(2)
index.build(corpus)

result = index.search_batch(queries, k=2)

print(result["values"])
print(result["indices"])
```

The output layout is flat:

```text
values[q * K + j]
indices[q * K + j]
```

## Python Tests

```bash
python3 -m pytest python_tests -q
```

## Minimal RAG Retrieval Demo

```bash
python3 examples/rag_demo.py
```

This demo uses a small bag-of-words embedding to demonstrate the retrieval pipeline without requiring external model downloads.

It shows:

```text
text → vector → cuRAG Index → top-k retrieved documents
```

Example output:

```text
Question:
What is CUDA shared memory?

Top retrieved documents:
1. CUDA shared memory is fast on-chip memory used by threads inside the same block.
```

## Benchmarks

Local benchmark environment:

- GPU: NVIDIA GeForce MX250
- Dimension: 768
- K: 10
- Queries: 100

### Python Binding Benchmark

| Corpus vectors | Repeated search | Batched search | Speedup |
| ---: | ---: | ---: | ---: |
| 10,000 | 3.193 ms/query | 1.813 ms/query | 1.761x |
| 100,000 | 28.430 ms/query | 17.602 ms/query | 1.615x |

### C++ Batch Benchmark

| Corpus vectors | Repeated search | Batched search | Speedup |
| ---: | ---: | ---: | ---: |
| 10,000 | 3.867 ms/query | 2.241 ms/query | 1.726x |
| 100,000 | 28.206 ms/query | 17.719 ms/query | 1.592x |

## Design Notes

cuRAG stores vectors in row-major layout:

```text
corpus[i * dim + j]
```

Batched search stores the full score matrix:

```text
scores[num_queries * num_vectors]
```

This makes the implementation simple and fast for moderate batch sizes, but memory usage grows as:

```text
num_queries * num_vectors * sizeof(float)
```

Examples:

| Queries | Corpus vectors | Score memory |
| ---: | ---: | ---: |
| 100 | 100,000 | ~40 MB |
| 100 | 1,000,000 | ~400 MB |

## Current Limitations

- Exact brute-force search only
- No IVF/HNSW/ANN index yet
- Maximum supported dimension is currently 1024
- Batched search materializes the full `Q x N` score matrix
- `Index::search()` and `Index::search_batch()` reuse internal mutable buffers and are not thread-safe on the same `Index` instance
- The minimal RAG demo uses toy embeddings, not real language model embeddings
- No FAISS comparison benchmark yet

## Future Work

- Add real embedding demo with SentenceTransformers
- Add FAISS comparison benchmark
- Add approximate indexing
- Improve top-k merge stage
- Add CUDA streams
- Add packaging wheels
- Add LLM-based answer generation demo

## Why This Project Exists

Most RAG demos use existing vector databases or libraries as black boxes. cuRAG is built to understand the retrieval backend itself:

- How vectors are stored on GPU
- How cosine similarity is computed at scale
- How top-k retrieval works
- How batching changes throughput
- How a C++/CUDA backend can be exposed to Python

This makes cuRAG a systems-focused learning project for GPU programming, vector search, and AI infrastructure.
