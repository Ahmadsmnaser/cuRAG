# cuRAG: Refined Project Plan

## Overview

**Name:** cuRAG

**One-line:** A GPU vector search library built from scratch in CUDA, designed to understand the internals of production systems like FAISS-GPU and to serve as a retrieval backend for RAG pipelines.

**Duration:** 6 weeks core + optional 2-week extension

**Identity:** Systems/CUDA learning project with honest framing. AI/RAG is the application layer.

**Hardware:** MX250 for development. Use Colab T4 for selected benchmarks where MX250 limits credibility.

---

## Success Criteria

By end of project you must have:

1. Working CUDA implementation: batched cosine similarity + top-K + memory management
2. Python bindings via pybind11
3. RAG pipeline using cuRAG as retrieval backend with real embedding model
4. **Honest benchmark suite:** cuRAG vs FAISS-CPU vs FAISS-GPU on 3+ corpus sizes
5. **Gap analysis document** explaining where FAISS is faster and why (the differentiator)
6. README + DESIGN.md with educational framing, architecture diagram, performance numbers, design decisions
7. Clean Git history showing progression

---

## Phase 1 — Foundation & Core Kernel (Week 1)

**Objective:** Working cosine similarity kernel with correct results, baseline performance numbers.

**Deliverables:**
- Repository skeleton with CMake build system
- `cosine_similarity.cu`: batched cosine similarity (1 query × N corpus vectors)
- `normalize.cu`: L2 normalization kernel
- Unit tests verifying correctness against NumPy reference
- Initial benchmark: kernel time vs naive PyTorch on 100K × 768 vectors

**Key technical decisions to make and document:**
- Memory layout: row-major (`corpus[i*dim + j]`) vs column-major
- Thread block size: start with 256, profile, justify final choice
- Pre-normalize vectors at index build time (yes — amortize cost)
- Float32 only for v1 (defer FP16 to extensions)

**Pitfalls:**
- Don't optimize prematurely. First version: one thread per corpus vector, dot product in registers
- Don't skip correctness tests — use `np.allclose` with rtol=1e-5

**Exit gate:** Kernel produces correct top-K identical to NumPy reference. Baseline timing recorded.

---

## Phase 2 — Top-K Selection (Week 2)

**Objective:** Implement efficient top-K selection on GPU. The hardest part.

**Deliverables:**
- `topk_bitonic.cu`: bitonic sort-based top-K
- Decision document: why bitonic over alternatives (radix select, partial sort, heap-based)
- Integrated pipeline: similarity → top-K → output
- Benchmark: top-K time for K ∈ {10, 100, 1000}

**Technical depth required:**
- Understand why naive sort is wasteful (sorting N when you need K)
- Implement bitonic merge for power-of-2 K, handle non-power-of-2
- Use shared memory for merge step
- Profile with Nsight Compute, identify bottleneck

**Alternatives to document:**
- Bitonic sort: good for small-medium K, GPU-friendly
- Radix select: faster for small K, complex
- Block-wise top-K + global merge: scales better for huge corpora
- Thrust/CUB built-ins: defeats project purpose, don't use

**Exit gate:** Top-K results match Python reference. Nsight-profiled. Choice justified in DESIGN.md draft.

---

## Phase 3 — System Architecture & Memory Management (Week 3)

**Objective:** Move from "kernels that work" to "library that's usable". Where systems engineering shows.

**Deliverables:**
- `memory_pool.cu`: simple memory pool to amortize `cudaMalloc` cost
- `index.cu`: Index class (build, search, save, load)
- Async query support: CUDA streams to overlap H2D transfer with compute
- Multi-query batching
- Benchmark: throughput (queries/sec) for batched vs sequential

**Architecture decisions:**
- Index ownership of GPU memory (RAII pattern)
- Error handling: CUDA errors → custom exception type
- Thread safety: document what's safe concurrently
- Memory budget: document the limit when corpus doesn't fit in GPU

**This phase differentiates "school project" from "library". Don't skip.**

**Exit gate:** Build index from NumPy array, search single/batched, save/load to disk. Memory profiled.

---

## Phase 4 — Python Bindings & Embedding Integration (Week 4)

**Objective:** Make cuRAG callable from Python, integrate with real embedding model.

**Deliverables:**
- `bindings/python_binding.cpp`: pybind11 wrapper
- Python API: `Index.build(vectors)`, `Index.search(queries, k)`, `Index.save(path)`, `Index.load(path)`
- Integration with Sentence-Transformers (`all-MiniLM-L6-v2`, 384 dim)
- Demo script: build index from corpus (Wikipedia snippets, 100K docs)
- Zero-copy from PyTorch tensors where possible

**Technical considerations:**
- Release GIL during CUDA calls
- Explicit memory ownership between Python and C++
- CUDA errors → Python exceptions with useful messages
- Fail fast on wrong input (non-contiguous, wrong dtype)

**Exit gate:** `pip install -e .` works. End-to-end Python pipeline produces correct results.

---

## Phase 5 — RAG Pipeline & Agent Layer (Week 5)

**Objective:** Wrap cuRAG as retrieval backend in real RAG pipeline. Keep thin — not the focus.

**Deliverables:**
- `agent/rag_pipeline.py`: simple RAG pipeline (no LangChain, write yourself, ~100 lines)
- Components: embedding → retrieval (cuRAG) → LLM call (Groq free API or local Ollama)
- Benchmark dataset: 50-100 questions over corpus, manual quality check
- Optional: PydanticAI agent that auto-tunes batch size and reports latency

**Why custom RAG over LangChain:**
- LangChain abstracts away systems details
- 100-line custom pipeline shows architecture understanding
- Easier to instrument and benchmark
- Mention this decision explicitly in README

**Exit gate:** End-to-end RAG works. Latency breakdown documented (embedding/retrieval/LLM times).

---

## Phase 6 — Benchmarks, Gap Analysis, Writeup (Week 6)

**Objective:** Tell the story honestly. This phase makes it a portfolio piece.

### Benchmark suite
- `benchmarks/vs_faiss_cpu.py`
- `benchmarks/vs_faiss_gpu.py` ← critical, honest comparison
- `benchmarks/vs_numpy.py`
- Three corpus sizes: 10K, 100K, 1M at 384 and 768 dim
- Metrics: build time, latency (p50/p95/p99), throughput, memory footprint
- Output: CSV + matplotlib charts in `benchmarks/results/`

### Gap analysis (the differentiator)

Create `GAP_ANALYSIS.md` documenting where FAISS-GPU is faster and why. Examples to investigate:

1. **Warp-level primitives** — does FAISS use `__shfl_xor_sync` for reductions? Measure.
2. **Kernel fusion** — FAISS likely fuses similarity + top-K into one kernel. You have them separate.
3. **Memory access patterns** — analyze with Nsight, compare access patterns.
4. **IVF indexing** — FAISS supports approximate search via IVF. You're flat-only. Explain the algorithmic difference.
5. **Persistent kernels** — does FAISS keep kernels resident? Investigate.

For each: 1-2 paragraphs explaining what FAISS does, why it's faster, why you didn't implement it (scope/learning focus).

**This document is your interview gold.** Any NVIDIA engineer reading it will see you understand GPU optimization at a real level.

### README structure (non-negotiable)

1. One-line description with honest positioning
2. Performance summary table (cuRAG vs FAISS-GPU, % of FAISS performance)
3. Architecture diagram
4. Technical highlights (3-5 bullets, systems-flavored)
5. Design decisions → DESIGN.md
6. Gap analysis → GAP_ANALYSIS.md
7. Build instructions
8. Usage example
9. Benchmarks with charts
10. Limitations (be honest)
11. Future work (HNSW, IVF, FP16)

### README opening (suggested framing)

> cuRAG is a CUDA implementation of GPU vector search built from scratch as a learning exercise. It achieves X% of FAISS-GPU performance on equivalent workloads; the analysis of *why* FAISS is faster is documented in GAP_ANALYSIS.md and constitutes the primary educational outcome of this project.

This framing is your shield against "but FAISS exists" and your sword for interview discussions.

**Exit gate:** Repository looks production-ready. README scannable in 30 seconds. Numbers reproducible. GAP_ANALYSIS.md is substantive (3+ pages).

---

## Phase 7 — HNSW on GPU (Optional Extension, Weeks 7-8)

**Only if time permits. Don't start until Phase 6 is fully done.**

**Why this matters:** HNSW is the dominant ANN algorithm on CPU. FAISS-GPU does *not* implement HNSW well — it's algorithmically hostile to GPUs (irregular memory access, sequential graph traversal). A GPU HNSW implementation is **a genuinely novel contribution**, not a reimplementation. This transforms the project from "good learning exercise" to "interesting work".

**Deliverables:**
- HNSW graph construction (can do on CPU initially)
- GPU-friendly batched search: process N queries in parallel through the graph
- Comparison: cuRAG-HNSW vs cuRAG-flat vs FAISS-CPU HNSW
- Documented design decisions (irregular memory access mitigation, warp divergence handling)

**Risks:**
- Could take 4+ weeks instead of 2
- Easy to get stuck — HNSW on GPU is an active research area
- If you start, commit to finishing or formally close as "work in progress"

---

## Weekly Time Allocation

| Week | Focus | Hours |
|------|-------|-------|
| 1 | Core kernel + tests | 10 |
| 2 | Top-K + profiling | 10 |
| 3 | Memory mgmt + architecture | 10 |
| 4 | Python bindings + integration | 8 |
| 5 | RAG pipeline | 6 |
| 6 | Benchmarks + GAP_ANALYSIS + writeup | 12 |
| **Core total** | | **~56 hours** |
| 7-8 | HNSW extension (optional) | 20-40 |

**If behind schedule:** Cut from Phase 5 (RAG can be minimal). Never cut Phases 1-3 or Phase 6. GAP_ANALYSIS is non-negotiable.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| MX250 too slow, results look bad | Frame as "demonstrates technique"; use Colab T4 for FAISS-GPU comparison; note hardware everywhere |
| FAISS-GPU dramatically faster | Expected. Gap analysis turns this into the project's strength |
| Stuck on bitonic top-K | Fallback: block-wise top-K + global merge |
| pybind11 memory issues | Start binding early, don't leave to Phase 4 |
| Scope creep (FP16, IVF, etc.) | Resist. Mark as "future work" |
| Time conflict with Siraj | Phases 3 and 6 non-skippable. Cut Phase 5 first |
| Tempted to start Phase 7 too early | Don't. Finish Phase 6 fully first |

---

## What This Project Communicates

**To NVIDIA/AMD systems recruiter:**
- Writes CUDA from scratch
- Understands memory hierarchy, profiling, optimization tradeoffs
- Can architect a library
- Has the maturity to analyze production code (FAISS) and understand why it's fast
- Benchmarks honestly, doesn't oversell

**To AI infrastructure recruiter:**
- Understands RAG at systems level
- Built retrieval backend, didn't just use FAISS
- Wrote custom RAG pipeline, understands the stack

**Universal signal:**
- Ships finished work with documentation
- Honest about limitations (rare and valuable)
- Self-aware about positioning
- Educational mindset, not ego-driven

**The honest positioning is itself a differentiator.** Most portfolio projects oversell. Yours undersells with substance, which reads as senior-engineer behavior.

---

## CV Bullet (Final Form)

> **cuRAG — GPU Vector Search Library** (CUDA, C++, Python)
> Built a GPU-accelerated vector similarity search library from scratch: custom batched cosine similarity kernels, bitonic top-K selection, async memory management via CUDA streams. Achieved X% of FAISS-GPU throughput on 1M 768-dim vectors; documented optimization gap analysis identifying warp-level primitives and kernel fusion as key FAISS advantages. Integrated as retrieval backend in custom RAG pipeline.

This bullet works for NVIDIA, AMD, Intel systems roles **and** AI infrastructure roles. Same line, two stories.

---

## Repository Structure

```
cuRAG/
├── CMakeLists.txt
├── README.md
├── DESIGN.md
├── GAP_ANALYSIS.md
├── LICENSE
├── .gitignore
│
├── kernels/
│   ├── cosine_similarity.cu
│   ├── normalize.cu
│   └── topk_bitonic.cu
│
├── src/
│   ├── index.cu
│   ├── search.cu
│   └── memory_pool.cu
│
├── include/
│   └── curag/
│       ├── index.hpp
│       └── types.hpp
│
├── bindings/
│   └── python_binding.cpp
│
├── agent/
│   ├── rag_pipeline.py
│   └── benchmark_agent.py
│
├── tests/
│   ├── test_similarity.cu
│   ├── test_topk.cu
│   └── test_python.py
│
└── benchmarks/
    ├── vs_faiss_cpu.py
    ├── vs_faiss_gpu.py
    ├── vs_numpy.py
    └── results/
```

---

## Next Steps

1. Create GitHub repo today, even empty. Name: `cuRAG`. Initialize with README skeleton (using the honest positioning) and MIT license.
2. Set up CMake build with one dummy kernel (`cudaMalloc + cudaFree`). Verify build pipeline works end-to-end before writing real code.
3. Start Phase 1.
