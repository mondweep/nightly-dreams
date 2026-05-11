# ADR-005: PDX Columnar Layout for SIMD-Accelerated Distance Computation

**Date:** 2026-05-11  
**Status:** Proposed  
**Deciders:** Nightly Research Agent  
**Technical Area:** Storage layer / distance kernel  
**Research doc:** [docs/research/nightly/2026-05-11-pdx-columnar-simd/README.md](../research/nightly/2026-05-11-pdx-columnar-simd/README.md)

---

## Context

Vector search throughput is bounded by memory bandwidth, not floating-point compute. A full scan of 10,000 × 1536-dimensional f32 vectors requires reading 61 MB, and this budget is consumed on every query for cold-cache data. The row-major storage layout used throughout ruvector — and by all major competitors (FAISS, Qdrant, Milvus) — exacerbates this: computing distance between a query and N candidates requires strided reads that defeat prefetch and prevent SIMD-width utilisation.

Kuffo et al. (CWI, SIGMOD 2025, arXiv 2503.04422) demonstrate that transposing the vector layout within fixed-width blocks — so that dimension *d* of candidates 0…B-1 is stored contiguously — eliminates the stride, enabling full-width SIMD auto-vectorisation with zero horizontal reductions in the hot path and permitting dimension-indexed early-exit pruning (ADSampling / BSA). Their C++ implementation achieves 2× average kernel speedup over hand-written AVX-512 horizontal kernels and 4.6× over FAISS IVF with ADSampling.

This ADR concerns integrating the PDX storage layout as a first-class primitive in ruvector's distance kernel layer, validated by a working Rust PoC showing **2.36× QPS improvement** at 768-dim and **2.10×** at 1536-dim versus row-major brute-force at 100% Recall@10.

Prior ruvector research has addressed index algorithms (DABS-HNSW, ADR-003), quantisation (RaBitQ+IVF, ADR-002), retrieval strategies (MRL cascaded, ADR-004), and query planning (adaptive filtered ANN, ADR-001). None of these touched the storage kernel layer, leaving a structural performance gap that PDX fills orthogonally.

---

## Decision

Introduce PDX columnar vector layout as a storage primitive in ruvector, beginning with this PoC and leading to a production `ruvector-pdx` crate:

1. **`Distance` trait** — unified interface with both `distance()` (row-major scalar) and `distance_columnar()` (column-major batch) methods. Concrete implementations (`L2Sq`, `NegDot`) provide LLVM-auto-vectorisable inner loops with no `std::arch` unsafe required at PoC stage.

2. **`ColumnarIndex`** — a flat brute-force index that stores vectors transposed: `columns[d][i]` == vector *i*, dimension *d*. Replaces the standard `Vec<Vec<f32>>` row-major storage. Demonstrates the layout effect on distance throughput without index-algorithm interference.

3. **`ChunkedColumnarIndex` (tiled PDX)** — partitions vectors into fixed-size tiles; distance is computed tile-by-tile with optional partial-distance pruning. Provides the early-exit foundation for the full PDX paper implementation. Production version replaces `Vec<&[f32]>` scratch with pre-allocated thread-local buffers.

4. **Trait-based design** — the `Distance` trait decouples metric from storage, enabling any existing or future index (IVF, HNSW, flat) to adopt the columnar kernel by calling `distance_columnar` over its candidate set rather than looping over row-major vectors.

5. **`recall_at_k` utility** — shared utility function for benchmark validation and acceptance testing, consistent with prior ADRs.

The production integration path (IVF bucket storage, HNSW candidate evaluation, const-generic tile, f16 columns, ADSampling early-exit) is deferred to follow-on work. This ADR scopes only the trait abstraction, two index implementations, and the flat-scan PoC demonstrating the layout benefit.

---

## Consequences

### Positive

- **2.36× throughput improvement** at 768-dim (BERT/all-MiniLM embeddings), **2.10×** at 1536-dim (OpenAI text-embedding-3), **1.83×** at 128-dim (MRL-truncated) — all at 100% Recall@10.
- **Zero algorithmic cost** — the gain is entirely from memory access pattern improvement. Index structure, recall, and distance metric semantics are unchanged.
- **Additive to all prior ADRs** — PDX accelerates the distance kernel that RaBitQ, MRL cascade, filtered ANN, and DABS-HNSW all invoke. It stacks with quantisation (f16 columns), cascaded search (coarse-pass columnar kernel), and graph pruning (candidate evaluation).
- **No unsafe code** — the PoC achieves auto-vectorisation with safe Rust. LLVM emits AVX2 (8-wide) or AVX-512 (16-wide) FMA chains from the plain double loop. Unsafe `std::arch` intrinsics are a follow-on optimisation for critical paths only.
- **Portability** — the scalar fallback path is correct on all targets. SIMD acceleration is an emergent property of the loop structure, not a hard dependency.

### Negative

- **Higher insertion overhead** — building the columnar index requires D separate append operations per vector (one per column) versus one contiguous `extend_from_slice`. At D=1536 this is measurably slower for write-heavy workloads.
- **Memory fragmentation** — the PoC uses `Vec<Vec<f32>>` (D allocations per index), which increases heap fragmentation vs. a single flat buffer. Production implementation must use a single contiguous allocation with column-pointer arithmetic.
- **ChunkedColumnarIndex PoC regression** — the chunked path is 2.5× *slower* than row-major in this PoC due to per-query `Vec<&[f32]>` allocations for `col_refs`. This is a known implementation artefact documented in the research note; it is resolved by thread-local scratch buffers in production.
- **No HNSW integration yet** — HNSW candidate evaluation (10–200 vectors per hop) is too small for PDX tiling to provide benefit. PDX is most effective at IVF bucket scale (1,000–50,000 vectors). Full HNSW integration requires a separate ADR.

### Neutral

- **No new dependencies** — the PoC adds only `rand` and `ordered-float`, already used in prior ADRs.
- **No breaking changes** — the `Distance` trait is additive; existing row-major code paths continue to work.

---

## Alternatives Considered

### A: Hand-written AVX-512 Intrinsics (Rejected for PoC)

Adding `std::arch` intrinsics would require `unsafe` blocks, feature detection at runtime, and separate ARM/NEON fallbacks. The auto-vectorised safe Rust achieves 80–95% of the manual intrinsics speedup according to the PDX paper's ablation (the compiler handles horizontal reduction elimination which is the dominant gain). Deferred to follow-on work for the final 10–20% performance headroom.

### B: Row-major with SIMD Loop (Rejected)

FAISS and Qdrant manually write AVX-512/AVX2 kernels for row-major storage. This requires horizontal adds at the end of each vector's distance computation: a `VHADDPS` tree that wastes 3 cycles per vector. At N=10,000 and D=768, that is 30M wasted cycles per query. PDX eliminates horizontal adds entirely.

### C: LanceDB-style columnar storage (Rejected as insufficient)

LanceDB uses Apache Arrow / Lance format for on-disk columnar storage, but converts to row-major for in-memory SIMD distance computation. This captures storage I/O bandwidth benefits but not the compute kernel benefit that PDX provides. PDX is orthogonal and complementary.

### D: Quantisation instead of layout change (Deferred)

Storing columns as f16 would halve memory bandwidth and is a natural follow-on. However, f16 requires careful handling of accumulation errors (Kahan summation or f32 accumulators). This is a separate ADR.

### E: Product Quantisation codebooks (Already done in ADR-002)

RaBitQ+IVF (ADR-002) addresses quantisation-based compression. PDX addresses storage layout for uncompressed or lightly compressed (f16) vectors. Both are necessary; they target different performance bottlenecks.

---

## References

1. Kuffo et al., "PDX: A Data Layout for Vector Similarity Search", SIGMOD 2025, arXiv 2503.04422.
2. cwida/PDX C++ reference: https://github.com/cwida/PDX
3. Lyu et al., "Bang for the Buck: Designing a Vector Search Engine for Cloud CPUs", arXiv 2505.07621.
4. Johnson et al., "Billion-scale similarity search with GPUs", IEEE TPAMI 2019.
5. Prior ADRs: ADR-001 (adaptive filtered ANN), ADR-002 (RaBitQ+IVF), ADR-003 (DABS-HNSW), ADR-004 (MRL cascaded search).
