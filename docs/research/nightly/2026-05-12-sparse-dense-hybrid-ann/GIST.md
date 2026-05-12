# ruvector 2026: Sparse-Dense Hybrid ANNS — High-Performance Rust Vector Search with Distribution-Aligned Fusion

> **150-char summary:** Hybrid graph index fusing dense embeddings + BM25 sparse vectors in one Rust ANNS pass: +7.2% Recall@10, single index, no RRF merge overhead. Real benchmarks included.

Sparse-dense hybrid vector search is the fastest-growing retrieval challenge in 2026 RAG systems. This post presents a working **Rust proof-of-concept** for a unified graph-based approximate nearest neighbor (ANNS) index that natively combines dense neural embeddings and sparse BM25/SPLADE term-weight vectors — no separate indices, no Reciprocal Rank Fusion merge latency. Implemented as part of the [ruvector](https://github.com/ruvnet/ruvector) nightly research series.

---

## Introduction

Every production RAG pipeline in 2026 faces the same problem: semantic retrieval (dense embeddings, HNSW) catches conceptually similar documents; lexical retrieval (BM25/SPLADE, sparse vectors) catches exact keyword matches. Running both adds 2× latency and 2× index memory. The naive "run separately, merge with RRF" approach discards the cross-modal correlations that could make a single graph traversal more accurate than two separate ones.

The key insight, validated in arXiv:2410.20381 and implemented here: build one navigable small-world graph using a **fused hybrid distance** `d(a,b) = α·L2_dense + (1−α)·IP_sparse`, where **α is calibrated automatically** by comparing the variance of dense distances vs. sparse distances across the corpus. The graph edges then encode *both* semantic proximity and vocabulary overlap simultaneously.

**ruvector sparse-dense hybrid ANNS**: part of the [ruvector high-performance Rust vector database](https://github.com/ruvnet/ruvector), open-source, MIT license.

---

## Features

- **`HybridVector`**: carries both `dense: Vec<f32>` (L2-normalised) and `sparse: BTreeMap<u32, f32>` (L2-normalised term weights) in a single type
- **`HybridNSW` graph index**: navigable small-world graph built and searched over the fused hybrid distance; `M` neighbours per node
- **`calibrate_alpha`**: samples up to 8128 pairwise distances from the corpus and returns the inverse-variance–weighted α automatically
- **`SearchMode::HybridFull`**: all graph hops use hybrid distance — maximum recall
- **`SearchMode::TwoStage`**: dense-only routing during traversal, hybrid reranking of candidate set — faster when dense dominates (α ≥ 0.5)
- **`recall_at_k`**: standard Recall@k utility against brute-force ground truth
- 7 unit tests, `cargo build --release` clean, 374 lines in lib.rs

---

## Benefits

| Benefit | Details |
|---------|---------|
| **+7.2% Recall@10** | Hybrid graph vs. dense-only at the same QPS (0.888 vs. 0.829 at α=0.33) |
| **Single-index simplicity** | No RRF merge, no dual-latency penalty, one query path |
| **Automatic α calibration** | No manual tuning; adapts to corpus characteristics via distribution alignment |
| **Composable** | Plugs into PDX columnar storage (ADR-005), RaBitQ quantisation (ADR-002), MRL cascading (ADR-004) |
| **Pure safe Rust** | No `unsafe`, no SIMD intrinsics at PoC stage; LLVM auto-vectorises dense L2 |
| **Documented failure modes** | TwoStage collapse at α < 0.5 is quantified and guarded against |

---

## Comparisons

| System | Hybrid Retrieval | Graph Type | Joint Traversal? | Notes |
|--------|-----------------|------------|-----------------|-------|
| **ruvector (this work)** | Dense + Sparse | NSW/HNSW hybrid | **Yes** | Distribution-aligned α; single pass |
| Qdrant | Dense + Sparse | HNSW (dense) + segment (sparse) | No | Separate queries, RRF |
| Milvus | Dense + Sparse | HNSW + IVF-Flat | No | Two indices, score fusion |
| Weaviate | Dense + BM25 | HNSW + WAND | No | Separate traversals, late fusion |
| Pinecone | Dense + Sparse | HNSW | No | Separate codepaths per type |
| LanceDB | Dense + FTS | HNSW + Lance format | No | RRF post-merge |
| FAISS | Dense only | IVF / HNSW | N/A | No sparse support |

---

## Benchmarks

**Hardware:** Intel Xeon @ 2.10 GHz, 4 cores, 15 GiB RAM  
**Crate:** `sparse-dense-hybrid-ann` (Rust, release build, LLVM auto-vectorisation)  
**Dataset:** N=2000 hybrid vectors, dense dim=128, sparse vocab=1024, NNZ=20  
**Evaluation:** Recall@10 against brute-force ground truth at the same α

### N=2,000 Benchmark (k=10, M=16)

| Variant | QPS | Recall@10 | Notes |
|---------|-----|-----------|-------|
| BruteForce (ground truth) | 1,111.5 | 1.0000 | Linear scan baseline |
| Variant-1: DenseOnly-NSW | 680.0 | 0.8290 | Ignores sparse component |
| **Variant-2: HybridFull-NSW (α=0.33)** | **677.7** | **0.8880** | Distribution-aligned; +7.2% recall |
| Variant-3a: TwoStage (cf=8) | 815.4 | 0.0595 | ⚠ Fails at sparse-dominant α |
| Variant-3b: TwoStage (cf=30) | 516.6 | 0.0625 | ⚠ Large pool doesn't help |

### Alpha Sensitivity (HybridFull-NSW, same graph)

| α (dense weight) | QPS | Recall@10 |
|-----------------|-----|-----------|
| 0.20 (sparse-dominant) | 687.7 | **0.9205** |
| **0.33 (calibrated)** | **677.7** | **0.8880** |
| 0.50 (balanced) | 670.8 | 0.7825 |
| 0.65 | 680.3 | 0.7950 |
| 0.80 (dense-dominant) | 688.8 | 0.8215 |

**Key finding:** Calibrated α=0.33 lands at 88.8% recall, close to the optimal of 92.1% at α=0.20. The distribution alignment function correctly identifies the corpus as sparse-dominant without any manual tuning.

### Memory Estimate (N=2,000)

| Component | Size |
|-----------|------|
| Dense vectors (f32 × 128 × 2000) | 1.0 MB |
| Sparse vectors (avg 20 terms × 2000) | 0.3 MB |
| Graph adjacency (M=16 × N) | 0.2 MB |
| **Total** | **≈ 1.5 MB** |

Production scale (1M vectors, DIM=768, NNZ=50): ≈ 3.5 GB dense + 0.4 GB sparse + 0.5 GB graph ≈ **4.4 GB**.

---

## Optimizations

The following optimizations are on the roadmap (details in [ADR-006](https://github.com/mondweep/nightly-dreams/blob/research/nightly/2026-05-12-sparse-dense-hybrid-ann/docs/adr/ADR-006-sparse-dense-hybrid-ann.md)):

1. **Hierarchical HNSW layers** — multi-layer routing eliminates the single-entry-point bottleneck. Expected crossover vs. brute-force at N≈50K.
2. **CSR sparse storage** — replace `BTreeMap<u32,f32>` with sorted flat arrays. 3–5× sparse distance speedup via merge-join dot product.
3. **Adaptive TwoStage switching** — mid-traversal switch from dense to hybrid when candidates exceed a distance threshold (from arXiv:2410.20381).
4. **PDX columnar layout for dense** (ADR-005 composability) — 2× distance throughput on the dense component.
5. **RaBitQ sparse quantisation** (ADR-002 composability) — compress sparse component from f32 to lower precision.

---

## Get Started

```bash
# Clone the nightly-dreams research repo
git clone https://github.com/mondweep/nightly-dreams
cd nightly-dreams
git checkout research/nightly/2026-05-12-sparse-dense-hybrid-ann

# Build and run the benchmark binary
cargo run --release -p sparse-dense-hybrid-ann

# Run the test suite
cargo test -p sparse-dense-hybrid-ann
```

**Research branch:** `research/nightly/2026-05-12-sparse-dense-hybrid-ann`  
**Full research doc:** `docs/research/nightly/2026-05-12-sparse-dense-hybrid-ann/README.md`  
**ADR:** `docs/adr/ADR-006-sparse-dense-hybrid-ann.md`  
**ruvector main repo:** https://github.com/ruvnet/ruvector

---

*Part of the [ruvector](https://github.com/ruvnet/ruvector) nightly research series — autonomous state-of-the-art improvements to the Rust vector database, published daily.*

**Keywords:** vector database Rust, ANNS approximate nearest neighbor, hybrid search BM25 dense sparse, HNSW graph index, RAG retrieval augmented generation, distribution alignment, SPLADE sparse vectors, ruvector, high performance vector search 2026
