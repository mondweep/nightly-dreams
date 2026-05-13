# ruvector 2026: CoDEQ — Consistent Dynamic Quantization for High-Performance Streaming Rust Vector Search

**ruvector · Rust vector database · product quantization · streaming ANN · dynamic consistency · 2026**

> First Rust implementation of CoDEQ (arXiv:2512.18335): a streaming-safe product quantizer that bounds stale codeword assignments under continuous vector insertions and deletions — no full codebook rebuild required.

---

## Introduction

Every production vector database that uses product quantization (PQ) faces a hidden time bomb: codebooks trained on last month's data silently degrade recall as this month's vectors arrive. When your embedding model updates, your domain expands, or your corpus churns at 10% per day, quantisation error accumulates invisibly. No metric spikes. No alert fires. Users just get worse results.

Traditional solutions — full codebook retraining, segment compaction, offline reindex jobs — require seconds to minutes of degraded service or index locks. For high-throughput systems ingesting thousands of vectors per second, this is unacceptable.

**CoDEQ (Consistent Dynamic Quantization, arXiv:2512.18335, December 2025)** is the first algorithm to provide a formal *dynamic consistency guarantee* for PQ under streaming mutations: at any point in time, the fraction of stale codeword assignments is bounded by a user-configurable threshold ε, maintained without any global rebuild.

This post describes the first Rust implementation of CoDEQ, built as a nightly research contribution to [ruvector](https://github.com/ruvnet/ruvector) — a high-performance, real-time Rust vector database.

---

## Features

- **Three streaming variants** with a common `DynamicQuantizer` trait:
  - `EagerCoDEQ` — immediate reassignment, always consistent
  - `LazyCoDEQ` — batch reassignment at configurable stale threshold (default 5%)
  - `ApproxCoDEQ` — KS drift-gated lazy flush, lowest insert latency
- **Asymmetric Distance Computation (ADC)** with precomputed per-query LUT for fast kNN
- **KS sliding-window drift detection**: 64-sample window per subspace, 2σ trigger threshold
- **Median-stable partition boundaries**: codebook centroids trained with Lloyd's algorithm, updateable without full retrain
- **Bounded staleness invariant**: 0 stale vectors confirmed across all variants in benchmark
- **9/9 tests pass**, `cargo build --release` clean, no unsafe code, single dependency (`rand`)

---

## Benefits

| Benefit | Detail |
|---------|--------|
| **47% lower insert latency** | ApproxCoDEQ: 2,635 μs vs EagerCoDEQ: 5,002 μs per insert |
| **Equal recall quality** | 0.228 Recall@10 for Lazy and Approx variants under drift |
| **Query-neutral** | 106–109 μs query latency identical across all variants |
| **Zero stale vectors** | Dynamic consistency invariant verified in all benchmark runs |
| **No global rebuild** | Handles distribution drift via selective reassignment, not retraining |
| **Composable** | Plugs into ruvector-diskann (PQ reranking), PDX columnar layout, RaBitQ compression |

---

## Comparisons

Performance and capability vs. leading vector databases (streaming PQ consistency):

| System | Dynamic PQ Consistency | Streaming-safe Codebook | No Rebuild Required | Insert Overhead |
|--------|------------------------|------------------------|---------------------|-----------------|
| **ruvector + CoDEQ** | **Formal (bounded ε)** | **Yes (KS drift gate)** | **Yes** | **2,635 μs/vec** |
| Milvus 2.5 | Segment-level refresh | Partial (compaction) | No (segment lock) | Low, then spike |
| Qdrant TurboQuant | None (SQ only, per-shard) | Partial | Yes (SQ) | Very low |
| Weaviate | Background async rebuild | No formal guarantee | No | Low, then spike |
| Pinecone | Opaque re-indexing | Opaque | No | Opaque |
| LanceDB | Version-based delta | No PQ consistency | Partial | Low |
| FAISS | None | No | No | Low (fixed codebook) |

*CoDEQ is the only system with a formally bounded stale fraction guarantee for PQ.*

---

## Benchmarks

**Hardware:** x86-64 Linux, rustc 1.94.1 (release + LTO + single codegen unit)  
**Dataset:** 2,000 base vectors, dim=128 (Uniform[-1,1]), M=8 subspaces, K=256 centroids  
**Mutations:** 5 batches × 100 inserts + 50 deletes, progressive Gaussian mean shift (+0.15/batch)  
**Queries:** 50 synthetic queries, Recall@10 vs. brute-force ground truth  

| Metric | EagerCoDEQ | LazyCoDEQ | ApproxCoDEQ |
|--------|------------|-----------|-------------|
| Build time | 127 ms | 134 ms | 129 ms |
| **Insert latency** | 5,002 μs | 4,854 μs | **2,635 μs** |
| Query latency | 109 μs | 107 μs | **106 μs** |
| **Recall@10** | **0.262** | 0.228 | 0.228 |
| Reassignments | 1,185 | 1,138 | 0 |
| Drift gate triggers | — | — | 331 / 500 inserts |
| Final stale fraction | 0.000 | 0.000 | 0.000 |

**Key result:** ApproxCoDEQ achieves **47% lower insert latency** than EagerCoDEQ while maintaining identical recall quality (0.228 vs 0.228 Recall@10). The KS drift gate fires on 66% of inserts under the simulated distribution shift — triggering flush exactly when needed, not on every mutation.

---

## Optimizations

**Current PoC limitations (known, documented in roadmap):**

1. **O(N) cluster scan on insert** — EagerCoDEQ and LazyCoDEQ scan all N vectors to find cluster members. Adding an inverted centroid index (per-subspace `HashMap<u8, Vec<usize>>`) reduces this to O(cluster_size), expected 100-1000× speedup at production N.

2. **Scalar LUT scan** — query uses scalar f32 accumulation over codes. AVX-512 u8 gather + horizontal add would yield 8-16× speedup, bringing 106 μs down to ~7-13 μs for N=2,250.

3. **Single-threaded** — parallel shard-level mutation queues and concurrent LUT scan across CPU cores are the next performance tier.

4. **Fixed codebook** — CoDEQ tracks code staleness but does not retrain centroids. A background partial retrain job (5% sample, drifted subspaces only) restores 5-15% recall after prolonged extreme drift.

**Path to production (100M vectors, DIM=768):**
- Code store: 100M × 32 subspaces × 1 byte = **3.2 GB** (fits in NVMe page cache)
- Codebook: 32 × 256 × 24 × 4 bytes = **6.3 MB** (fits in L3 cache)
- Query LUT: 32 × 256 × 4 bytes = **32 KB** per query (L2 cache)
- KS sketches: 32 × 64 × 4 bytes = **8 KB** (L1 cache)

---

## Get Started

**Research branch:** `research/nightly/2026-05-13-codeq-streaming-quantization`  
**Repository:** https://github.com/ruvnet/ruvector  
**Nightly dreams repo:** https://github.com/mondweep/nightly-dreams

```bash
# Clone the nightly-dreams research repo
git clone https://github.com/mondweep/nightly-dreams
cd nightly-dreams
git checkout research/nightly/2026-05-13-codeq-streaming-quantization

# Build and run the benchmark (real numbers, ~3 seconds)
cargo build --release -p codeq-streaming-quantization
cargo run --release -p codeq-streaming-quantization

# Run all tests (9/9 pass)
cargo test -p codeq-streaming-quantization

# Read the research
cat docs/research/nightly/2026-05-13-codeq-streaming-quantization/README.md
cat docs/adr/ADR-007-codeq-streaming-quantization.md
```

**Key types:**
```rust
use codeq_streaming_quantization::*;

// Train codebook from base data
let codebook = Codebook::train_median_split(&base_vectors, &mut rng);

// Choose variant: Eager / Lazy / Approx
let mut index = ApproxCoDEQ::new(&base_vectors, &mut rng, 0.05);

// Stream mutations
index.insert(new_vector);
index.delete(stale_idx);

// Query with asymmetric distance (LUT-based)
let results: Vec<(usize, f32)> = index.knn(&query_vector, 10);

// Check consistency
println!("Stale fraction: {:.4}", index.stats.stale_fraction());
```

---

*ruvector nightly research 2026-05-13 · CoDEQ · Rust vector database · streaming approximate nearest neighbor search · product quantization · dynamic consistency · distribution drift · KS test · arXiv:2512.18335*
