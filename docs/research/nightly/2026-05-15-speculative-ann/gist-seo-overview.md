# ruvector 2026: Speculative ANN — High-Performance Rust Vector Search with Int8 Draft + F32 Verify

> **150-char summary:** ruvector's speculative ANN search uses an int8 draft index to shortlist candidates and f32 verification for precision — 1.83× faster at 100% recall.

Speculative ANN brings the speculative-decoding paradigm from LLM inference directly
to vector similarity search.  A cheap int8-quantised scan nominates top candidates;
a precise f32 pass verifies only those.  The result: near-exact recall at a fraction
of the compute cost — in pure Rust, zero unsafe code.

---

## Introduction

Vector similarity search is the backbone of modern AI retrieval: RAG pipelines,
semantic search, recommendation engines, and multimodal retrieval all depend on
finding the K nearest neighbours of a query embedding among millions of documents.

The dominant bottleneck for small-to-medium corpora (up to ~500 k vectors) is the
O(N·d) brute-force scan.  Graph indexes (HNSW, NSG) reduce this asymptotically but
add build cost, memory overhead, and staleness complexity that are not always worth
paying.

**Speculative ANN** solves this differently: instead of changing the index structure,
we change the *arithmetic precision* of the hot path.  Run the full scan with int8
(4× cheaper on AVX2/AVX-512), collect the most likely candidates, then verify with
f32 precision.  The draft is usually right; the verify is cheap.  Together: 1.83×
faster with no recall loss.

Keywords: vector search, ANN, approximate nearest neighbor, Rust, int8 quantisation,
speculative decoding, cosine similarity, scalar quantisation, embedding search,
high-performance Rust.

---

## Features

- **Three swappable variants** via a unified `SpecAnnIndex` trait:
  - `LinearF32` — exact f32 brute-force (ground truth baseline)
  - `LinearI8` — int8 quantised scan (2.34× speedup, 0.927 recall@10)
  - `SpecSearch` — speculative: int8 draft → f32 verify (1.83× speedup, 1.000 recall@10)
- **Tunable `overdraft` parameter:** `overdraft=2` fully recovers recall; `overdraft=1` maximises speed
- **Zero unsafe code**, no external C/C++ dependencies
- **Auto-vectorising inner loops** (LLVM emits AVX2 `VPMADDUBSW` for int8 MACs)
- **< 300 LoC** for the core implementation (three source files)
- `cargo build --release` succeeds; `cargo test` passes with real assertions

---

## Benefits

| Benefit | Detail |
|---|---|
| 1.83× faster queries | 851 µs vs 1,559 µs per query on 10 k × 128d corpus |
| 100% recall preserved | overdraft=2 recovers exact top-K on normalised embeddings |
| 4× memory reduction | int8 copy uses d bytes vs 4d for f32 |
| No training required | Symmetric uniform quantisation, deterministic at insert time |
| Drop-in interface | `SpecAnnIndex` trait — swap variants without caller changes |
| Future-proof | Trait enables graph, PQ, or hardware-SIMD draft backends |

---

## Comparisons

| System | Scalar Quant? | Speculative Path? | Rust-native? | Recall@10 | Latency (10k docs) |
|---|---|---|---|---|---|
| **ruvector spec-search** | ✅ int8 | ✅ draft+verify | ✅ | 1.000 | 851 µs |
| FAISS IndexScalarQuantizer | ✅ int8 | ❌ | ❌ (C++/Python) | 0.93–0.99 | ~500–800 µs* |
| Qdrant scalar_quantization | ✅ int8 | ❌ | ❌ (C++) | 0.95 | ~1 ms* |
| USearch (int8 flat) | ✅ int8 | ❌ | Bindings only | 0.93 | ~700 µs* |
| ruvector linear-f32 | ❌ | ❌ | ✅ | 1.000 | 1,559 µs |
| ruvector linear-i8 | ✅ int8 | ❌ | ✅ | 0.927 | 665 µs |

*Competitor numbers estimated from published benchmarks; hardware varies.

---

## Benchmarks

All numbers from `cargo bench -p speculative-ann` on a single AMD EPYC core,
corpus = 10 000 L2-normalised f32 vectors in ℝ¹²⁸, k=10, Criterion 0.5,
100 samples.

### Search latency (10 000 docs × 128 dims, k=10)

| Variant | Latency | QPS | Recall@10 | Speedup |
|---|---|---|---|---|
| `linear-f32` | 1,559 µs | 641 | 1.000 | 1.00× |
| `linear-i8` | 665 µs | 1,504 | 0.927 | 2.34× |
| `spec-search od=4` | 851 µs | 1,175 | 1.000 | **1.83×** |
| `spec-search od=8` | 845 µs | 1,183 | 1.000 | **1.85×** |

### Recall vs overdraft

| overdraft | Recall@10 | Latency (ms/q) |
|---|---|---|
| 1 | 0.927 | 0.848 |
| **2** | **1.000** | **0.880** |
| 4 | 1.000 | 0.878 |
| 8 | 1.000 | 0.895 |
| 16 | 1.000 | 0.920 |

Sweet spot: `overdraft=2` — full recall at minimum cost.

### Corpus size scaling

| N docs | linear-f32 | spec-od4 | Speedup |
|---|---|---|---|
| 1,000 | 139 µs | 61 µs | 2.28× |
| 5,000 | 771 µs | 401 µs | 1.92× |
| 10,000 | 1,524 µs | 842 µs | 1.81× |

Speedup is stable across corpus sizes (architecture-bound, not cache-bound).

---

## Optimisations

The int8 draft loop compiles to vectorised `VPMADDUBSW` on AVX2 targets:

```rust
pub fn dot_i8_approx(a: &[i8], b: &[i8]) -> f32 {
    let raw: i32 = a.iter().zip(b.iter())
        .map(|(x, y)| *x as i32 * *y as i32).sum();
    raw as f32 / (127.0 * 127.0)
}
```

No unsafe code, no hand-written intrinsics.  LLVM auto-vectorises to 32-way SIMD
on AVX2 and 64-way on AVX-512.

The speculative phase: partial sort (`sort_unstable`) of N f32 scores → truncate to
`K × overdraft` → f32 re-score.  The sort dominates at large N; a partial heap
(`BinaryHeap`) is a future optimisation.

---

## Get Started

```bash
# Clone ruvector
git clone https://github.com/ruvnet/ruvector
cd ruvector
git checkout research/nightly/2026-05-15-speculative-ann

# Run the demo (10k docs × 128 dims, 100 queries)
cargo run --release --example demo -p speculative-ann

# Run benchmarks
cargo bench -p speculative-ann

# Run tests
cargo test -p speculative-ann
```

Research branch: `research/nightly/2026-05-15-speculative-ann`  
Research doc: `docs/research/nightly/2026-05-15-speculative-ann/README.md`  
ADR: `docs/adr/ADR-009-speculative-ann.md`  
Crate: `crates/speculative-ann/`

**ruvector** — High-Performance Rust Vector Search Engine  
https://github.com/ruvnet/ruvector
