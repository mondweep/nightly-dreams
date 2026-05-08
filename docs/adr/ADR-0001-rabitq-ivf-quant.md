# ADR-0001: RaBitQ 1-Bit Quantization with IVF Partitioning

**Status:** Proposed  
**Date:** 2026-05-08  
**Deciders:** Nightly Research Agent  
**Supersedes:** —  
**Related:** docs/research/nightly/2026-05-08-rabitq-ivf-quant/README.md

---

## Context

ruvector already offers HNSW, DiskANN/Vamana, OPQ, and scalar (SQ8) quantization. However,
it currently lacks a **theoretically-grounded 1-bit quantization** path — one where the
memory/recall trade-off is characterised by a proven error bound rather than empirical
tuning alone.

Three trends motivate addressing this now:

1. **Scale pressure**: embedding corpora are growing to 100 M–1 B vectors. At 512 B/vector
   (128-d f32), that is 500 GB of RAM. A 20× compression factor brings this to 25 GB —
   the sweet spot for a single high-memory server node.

2. **Theoretical guarantees**: users in regulated industries (finance, healthcare) need
   computable recall bounds, not just "usually works well" empirics. RaBitQ (SIGMOD 2024)
   provides a sharp, closed-form error bound on inner-product estimation from 1-bit codes.

3. **Competitor parity**: LanceDB shipped RaBitQ in 2025. Qdrant, Milvus, and
   ElasticSearch all expose binary quantization with varying levels of theory. ruvector
   should offer at least one fully-grounded alternative.

---

## Decision

**Add `ruvector-quant` as a first-class crate** implementing RaBitQ 1-bit quantization
combined with an IVF partition index.

Specifically, this ADR approves:

1. **`crates/rabitq-ivf-quant`** (this PoC) as the research artefact demonstrating
   feasibility.

2. A future promotion path to **`crates/ruvector-quant`** with:
   - Full Walsh-Hadamard rotation (replacing the PoC's sign-flip approximation)
   - SIMD popcount (AVX2 + AVX-512 backends)
   - Trait-based codec generics (`Codec<const BITS: u8>`) allowing 1, 2, 4-bit variants
   - `IvfIndex<C: Codec>` wrapper usable with any codec

3. **Integration point**: expose as a `QuantBackend` trait so ruvector-core's HNSW can
   optionally use bit-distance traversal on its graph edges before exact f32 rescore.

The PoC shows (on 10K × 128-d synthetic corpus, single-threaded):

| Variant | QPS | Memory | Recall@10 |
|---|---|---|---|
| Exact f32 (baseline) | 541 | 5 000 KB | 1.000 |
| RaBitQ 1-bit raw | 2 250 (+4.2×) | 273 KB (−18×) | 0.221 |
| RaBitQ + rescore-64 | 1 796 (+3.3×) | 5 273 KB | 0.556 |
| IVF-32 + RaBitQ | 9 322 (+17×) | 289 KB (−17×) | 0.338 |

The recall numbers are honest and below production targets. They are expected to improve
significantly (to >0.85) once the Walsh-Hadamard rotation replaces the sign-flip
approximation, per the theoretical analysis in the reference paper.

---

## Consequences

### Positive

- **Memory reduction**: 17–18× fewer bytes for the quantized index, enabling larger corpora
  in the same RAM footprint.
- **Throughput**: popcount-based distance estimation is 4–17× faster than f32 dot-product
  for the Hamming scan phase.
- **Theoretical grounding**: the RaBitQ error bound gives users and auditors a closed-form
  expression for worst-case recall degradation at a given bit budget.
- **Composability**: the `VectorIndex` trait defined in this PoC is identical in signature
  to ruvector-core's existing indexing abstractions, making integration straightforward.

### Negative / Risks

- **Recall with sign-flip only**: the current PoC uses a simplified rotation (random sign
  flip, not full WHT). On correlated real-world embeddings, recall may be lower than the
  0.556 seen on synthetic data.
- **Build cost of IVF**: k-means (15 iterations, 32 clusters, 10K vectors) takes 720 ms
  on a single thread. At 100 M vectors this becomes ~2 hours. Mitigation: mini-batch
  k-means or GPU clustering (cuML k-means).
- **rescore_k sensitivity**: recall scales with `rescore_k / corpus_size`. Users must tune
  this parameter or accept variable recall guarantees.
- **No incremental update**: the IVF index requires a full rebuild when the corpus changes.
  Addressed in the roadmap by LSM-VEC-style delta layers (arXiv 2505.17152).

---

## Alternatives Considered

### A. Scalar Quantization (SQ8) only

ruvector already has SQ8. Adding it would be redundant; the 4× compression factor does
not address the billion-scale memory pressure that motivates this ADR.

### B. Product Quantization (PQ)

PQ offers ~8× compression with codebook lookup and asymmetric distance computation. Recall
is higher than 1-bit RaBitQ at equivalent compression, but:
- No closed-form error bound
- Slower to compute (table lookup vs. popcount)
- Already considered for future work in existing ruvector roadmap

### C. MSTG (Multi-Scale Tree Graph) from rabitq-rs

rabitq-rs v0.9.0 ships MSTG — a hierarchical clustering + HNSW hybrid that outperforms
plain IVF on recall for equivalent compression. It is more complex to implement from
scratch and would exceed the line-count and time budget for a nightly research artefact.
Recommended as follow-on work once the RaBitQ codec itself is proven in production.

### D. DiskANN (already in ruvector)

DiskANN solves the billion-scale problem via SSD-backed graph traversal. RaBitQ is
complementary: it provides an in-RAM compression option for datasets that fit in memory
once quantized, without the SSD I/O latency of DiskANN.

---

## Implementation Path

1. **This PoC** (merged from `research/nightly/2026-05-08-rabitq-ivf-quant`)
2. **v1** (sprint +2 weeks): Replace sign-flip with WHT, add AVX2 popcount, 
   target >0.85 recall@10 at 18× compression on ANN benchmark datasets.
3. **v2** (sprint +4 weeks): Multi-bit codec (2, 4 bits), `IvfIndex<C: Codec>` generic.
4. **v3** (sprint +8 weeks): Integration into ruvector-core as optional HNSW quantization
   backend; LSM-VEC delta layer for incremental updates.
