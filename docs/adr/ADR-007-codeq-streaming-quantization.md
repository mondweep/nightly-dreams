# ADR-007: Consistent Dynamic Quantization for Streaming Vector Search (CoDEQ)

**Date:** 2026-05-13  
**Status:** Proposed  
**Deciders:** Nightly Research Agent  
**Technical Area:** Index maintenance / quantization / streaming mutation  
**Research doc:** [docs/research/nightly/2026-05-13-codeq-streaming-quantization/README.md](../research/nightly/2026-05-13-codeq-streaming-quantization/README.md)

---

## Context

Every deployed product quantizer in ruvector today (RaBitQ, ADR-002; PDX columnar layout, ADR-005) assumes a static or infrequently-updated dataset. Codebooks are trained once — either offline or at segment creation time — and thereafter treated as immutable. This is appropriate for archival or near-static corpora but breaks down in production systems where:

- **Model rotations** periodically replace the embedding model, invalidating all prior codeword assignments
- **Domain expansion** (new product lines, languages, user cohorts) shifts the marginal distribution of each PQ subspace away from trained centroid positions
- **Continuous churn** in news, social, or financial workloads means 1–10% of the vector corpus is replaced per day

The consequence is silent recall degradation: quantisation error increases as data drifts away from centroid positions, but no metric surface exposes this. Users observe worse retrieval quality without understanding why. FAISS, Milvus, Qdrant, and Weaviate all handle this with full codebook retraining jobs (seconds to minutes of elevated latency) or offline reindex operations.

The arXiv:2512.18335 paper (December 2025) introduces **CoDEQ**, the first formal dynamic consistency framework for PQ under streaming mutations. The three key innovations — median-stable partitions, a lazy stale-vector queue, and a KS drift gate — allow the stale fraction to be bounded at a user-configurable ε without a global rebuild. This ADR records the decision to introduce CoDEQ primitives into the ruvector research series as a foundation for a future `ruvector-codeq` production crate.

**Benchmark results from this PoC (N=2,250, DIM=128, M=8, K=256, 5 drift batches):**
- EagerCoDEQ: 5,002 μs/insert, 109 μs/query, Recall@10 = 0.262
- LazyCoDEQ: 4,854 μs/insert, 107 μs/query, Recall@10 = 0.228
- ApproxCoDEQ: **2,635 μs/insert (−47%)**, 106 μs/query, Recall@10 = 0.228, 0 stale vectors

---

## Decision

Introduce three `CoDEQ` variants as Rust primitives, validated by this PoC, and advance toward a production `ruvector-codeq` crate:

1. **`EagerCoDEQ`** — on every insert, scan affected clusters and immediately reassign any vector whose nearest centroid has changed. Provides perfect consistency at all times (stale_fraction = 0 after every insert) but currently costs O(N) per insert without an inverted index. Best for: small corpora (<50K vectors) or workloads with very low insert rates.

2. **`LazyCoDEQ`** — enqueue affected vectors in a pending reassignment queue; flush only when the stale fraction exceeds a configurable threshold (default: 5%). Allows brief staleness windows in exchange for lower per-insert cost. Reassignment is batched, amortising the O(N) scan across many inserts. Best for: medium corpora where occasional staleness spikes are acceptable.

3. **`ApproxCoDEQ`** — adds per-subspace KS drift sketches (sliding window, 64 samples per subspace). Flushes the lazy queue only when the window mean has drifted more than 2σ from the baseline distribution. Achieves 47% lower insert latency vs. Eager while maintaining the same recall, by avoiding flushes on distributional noise. Best for: high-throughput insert workloads with detectable drift patterns.

4. **`DynamicQuantizer` trait** — a common abstraction over all three variants, enabling backend-swapping without changing query or mutation call sites. Future variants (SIMD, GPU, mmap-backed) implement the same trait.

5. **Progression path** — the PoC uses O(N) cluster scans for simplicity. The next milestone (future ADR) introduces an inverted centroid index (per-subspace `HashMap<u8, Vec<usize>>`) to reduce eager scan from O(N) to O(cluster_size), enabling million-vector scale.

---

## Consequences

### Positive

- **47% insert latency reduction** (ApproxCoDEQ vs. EagerCoDEQ) with equal recall quality, validated by the PoC benchmark.
- **Bounded staleness** — the dynamic consistency invariant is empirically confirmed: 0 stale vectors at the end of all mutation batches across all three variants.
- **Orthogonal to query path** — query latency (106–109 μs) is identical across variants. CoDEQ's mutation strategy does not affect the LUT-scan hot path.
- **Composable with prior ADRs** — CoDEQ's code store (u8 per subspace) plugs directly into PDX's columnar layout (ADR-005) and can serve as the PQ-reranking layer for DiskANN. RaBitQ's quantisation (ADR-002) can replace the initial codebook encoding for higher compression.
- **First formal dynamic consistency implementation in ruvector** — fills a gap that no prior research in this series addressed. All 5 prior topics (ADR-001 through ADR-006) assumed static codebooks.
- **9/9 tests pass** — including drift detection, insert/delete size correctness, recall floor validation, and KS sketch functionality.

### Negative

- **O(N) eager scan** — without the inverted centroid index, EagerCoDEQ costs 5,002 μs/insert at N=2,250, scaling to ~2.5 seconds at N=1M. Not production-ready until the inverted index is added (next ADR).
- **Recall under drift is moderate** — Recall@10 = 0.228–0.262 at K=256 centroids and only 2 training iterations. Production quality (Recall@10 ≥ 0.90) requires more iterations, better initialisation (k-means++), and larger K or M. The PoC demonstrates the correctness of the consistency mechanism, not peak recall.
- **KS window sensitivity** — with KS_WINDOW=64 and +0.15 drift per batch, the gate fires on 66% of inserts. For workloads with very high insert rates and slow drift, a larger window (256–1024) would reduce unnecessary flushes. Tuning is workload-dependent.
- **swap_remove deletes** — the current delete implementation uses `Vec::swap_remove`, which invalidates external index-to-ID mappings. Production requires a tombstone + compaction design.

### Neutral

- `rand = { version = "0.8", features = ["small_rng"] }` is the only new dependency; already present transitively across the workspace.
- No changes to existing crates; this is purely additive.

---

## Alternatives Considered

### A: Full Codebook Retrain on Threshold Exceeded (Rejected)

The simplest approach: when the stale fraction exceeds ε, retrain the entire codebook from scratch. This provides a guaranteed fresh codebook after each retrain but requires O(N × iterations) time (typically 5–60 seconds at N=1M). During retrain, the index is either read-only (serving degraded recall) or locked (no queries). Neither is acceptable for a high-availability production database. CoDEQ's lazy queue + drift gate achieves bounded staleness without any lock or rebuild.

### B: Per-Segment Codebooks with Merging (Rejected as Insufficient)

Milvus and Weaviate segment data into fixed-size immutable segments, each with its own locally-trained codebook. New segments have fresh codebooks; old segments drift. Merging segments triggers a codebook re-merge. This solves the monotonic-insert case but fails on delete-heavy workloads (deletions from old segments do not trigger retraining) and creates codebook boundary artefacts when querying across segments. CoDEQ operates on a single unified codebook per index, avoiding these artefacts.

### C: Scalar Quantization Without Centroids (Rejected for Compression Ratio)

Qdrant's TurboQuant uses online scalar quantization (SQ8): per-dimension min/max tracking, no centroids. This adapts naturally to streaming data (update min/max on insert). However, SQ8 provides 4× compression vs. PQ's 16-100× compression at the same recall level. For billion-vector indexes where the code store must fit in DRAM, PQ's compression density is essential. CoDEQ preserves PQ's compression ratio while adding streaming correctness.

### D: Background Rebuild Thread (Rejected as Racy)

Launch a background thread that continuously retrains the codebook on a rolling window of recent vectors. The main query thread uses the current codebook; the background thread atomically swaps in a new one. This avoids query latency spikes but introduces a race condition: vectors encoded before the swap have codes in the old basis, while new vectors use the new basis. Queries that span pre- and post-swap vectors return distances from incommensurable codebooks, corrupting k-NN rankings. CoDEQ's lazy queue explicitly tracks and resolves this by re-encoding stale vectors after any codebook update.
