# ADR-008: Late-Interaction MaxSim Multi-Vector ANN Index

**Status:** Accepted  
**Date:** 2026-05-14  
**Branch:** research/nightly/2026-05-14-late-interaction-maxsim

---

## Context

Modern neural retrieval systems — ColBERT, ColPALI, ColQwen, and their multimodal
descendants — represent documents and queries as *sets* of token-level vectors rather
than a single embedding.  The relevance score is the Chamfer / MaxSim distance:

```
score(Q, D) = Σ_{q ∈ Q}  max_{d ∈ D}  cosine(q, d)
```

Single-vector ANN indexes (HNSW, IVF-PQ) are blind to this structure: they require
either pre-aggregating token vectors (losing expressivity) or expensive per-token
ANN lookups followed by union scoring.

The GEM paper (arXiv 2603.20336, March 2026) demonstrates that native multi-vector
graph indexes can achieve up to 16× speedup over PLAID at the same recall on MS MARCO
(597 M token vectors).  The MUVERA approach (2024) showed that Fixed-Dimensional
Encoding (FDE) reduces multi-vector retrieval to single-vector ANN, reaching 0.54 ms
query latency with ColBERT-style encoders.

A dedicated ECIR 2026 workshop ("Late Interaction and Multi-Vector Retrieval") confirms
this is the dominant frontier of neural IR research.

ruvector lacks any first-class support for multi-vector / late-interaction retrieval.
This ADR records the decision to add it as a PoC crate.

---

## Decision

Implement `crates/late-interaction-maxsim` with three measured variants:

| Variant | Algorithm | Source inspiration |
|---|---|---|
| `linear` | Brute-force O(N·K·Q·D) MaxSim | baseline / ground truth |
| `centroid` | k-means inverted list + MaxSim re-rank | PLAID (Santhanam et al. 2022) |
| `fde` | Random projection mean-pooling + MaxSim re-rank | MUVERA (Faggella et al. 2024) |

All variants expose a common `MaxSimIndex` trait so backends are swappable without
caller changes.

---

## Consequences

### Positive
- ruvector gains a first-class multi-vector retrieval primitive.
- The `MaxSimIndex` trait enables future graph-native (GEM-style) backends.
- Centroid and FDE variants demonstrate the speed-recall trade-off quantitatively.
- All three compile with `cargo build --release` and pass `cargo test`.

### Negative / Risks
- FDE recall is low on uniform synthetic data (~25 % @10).  On real corpora with
  semantic clustering, recall is substantially higher (MUVERA reports >0.95 @100).
- k-means in `centroid` is single-threaded Lloyd's; production use warrants a
  parallel implementation or `faer`/`ndarray` integration.
- The centroid index becomes slower than linear below ~10 k docs; break-even depends
  on `n_probe` and token distribution.

### Neutral
- No unsafe code; no SIMD intrinsics (SIMD MaxSim is left for a follow-up ADR).
- The FDE projection matrix is deterministically seeded, so results are reproducible.

---

## Alternatives Considered

| Alternative | Why rejected |
|---|---|
| Per-token HNSW + union scoring | Requires an HNSW crate dependency; adds 1000+ LoC; high build complexity for a PoC |
| Graph-native GEM approach | Full GEM requires two-stage clustering + bridge construction; estimated 2000+ LoC — out of scope for a nightly slot |
| Residual quantisation (ColBERTv2 style) | Quantisation on multi-vectors is covered by prior ADRs (ADR-001 RaBitQ, ADR-007 CoDEQ); avoid duplication |
| Binary sketching (MinHash MaxSim) | Loses cosine semantics; only appropriate for set-Jaccard, not inner-product MaxSim |

---

## References

- GEM: arXiv 2603.20336 (2026)
- PLAID: Santhanam et al., EMNLP 2022
- MUVERA: Faggella et al., NeurIPS 2024 (MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings)
- ColBERTv2: Santhanam et al., NAACL 2022
- LIR Workshop: arXiv 2511.00444 (ECIR 2026)
