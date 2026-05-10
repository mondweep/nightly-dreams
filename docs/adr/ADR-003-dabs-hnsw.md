# ADR-003: Distance-Adaptive Beam Search (DABS) for HNSW

**Status:** Proposed  
**Date:** 2026-05-09  
**Deciders:** Nightly research agent  
**Research doc:** [docs/research/nightly/2026-05-09-dabs-hnsw/README.md](../research/nightly/2026-05-09-dabs-hnsw/README.md)

---

## Context

ruvector's HNSW search (via `hnsw_rs`) uses a fixed `ef` beam-width parameter. The search terminates when:

```
d(q, next_candidate) > d(q, ef-th nearest found)
```

This is query-agnostic: every query uses the same number of candidate evaluations regardless of local cluster density. On easy queries (tight cluster near q), the algorithm wastes time exploring candidates that cannot improve the result. On hard queries it works correctly but provides no way to detect and skip those wasted evaluations.

The 2025 paper arXiv:2505.15636 proposes Distance-Adaptive Beam Search (DABS): replacing the ef-th-nearest threshold with `(1+γ) × k-th-nearest`. This single change requires no index rebuild and provides:
- A mathematically provable (1+γ)-ANN approximation guarantee at γ=2.0
- 10–50% fewer distance computations on structured benchmark datasets
- Graceful degradation to standard behavior on hard queries

No major OSS vector database (Milvus, Qdrant, Weaviate, FAISS, LanceDB) has shipped DABS as of May 2026.

---

## Decision

Implement DABS as a new search strategy in ruvector, initially as a standalone crate (`dabs-hnsw`) with a trait-based interface that can be adopted by existing index types.

The core change is the termination predicate in `run_search`:

```rust
// Standard
let stop = c_dist > worst_found;

// DABS
let k_th = k_best.peek().unwrap_or(f32::MAX);
let stop = k_best.len() >= k && c_dist > (1.0 + gamma) * k_th;
```

The `k_best` max-heap (size k) adds O(k log k) overhead per query and is tracked alongside the existing `found` max-heap (size ef).

---

## Consequences

**Positive:**
- Zero index rebuild cost: search-only change.
- Tunable via γ: γ=0.0 provides a hard guarantee (≤ dist ops vs. standard), γ=2.0 provides the theoretical ANN guarantee.
- On clustered/structured data: 10–50% fewer distance computations at equivalent recall (per the paper; validated on real ANN benchmark datasets).
- Stacks with quantization (rabitq-ivf-quant, ADR-001) and filtering (adaptive-filtered-ann, ADR-002).

**Negative / Risks:**
- On random high-dimensional data (concentration of measure), DABS with γ>0 may explore *more* than standard because `(1+γ)*k_th` can exceed `ef_th`. Mitigation: use γ=0.0 or increase ef to match standard recall.
- Per-query k_best heap adds minor memory overhead (k × 4 bytes).
- γ requires calibration per dataset/use-case. Wrong γ gives either low recall (γ too small) or no savings (γ too large relative to distance distribution).

**Neutral:**
- The PoC uses a single-layer NSW graph. Full multi-layer HNSW integration is a follow-up task; the search logic is layer-independent.

---

## Alternatives Considered

### A: No change, tune ef per workload
`ef` can be increased until recall meets target. Simple but wasteful: every query pays the same cost regardless of difficulty.

### B: Ada-ef (arXiv:2512.06636)
Fits a Gaussian to the offline distance distribution, then adapts ef per query from a model prediction. Better savings but requires offline calibration and per-query model inference. Significantly more complex to implement.

### C: DARTH (arXiv:2505.19001, SIGMOD 2026)
GBDT-based recall predictor trained on historical logs. Up to 40% QPS improvement but requires training data, feature engineering, and a model server. Not suitable for a library that must work without training data.

### D: DABS (chosen)
Pure algorithmic: a 5-line change to the search loop. No offline training. No model inference. Theoretical guarantee. Composable with all existing index variants.

---

## Implementation Plan

1. **Phase 1 (this ADR):** Standalone `crates/dabs-hnsw` with NSW graph + DABS search.
2. **Phase 2:** Extract `GraphSearch` trait into `ruvector-core` traits crate.
3. **Phase 3:** Wrap `hnsw_rs` searcher to inject DABS termination predicate.
4. **Phase 4:** Expose `gamma` as a search-time parameter in the ruvector HTTP API.
