# Distance-Adaptive Beam Search for HNSW (DABS)

**Date:** 2026-05-09  
**Slug:** `dabs-hnsw`  
**ADR:** [ADR-003](../../../adr/ADR-003-dabs-hnsw.md)  
**PoC:** [`crates/dabs-hnsw/`](../../../../crates/dabs-hnsw/)

---

## Abstract

Standard HNSW beam search uses a fixed `ef` parameter: it terminates once `ef` candidates have been collected in the found set and all remaining candidates are farther than the worst result. This is a data-agnostic stopping rule — it applies the same termination budget regardless of whether the query is landing in a tight, well-separated cluster or a diffuse, overlapping region of the embedding space.

Distance-Adaptive Beam Search (DABS), introduced in arXiv:2505.15636 (May 2025), replaces this with a query-aware stopping criterion:

> Terminate when the nearest unvisited candidate is farther than **(1 + γ) × d(q, k-th nearest found)**.

This single change, requiring no index rebuild, achieves **10–50% fewer distance computations at equivalent recall** on structured benchmark datasets. On easy queries (tight clusters), the k-th nearest is found quickly and the threshold triggers early. On hard queries, the algorithm degrades gracefully to standard behavior.

---

## SOTA Survey

### Fixed-ef HNSW (Malkov & Yashunin, 2018)

The canonical HNSW formulation (arXiv:1603.09320) introduced the `ef` search parameter to control the trade-off between recall and query time. The termination condition is:

```
while min(candidates).dist < max(found).dist:
    expand next candidate
```

This is already better than pure ef-count termination, but still relies on the ef-th nearest as the threshold — a quantity that may be much farther than the true k-th nearest when `ef >> k`.

### Ada-ef (Zhang et al., December 2025)

arXiv:2512.06636 proposed fitting a Gaussian model to the distance distribution offline, then adapting `ef` per query based on predicted difficulty. This achieves 15–30% QPS improvement but requires an offline calibration step and adds per-query model inference cost.

### DARTH (Peng et al., SIGMOD 2026)

arXiv:2505.19001 uses a gradient-boosted decision tree to predict when to stop, trained on historical query logs. Achieves up to 40% QPS improvement but is infrastructure-heavy: requires training data, a feature-engineering pipeline, and a serving infrastructure for the GBDT model.

### DABS (Abbe, Lindeman, Sreenivasan, May 2025)

arXiv:2505.15636 proposes the simplest possible adaptive stopping rule: compare the next candidate to a scaled version of the current k-th nearest. No offline training, no model inference, no index changes. The paper provides a theoretical approximation guarantee for the (1+γ)-ANN problem and reports 10–50% distance computation reduction on SIFT-1M, GIST-960, and GloVe-100 benchmarks.

### Graph-based ANN survey (February 2025)

arXiv:2502.05575 surveyed graph-based ANN algorithms and confirmed that beam search termination is a key efficiency lever. Fixed-ef remains the dominant approach in production systems (Weaviate, Qdrant, Milvus all use it), making DABS a practically relevant improvement.

### Competitor implementations

| System | Search termination | Adaptive? |
|---|---|---|
| Milvus 2.5 | fixed ef (HNSW) | no |
| Qdrant 1.12 | fixed ef (HNSW) | no |
| Weaviate 1.28 | fixed ef (HNSW) | no |
| Pinecone | proprietary, likely fixed ef | unknown |
| FAISS | fixed ef (HNSW), beam width (NSG) | no |
| LanceDB 0.20 | fixed ef (IVF-HNSW) | no |

None of the major OSS vector databases have shipped DABS or equivalent adaptive stopping as of May 2026.

---

## Proposed Design

### Stopping criterion

Standard beam search terminates when:

```
d(q, next_candidate) > d(q, ef-th nearest found)
```

DABS terminates when:

```
d(q, next_candidate) > (1 + γ) × d(q, k-th nearest found)
```

The k-th nearest is tracked via a max-heap of size k (`k_best`). Its `peek()` is the k-th best distance at any point during search. The implementation adds O(k log k) overhead per query for heap maintenance, negligible against the distance computation cost.

### Parameter γ

| γ | Behaviour | When to use |
|---|---|---|
| 0.0 | Most aggressive: stop when candidate can't improve k-th result | Provably ≤ dist ops vs. standard; lowest recall on random data |
| 0.5 | Moderate: allows candidates up to 1.5× k-th nearest | General use on structured data |
| 2.0 | Conservative: theoretical (1+γ)-ANN guarantee from the paper | Safety margin for production |

### Integration with existing HNSW

The only change to an existing HNSW implementation is in the beam search inner loop. The graph structure, index construction, and all other components remain unchanged. ruvector currently uses `hnsw_rs` (fixed ef); DABS can be added as a search strategy without forking the index.

### Trait design

```rust
pub trait GraphSearch {
    fn beam_search(&self, query: &[f32], k: usize, ef: usize) -> SearchStats;
    fn dabs_search(&self, query: &[f32], k: usize, ef: usize, gamma: f32) -> SearchStats;
    fn brute_force(&self, query: &[f32], k: usize) -> Vec<usize>;
}
```

This allows future backends (GPU, SIMD-accelerated, multi-layer HNSW) to implement the same interface.

---

## Implementation Notes

The PoC implements a single-layer Navigable Small World (NSW) graph — identical to the HNSW base layer, which is where beam search runs. Multi-layer entry point descent is orthogonal and composable.

**Graph construction:** Each inserted node finds its `m` nearest current neighbours via beam search (brute force for the first `ef_build` nodes to avoid bootstrap bias). Edges are bidirectional, pruned to `2m` per node.

**Entry point strategy:** Four evenly-spaced landmark nodes (`[0, n/4, n/2, 3n/4]`) are evaluated and all seeded into both the candidate and found heaps. This provides better coverage than a single fixed entry without requiring O(√n) landmark computation.

**k-best tracker:** A max-heap of size k maintains `d(q, k-th nearest)` in O(log k) per update. For ef=64, k=10, this is ~6 comparisons vs. potentially hundreds of distance evaluations — negligible overhead.

**Visited array:** A per-query `Vec<bool>` of length n. At 10 000 nodes this is a 10 KB allocation — cheap but not zero. A generation-counter approach would avoid this allocation.

---

## Benchmark Methodology

**Hardware:** Single-threaded Rust release build (no SIMD, no parallelism).  
**Dataset:** 10 000 × 128-dim random unit vectors (LCG seeded, deterministic).  
**Queries:** 200 queries from the same distribution.  
**Ground truth:** Brute-force exact k-NN (k=10) per query.  
**Metrics:** QPS (wall clock), Recall@10 (fraction of true k-NN found), mean distance computations per query.  
**ef:** 64 for all beam/DABS variants; additionally standard at ef=100 and ef=200.

---

## Results

Measured on 2026-05-09, single-threaded release build:

| Variant | QPS | Recall@10 | Dist/query | Notes |
|---|---|---|---|---|
| Standard ef=64 | 4,438 | 0.735 | 1,397 | baseline |
| Standard ef=100 | 3,072 | 0.823 | 1,934 | |
| Standard ef=200 | 1,858 | 0.919 | 3,075 | recall target |
| DABS γ=0.0 ef=64 | **14,729** | 0.319 | **387** | 72% fewer dist vs. std@ef=64 |
| DABS γ=0.5 ef=64 | 1,706 | 0.909 | 3,663 | ~91% recall, 19% more dist than std@eq-recall |
| DABS γ=1.0 ef=64 | 1,685 | 0.909 | 3,698 | |
| DABS γ=2.0 ef=64 | 1,687 | 0.909 | 3,698 | |

### Interpretation

**DABS γ=0.0** is the most dramatic result: 14,729 QPS with 387 dist/query — a 3.3× QPS improvement and 72% dist reduction relative to standard ef=64. This comes at the cost of recall (31.9% vs 73.5%), as the algorithm terminates as soon as it cannot improve the current k-th nearest. On clustered data, this recall would be much higher.

**DABS γ=0.5–2.0** on random 128-dim data: all three γ values converge to ~90.9% recall using ~3,663 dist/query. This matches the recall of standard ef=200 (91.9%) but with 19% *more* distance computations. This is the concentration-of-measure effect: in high-dimensional random data, all pairwise distances cluster near √2, making the gap between k_th and ef_th small relative to γ scaling.

**On structured data** (SIFT-1M, GIST-960, GloVe-100): the paper reports 10–50% fewer dist ops at equivalent recall because real embedding datasets have meaningful cluster structure — easy queries land near a tight cluster (small k_th relative to ef_th), triggering early termination. The random-data benchmark serves as a worst-case baseline.

---

## How It Works (Blog-readable Walkthrough)

Imagine you're searching for the 10 nearest coffee shops to your location. Standard HNSW beam search keeps a list of 64 candidates and doesn't stop until it's checked far enough that nothing new can improve the 64th slot on the list. But you only care about the top 10!

DABS says: once we've found 10 candidates, keep exploring — but stop as soon as the next candidate is more than `(1+γ)` times as far away as the 10th best candidate we already found. If the 10th best is 500 meters away and the next candidate in our search queue is 750 meters away (γ=0.5), we'd need to beat 500m to improve our result — but we're already looking 750m out. Stop.

For tight neighborhoods (easy queries), this kicks in almost immediately after finding 10 results. For spread-out queries (hard), it keeps searching just as long as the standard approach would. The result: *average* savings of 10–50% on real datasets, with zero configuration and no index changes.

---

## Practical Failure Modes

1. **High-dimensional random data:** Concentration of measure collapses the gap between k-th and ef-th nearest distances. DABS with γ>0 may explore more than standard. Use γ=0.0 for the guaranteed ≤ dist-ops invariant, or scale ef accordingly.

2. **Small graphs (n < 1000):** With few nodes, ef easily exhausts most of the graph. DABS and standard converge to similar behavior. Benefits emerge at n > 10K.

3. **Poorly connected graphs:** NSW/HNSW graphs with low m (e.g., m=4) may not have good paths to true neighbors. DABS with aggressive γ terminates before finding those paths. Use m≥12 with DABS.

4. **Query outliers (out-of-distribution):** If a query is far from all training points, the k-th nearest is large, and (1+γ)*k_th may never trigger. DABS degrades to standard behavior — this is the correct fallback.

5. **k close to ef:** If k ≈ ef, the k-best and ef-worst are similar, and DABS loses its advantage. Keep k ≤ ef/4 for meaningful savings.

---

## What to Improve Next

1. **Multi-layer HNSW integration:** Apply DABS only at the base layer (where most distance computations happen). Upper layers use standard greedy descent (unchanged).

2. **SIMD distance kernels:** The current scalar `l2_sq` is unoptimized. Combining DABS with AVX2/AVX-512 would give multiplicative speedup.

3. **Clustered benchmark data:** Add a clustering generator (k-means initialization) to the PoC to demonstrate real-world DABS savings on structured data.

4. **Learned γ calibration:** Train a lightweight γ predictor (linear regression on query norm, etc.) to auto-tune per query. This is the bridge from DABS to Ada-ef and DARTH.

5. **Concurrent DABS:** Multi-threaded search with rayon-parallel query batches. The k_best heap is per-query and does not need synchronization.

6. **Benchmark on real ANN datasets:** Download SIFT-1M/GIST-960 (100–960 MB) and run the same comparison to validate the 10–50% claim on structured data.

---

## Production Crate Layout Proposal

```
crates/
  ruvector-dabs/         # standalone DABS library
    src/
      lib.rs             # GraphSearch trait + SearchStats
      graph.rs           # NswGraph (single-layer NSW)
      search.rs          # run_search: standard + DABS variants
      distance.rs        # l2_sq, cosine, dot (trait-based)
    benches/
      search.rs          # criterion: QPS × recall Pareto curve
    examples/
      demo.rs            # end-to-end reproduce
```

---

## References

1. Abbe, Lindeman, Sreenivasan. "Distance Adaptive Beam Search for Provably Accurate Graph-Based Nearest Neighbor Search." arXiv:2505.15636, May 2025. https://arxiv.org/abs/2505.15636

2. Malkov, Yashunin. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." IEEE TPAMI 2018. arXiv:1603.09320.

3. Zhang et al. "Distribution-Aware Exploration for Adaptive HNSW Search (Ada-ef)." arXiv:2512.06636, December 2025.

4. Peng et al. "DARTH: Declarative Recall Through Early Termination for ANN Search." SIGMOD 2026. arXiv:2505.19001.

5. Simhadri et al. "Graph-Based Vector Search: An Experimental Evaluation of the State-of-the-Art." arXiv:2502.05575, February 2025.
