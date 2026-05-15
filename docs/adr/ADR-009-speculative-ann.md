# ADR-009: Speculative ANN Search with Int8 Draft Index

**Status:** Accepted  
**Date:** 2026-05-15  
**Branch:** research/nightly/2026-05-15-speculative-ann

---

## Context

Vector similarity search over large corpora is dominated by exact brute-force cost
O(N·d) per query.  Graph indexes (HNSW, NSG) reduce this asymptotically but carry
high build cost and memory overhead that is not always acceptable.  When a flat
(brute-force) index is used — as is the case for small-to-medium corpora (<500 k docs)
or when freshness demands prevent graph maintenance — the query path is a bottleneck.

Hardware has an under-exploited lever: SIMD integer arithmetic.  On modern x86-64 CPUs
(AVX2/AVX-512) and ARM (NEON/SVE2), int8 dot-product throughput is approximately 4×
greater than f32, and int8 memory bandwidth is 4× narrower.  Scalar quantisation of
embedding vectors to int8 introduces a bounded error of ≈ 0.008 per dimension (1/127),
and for high-dimensional embeddings (d≥64) the rank-order of dot products is largely
preserved.

The idea of *speculative execution* — draft cheaply, verify selectively — is well
established in hardware (branch prediction, out-of-order execution) and has been
popularised in LLM inference as *speculative decoding* (Chen et al., 2023; Leviathan
et al., 2023).  Applying the same paradigm to ANN: use an int8 draft index to rank
all N documents cheaply, then verify only the top `K × overdraft` candidates with
full f32 precision to recover exact top-K results.

ruvector currently has no int8 scalar-quantised index and no speculative search path.

---

## Decision

Implement `crates/speculative-ann` with three variants:

| Variant | Algorithm | Role |
|---|---|---|
| `linear-f32` | brute-force f32 dot products | baseline / ground truth |
| `linear-i8` | quantise vectors to int8, brute-force int8 dot products | pure draft (fast, approximate) |
| `spec-search` | int8 draft → top-K×overdraft candidates → f32 verify | speculative (fast + exact) |

The `SpecAnnIndex` trait provides a uniform interface; variants are interchangeable.

Scalar quantisation: `q = clamp(x × 127, −127, 127) as i8`.  Approximate dot:
`Σ aᵢ·bᵢ / 127²`.  No per-vector scale factor (symmetric uniform quantisation).

---

## Consequences

### Positive
- `spec-search` with `overdraft=2` achieves **recall@10 = 1.000** at **1.78×** the
  throughput of pure f32 on 10 k corpus × 128 dims (criterion: 878 µs vs 1,559 µs).
- Memory usage: int8 vectors are 4× smaller than f32; `spec-search` doubles this
  via dual storage, but the int8 copy still accelerates the O(N·d) hot path.
- The `overdraft` parameter is runtime-tunable: `overdraft=1` maximises speed at
  ~7% recall cost; `overdraft≥2` fully recovers recall in these experiments.
- No unsafe code, no SIMD intrinsics (compiler auto-vectorisation with AVX2 is
  sufficient; explicit SIMD is a follow-up).
- The `SpecAnnIndex` trait enables future graph-backed or PQ-backed draft indexes
  without changing the caller interface.

### Negative / Risks
- On corpora with near-identical cosine scores (adversarial or highly clustered),
  quantisation errors can cause rank swaps outside the overdraft window, reducing
  recall below 100%.  `overdraft=4` is a safer default.
- `spec-search` stores both f32 and int8 representations, increasing peak memory
  by 1.25× compared to f32-only.
- At very small N (<200 docs), the overhead of dual allocation and the int8 partial
  sort can be slower than pure f32.  A corpus-size gate should prefer `linear-f32`
  below a configurable threshold.

### Neutral
- The current implementation is single-threaded.  Parallelising the draft scan with
  rayon is straightforward and is left for a follow-up ADR.
- Symmetric uniform quantisation (no per-vector scale) is chosen for simplicity.
  Asymmetric or per-channel quantisation would improve accuracy at the cost of an
  additional dequant multiply.

---

## Alternatives Considered

| Alternative | Why rejected |
|---|---|
| Product Quantisation (PQ) draft | PQ requires a training phase (k-means); adds build complexity. Covered by prior ADRs (RaBitQ, CoDEQ). |
| Binary (1-bit) draft | 128× compression but recall drops ~20% even at overdraft=8; too low for a general-purpose draft index. |
| Random projection draft | Preserves order better than binary but requires O(d×P) projection matrix; similar speed to int8 with worse accuracy. |
| HNSW draft | High build cost and recall-performance depends on graph connectivity; not appropriate for a nightly-scope PoC. |
| Async prefetch (sw pipelining) | Orthogonal optimisation; can be layered on top of `spec-search` in a follow-up. |

---

## References

- Speculative Decoding: Chen et al., arXiv 2302.01318 (2023); Leviathan et al., ICML 2023
- ScaNN (Google): Guo et al., ICML 2020 — asymmetric quantised scoring
- FAISS int8 support: Johnson et al., IEEE TPAMI 2021
- USearch (Unum): symmetric int8 scan, GitHub 2023 — nearest prior art in Rust/C++
- Auto-vectorisation of integer reductions: LLVM codegen docs; GCC trunk notes 2024
- ruvector prior ADRs: ADR-001 (RaBitQ), ADR-007 (CoDEQ), ADR-008 (MaxSim)
