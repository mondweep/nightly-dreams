//! Distance-Adaptive Beam Search (DABS) for navigable small-world graphs.
//!
//! Replaces the fixed-`ef` termination in standard HNSW beam search with
//! a distance-based stopping criterion (arXiv:2505.15636, May 2025):
//!
//!   terminate when d(q, next_candidate) > (1 + γ) × d(q, k-th_nearest_found)
//!
//! Easy queries (tight cluster near q) terminate immediately after finding k
//! good results; hard queries continue as long as standard ef-based search.
//! The result is 10–50% fewer distance computations at equivalent recall.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

// ── distance ──────────────────────────────────────────────────────────────────

#[inline(always)]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

// ── ordered-float wrapper (no external dep) ───────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
pub struct F32(pub f32);

impl Eq for F32 {}
impl PartialOrd for F32 {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&o.0)
    }
}
impl Ord for F32 {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.partial_cmp(o).unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ── search result ─────────────────────────────────────────────────────────────

/// A single nearest-neighbor result.
#[derive(Debug, Clone, Copy)]
pub struct Hit {
    pub id: usize,
    pub dist_sq: f32,
}

/// Output from every search variant.
pub struct SearchStats {
    /// k results, ascending by dist_sq.
    pub hits: Vec<Hit>,
    /// Number of L2-squared evaluations performed (distance computation count).
    pub dist_ops: u64,
}

// ── search trait ──────────────────────────────────────────────────────────────

pub trait GraphSearch {
    /// Standard HNSW beam search: terminate when the nearest unvisited
    /// candidate is farther than the worst result in the `ef`-sized found set.
    fn beam_search(&self, query: &[f32], k: usize, ef: usize) -> SearchStats;

    /// DABS (arXiv:2505.15636): terminate when nearest candidate >
    /// (1 + γ) × current k-th nearest found.  γ=0 is most aggressive;
    /// γ=2.0 provides the theoretical approximation guarantee from the paper.
    fn dabs_search(&self, query: &[f32], k: usize, ef: usize, gamma: f32) -> SearchStats;

    /// Exact brute-force k-NN; used to compute ground-truth recall.
    fn brute_force(&self, query: &[f32], k: usize) -> Vec<usize>;
}

// ── NSW graph ─────────────────────────────────────────────────────────────────

/// Navigable Small World graph (equivalent to the HNSW base layer).
/// Insertion greedily connects each new node to its m nearest current
/// neighbours and maintains bidirectional edges (capped at 2m per node).
pub struct NswGraph {
    pub vectors: Vec<Vec<f32>>,
    pub neighbors: Vec<Vec<usize>>,
    pub dim: usize,
    pub m: usize,
}

impl NswGraph {
    pub fn new(dim: usize, m: usize) -> Self {
        Self {
            vectors: Vec::new(),
            neighbors: Vec::new(),
            dim,
            m,
        }
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Insert one vector; connects it to m nearest current neighbours.
    pub fn insert(&mut self, vec: Vec<f32>) {
        debug_assert_eq!(vec.len(), self.dim);
        let id = self.len();
        self.vectors.push(vec);
        self.neighbors.push(Vec::new());
        if id == 0 {
            return;
        }

        // Brute-force for the first ef_build nodes; beam search thereafter.
        let ef_build = (self.m * 4).max(32);
        let new_nbrs: Vec<usize> = if id <= ef_build {
            let mut all: Vec<(F32, usize)> = (0..id)
                .map(|j| (F32(l2_sq(&self.vectors[id], &self.vectors[j])), j))
                .collect();
            all.sort_unstable();
            all.truncate(self.m);
            all.into_iter().map(|(_, j)| j).collect()
        } else {
            let stats = self.run_search(&self.vectors[id].clone(), self.m, ef_build, None);
            stats.hits.into_iter().map(|h| h.id).collect()
        };

        for &nb in &new_nbrs {
            self.neighbors[id].push(nb);
            if !self.neighbors[nb].contains(&id) {
                if self.neighbors[nb].len() < self.m * 2 {
                    self.neighbors[nb].push(id);
                } else {
                    // Prune nb's neighbour list to 2m nearest.
                    let nb_vec = self.vectors[nb].clone();
                    let mut cands: Vec<usize> = self.neighbors[nb].clone();
                    cands.push(id);
                    cands.sort_unstable_by(|&a, &b| {
                        F32(l2_sq(&nb_vec, &self.vectors[a]))
                            .cmp(&F32(l2_sq(&nb_vec, &self.vectors[b])))
                    });
                    cands.truncate(self.m * 2);
                    self.neighbors[nb] = cands;
                }
            }
        }
    }

    /// Core search engine shared by both beam_search and dabs_search.
    ///
    /// `gamma = None`  → standard termination (worst-of-ef threshold).
    /// `gamma = Some(g)` → DABS termination ((1+g) × k-th nearest threshold).
    fn run_search(&self, query: &[f32], k: usize, ef: usize, gamma: Option<f32>) -> SearchStats {
        let n = self.len();
        if n == 0 {
            return SearchStats { hits: vec![], dist_ops: 0 };
        }

        let mut dist_ops = 0u64;
        let mut visited = vec![false; n];

        // Multi-start entry: 4 evenly-spaced landmark nodes, all seeded into
        // both heaps so their neighbourhoods are explored from the start.
        // Identical strategy for every variant → fair comparison.
        let mut cands: BinaryHeap<Reverse<(F32, usize)>> = BinaryHeap::new();
        let mut found: BinaryHeap<(F32, usize)> = BinaryHeap::new();
        let mut k_best: BinaryHeap<F32> = BinaryHeap::new();

        let starts = [0, n / 4, n / 2, 3 * n / 4];
        for &s in &starts {
            let d = l2_sq(query, &self.vectors[s]);
            dist_ops += 1;
            visited[s] = true;
            cands.push(Reverse((F32(d), s)));
            found.push((F32(d), s));
            if found.len() > ef {
                found.pop();
            }
            let k_th = k_best.peek().map(|F32(kd)| *kd).unwrap_or(f32::MAX);
            if d < k_th || k_best.len() < k {
                k_best.push(F32(d));
                if k_best.len() > k {
                    k_best.pop();
                }
            }
        }

        loop {
            let c_dist = match cands.peek() {
                Some(Reverse((F32(d), _))) => *d,
                None => break,
            };

            let worst_found = found.peek().map(|(F32(d), _)| *d).unwrap_or(f32::MAX);

            let stop = match gamma {
                None => c_dist > worst_found,
                Some(g) => {
                    if k_best.len() < k {
                        // Haven't found k results yet — never stop early.
                        false
                    } else {
                        let k_th = k_best.peek().map(|F32(d)| *d).unwrap_or(f32::MAX);
                        c_dist > (1.0 + g) * k_th
                    }
                }
            };
            if stop {
                break;
            }

            let Reverse((_, c_id)) = cands.pop().unwrap();
            for &nb in &self.neighbors[c_id] {
                if visited[nb] {
                    continue;
                }
                visited[nb] = true;
                let d = l2_sq(query, &self.vectors[nb]);
                dist_ops += 1;

                let worst = found.peek().map(|(F32(w), _)| *w).unwrap_or(f32::MAX);
                if d < worst || found.len() < ef {
                    cands.push(Reverse((F32(d), nb)));
                    found.push((F32(d), nb));
                    if found.len() > ef {
                        found.pop();
                    }
                    // Update k-best.
                    let k_th = k_best.peek().map(|F32(d)| *d).unwrap_or(f32::MAX);
                    if d < k_th || k_best.len() < k {
                        k_best.push(F32(d));
                        if k_best.len() > k {
                            k_best.pop();
                        }
                    }
                }
            }
        }

        let mut hits: Vec<Hit> = found
            .into_iter()
            .map(|(F32(d), id)| Hit { id, dist_sq: d })
            .collect();
        hits.sort_unstable_by(|a, b| a.dist_sq.partial_cmp(&b.dist_sq).unwrap());
        hits.truncate(k);
        SearchStats { hits, dist_ops }
    }
}

impl GraphSearch for NswGraph {
    fn beam_search(&self, query: &[f32], k: usize, ef: usize) -> SearchStats {
        self.run_search(query, k, ef, None)
    }

    fn dabs_search(&self, query: &[f32], k: usize, ef: usize, gamma: f32) -> SearchStats {
        self.run_search(query, k, ef, Some(gamma))
    }

    fn brute_force(&self, query: &[f32], k: usize) -> Vec<usize> {
        let mut all: Vec<(F32, usize)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (F32(l2_sq(query, v)), i))
            .collect();
        all.sort_unstable();
        all.truncate(k);
        all.into_iter().map(|(_, id)| id).collect()
    }
}

// ── recall ────────────────────────────────────────────────────────────────────

/// Fraction of true-k-NN ids present in `approx`.
pub fn recall_at_k(truth: &[usize], approx: &[Hit]) -> f64 {
    let approx_set: std::collections::HashSet<usize> = approx.iter().map(|h| h.id).collect();
    truth.iter().filter(|&&id| approx_set.contains(&id)).count() as f64 / truth.len() as f64
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> f32 {
        *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((*s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    fn make_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        let v: Vec<f32> = (0..dim).map(|_| lcg(&mut s)).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        v.into_iter().map(|x| x / norm).collect()
    }

    #[test]
    fn beam_and_dabs_return_k_results() {
        let mut g = NswGraph::new(16, 8);
        for i in 0..200u64 {
            g.insert(make_vec(16, i * 1337 + 1));
        }
        let q = make_vec(16, 9_999);
        let k = 10;
        let ef = 64;
        let bs = g.beam_search(&q, k, ef);
        let dabs = g.dabs_search(&q, k, ef, 2.0);
        assert_eq!(bs.hits.len(), k);
        assert_eq!(dabs.hits.len(), k);
    }

    #[test]
    fn dabs_gamma0_always_fewer_or_equal_dist_ops() {
        // γ=0 is the tightest DABS variant: terminate when candidate > k-th nearest.
        // Since k-th nearest ≤ ef-th nearest (standard threshold), DABS(γ=0) always
        // terminates at the same point or earlier — a hard mathematical invariant.
        let mut g = NswGraph::new(32, 12);
        for i in 0..500u64 {
            g.insert(make_vec(32, i * 1337 + 7));
        }
        let nq = 50;
        for qi in 0..nq as u64 {
            let q = make_vec(32, qi * 9_999 + 3);
            let std_ops = g.beam_search(&q, 10, 64).dist_ops;
            let dabs_ops = g.dabs_search(&q, 10, 64, 0.0).dist_ops;
            assert!(
                dabs_ops <= std_ops,
                "query {qi}: DABS(γ=0) used {dabs_ops} ops but standard used only {std_ops}"
            );
        }
    }

    #[test]
    fn dabs_gamma2_maintains_recall() {
        let mut g = NswGraph::new(32, 12);
        let n = 1_000u64;
        for i in 0..n {
            g.insert(make_vec(32, i * 1337 + 1));
        }
        let nq = 100;
        let k = 10;
        let ef = 64;
        let mut total_recall_std = 0.0f64;
        let mut total_recall_dabs = 0.0f64;
        for qi in 0..nq as u64 {
            let q = make_vec(32, qi * 9_999 + 5);
            let truth = g.brute_force(&q, k);
            let std_hits = g.beam_search(&q, k, ef).hits;
            let dabs_hits = g.dabs_search(&q, k, ef, 2.0).hits;
            total_recall_std += recall_at_k(&truth, &std_hits);
            total_recall_dabs += recall_at_k(&truth, &dabs_hits);
        }
        let avg_std = total_recall_std / nq as f64;
        let avg_dabs = total_recall_dabs / nq as f64;
        // DABS γ=2.0 should stay within 5 pp of standard recall.
        assert!(
            avg_dabs >= avg_std - 0.05,
            "DABS recall {avg_dabs:.3} too far below standard {avg_std:.3}"
        );
    }
}
