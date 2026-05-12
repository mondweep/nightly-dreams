use std::collections::{BTreeMap, HashSet};
use std::cmp::Ordering;

// ─────────────────────── Vector types ────────────────────────────────────────

/// Dense f32 embedding + sparse BM25-style term weights.
#[derive(Clone, Debug)]
pub struct HybridVector {
    pub dense: Vec<f32>,
    /// Sparse component: term_id → weight (L2-normalised term vector).
    pub sparse: BTreeMap<u32, f32>,
}

impl HybridVector {
    pub fn new(dense: Vec<f32>, sparse: BTreeMap<u32, f32>) -> Self {
        Self { dense, sparse }
    }

    /// L2 distance on the dense component.
    pub fn dense_l2(&self, other: &Self) -> f32 {
        self.dense
            .iter()
            .zip(other.dense.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt()
    }

    /// Sparse inner-product distance: 1 − dot(a, b).
    /// Both sparse vectors should be L2-normalised so this ∈ [0, 2].
    pub fn sparse_ip_dist(&self, other: &Self) -> f32 {
        let mut dot = 0.0f32;
        for (k, v) in &self.sparse {
            if let Some(ov) = other.sparse.get(k) {
                dot += v * ov;
            }
        }
        1.0 - dot
    }

    /// Fused hybrid distance: alpha × dense_L2 + (1−alpha) × sparse_IP_dist.
    pub fn hybrid_dist(&self, other: &Self, alpha: f32) -> f32 {
        alpha * self.dense_l2(other) + (1.0 - alpha) * self.sparse_ip_dist(other)
    }
}

// ─────────────────────── Brute-force baseline ────────────────────────────────

pub struct BruteForce {
    data: Vec<HybridVector>,
}

impl BruteForce {
    pub fn new(data: Vec<HybridVector>) -> Self {
        Self { data }
    }

    /// Return indices of top-k nearest by hybrid distance.
    pub fn search(&self, query: &HybridVector, k: usize, alpha: f32) -> Vec<usize> {
        let mut dists: Vec<(f32, usize)> = self
            .data
            .iter()
            .enumerate()
            .map(|(i, v)| (v.hybrid_dist(query, alpha), i))
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        dists.iter().take(k).map(|(_, i)| *i).collect()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

// ─────────────────────── NSW graph index ─────────────────────────────────────

/// Controls which distance is computed during graph traversal.
#[derive(Clone, Copy, Debug)]
pub enum SearchMode {
    /// Always use the full hybrid distance during traversal and reranking.
    HybridFull,
    /// Use only the dense component during traversal, then hybrid-rerank top candidates.
    TwoStage { candidates_factor: usize },
}

pub struct HybridNSW {
    pub data: Vec<HybridVector>,
    adj: Vec<Vec<usize>>,
    pub m: usize,
    pub alpha: f32,
}

impl HybridNSW {
    pub fn new(m: usize, alpha: f32) -> Self {
        Self { data: Vec::new(), adj: Vec::new(), m, alpha }
    }

    fn traversal_dist(&self, a: &HybridVector, b: &HybridVector, mode: SearchMode) -> f32 {
        match mode {
            SearchMode::HybridFull => a.hybrid_dist(b, self.alpha),
            SearchMode::TwoStage { .. } => a.dense_l2(b),
        }
    }

    /// Insert a vector, connecting it to M nearest neighbours (brute-force during build).
    pub fn insert(&mut self, v: HybridVector) {
        let id = self.data.len();
        self.data.push(v);
        self.adj.push(Vec::new());
        if id == 0 {
            return;
        }

        let mut dists: Vec<(f32, usize)> = (0..id)
            .map(|j| (self.data[id].hybrid_dist(&self.data[j], self.alpha), j))
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        let neighbours: Vec<usize> = dists.iter().take(self.m).map(|(_, j)| *j).collect();
        for &nb in &neighbours {
            if !self.adj[nb].contains(&id) {
                self.adj[nb].push(id);
            }
        }
        self.adj[id] = neighbours;
    }

    /// Greedy NSW search returning up to k nearest neighbours.
    pub fn search(&self, query: &HybridVector, k: usize, mode: SearchMode) -> Vec<usize> {
        let n = self.data.len();
        if n == 0 {
            return vec![];
        }
        let ef = match mode {
            SearchMode::HybridFull => k * 4,
            SearchMode::TwoStage { candidates_factor } => k * candidates_factor,
        };

        let mut visited = vec![false; n];
        visited[0] = true;

        let d0 = self.traversal_dist(query, &self.data[0], mode);
        // candidates: (dist, id) — we want to process the nearest first
        let mut cands: Vec<(f32, usize)> = vec![(d0, 0)];
        // results: (dist, id) — keep ef closest, drop the farthest when over-full
        let mut results: Vec<(f32, usize)> = vec![(d0, 0)];

        loop {
            // Nearest unprocessed candidate
            let Some((min_pos, _)) = cands.iter().enumerate()
                .min_by(|(_, a), (_, b)| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal))
            else {
                break;
            };
            let (min_dist, cur) = cands.swap_remove(min_pos);

            let worst_dist = results
                .iter()
                .map(|(d, _)| *d)
                .fold(f32::NEG_INFINITY, f32::max);
            if min_dist > worst_dist && results.len() >= ef {
                break;
            }

            for &nb in &self.adj[cur] {
                if visited[nb] {
                    continue;
                }
                visited[nb] = true;
                let nd = self.traversal_dist(query, &self.data[nb], mode);
                let worst = results.iter().map(|(d, _)| *d).fold(f32::NEG_INFINITY, f32::max);
                if nd < worst || results.len() < ef {
                    cands.push((nd, nb));
                    results.push((nd, nb));
                    if results.len() > ef {
                        let max_pos = results
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| {
                                a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal)
                            })
                            .map(|(i, _)| i)
                            .unwrap();
                        results.swap_remove(max_pos);
                    }
                }
            }
        }

        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        let mut ids: Vec<usize> = results.iter().take(k).map(|(_, i)| *i).collect();

        // Two-stage: re-rank the candidate set with full hybrid distance
        if let SearchMode::TwoStage { .. } = mode {
            ids.sort_by(|&a, &b| {
                query
                    .hybrid_dist(&self.data[a], self.alpha)
                    .partial_cmp(&query.hybrid_dist(&self.data[b], self.alpha))
                    .unwrap_or(Ordering::Equal)
            });
        }

        ids
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

// ─────────────────────── Distribution alignment ──────────────────────────────

/// Estimate the optimal fusion weight alpha by aligning the variance of
/// dense-L2 and sparse-IP distance distributions across a random sample.
///
/// Principle: weight each modality inversely to its standard deviation so that
/// neither component dominates the fused score by accident of scale.
pub fn calibrate_alpha(sample: &[HybridVector]) -> f32 {
    let n = sample.len().min(128);
    if n < 4 {
        return 0.5;
    }

    let mut dense_d: Vec<f32> = Vec::with_capacity(n * n / 2);
    let mut sparse_d: Vec<f32> = Vec::with_capacity(n * n / 2);

    for i in 0..n {
        for j in (i + 1)..n {
            dense_d.push(sample[i].dense_l2(&sample[j]));
            sparse_d.push(sample[i].sparse_ip_dist(&sample[j]));
        }
    }

    let std_fn = |xs: &[f32]| -> f32 {
        let mean = xs.iter().sum::<f32>() / xs.len() as f32;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / xs.len() as f32;
        var.sqrt()
    };

    let std_dense = std_fn(&dense_d);
    let std_sparse = std_fn(&sparse_d);

    if std_dense + std_sparse < 1e-9 {
        return 0.5;
    }
    // Assign dense the fraction of total "spread" owned by sparse (normalise them both)
    let alpha = std_sparse / (std_dense + std_sparse);
    alpha.clamp(0.15, 0.85)
}

// ─────────────────────── Recall metric ───────────────────────────────────────

pub fn recall_at_k(ground_truth: &[usize], retrieved: &[usize]) -> f32 {
    if ground_truth.is_empty() {
        return 1.0;
    }
    let gt_set: HashSet<usize> = ground_truth.iter().cloned().collect();
    let hits = retrieved.iter().filter(|i| gt_set.contains(i)).count();
    hits as f32 / ground_truth.len() as f32
}

// ─────────────────────── Tests ───────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_vec(dense: &[f32], sparse: &[(u32, f32)]) -> HybridVector {
        let d = dense.to_vec();
        let s: BTreeMap<u32, f32> = sparse.iter().cloned().collect();
        HybridVector::new(d, s)
    }

    #[test]
    fn dense_l2_zero_self_distance() {
        let v = toy_vec(&[1.0, 0.0, 0.0], &[(0, 1.0)]);
        assert!(v.dense_l2(&v) < 1e-6);
    }

    #[test]
    fn sparse_ip_dist_orthogonal_is_one() {
        let a = toy_vec(&[], &[(0, 1.0)]);
        let b = toy_vec(&[], &[(1, 1.0)]);
        let d = a.sparse_ip_dist(&b);
        assert!((d - 1.0).abs() < 1e-6, "expected 1.0, got {d}");
    }

    #[test]
    fn sparse_ip_dist_identical_normalised() {
        let a = toy_vec(&[], &[(0, 1.0)]);
        let d = a.sparse_ip_dist(&a);
        assert!(d < 1e-6, "expected ~0.0, got {d}");
    }

    #[test]
    fn brute_force_finds_nearest() {
        let data: Vec<HybridVector> = (0..20)
            .map(|i| toy_vec(&[i as f32, 0.0], &[]))
            .collect();
        let query = toy_vec(&[3.1, 0.0], &[]);
        let bf = BruteForce::new(data);
        let res = bf.search(&query, 1, 1.0);
        assert_eq!(res[0], 3, "nearest should be index 3 (vec at x=3)");
    }

    #[test]
    fn nsw_recall_better_than_random() {
        let data: Vec<HybridVector> = (0..100)
            .map(|i| {
                let v = vec![(i as f32) / 100.0, ((i * 7) % 100) as f32 / 100.0];
                let s: BTreeMap<u32, f32> = [(i as u32 % 10, 1.0)].iter().cloned().collect();
                HybridVector::new(v, s)
            })
            .collect();

        let query = HybridVector::new(vec![0.5, 0.5], BTreeMap::new());
        let bf = BruteForce::new(data.clone());
        let gt = bf.search(&query, 5, 0.5);

        let mut nsw = HybridNSW::new(8, 0.5);
        for v in &data {
            nsw.insert(v.clone());
        }
        let res = nsw.search(&query, 5, SearchMode::HybridFull);
        let r = recall_at_k(&gt, &res);
        assert!(r > 0.4, "NSW recall@5 should be > 40%, got {r:.2}");
    }

    #[test]
    fn calibrate_alpha_range() {
        let data: Vec<HybridVector> = (0..40)
            .map(|i| {
                let d = vec![(i as f32) / 40.0, (i as f32).sin()];
                let s: BTreeMap<u32, f32> = [(i as u32 % 5, 0.5)].iter().cloned().collect();
                HybridVector::new(d, s)
            })
            .collect();
        let a = calibrate_alpha(&data);
        assert!(a >= 0.15 && a <= 0.85, "alpha={a} out of clamped range");
    }

    #[test]
    fn two_stage_recall_comparable_to_full() {
        let data: Vec<HybridVector> = (0..200)
            .map(|i| {
                let v = vec![(i as f32) / 200.0, ((i * 3) % 200) as f32 / 200.0];
                let s: BTreeMap<u32, f32> = [(i as u32 % 20, 1.0)].iter().cloned().collect();
                HybridVector::new(v, s)
            })
            .collect();

        let alpha = calibrate_alpha(&data);
        let query = HybridVector::new(vec![0.25, 0.75], BTreeMap::new());
        let bf = BruteForce::new(data.clone());
        let gt = bf.search(&query, 10, alpha);

        let mut nsw = HybridNSW::new(12, alpha);
        for v in &data {
            nsw.insert(v.clone());
        }

        let r_full = recall_at_k(
            &gt,
            &nsw.search(&query, 10, SearchMode::HybridFull),
        );
        let r_two = recall_at_k(
            &gt,
            &nsw.search(&query, 10, SearchMode::TwoStage { candidates_factor: 8 }),
        );

        assert!(r_full > 0.4, "HybridFull recall={r_full:.2}");
        assert!(r_two > 0.4, "TwoStage recall={r_two:.2}");
    }
}
