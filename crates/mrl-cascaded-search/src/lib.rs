//! Matryoshka Representation Learning (MRL) cascaded ANN search.
//!
//! Two-stage retrieval: fast ANN at a truncated dimension (coarse pass),
//! then exact re-scoring at the full dimension (fine pass).
//! Faithfully implements the MRL cascaded retrieval strategy from
//! Kusupati et al. NeurIPS 2022 and subsequent Weaviate / OpenAI adoption.

use std::collections::BinaryHeap;

use ordered_float::OrderedFloat;

/// Trait every distance kernel must implement.
pub trait Distance: Send + Sync {
    /// Returns distance between two equal-length slices (lower = more similar).
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
}

/// Squared Euclidean distance (avoids sqrt, monotone with L2).
pub struct L2Sq;
impl Distance for L2Sq {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
    }
}

/// Dot-product similarity expressed as a distance (1 - dot, assumes unit norms).
pub struct DotProduct;
impl Distance for DotProduct {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        1.0 - a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
    }
}

/// A single stored vector plus its id.
struct Entry {
    id: u64,
    /// Full-dimension embedding, owned.
    embedding: Vec<f32>,
}

/// Flat (brute-force) index operating on a configurable prefix of each vector.
/// Real production code would use HNSW here; we keep it simple so benchmarks
/// measure the cascade algorithm rather than the index internals.
pub struct FlatIndex {
    entries: Vec<Entry>,
    full_dim: usize,
}

impl FlatIndex {
    pub fn new(full_dim: usize) -> Self {
        Self { entries: Vec::new(), full_dim }
    }

    pub fn insert(&mut self, id: u64, embedding: Vec<f32>) {
        assert_eq!(
            embedding.len(),
            self.full_dim,
            "embedding dim mismatch: got {}, expected {}",
            embedding.len(),
            self.full_dim
        );
        self.entries.push(Entry { id, embedding });
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Brute-force ANN at `search_dim` (must be ≤ full_dim).
    /// Returns at most `k` (id, distance) pairs sorted by ascending distance.
    pub fn search_at_dim(
        &self,
        query: &[f32],
        search_dim: usize,
        k: usize,
        dist: &dyn Distance,
    ) -> Vec<(u64, f32)> {
        assert!(search_dim <= self.full_dim, "search_dim > full_dim");
        assert!(search_dim <= query.len(), "query shorter than search_dim");

        // Max-heap keyed by distance: pop the worst when heap exceeds k,
        // keeping only the k nearest neighbours.
        let mut heap: BinaryHeap<(OrderedFloat<f32>, u64)> = BinaryHeap::with_capacity(k + 1);

        for entry in &self.entries {
            let d = dist.distance(&query[..search_dim], &entry.embedding[..search_dim]);
            heap.push((OrderedFloat(d), entry.id));
            if heap.len() > k {
                heap.pop(); // removes largest distance — the worst match
            }
        }

        let mut results: Vec<(u64, f32)> = heap
            .into_iter()
            .map(|(OrderedFloat(d), id)| (id, d))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results
    }

    /// Exact re-score of a candidate set at `rescore_dim`.
    pub fn rescore(
        &self,
        query: &[f32],
        candidates: &[(u64, f32)],
        rescore_dim: usize,
        k: usize,
        dist: &dyn Distance,
    ) -> Vec<(u64, f32)> {
        assert!(rescore_dim <= self.full_dim);
        // Build id → entry map (linear scan over candidates, small set)
        let mut rescored: Vec<(u64, f32)> = candidates
            .iter()
            .filter_map(|(id, _)| {
                self.entries.iter().find(|e| e.id == *id).map(|e| {
                    let d = dist.distance(&query[..rescore_dim], &e.embedding[..rescore_dim]);
                    (*id, d)
                })
            })
            .collect();
        rescored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        rescored.truncate(k);
        rescored
    }
}

/// Configuration for a cascaded MRL search pass.
#[derive(Clone, Debug)]
pub struct CascadeConfig {
    /// Dimension used in the fast coarse ANN pass.
    pub coarse_dim: usize,
    /// Oversample factor: retrieve `k * oversample` candidates in coarse pass.
    pub oversample: usize,
    /// Final dimension used for re-scoring.
    pub fine_dim: usize,
}

/// Cascaded MRL search: coarse ANN → fine re-score.
pub struct MrlCascadedIndex {
    inner: FlatIndex,
    config: CascadeConfig,
}

impl MrlCascadedIndex {
    pub fn new(full_dim: usize, config: CascadeConfig) -> Self {
        assert!(config.coarse_dim <= config.fine_dim, "coarse_dim must be ≤ fine_dim");
        assert!(config.fine_dim <= full_dim, "fine_dim must be ≤ full_dim");
        Self { inner: FlatIndex::new(full_dim), config }
    }

    pub fn insert(&mut self, id: u64, embedding: Vec<f32>) {
        self.inner.insert(id, embedding);
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Run cascaded search and return top-k (id, distance at fine_dim).
    pub fn search(&self, query: &[f32], k: usize, dist: &dyn Distance) -> Vec<(u64, f32)> {
        let coarse_k = k * self.config.oversample;
        let candidates =
            self.inner.search_at_dim(query, self.config.coarse_dim, coarse_k, dist);
        self.inner.rescore(query, &candidates, self.config.fine_dim, k, dist)
    }

    /// Baseline: direct search at full dimension (no cascade).
    pub fn search_full(&self, query: &[f32], k: usize, dist: &dyn Distance) -> Vec<(u64, f32)> {
        self.inner
            .search_at_dim(query, self.inner.full_dim, k, dist)
    }
}

/// Compute recall@k: fraction of ground-truth ids that appear in results.
pub fn recall_at_k(ground_truth: &[(u64, f32)], results: &[(u64, f32)], k: usize) -> f64 {
    let gt_ids: std::collections::HashSet<u64> =
        ground_truth.iter().take(k).map(|(id, _)| *id).collect();
    let found = results.iter().take(k).filter(|(id, _)| gt_ids.contains(id)).count();
    found as f64 / k.min(gt_ids.len()) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use rand::Rng;

    fn make_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect()
    }

    #[test]
    fn test_flat_index_exact_recall() {
        let dim = 128;
        let n = 500;
        let vecs = make_random_vectors(n, dim, 42);
        let mut idx = FlatIndex::new(dim);
        for (i, v) in vecs.iter().enumerate() {
            idx.insert(i as u64, v.clone());
        }
        // query is the first vector — should be top-1
        let query = vecs[0].clone();
        let results = idx.search_at_dim(&query, dim, 5, &L2Sq);
        assert_eq!(results[0].0, 0, "first result must be query vector itself");
        assert!(results[0].1 < 1e-6, "self-distance must be ~0");
    }

    /// Generate MRL-structured vectors: cluster_center (low-dim signal) + noise.
    /// The first `coarse_dim` dimensions are dominated by the cluster center,
    /// making them predictive of the full distance — matching MRL training behaviour.
    fn make_mrl_vectors(
        n: usize,
        full_dim: usize,
        n_clusters: usize,
        signal_weight: f32,
        seed: u64,
    ) -> Vec<Vec<f32>> {
        let mut rng = SmallRng::seed_from_u64(seed);
        // generate cluster centres
        let centers: Vec<Vec<f32>> = (0..n_clusters)
            .map(|_| (0..full_dim).map(|_| rng.gen::<f32>()).collect())
            .collect();
        (0..n)
            .map(|_| {
                let c = &centers[rng.gen_range(0..n_clusters)];
                (0..full_dim)
                    .map(|d| signal_weight * c[d] + (1.0 - signal_weight) * rng.gen::<f32>())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_cascade_recall_above_threshold() {
        let full_dim = 256;
        let n = 1000;
        let k = 10;
        // MRL-structured data: strong cluster signal in all dims so prefix is predictive
        let vecs = make_mrl_vectors(n, full_dim, 20, 0.85, 7);

        let config = CascadeConfig { coarse_dim: 64, oversample: 10, fine_dim: full_dim };
        let mut idx = MrlCascadedIndex::new(full_dim, config);
        for (i, v) in vecs.iter().enumerate() {
            idx.insert(i as u64, v.clone());
        }

        let query = make_mrl_vectors(1, full_dim, 20, 0.85, 99)[0].clone();
        let ground_truth = idx.search_full(&query, k, &L2Sq);
        let cascade_results = idx.search(&query, k, &L2Sq);

        let recall = recall_at_k(&ground_truth, &cascade_results, k);
        // With cluster-structured data and oversample=10, recall should be ≥ 0.70
        assert!(
            recall >= 0.70,
            "recall@{k} = {recall:.2} below 0.70 (MRL-structured data, oversample=10)"
        );
    }

    #[test]
    fn test_rescore_subset() {
        let dim = 64;
        let n = 100;
        let vecs = make_random_vectors(n, dim, 13);
        let mut idx = FlatIndex::new(dim);
        for (i, v) in vecs.iter().enumerate() {
            idx.insert(i as u64, v.clone());
        }
        let query = vecs[5].clone();
        let coarse = idx.search_at_dim(&query, 32, 20, &L2Sq);
        let fine = idx.rescore(&query, &coarse, dim, 5, &L2Sq);
        assert_eq!(fine.len(), 5);
        // The exact match (id=5) must be in fine results since it's in coarse
        assert!(fine.iter().any(|(id, _)| *id == 5));
    }

    #[test]
    fn test_dot_product_distance() {
        // Unit vectors: dot = cos, distance = 1 - cos
        let a = vec![1.0_f32, 0.0];
        let b = vec![1.0_f32, 0.0];
        assert!((DotProduct.distance(&a, &b)).abs() < 1e-6);
    }
}
