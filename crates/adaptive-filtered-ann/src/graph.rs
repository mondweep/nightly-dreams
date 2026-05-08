use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use crate::Vector;

/// Approximate k-NN graph built via random-sampling neighbourhood selection.
/// Each node stores its M nearest neighbours (by L2²) among a large random sample.
pub struct KnnGraph {
    pub adj: Vec<Vec<usize>>,
    pub m: usize,
}

impl KnnGraph {
    /// `m` — out-degree (number of neighbours per node).
    /// Build cost: O(N * sample_per_node * D).
    /// With sample_size=2000 at N=10K, D=128 → ~2.56B fp ops (< 2s release).
    pub fn build(vectors: &[Vector], m: usize) -> Self {
        let n = vectors.len();
        assert!(n > m, "corpus must be larger than m");
        let mut rng = SmallRng::seed_from_u64(42);
        // Larger sample = better graph connectivity, up to full exact search.
        let sample_size = 2000usize.min(n.saturating_sub(1));

        let adj: Vec<Vec<usize>> = (0..n)
            .map(|i| {
                // Reservoir-sample `sample_size` distinct indices ≠ i.
                let mut pool: Vec<usize> = (0..n).filter(|&j| j != i).collect();
                for k in 0..sample_size.min(pool.len()) {
                    let j = rng.gen_range(k..pool.len());
                    pool.swap(k, j);
                }
                pool.truncate(sample_size.min(pool.len()));

                // Keep m nearest by L2².
                let mut scored: Vec<(f32, usize)> = pool
                    .iter()
                    .map(|&j| (vectors[i].l2_sq(&vectors[j]), j))
                    .collect();
                scored.sort_by(|a, b| {
                    a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                });
                scored.into_iter().take(m).map(|(_, j)| j).collect()
            })
            .collect();

        Self { adj, m }
    }

    pub fn len(&self) -> usize { self.adj.len() }
}
