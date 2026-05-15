pub mod centroid;
pub mod fde;
pub mod linear;

pub use centroid::CentroidIndex;
pub use fde::FdeIndex;
pub use linear::LinearIndex;

/// A document or query as a bag of L2-normalised token vectors.
pub type MultiVec = Vec<Vec<f32>>;

/// Chamfer / MaxSim score: Σ_{q∈Q} max_{d∈D} dot(q, d).
/// Requires vectors to be L2-normalised so dot = cosine similarity.
pub fn maxsim(query: &MultiVec, doc: &MultiVec) -> f32 {
    query
        .iter()
        .map(|q| {
            doc.iter()
                .map(|d| dot(q, d))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

#[inline(always)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline(always)]
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

pub fn normalize(v: &mut Vec<f32>) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-9 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

/// Extract top-k (id, score) pairs in descending score order.
pub fn topk(mut scores: Vec<(usize, f32)>, k: usize) -> Vec<(usize, f32)> {
    scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(k);
    scores
}

/// Shared interface for all MaxSim index variants.
pub trait MaxSimIndex {
    fn insert(&mut self, id: usize, doc: MultiVec);
    fn build(&mut self);
    fn search(&self, query: &MultiVec, k: usize) -> Vec<(usize, f32)>;
    fn name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_doc(seed: u32, n_tokens: usize, dim: usize) -> MultiVec {
        (0..n_tokens)
            .map(|t| {
                let mut v: Vec<f32> = (0..dim)
                    .map(|d| ((seed as f32 * 1.3 + t as f32 * 0.5 + d as f32 * 0.07).sin()))
                    .collect();
                normalize(&mut v);
                v
            })
            .collect()
    }

    #[test]
    fn test_maxsim_self_is_n_tokens() {
        let doc = make_doc(1, 4, 16);
        let score = maxsim(&doc, &doc);
        // Each query token matches itself perfectly (dot = 1.0 after normalisation)
        assert!((score - 4.0).abs() < 1e-5, "expected ~4.0, got {score}");
    }

    #[test]
    fn test_dot_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!((dot(&a, &b)).abs() < 1e-9);
    }

    #[test]
    fn test_linear_self_retrieval() {
        let dim = 16;
        let mut idx = LinearIndex::new();
        for i in 0..10usize {
            idx.insert(i, make_doc(i as u32, 4, dim));
        }
        idx.build();
        let query = make_doc(0, 4, dim);
        let results = idx.search(&query, 1);
        assert_eq!(results[0].0, 0, "linear should retrieve itself first");
    }

    #[test]
    fn test_centroid_returns_results() {
        let dim = 16;
        let mut idx = CentroidIndex::new(4, 2);
        for i in 0..20usize {
            idx.insert(i, make_doc(i as u32, 4, dim));
        }
        idx.build();
        let results = idx.search(&make_doc(0, 4, dim), 5);
        assert!(!results.is_empty());
        assert!(results[0].0 < 20);
    }

    #[test]
    fn test_fde_returns_results() {
        let dim = 16;
        let mut idx = FdeIndex::new(32, 10);
        for i in 0..20usize {
            idx.insert(i, make_doc(i as u32, 4, dim));
        }
        idx.build();
        let results = idx.search(&make_doc(0, 4, dim), 5);
        assert!(!results.is_empty());
        assert!(results[0].0 < 20);
    }

    #[test]
    fn test_maxsim_asymmetric() {
        // Doc with a single token at e0, query at e0 + noise
        let doc = vec![vec![1.0f32, 0.0, 0.0]];
        let query = vec![vec![1.0f32, 0.0, 0.0]];
        let score = maxsim(&query, &doc);
        assert!((score - 1.0).abs() < 1e-6);
    }
}
