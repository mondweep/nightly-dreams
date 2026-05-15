pub mod linear_f32;
pub mod linear_i8;
pub mod spec_search;

pub use linear_f32::LinearF32Index;
pub use linear_i8::LinearI8Index;
pub use spec_search::SpecSearch;

/// Common trait for all speculative-ANN index variants.
pub trait SpecAnnIndex: Send + Sync {
    fn insert(&mut self, id: usize, vec: &[f32]);
    fn build(&mut self);
    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)>;
    fn name(&self) -> &str;
    fn dim(&self) -> usize;
}

// ── shared scalar quantisation (symmetric int8) ──────────────────────────────

pub const QUANT_SCALE: f32 = 127.0;

#[inline]
pub fn quantize(v: &[f32]) -> Vec<i8> {
    v.iter()
        .map(|&x| (x * QUANT_SCALE).clamp(-127.0, 127.0) as i8)
        .collect()
}

/// Dequantised approximate dot: (Σ aᵢ·bᵢ) / scale²
#[inline]
pub fn dot_i8_approx(a: &[i8], b: &[i8]) -> f32 {
    let raw: i32 = a.iter().zip(b.iter()).map(|(x, y)| *x as i32 * *y as i32).sum();
    raw as f32 / (QUANT_SCALE * QUANT_SCALE)
}

#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
pub fn normalize(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-9 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

/// Retain top-k by descending score.
pub fn topk(mut scores: Vec<(usize, f32)>, k: usize) -> Vec<(usize, f32)> {
    scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(k);
    scores
}

// ── recall helper ─────────────────────────────────────────────────────────────

/// Recall@k: |approx ∩ ground_truth| / k
pub fn recall_at_k(approx: &[(usize, f32)], ground_truth: &[(usize, f32)]) -> f32 {
    let k = approx.len().min(ground_truth.len());
    if k == 0 {
        return 0.0;
    }
    let gt_ids: std::collections::HashSet<usize> =
        ground_truth.iter().take(k).map(|(id, _)| *id).collect();
    let hits = approx.iter().take(k).filter(|(id, _)| gt_ids.contains(id)).count();
    hits as f32 / k as f32
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn random_unit_vec(rng: &mut SmallRng, dim: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        normalize(&mut v);
        v
    }

    fn make_corpus(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..n).map(|_| random_unit_vec(&mut rng, dim)).collect()
    }

    fn test_variant(mut index: Box<dyn SpecAnnIndex>, corpus: &[Vec<f32>], dim: usize) {
        for (id, v) in corpus.iter().enumerate() {
            index.insert(id, v);
        }
        index.build();
        assert_eq!(index.dim(), dim);

        // query = first doc → top-1 must be itself
        let results = index.search(&corpus[0], 1);
        assert!(!results.is_empty(), "{}: empty results", index.name());
        assert_eq!(results[0].0, 0, "{}: self not top-1", index.name());

        // top-k must be sorted descending
        let results10 = index.search(&corpus[0], 10);
        for w in results10.windows(2) {
            assert!(w[0].1 >= w[1].1, "{}: results not sorted", index.name());
        }
    }

    #[test]
    fn test_all_variants() {
        let dim = 64;
        let n = 200;
        let corpus = make_corpus(n, dim, 42);

        test_variant(Box::new(LinearF32Index::new(dim)), &corpus, dim);
        test_variant(Box::new(LinearI8Index::new(dim)), &corpus, dim);
        test_variant(
            Box::new(SpecSearch::new(dim, 4)),
            &corpus,
            dim,
        );
    }

    #[test]
    fn test_recall_spec_vs_linear() {
        let dim = 64;
        let n = 500;
        let corpus = make_corpus(n, dim, 99);

        let mut gt_idx = LinearF32Index::new(dim);
        let mut spec_idx = SpecSearch::new(dim, 5);

        for (id, v) in corpus.iter().enumerate() {
            gt_idx.insert(id, v);
            spec_idx.insert(id, v);
        }
        gt_idx.build();
        spec_idx.build();

        let mut rng = SmallRng::seed_from_u64(7);
        let mut total_recall = 0.0f32;
        let queries = 50;
        for _ in 0..queries {
            let q = (0..dim)
                .map(|_| rng.gen_range(-1.0f32..1.0))
                .collect::<Vec<_>>();
            let mut q = q;
            normalize(&mut q);
            let gt = gt_idx.search(&q, 10);
            let sp = spec_idx.search(&q, 10);
            total_recall += recall_at_k(&sp, &gt);
        }
        let avg_recall = total_recall / queries as f32;
        // SpecSearch with overdraft=5 should achieve ≥80% recall@10
        assert!(
            avg_recall >= 0.80,
            "recall@10 too low: {:.3}",
            avg_recall
        );
    }

    #[test]
    fn test_quantize_roundtrip() {
        let v = vec![0.5f32, -0.3, 1.0, -1.0, 0.0];
        let q = quantize(&v);
        for (orig, qi) in v.iter().zip(q.iter()) {
            let restored = *qi as f32 / QUANT_SCALE;
            assert!((orig - restored).abs() < 0.01, "quant error too large");
        }
    }
}
