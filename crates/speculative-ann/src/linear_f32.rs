use crate::{dot_f32, topk, SpecAnnIndex};

/// Baseline: brute-force linear scan with full f32 dot products.
/// O(N·d) per query, exact results.  Ground-truth reference.
pub struct LinearF32Index {
    dim: usize,
    vecs: Vec<Vec<f32>>,
    ids: Vec<usize>,
}

impl LinearF32Index {
    pub fn new(dim: usize) -> Self {
        Self { dim, vecs: Vec::new(), ids: Vec::new() }
    }
}

impl SpecAnnIndex for LinearF32Index {
    fn insert(&mut self, id: usize, vec: &[f32]) {
        assert_eq!(vec.len(), self.dim, "dimension mismatch");
        self.ids.push(id);
        self.vecs.push(vec.to_vec());
    }

    fn build(&mut self) {
        // no-op: linear scan needs no index structure
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let scores: Vec<(usize, f32)> = self
            .vecs
            .iter()
            .zip(self.ids.iter())
            .map(|(v, &id)| (id, dot_f32(query, v)))
            .collect();
        topk(scores, k)
    }

    fn name(&self) -> &str {
        "linear-f32"
    }

    fn dim(&self) -> usize {
        self.dim
    }
}
