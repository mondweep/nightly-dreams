use crate::{dot_i8_approx, quantize, topk, SpecAnnIndex};

/// Draft index: brute-force linear scan with int8-quantised dot products.
/// ~4× faster arithmetic throughput than f32 on modern CPUs.
/// Results are approximate (quantisation error ~±1/127 per dimension).
pub struct LinearI8Index {
    dim: usize,
    vecs: Vec<Vec<i8>>,
    ids: Vec<usize>,
}

impl LinearI8Index {
    pub fn new(dim: usize) -> Self {
        Self { dim, vecs: Vec::new(), ids: Vec::new() }
    }
}

impl SpecAnnIndex for LinearI8Index {
    fn insert(&mut self, id: usize, vec: &[f32]) {
        assert_eq!(vec.len(), self.dim, "dimension mismatch");
        self.ids.push(id);
        self.vecs.push(quantize(vec));
    }

    fn build(&mut self) {
        // no-op: quantisation happens at insert time
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let qq = quantize(query);
        let scores: Vec<(usize, f32)> = self
            .vecs
            .iter()
            .zip(self.ids.iter())
            .map(|(v, &id)| (id, dot_i8_approx(&qq, v)))
            .collect();
        topk(scores, k)
    }

    fn name(&self) -> &str {
        "linear-i8"
    }

    fn dim(&self) -> usize {
        self.dim
    }
}
