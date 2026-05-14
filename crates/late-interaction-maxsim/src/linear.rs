use crate::{maxsim, topk, MaxSimIndex, MultiVec};

/// Brute-force O(N·K·Q·D) MaxSim scan — ground-truth baseline.
pub struct LinearIndex {
    docs: Vec<(usize, MultiVec)>,
}

impl LinearIndex {
    pub fn new() -> Self {
        Self { docs: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.docs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }
}

impl Default for LinearIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl MaxSimIndex for LinearIndex {
    fn insert(&mut self, id: usize, doc: MultiVec) {
        self.docs.push((id, doc));
    }

    fn build(&mut self) {}

    fn search(&self, query: &MultiVec, k: usize) -> Vec<(usize, f32)> {
        let scores = self
            .docs
            .iter()
            .map(|(id, doc)| (*id, maxsim(query, doc)))
            .collect();
        topk(scores, k)
    }

    fn name(&self) -> &str {
        "linear"
    }
}
