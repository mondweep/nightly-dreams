use crate::encoder::{BitVector, RaBitQEncoder};
use crate::{l2_distance, VectorIndex};

/// Exact brute-force flat index over f32 vectors. Baseline for recall comparison.
pub struct FlatStore {
    ids: Vec<usize>,
    vecs: Vec<Vec<f32>>,
    dim: usize,
}

impl FlatStore {
    pub fn new(dim: usize) -> Self {
        FlatStore { ids: Vec::new(), vecs: Vec::new(), dim }
    }
}

impl VectorIndex for FlatStore {
    fn insert(&mut self, id: usize, vector: &[f32]) {
        assert_eq!(vector.len(), self.dim);
        self.ids.push(id);
        self.vecs.push(vector.to_vec());
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut dists: Vec<(usize, f32)> = self
            .vecs
            .iter()
            .enumerate()
            .map(|(pos, v)| (self.ids[pos], l2_distance(query, v)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k);
        dists
    }

    fn memory_bytes(&self) -> usize {
        self.vecs.len() * self.dim * std::mem::size_of::<f32>()
    }
}

/// 1-bit RaBitQ flat index. Optional two-phase search: bit-scan then exact rescore.
///
/// Phase 1 – Hamming scan: encodes the query, computes estimated L2 for every stored
///           code word in O(D/64) per vector via popcount.
/// Phase 2 – Exact rescore (optional): retrieves `rescore_k` candidates, rescores
///           with true f32 L2 distance, returns top-k.
pub struct BitFlatStore {
    ids: Vec<usize>,
    encoded: Vec<BitVector>,
    /// Kept only when `rescore == true`; parallel to `ids`/`encoded`.
    originals: Vec<Vec<f32>>,
    encoder: RaBitQEncoder,
    dim: usize,
    rescore: bool,
    rescore_k: usize,
}

impl BitFlatStore {
    pub fn new(dim: usize, rescore: bool, rescore_k: usize) -> Self {
        let encoder = RaBitQEncoder::new(dim, 0xdeadbeef);
        BitFlatStore {
            ids: Vec::new(),
            encoded: Vec::new(),
            originals: Vec::new(),
            encoder,
            dim,
            rescore,
            rescore_k,
        }
    }

    pub fn compression_ratio(&self, baseline_bytes: usize) -> f32 {
        let my_bytes = self.memory_bytes();
        if my_bytes == 0 {
            return 0.0;
        }
        baseline_bytes as f32 / my_bytes as f32
    }
}

impl VectorIndex for BitFlatStore {
    fn insert(&mut self, id: usize, vector: &[f32]) {
        assert_eq!(vector.len(), self.dim);
        self.ids.push(id);
        self.encoded.push(self.encoder.encode(vector));
        if self.rescore {
            self.originals.push(vector.to_vec());
        }
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let q_bits = self.encoder.encode(query);
        let candidate_k = if self.rescore { self.rescore_k.max(k) } else { k };

        // Phase 1: rank all codes by estimated L2 (positions into ids/encoded/originals)
        let mut dists: Vec<(usize, f32)> = self
            .encoded
            .iter()
            .enumerate()
            .map(|(pos, bv)| (pos, RaBitQEncoder::estimated_l2_sq(&q_bits, bv)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(candidate_k);

        if self.rescore {
            // Phase 2: exact L2 rescore of top candidates
            let mut rescored: Vec<(usize, f32)> = dists
                .iter()
                .map(|(pos, _)| {
                    let id = self.ids[*pos];
                    let dist = l2_distance(query, &self.originals[*pos]);
                    (id, dist)
                })
                .collect();
            rescored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            rescored.truncate(k);
            rescored
        } else {
            dists
                .iter()
                .take(k)
                .map(|(pos, dist)| (self.ids[*pos], *dist))
                .collect()
        }
    }

    fn memory_bytes(&self) -> usize {
        // Each BitVector: n_words * 8 bytes (bits) + 4 bytes (norm) + 8 bytes (dim)
        let n_words = (self.dim + 63) / 64;
        let bits_bytes = self.encoded.len() * (n_words * 8 + 4 + 8);
        let orig_bytes = if self.rescore {
            self.originals.len() * self.dim * std::mem::size_of::<f32>()
        } else {
            0
        };
        bits_bytes + orig_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    fn random_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0f32)).collect())
            .collect()
    }

    fn recall_at_k(pred: &[(usize, f32)], gt: &[(usize, f32)], k: usize) -> f32 {
        let gt_ids: std::collections::HashSet<usize> =
            gt.iter().take(k).map(|(id, _)| *id).collect();
        let pred_ids: std::collections::HashSet<usize> =
            pred.iter().take(k).map(|(id, _)| *id).collect();
        gt_ids.intersection(&pred_ids).count() as f32 / k as f32
    }

    #[test]
    fn flat_returns_exact_nearest_neighbor() {
        let vecs = random_vecs(200, 32, 1);
        let mut store = FlatStore::new(32);
        for (i, v) in vecs.iter().enumerate() {
            store.insert(i, v);
        }
        let results = store.search(&vecs[0], 1);
        assert_eq!(results[0].0, 0, "exact NN must be the query itself");
        assert!(results[0].1 < 1e-5, "distance to self must be ~0");
    }

    #[test]
    fn bit_flat_rescore_achieves_reasonable_recall() {
        let vecs = random_vecs(500, 64, 7);
        let mut flat = FlatStore::new(64);
        let mut bit = BitFlatStore::new(64, true, 64);
        for (i, v) in vecs.iter().enumerate() {
            flat.insert(i, v);
            bit.insert(i, v);
        }
        let mut total = 0.0f32;
        let n_q = 30;
        for qi in 0..n_q {
            let q = &vecs[qi * 13 % 500];
            let gt = flat.search(q, 10);
            let ap = bit.search(q, 10);
            total += recall_at_k(&ap, &gt, 10);
        }
        let avg = total / n_q as f32;
        assert!(avg >= 0.25, "recall@10 too low: {avg:.3} (expected ≥0.25)");
    }

    #[test]
    fn bit_flat_memory_less_than_flat() {
        let vecs = random_vecs(1000, 128, 3);
        let mut flat = FlatStore::new(128);
        let mut bit = BitFlatStore::new(128, false, 0);
        for (i, v) in vecs.iter().enumerate() {
            flat.insert(i, v);
            bit.insert(i, v);
        }
        assert!(
            bit.memory_bytes() < flat.memory_bytes(),
            "bit store must use less memory than flat f32 store"
        );
    }
}
