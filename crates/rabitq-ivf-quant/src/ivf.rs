use crate::encoder::{BitVector, RaBitQEncoder};
use crate::l2_distance;

/// IVF (Inverted File) index with RaBitQ 1-bit quantization per cell.
///
/// Build pipeline:
///   1. Run Lloyd's k-means on the full corpus to get `nlist` centroids.
///   2. Assign each vector to its nearest centroid cell.
///   3. Encode each cell's vectors with RaBitQ.
///
/// Query pipeline:
///   1. Find the `nprobe` nearest centroids (by exact L2).
///   2. Scan those cells using estimated L2 from bit codes.
///   3. Rescore top `rescore_k` candidates with exact f32 L2 and return top-k.
pub struct IvfBitStore {
    centroids: Vec<Vec<f32>>,
    /// Per cell: (global_index_into_all_vecs, bit_vector)
    cells: Vec<Vec<(usize, BitVector)>>,
    /// All original vectors, indexed positionally (parallel to insert order).
    all_ids: Vec<usize>,
    all_vecs: Vec<Vec<f32>>,
    encoder: RaBitQEncoder,
    dim: usize,
    nlist: usize,
    nprobe: usize,
    rescore_k: usize,
}

impl IvfBitStore {
    pub fn new(dim: usize, nlist: usize, nprobe: usize, rescore_k: usize) -> Self {
        IvfBitStore {
            centroids: Vec::new(),
            cells: Vec::new(),
            all_ids: Vec::new(),
            all_vecs: Vec::new(),
            encoder: RaBitQEncoder::new(dim, 0xdeadbeef),
            dim,
            nlist,
            nprobe,
            rescore_k,
        }
    }

    /// Bulk-build the index from a slice of (id, vector) pairs.
    /// Runs k-means, assigns vectors to cells, and encodes them.
    pub fn build(&mut self, vectors: &[(usize, Vec<f32>)]) {
        assert!(!vectors.is_empty(), "cannot build on empty corpus");
        let effective_nlist = self.nlist.min(vectors.len());

        // Store originals
        for (id, v) in vectors {
            self.all_ids.push(*id);
            self.all_vecs.push(v.clone());
        }

        // Initialize centroids from uniformly sampled points
        let step = vectors.len() / effective_nlist;
        let mut centroids: Vec<Vec<f32>> =
            (0..effective_nlist).map(|i| vectors[i * step].1.clone()).collect();

        // Lloyd's k-means (15 iterations)
        for _ in 0..15 {
            let assignments = self.assign(&self.all_vecs, &centroids);
            let updated = self.update_centroids(&self.all_vecs, &assignments, &centroids, effective_nlist);
            centroids = updated;
        }
        self.centroids = centroids;

        // Build cells
        self.cells = vec![Vec::new(); effective_nlist];
        let assignments = self.assign(&self.all_vecs, &self.centroids);
        for (pos, cell) in assignments.iter().enumerate() {
            let bv = self.encoder.encode(&self.all_vecs[pos]);
            self.cells[*cell].push((pos, bv));
        }
    }

    fn assign(&self, vecs: &[Vec<f32>], centroids: &[Vec<f32>]) -> Vec<usize> {
        vecs.iter()
            .map(|v| {
                centroids
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, l2_distance(v, c)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect()
    }

    fn update_centroids(
        &self,
        vecs: &[Vec<f32>],
        assignments: &[usize],
        old_centroids: &[Vec<f32>],
        nlist: usize,
    ) -> Vec<Vec<f32>> {
        let mut sums = vec![vec![0.0f32; self.dim]; nlist];
        let mut counts = vec![0usize; nlist];
        for (v, &cell) in vecs.iter().zip(assignments) {
            for (s, x) in sums[cell].iter_mut().zip(v.iter()) {
                *s += x;
            }
            counts[cell] += 1;
        }
        sums.iter()
            .enumerate()
            .map(|(i, s)| {
                if counts[i] > 0 {
                    s.iter().map(|x| x / counts[i] as f32).collect()
                } else {
                    // keep old centroid if cell is empty
                    old_centroids[i].clone()
                }
            })
            .collect()
    }

    fn nearest_centroids(&self, query: &[f32], n: usize) -> Vec<usize> {
        let mut dists: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, l2_distance(query, c)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(n);
        dists.iter().map(|(i, _)| *i).collect()
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.centroids.is_empty() {
            return Vec::new();
        }
        let nprobe = self.nprobe.min(self.centroids.len());
        let probe_cells = self.nearest_centroids(query, nprobe);
        let q_bits = self.encoder.encode(query);
        let candidate_k = self.rescore_k.max(k);

        let mut candidates: Vec<(usize, f32)> = probe_cells
            .iter()
            .flat_map(|&cell| &self.cells[cell])
            .map(|(pos, bv)| (*pos, RaBitQEncoder::estimated_l2_sq(&q_bits, bv)))
            .collect();

        // Deduplicate (a vector can end up in multiple probe cells if nprobe > 1)
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.dedup_by_key(|x| x.0);
        candidates.truncate(candidate_k);

        // Exact rescore
        let mut rescored: Vec<(usize, f32)> = candidates
            .iter()
            .map(|(pos, _)| {
                let id = self.all_ids[*pos];
                let dist = l2_distance(query, &self.all_vecs[*pos]);
                (id, dist)
            })
            .collect();
        rescored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        rescored.truncate(k);
        rescored
    }

    pub fn memory_bytes(&self) -> usize {
        let centroid_bytes = self.centroids.len() * self.dim * std::mem::size_of::<f32>();
        let n_words = (self.dim + 63) / 64;
        let cell_bytes: usize = self
            .cells
            .iter()
            .map(|c| c.len() * (n_words * 8 + 4 + 8))
            .sum();
        centroid_bytes + cell_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{flat::FlatStore, VectorIndex};
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
    fn ivf_builds_without_panic() {
        let vecs = random_vecs(300, 64, 99);
        let corpus: Vec<(usize, Vec<f32>)> = vecs.into_iter().enumerate().collect();
        let mut ivf = IvfBitStore::new(64, 10, 3, 30);
        ivf.build(&corpus);
        assert_eq!(ivf.centroids.len(), 10);
    }

    #[test]
    fn ivf_returns_k_results() {
        let vecs = random_vecs(400, 64, 55);
        let corpus: Vec<(usize, Vec<f32>)> = vecs.iter().cloned().enumerate().collect();
        let mut ivf = IvfBitStore::new(64, 8, 2, 32);
        ivf.build(&corpus);
        let results = ivf.search(&vecs[0], 10);
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn ivf_achieves_reasonable_recall() {
        let vecs = random_vecs(600, 64, 13);
        let corpus: Vec<(usize, Vec<f32>)> = vecs.iter().cloned().enumerate().collect();

        let mut flat = FlatStore::new(64);
        for (i, v) in vecs.iter().enumerate() {
            flat.insert(i, v);
        }
        let mut ivf = IvfBitStore::new(64, 15, 4, 40);
        ivf.build(&corpus);

        let mut total = 0.0f32;
        let n_q = 30;
        for qi in 0..n_q {
            let q = &vecs[qi * 17 % 600];
            let gt = flat.search(q, 10);
            let ap = ivf.search(q, 10);
            total += recall_at_k(&ap, &gt, 10);
        }
        let avg = total / n_q as f32;
        assert!(avg >= 0.25, "IVF recall@10 too low: {avg:.3} (expected ≥0.25)");
    }
}
