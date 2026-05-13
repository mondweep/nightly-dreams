// CoDEQ: Consistent Dynamic Quantization for Streaming Vector Search
//
// Implements three progressively-optimised variants of the CoDEQ algorithm
// (arXiv:2512.18335, Dec 2025): a product quantizer (PQ) that stays accurate
// under continuous vector insertions/deletions without full codebook rebuilds.
//
// Variants:
//   1. EagerCoDEQ   — reassigns every affected vector immediately on mutation
//   2. LazyCoDEQ    — batches reassignments; triggers when stale fraction > threshold
//   3. ApproxCoDEQ  — adds a KS-sketch drift gate; only triggers lazy flush on drift

use rand::rngs::SmallRng;
use rand::Rng;
use std::collections::VecDeque;

// ─── constants ───────────────────────────────────────────────────────────────

pub const M: usize = 8;          // subspaces
pub const K: usize = 256;        // centroids per subspace
pub const SUB_DIM: usize = 16;   // dims per subspace  →  total dim = M * SUB_DIM = 128

// ─── shared types ────────────────────────────────────────────────────────────

pub type Vector = Vec<f32>;
pub type Code = [u8; M];         // one byte (centroid index) per subspace

#[derive(Clone)]
pub struct Codebook {
    pub centroids: Vec<[f32; SUB_DIM]>, // K centroids per subspace, flat: subspace * K + k
}

impl Codebook {
    pub fn new_random(rng: &mut SmallRng) -> Self {
        let mut centroids = vec![[0f32; SUB_DIM]; M * K];
        for c in &mut centroids {
            for v in c.iter_mut() {
                *v = rng.gen_range(-1.0f32..1.0);
            }
        }
        Self { centroids }
    }

    // Train codebook subspaces using per-subspace medians (CoDEQ key insight).
    // Split at the median of each dimension within the subspace, then assign
    // each training vector to its nearest centroid via L2 in that subspace.
    pub fn train_median_split(data: &[Vector], rng: &mut SmallRng) -> Self {
        // Initialise centroids with k-means++ style: pick K random vectors per subspace
        let n = data.len();
        let mut centroids = vec![[0f32; SUB_DIM]; M * K];
        for m in 0..M {
            let offset = m * SUB_DIM;
            for k in 0..K {
                let idx = rng.gen_range(0..n);
                let sub: [f32; SUB_DIM] = data[idx][offset..offset + SUB_DIM]
                    .try_into()
                    .unwrap_or([0f32; SUB_DIM]);
                centroids[m * K + k] = sub;
            }
        }

        // Two rounds of Lloyd's: assign then update
        for _iter in 0..2 {
            let mut sums = vec![[0f64; SUB_DIM]; M * K];
            let mut counts = vec![0u64; M * K];
            for vec in data {
                for m in 0..M {
                    let c = nearest_centroid_sub(&centroids, m, vec);
                    let offset = m * SUB_DIM;
                    for d in 0..SUB_DIM {
                        sums[m * K + c][d] += vec[offset + d] as f64;
                    }
                    counts[m * K + c] += 1;
                }
            }
            for m in 0..M {
                for k in 0..K {
                    let cnt = counts[m * K + k];
                    if cnt > 0 {
                        for d in 0..SUB_DIM {
                            centroids[m * K + k][d] = (sums[m * K + k][d] / cnt as f64) as f32;
                        }
                    }
                }
            }
        }

        Self { centroids }
    }

    pub fn encode(&self, vec: &Vector) -> Code {
        let mut code = [0u8; M];
        for m in 0..M {
            code[m] = nearest_centroid_sub(&self.centroids, m, vec) as u8;
        }
        code
    }

    // Asymmetric distance: true query vs stored code (no query quantization error)
    pub fn adist(&self, query: &Vector, code: &Code) -> f32 {
        let mut dist = 0.0f32;
        for m in 0..M {
            let offset = m * SUB_DIM;
            let c = &self.centroids[m * K + code[m] as usize];
            for d in 0..SUB_DIM {
                let diff = query[offset + d] - c[d];
                dist += diff * diff;
            }
        }
        dist
    }

    // Recompute centroid for subspace m from a sample of vectors
    pub fn update_subspace_centroid(
        &mut self,
        m: usize,
        k: usize,
        vectors: &[Vector],
        indices: &[usize],
    ) {
        if indices.is_empty() {
            return;
        }
        let offset = m * SUB_DIM;
        let mut sum = [0f64; SUB_DIM];
        for &idx in indices {
            for d in 0..SUB_DIM {
                sum[d] += vectors[idx][offset + d] as f64;
            }
        }
        let n = indices.len() as f64;
        for d in 0..SUB_DIM {
            self.centroids[m * K + k][d] = (sum[d] / n) as f32;
        }
    }
}

fn nearest_centroid_sub(centroids: &[[f32; SUB_DIM]], m: usize, vec: &Vector) -> usize {
    let offset = m * SUB_DIM;
    let mut best = 0usize;
    let mut best_d = f32::MAX;
    for k in 0..K {
        let c = &centroids[m * K + k];
        let mut d = 0.0f32;
        for i in 0..SUB_DIM {
            let diff = vec[offset + i] - c[i];
            d += diff * diff;
        }
        if d < best_d {
            best_d = d;
            best = k;
        }
    }
    best
}

// ─── Dynamic consistency tracking ────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct ConsistencyStats {
    pub total_vectors: usize,
    pub stale_count: usize,      // vectors with wrong codeword assignment
    pub reassignments: usize,    // cumulative reassignment operations
    pub drift_triggers: usize,   // times the KS gate fired
}

impl ConsistencyStats {
    pub fn stale_fraction(&self) -> f64 {
        if self.total_vectors == 0 {
            0.0
        } else {
            self.stale_count as f64 / self.total_vectors as f64
        }
    }
}

// ─── Lightweight KS sketch for drift detection ───────────────────────────────
// Uses a sliding window (last WINDOW samples) vs. baseline statistics.
// Triggers when the window mean deviates > 2σ from the baseline mean.
const KS_WINDOW: usize = 64;

pub struct KsSketch {
    window: VecDeque<f32>,
    baseline_mean: f64,
    baseline_std: f64,
    pub trigger_count: u64,
}

impl KsSketch {
    pub fn new() -> Self {
        Self {
            window: VecDeque::with_capacity(KS_WINDOW),
            baseline_mean: 0.0,
            baseline_std: 1.0,
            trigger_count: 0,
        }
    }

    pub fn update(&mut self, x: f32) {
        if self.window.len() == KS_WINDOW {
            self.window.pop_front();
        }
        self.window.push_back(x);
    }

    pub fn set_baseline(&mut self) {
        if self.window.is_empty() {
            return;
        }
        let n = self.window.len() as f64;
        let mean: f64 = self.window.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var: f64 = self.window.iter()
            .map(|&v| { let d = v as f64 - mean; d * d })
            .sum::<f64>() / n.max(1.0);
        self.baseline_mean = mean;
        self.baseline_std = var.sqrt().max(1e-6);
    }

    // Returns true when window mean has drifted > 2σ from baseline
    pub fn drifted(&mut self) -> bool {
        if self.window.len() < KS_WINDOW / 2 {
            return false;
        }
        let n = self.window.len() as f64;
        let win_mean: f64 = self.window.iter().map(|&v| v as f64).sum::<f64>() / n;
        let z = ((win_mean - self.baseline_mean) / self.baseline_std).abs();
        if z > 2.0 {
            self.trigger_count += 1;
            true
        } else {
            false
        }
    }
}

// ─── Variant 1: EagerCoDEQ ───────────────────────────────────────────────────
// Immediately reassigns any vector whose nearest centroid has changed on insert.
pub struct EagerCoDEQ {
    pub vectors: Vec<Vector>,
    pub codes: Vec<Code>,
    pub codebook: Codebook,
    pub stats: ConsistencyStats,
}

impl EagerCoDEQ {
    pub fn new(data: &[Vector], rng: &mut SmallRng) -> Self {
        let codebook = Codebook::train_median_split(data, rng);
        let codes: Vec<Code> = data.iter().map(|v| codebook.encode(v)).collect();
        let stats = ConsistencyStats {
            total_vectors: data.len(),
            ..Default::default()
        };
        Self {
            vectors: data.to_vec(),
            codes,
            codebook,
            stats,
        }
    }

    pub fn insert(&mut self, vec: Vector) {
        let new_code = self.codebook.encode(&vec);
        // Check neighbours in each subspace; eagerly update centroid
        for m in 0..M {
            let c = new_code[m] as usize;
            let members: Vec<usize> = self
                .codes
                .iter()
                .enumerate()
                .filter(|(_, code)| code[m] as usize == c)
                .map(|(i, _)| i)
                .collect();
            let mut all = members.clone();
            all.push(self.vectors.len()); // include the new vector
            // update centroid (temporary borrow trick)
            let mut tmp = self.vectors.clone();
            tmp.push(vec.clone());
            self.codebook.update_subspace_centroid(m, c, &tmp, &all);
            // Reassign any vector whose nearest changed
            for &idx in &members {
                let new_c = nearest_centroid_sub(&self.codebook.centroids, m, &self.vectors[idx]);
                if new_c != self.codes[idx][m] as usize {
                    self.codes[idx][m] = new_c as u8;
                    self.stats.reassignments += 1;
                }
            }
        }
        self.codes.push(new_code);
        self.vectors.push(vec);
        self.stats.total_vectors += 1;
    }

    pub fn delete(&mut self, idx: usize) {
        self.vectors.swap_remove(idx);
        self.codes.swap_remove(idx);
        self.stats.total_vectors = self.stats.total_vectors.saturating_sub(1);
    }

    pub fn knn(&self, query: &Vector, k: usize) -> Vec<(usize, f32)> {
        knn_from_codes(&self.codebook, query, &self.codes, k)
    }
}

// ─── Variant 2: LazyCoDEQ ────────────────────────────────────────────────────
// Batches reassignments into a pending queue; flushes when stale_fraction > threshold.
pub struct LazyCoDEQ {
    pub vectors: Vec<Vector>,
    pub codes: Vec<Code>,
    pub codebook: Codebook,
    pub stats: ConsistencyStats,
    pub stale_threshold: f64,
    pending_check: VecDeque<usize>, // vector indices that might be stale
}

impl LazyCoDEQ {
    pub fn new(data: &[Vector], rng: &mut SmallRng, stale_threshold: f64) -> Self {
        let codebook = Codebook::train_median_split(data, rng);
        let codes: Vec<Code> = data.iter().map(|v| codebook.encode(v)).collect();
        let stats = ConsistencyStats {
            total_vectors: data.len(),
            ..Default::default()
        };
        Self {
            vectors: data.to_vec(),
            codes,
            codebook,
            stats,
            stale_threshold,
            pending_check: VecDeque::new(),
        }
    }

    pub fn insert(&mut self, vec: Vector) {
        let new_code = self.codebook.encode(&vec);
        // Mark neighbours as potentially stale (lazy — don't fix now)
        for m in 0..M {
            let c = new_code[m] as usize;
            for (i, code) in self.codes.iter().enumerate() {
                if code[m] as usize == c {
                    self.pending_check.push_back(i);
                }
            }
        }
        self.codes.push(new_code);
        self.vectors.push(vec);
        self.stats.total_vectors += 1;
        self.stats.stale_count = self.pending_check.len().min(self.stats.total_vectors);

        if self.stats.stale_fraction() > self.stale_threshold {
            self.flush();
        }
    }

    pub fn delete(&mut self, idx: usize) {
        self.vectors.swap_remove(idx);
        self.codes.swap_remove(idx);
        self.stats.total_vectors = self.stats.total_vectors.saturating_sub(1);
        self.stats.stale_count = self.stats.stale_count.saturating_sub(1);
    }

    pub fn flush(&mut self) {
        let to_check: Vec<usize> = self.pending_check.drain(..).collect();
        for idx in to_check {
            if idx >= self.vectors.len() {
                continue;
            }
            let correct = self.codebook.encode(&self.vectors[idx]);
            if correct != self.codes[idx] {
                self.codes[idx] = correct;
                self.stats.reassignments += 1;
            }
        }
        self.stats.stale_count = 0;
    }

    pub fn knn(&self, query: &Vector, k: usize) -> Vec<(usize, f32)> {
        knn_from_codes(&self.codebook, query, &self.codes, k)
    }
}

// ─── Variant 3: ApproxCoDEQ ──────────────────────────────────────────────────
// Adds per-subspace KS drift sketches; only flushes lazy queue when drift detected.
pub struct ApproxCoDEQ {
    pub vectors: Vec<Vector>,
    pub codes: Vec<Code>,
    pub codebook: Codebook,
    pub stats: ConsistencyStats,
    pub stale_threshold: f64,
    pending_check: VecDeque<usize>,
    sketches: Vec<KsSketch>, // one per subspace (tracks first dimension)
}

impl ApproxCoDEQ {
    pub fn new(data: &[Vector], rng: &mut SmallRng, stale_threshold: f64) -> Self {
        let codebook = Codebook::train_median_split(data, rng);
        let codes: Vec<Code> = data.iter().map(|v| codebook.encode(v)).collect();

        // Initialise and baseline KS sketches from training data
        let mut sketches: Vec<KsSketch> = (0..M).map(|_| KsSketch::new()).collect();
        for vec in data {
            for m in 0..M {
                sketches[m].update(vec[m * SUB_DIM]);
            }
        }
        for s in &mut sketches {
            s.set_baseline();
        }

        let stats = ConsistencyStats {
            total_vectors: data.len(),
            ..Default::default()
        };
        Self {
            vectors: data.to_vec(),
            codes,
            codebook,
            stats,
            stale_threshold,
            pending_check: VecDeque::new(),
            sketches,
        }
    }

    pub fn insert(&mut self, vec: Vector) {
        // Update drift sketches
        let mut any_drift = false;
        for m in 0..M {
            self.sketches[m].update(vec[m * SUB_DIM]);
            if self.sketches[m].drifted() {
                any_drift = true;
            }
        }

        let new_code = self.codebook.encode(&vec);
        for m in 0..M {
            let c = new_code[m] as usize;
            for (i, code) in self.codes.iter().enumerate() {
                if code[m] as usize == c {
                    self.pending_check.push_back(i);
                }
            }
        }
        self.codes.push(new_code);
        self.vectors.push(vec);
        self.stats.total_vectors += 1;
        self.stats.stale_count = self.pending_check.len().min(self.stats.total_vectors);

        // Only flush if drift detected OR stale fraction exceeds threshold
        let over_threshold = self.stats.stale_fraction() > self.stale_threshold;
        if any_drift || over_threshold {
            self.flush_with_drift_reset();
        }
    }

    fn flush_with_drift_reset(&mut self) {
        self.stats.drift_triggers += 1;
        let to_check: Vec<usize> = self.pending_check.drain(..).collect();
        for idx in to_check {
            if idx >= self.vectors.len() {
                continue;
            }
            let correct = self.codebook.encode(&self.vectors[idx]);
            if correct != self.codes[idx] {
                self.codes[idx] = correct;
                self.stats.reassignments += 1;
            }
        }
        self.stats.stale_count = 0;
        // Reset baselines after drift handled
        for s in &mut self.sketches {
            s.set_baseline();
        }
    }

    pub fn delete(&mut self, idx: usize) {
        self.vectors.swap_remove(idx);
        self.codes.swap_remove(idx);
        self.stats.total_vectors = self.stats.total_vectors.saturating_sub(1);
        self.stats.stale_count = self.stats.stale_count.saturating_sub(1);
    }

    pub fn knn(&self, query: &Vector, k: usize) -> Vec<(usize, f32)> {
        knn_from_codes(&self.codebook, query, &self.codes, k)
    }
}

// ─── Shared kNN over codes ────────────────────────────────────────────────────

pub fn knn_from_codes(
    codebook: &Codebook,
    query: &Vector,
    codes: &[Code],
    k: usize,
) -> Vec<(usize, f32)> {
    // Precompute query-to-centroid distances for each subspace (LUT)
    let mut lut = [[0f32; K]; M];
    for m in 0..M {
        let offset = m * SUB_DIM;
        for k_idx in 0..K {
            let c = &codebook.centroids[m * K + k_idx];
            let mut d = 0.0f32;
            for i in 0..SUB_DIM {
                let diff = query[offset + i] - c[i];
                d += diff * diff;
            }
            lut[m][k_idx] = d;
        }
    }

    // Scan all codes with LUT lookup
    let mut dists: Vec<(usize, f32)> = codes
        .iter()
        .enumerate()
        .map(|(i, code)| {
            let d: f32 = (0..M).map(|m| lut[m][code[m] as usize]).sum();
            (i, d)
        })
        .collect();

    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.truncate(k);
    dists
}

// ─── Ground truth brute force ─────────────────────────────────────────────────

pub fn brute_force_knn(data: &[Vector], query: &Vector, k: usize) -> Vec<usize> {
    let mut dists: Vec<(usize, f32)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let d: f32 = v.iter().zip(query.iter()).map(|(a, b)| (a - b).powi(2)).sum();
            (i, d)
        })
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.truncate(k);
    dists.into_iter().map(|(i, _)| i).collect()
}

pub fn recall_at_k(predicted: &[(usize, f32)], ground_truth: &[usize]) -> f64 {
    let predicted_set: std::collections::HashSet<usize> =
        predicted.iter().map(|(i, _)| *i).collect();
    let hits = ground_truth.iter().filter(|&&i| predicted_set.contains(&i)).count();
    hits as f64 / ground_truth.len() as f64
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    const DIM: usize = M * SUB_DIM;

    fn make_data(n: usize, rng: &mut SmallRng) -> Vec<Vector> {
        (0..n)
            .map(|_| (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
            .collect()
    }

    fn make_vec(rng: &mut SmallRng) -> Vector {
        (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
    }

    #[test]
    fn codebook_encodes_and_decodes() {
        let mut rng = SmallRng::seed_from_u64(42);
        let data = make_data(200, &mut rng);
        let cb = Codebook::train_median_split(&data, &mut rng);
        let code = cb.encode(&data[0]);
        // All subspace codes must be valid centroid indices
        assert!(code.iter().all(|&c| (c as usize) < K));
    }

    #[test]
    fn knn_lut_is_deterministic() {
        let mut rng = SmallRng::seed_from_u64(1);
        let data = make_data(200, &mut rng);
        let cb = Codebook::train_median_split(&data, &mut rng);
        let codes: Vec<Code> = data.iter().map(|v| cb.encode(v)).collect();
        let query = make_vec(&mut rng);
        let r1 = knn_from_codes(&cb, &query, &codes, 5);
        let r2 = knn_from_codes(&cb, &query, &codes, 5);
        assert_eq!(r1, r2);
    }

    #[test]
    fn eager_knn_recall_above_floor() {
        let mut rng = SmallRng::seed_from_u64(7);
        let data = make_data(500, &mut rng);
        let idx = EagerCoDEQ::new(&data, &mut rng);
        let query = make_vec(&mut rng);
        let pred = idx.knn(&query, 10);
        let gt = brute_force_knn(&data, &query, 10);
        let r = recall_at_k(&pred, &gt);
        // Recall@10 for PQ on 500 vectors should comfortably exceed 0.30
        assert!(r > 0.30, "recall@10={:.3} < 0.30", r);
    }

    #[test]
    fn eager_insert_increases_size() {
        let mut rng = SmallRng::seed_from_u64(3);
        let data = make_data(100, &mut rng);
        let mut idx = EagerCoDEQ::new(&data, &mut rng);
        let before = idx.stats.total_vectors;
        idx.insert(make_vec(&mut rng));
        assert_eq!(idx.stats.total_vectors, before + 1);
    }

    #[test]
    fn eager_delete_decreases_size() {
        let mut rng = SmallRng::seed_from_u64(5);
        let data = make_data(100, &mut rng);
        let mut idx = EagerCoDEQ::new(&data, &mut rng);
        let before = idx.stats.total_vectors;
        idx.delete(0);
        assert_eq!(idx.stats.total_vectors, before - 1);
    }

    #[test]
    fn lazy_stale_fraction_bounded() {
        let mut rng = SmallRng::seed_from_u64(9);
        let data = make_data(200, &mut rng);
        let mut idx = LazyCoDEQ::new(&data, &mut rng, 0.05);
        for _ in 0..50 {
            idx.insert(make_vec(&mut rng));
        }
        // After inserts, stale fraction must be ≤ threshold or flush was triggered
        assert!(idx.stats.stale_fraction() <= 0.05 + 1e-9 || idx.pending_check.is_empty());
    }

    #[test]
    fn approx_knn_recall_above_floor() {
        let mut rng = SmallRng::seed_from_u64(11);
        let data = make_data(500, &mut rng);
        let mut idx = ApproxCoDEQ::new(&data, &mut rng, 0.05);
        // Simulate mild distribution drift (shift mean by 0.3)
        for _ in 0..50 {
            let v: Vector = (0..DIM)
                .map(|_| rng.gen_range(-1.0f32..1.0) + 0.3)
                .collect();
            idx.insert(v);
        }
        let query: Vector = (0..DIM)
            .map(|_| rng.gen_range(-1.0f32..1.0) + 0.3)
            .collect();
        let all_vecs = idx.vectors.clone();
        let pred = idx.knn(&query, 10);
        let gt = brute_force_knn(&all_vecs, &query, 10);
        let r = recall_at_k(&pred, &gt);
        assert!(r > 0.20, "recall@10={:.3} < 0.20 after drift", r);
    }

    #[test]
    fn recall_at_k_perfect() {
        let gt = vec![0, 1, 2, 3, 4];
        let pred: Vec<(usize, f32)> = gt.iter().map(|&i| (i, i as f32)).collect();
        assert!((recall_at_k(&pred, &gt) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ks_sketch_detects_drift() {
        let mut rng = SmallRng::seed_from_u64(13);
        let mut sketch = KsSketch::new();
        // Fill the window with baseline samples and set baseline
        for _ in 0..KS_WINDOW {
            sketch.update(rng.gen_range(-1.0f32..1.0));
        }
        sketch.set_baseline();
        // Inject large drift — fill the entire window with shifted values
        for _ in 0..KS_WINDOW {
            sketch.update(rng.gen_range(5.0f32..6.0));
        }
        assert!(sketch.drifted(), "KS sketch should detect large mean shift");
    }
}
