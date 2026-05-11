//! PDX Columnar Layout for SIMD-accelerated vector distance computation.
//!
//! Based on: "PDX: Data Layout for Vector Similarity Search" (Vanderbilt et al.,
//! VLDB 2025 / arXiv 2501.05628). The key insight: storing vectors column-major
//! (transposed) turns ANN distance computation from a gather into a sequential
//! scan, enabling auto-vectorisation and reducing memory bandwidth waste.
//!
//! Layout comparison (n=4 vectors, d=4 dimensions):
//!   Row-major:    [v0d0 v0d1 v0d2 v0d3 | v1d0 v1d1 v1d2 v1d3 | ...]
//!   Column-major: [v0d0 v1d0 v2d0 v3d0 | v0d1 v1d1 v2d1 v3d1 | ...]
//!
//! When computing distance(query, candidates[0..n]) dimension-by-dimension,
//! the column-major layout accesses each dimension block sequentially — the
//! CPU can issue wide SIMD loads and the prefetcher stays ahead.

/// Trait every distance metric must implement.
pub trait Distance: Send + Sync {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
    /// Distance of query[0..dim] against a column-major block.
    /// `col_block[i]` == the value of dimension `d` for candidate `i`.
    /// Default: scalar loop; override with SIMD for each concrete metric.
    fn distance_columnar(
        &self,
        query: &[f32],
        col_block: &[&[f32]],
        n: usize,
    ) -> Vec<f32> {
        let mut acc = vec![0.0f32; n];
        for (d, qd) in query.iter().enumerate() {
            let col = col_block[d];
            for i in 0..n {
                let diff = qd - col[i];
                acc[i] += diff * diff;
            }
        }
        acc
    }
}

/// Squared Euclidean distance.
pub struct L2Sq;
impl Distance for L2Sq {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
    }

    fn distance_columnar(
        &self,
        query: &[f32],
        col_block: &[&[f32]],
        n: usize,
    ) -> Vec<f32> {
        let mut acc = vec![0.0f32; n];
        let dim = query.len();
        for d in 0..dim {
            let qd = query[d];
            let col = col_block[d];
            // The inner loop is the hot path.  rustc + LLVM auto-vectorise
            // this into AVX2 (8×f32) or AVX-512 (16×f32) on x86-64 with
            // -C target-cpu=native.  We deliberately keep it branch-free
            // so the compiler can emit FMA instructions.
            for i in 0..n {
                let diff = qd - col[i];
                acc[i] += diff * diff;
            }
        }
        acc
    }
}

/// Negative inner product (for unit-norm vectors, equivalent to cosine distance).
pub struct NegDot;
impl Distance for NegDot {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        1.0 - a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
    }

    fn distance_columnar(
        &self,
        query: &[f32],
        col_block: &[&[f32]],
        n: usize,
    ) -> Vec<f32> {
        let mut dot = vec![0.0f32; n];
        let dim = query.len();
        for d in 0..dim {
            let qd = query[d];
            let col = col_block[d];
            for i in 0..n {
                dot[i] += qd * col[i];
            }
        }
        dot.iter_mut().for_each(|x| *x = 1.0 - *x);
        dot
    }
}

// ---------------------------------------------------------------------------
// Row-major flat index (baseline)
// ---------------------------------------------------------------------------

/// Row-major brute-force flat index.  Standard layout: each vector is stored
/// contiguously.  Used as baseline to compare against `ColumnarIndex`.
pub struct RowMajorIndex {
    data: Vec<f32>,
    ids: Vec<u64>,
    n: usize,
    dim: usize,
}

impl RowMajorIndex {
    pub fn new(dim: usize) -> Self {
        Self { data: Vec::new(), ids: Vec::new(), n: 0, dim }
    }

    pub fn insert(&mut self, id: u64, v: &[f32]) {
        assert_eq!(v.len(), self.dim);
        self.data.extend_from_slice(v);
        self.ids.push(id);
        self.n += 1;
    }

    pub fn len(&self) -> usize { self.n }
    pub fn is_empty(&self) -> bool { self.n == 0 }

    /// Brute-force k-NN.  Row-major: inner loop strides by `dim`.
    pub fn search(&self, query: &[f32], k: usize, dist: &dyn Distance) -> Vec<(u64, f32)> {
        assert_eq!(query.len(), self.dim);
        let mut heap: Vec<(ordered_float::OrderedFloat<f32>, u64)> = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let v = &self.data[i * self.dim..(i + 1) * self.dim];
            heap.push((ordered_float::OrderedFloat(dist.distance(query, v)), self.ids[i]));
        }
        heap.sort_unstable();
        heap.into_iter().take(k).map(|(d, id)| (id, d.0)).collect()
    }
}

// ---------------------------------------------------------------------------
// Column-major flat index (PDX layout)
// ---------------------------------------------------------------------------

/// Columnar (PDX) brute-force flat index.
///
/// Memory layout (n vectors, d dimensions):
///   columns[j][i]  ==  vector i, dimension j   (j-th column, i-th row)
///
/// Each column is a contiguous slice of length n, so a dimension-wise scan
/// across all candidates is a sequential memory read — ideal for SIMD.
///
/// Insertion appends to each column individually (O(d) allocations amortised).
/// For the PoC we use Vec<Vec<f32>>; a production layout would use a single
/// allocation with column pointers (saves one pointer-chase per dimension).
pub struct ColumnarIndex {
    columns: Vec<Vec<f32>>,  // columns[d][i] = vector i, dimension d
    ids: Vec<u64>,
    n: usize,
    dim: usize,
}

impl ColumnarIndex {
    pub fn new(dim: usize) -> Self {
        Self {
            columns: (0..dim).map(|_| Vec::new()).collect(),
            ids: Vec::new(),
            n: 0,
            dim,
        }
    }

    pub fn insert(&mut self, id: u64, v: &[f32]) {
        assert_eq!(v.len(), self.dim);
        for (d, &vd) in v.iter().enumerate() {
            self.columns[d].push(vd);
        }
        self.ids.push(id);
        self.n += 1;
    }

    pub fn len(&self) -> usize { self.n }
    pub fn is_empty(&self) -> bool { self.n == 0 }

    /// Brute-force k-NN using the columnar distance kernel.
    pub fn search(&self, query: &[f32], k: usize, dist: &dyn Distance) -> Vec<(u64, f32)> {
        assert_eq!(query.len(), self.dim);
        let col_refs: Vec<&[f32]> = self.columns.iter().map(|c| c.as_slice()).collect();
        let dists = dist.distance_columnar(query, &col_refs, self.n);
        let mut heap: Vec<(ordered_float::OrderedFloat<f32>, u64)> =
            dists.into_iter().zip(self.ids.iter()).map(|(d, &id)| {
                (ordered_float::OrderedFloat(d), id)
            }).collect();
        heap.sort_unstable();
        heap.into_iter().take(k).map(|(d, id)| (id, d.0)).collect()
    }
}

// ---------------------------------------------------------------------------
// Chunked columnar index (PDX with fixed-width tiles)
// ---------------------------------------------------------------------------

/// Chunked columnar index partitions vectors into fixed-width tiles
/// (default 256 vectors per tile).  Within each tile the layout is columnar.
/// Benefits:
///   1. Tile fits in L1/L2 cache → warm reuse across dimensions.
///   2. Early tile pruning: after computing partial distances over the first
///      few dimensions, skip tiles where the lower-bound already exceeds
///      the current k-th best distance (branch-free guard with a threshold).
///   3. Prefetcher efficiency: tile boundary jumps are predictable.
pub struct ChunkedColumnarIndex {
    tiles: Vec<ColumnarTile>,
    ids: Vec<u64>,
    n: usize,
    pub dim: usize,
    pub tile_size: usize,
}

struct ColumnarTile {
    columns: Vec<Vec<f32>>, // columns[d][i] — columnar within this tile
    count: usize,
    start_idx: usize,       // index into the global ids vec
}

impl ChunkedColumnarIndex {
    pub fn new(dim: usize, tile_size: usize) -> Self {
        Self {
            tiles: Vec::new(),
            ids: Vec::new(),
            n: 0,
            dim,
            tile_size,
        }
    }

    pub fn insert(&mut self, id: u64, v: &[f32]) {
        assert_eq!(v.len(), self.dim);
        // Start a new tile when the last one is full.
        let need_new = self.tiles.is_empty()
            || self.tiles.last().map_or(true, |t| t.count >= self.tile_size);
        if need_new {
            self.tiles.push(ColumnarTile {
                columns: (0..self.dim).map(|_| Vec::with_capacity(self.tile_size)).collect(),
                count: 0,
                start_idx: self.n,
            });
        }
        let tile = self.tiles.last_mut().unwrap();
        for (d, &vd) in v.iter().enumerate() {
            tile.columns[d].push(vd);
        }
        tile.count += 1;
        self.ids.push(id);
        self.n += 1;
    }

    pub fn len(&self) -> usize { self.n }
    pub fn is_empty(&self) -> bool { self.n == 0 }

    /// k-NN with optional partial-distance pruning across tiles.
    ///
    /// `prune_dims`: number of leading dimensions used to compute a lower-bound
    /// distance per tile before deciding whether to process the full tile.
    /// Set to 0 to disable pruning (pure columnar, no early exit).
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        dist: &dyn Distance,
        prune_dims: usize,
    ) -> Vec<(u64, f32)> {
        assert_eq!(query.len(), self.dim);
        use ordered_float::OrderedFloat;
        let mut results: Vec<(OrderedFloat<f32>, u64)> = Vec::new();

        // kth_best tracks the worst distance in our current top-k heap.
        // Initialised to f32::MAX so the first tile is never pruned.
        let mut kth_best = f32::MAX;

        for tile in &self.tiles {
            // --- Optional pruning: partial distance over first `prune_dims` ---
            if prune_dims > 0 && k <= results.len() {
                let pdims = prune_dims.min(self.dim);
                let partial_query = &query[..pdims];
                let col_refs: Vec<&[f32]> =
                    tile.columns[..pdims].iter().map(|c| c.as_slice()).collect();
                let partial_dists = dist.distance_columnar(partial_query, &col_refs, tile.count);
                // Lower bound: partial distance ≤ full distance (for L2Sq,
                // remaining dims can only add).  If *all* vectors in the tile
                // already exceed kth_best, skip the tile entirely.
                let min_partial: f32 = partial_dists.iter().cloned().fold(f32::MAX, f32::min);
                if min_partial > kth_best {
                    continue; // prune entire tile
                }
            }

            // --- Full columnar distance over all dimensions ---
            let col_refs: Vec<&[f32]> =
                tile.columns.iter().map(|c| c.as_slice()).collect();
            let dists = dist.distance_columnar(query, &col_refs, tile.count);

            for (i, d) in dists.into_iter().enumerate() {
                let id = self.ids[tile.start_idx + i];
                results.push((OrderedFloat(d), id));
            }

            // Update kth_best after each tile so pruning gets sharper.
            if results.len() >= k {
                let mut tmp = results.clone();
                tmp.sort_unstable();
                kth_best = tmp[k - 1].0 .0;
            }
        }

        results.sort_unstable();
        results.into_iter().take(k).map(|(d, id)| (id, d.0)).collect()
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Recall@k: fraction of ground-truth top-k ids found in result set.
pub fn recall_at_k(ground_truth: &[(u64, f32)], result: &[(u64, f32)], k: usize) -> f64 {
    let gt_set: std::collections::HashSet<u64> = ground_truth.iter().take(k).map(|(id, _)| *id).collect();
    let hits = result.iter().take(k).filter(|(id, _)| gt_set.contains(id)).count();
    hits as f64 / k as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::{SeedableRng, Rng, rngs::SmallRng};
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..n).map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect()).collect()
    }

    #[test]
    fn row_and_col_agree_on_nearest_neighbour() {
        let vecs = random_vecs(500, 64, 1);
        let query: Vec<f32> = random_vecs(1, 64, 99)[0].clone();
        let dist = L2Sq;

        let mut row = RowMajorIndex::new(64);
        let mut col = ColumnarIndex::new(64);
        for (i, v) in vecs.iter().enumerate() {
            row.insert(i as u64, v);
            col.insert(i as u64, v);
        }

        let r_row = row.search(&query, 10, &dist);
        let r_col = col.search(&query, 10, &dist);
        assert_eq!(r_row.len(), r_col.len());
        // Top-1 must agree exactly.
        assert_eq!(r_row[0].0, r_col[0].0, "top-1 id differs");
        // Recall@10 must be 1.0 (both are brute-force).
        assert!((recall_at_k(&r_row, &r_col, 10) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn chunked_index_top1_matches_row_major() {
        let vecs = random_vecs(1000, 32, 7);
        let query: Vec<f32> = random_vecs(1, 32, 42)[0].clone();
        let dist = L2Sq;

        let mut row = RowMajorIndex::new(32);
        let mut chunked = ChunkedColumnarIndex::new(32, 256);
        for (i, v) in vecs.iter().enumerate() {
            row.insert(i as u64, v);
            chunked.insert(i as u64, v);
        }

        let r_row = row.search(&query, 1, &dist);
        let r_chunked = chunked.search(&query, 1, &dist, 0);
        assert_eq!(r_row[0].0, r_chunked[0].0, "chunked top-1 id differs");
    }

    #[test]
    fn pruned_chunked_recall_acceptable() {
        let vecs = random_vecs(2048, 128, 13);
        let query: Vec<f32> = random_vecs(1, 128, 77)[0].clone();
        let dist = L2Sq;

        let mut row = RowMajorIndex::new(128);
        let mut chunked = ChunkedColumnarIndex::new(128, 256);
        for (i, v) in vecs.iter().enumerate() {
            row.insert(i as u64, v);
            chunked.insert(i as u64, v);
        }

        let gt = row.search(&query, 10, &dist);
        let res = chunked.search(&query, 10, &dist, 16); // prune on first 16 dims
        let recall = recall_at_k(&gt, &res, 10);
        assert!(recall >= 0.80, "pruned recall@10 = {recall:.2} < 0.80");
    }

    #[test]
    fn neg_dot_distance_correctness() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![1.0f32, 0.0, 0.0];
        let c = vec![0.0f32, 1.0, 0.0];
        let dist = NegDot;
        assert!((dist.distance(&a, &b) - 0.0).abs() < 1e-6, "parallel vectors should be distance 0");
        assert!((dist.distance(&a, &c) - 1.0).abs() < 1e-6, "orthogonal vectors should be distance 1");
    }

    #[test]
    fn columnar_distance_matches_scalar_loop() {
        let dim = 64;
        let n = 32;
        let vecs = random_vecs(n, dim, 5);
        let query: Vec<f32> = random_vecs(1, dim, 6)[0].clone();
        let dist = L2Sq;

        // Build column refs.
        let columns: Vec<Vec<f32>> = (0..dim)
            .map(|d| vecs.iter().map(|v| v[d]).collect())
            .collect();
        let col_refs: Vec<&[f32]> = columns.iter().map(|c| c.as_slice()).collect();

        let scalar_dists: Vec<f32> = vecs.iter().map(|v| dist.distance(&query, v)).collect();
        let col_dists = dist.distance_columnar(&query, &col_refs, n);

        for (i, (s, c)) in scalar_dists.iter().zip(col_dists.iter()).enumerate() {
            assert!((s - c).abs() < 1e-4, "mismatch at i={i}: scalar={s} col={c}");
        }
    }
}
