use crate::{dot, l2sq, maxsim, topk, MaxSimIndex, MultiVec};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;

/// PLAID-inspired centroid-pruned MaxSim index.
///
/// Build: k-means on all token vectors → inverted lists (centroid → doc IDs).
/// Search: find top-`n_probe` centroids per query token, union candidate docs,
///         exact MaxSim re-rank.
pub struct CentroidIndex {
    n_centroids: usize,
    n_probe: usize,
    centroids: Vec<Vec<f32>>,
    /// centroid_idx → sorted-unique list of doc indices
    inv_lists: Vec<Vec<usize>>,
    docs: Vec<(usize, MultiVec)>,
    /// accumulated (token_vec, doc_idx) before build()
    pending: Vec<(Vec<f32>, usize)>,
}

impl CentroidIndex {
    pub fn new(n_centroids: usize, n_probe: usize) -> Self {
        Self {
            n_centroids,
            n_probe,
            centroids: Vec::new(),
            inv_lists: Vec::new(),
            docs: Vec::new(),
            pending: Vec::new(),
        }
    }
}

impl MaxSimIndex for CentroidIndex {
    fn insert(&mut self, id: usize, doc: MultiVec) {
        let doc_idx = self.docs.len();
        for token in &doc {
            self.pending.push((token.clone(), doc_idx));
        }
        self.docs.push((id, doc));
    }

    fn build(&mut self) {
        assert!(!self.pending.is_empty(), "insert documents before build()");
        let vecs: Vec<&Vec<f32>> = self.pending.iter().map(|(v, _)| v).collect();
        let k = self.n_centroids.min(vecs.len());

        let mut rng = SmallRng::seed_from_u64(42);
        self.centroids = kmeans(&vecs, k, 25, &mut rng);

        // Build inverted lists with doc-level deduplication per centroid
        let n_docs = self.docs.len();
        self.inv_lists = vec![Vec::new(); k];
        let mut seen: Vec<HashSet<usize>> = vec![HashSet::new(); k];

        for (token, doc_idx) in &self.pending {
            let c = nearest_centroid(token, &self.centroids);
            if seen[c].insert(*doc_idx) {
                self.inv_lists[c].push(*doc_idx);
            }
        }
        drop(seen);

        // Keep pending memory; it's no longer needed after build.
        let _ = n_docs;
        self.pending.clear();
        self.pending.shrink_to_fit();
    }

    fn search(&self, query: &MultiVec, k: usize) -> Vec<(usize, f32)> {
        debug_assert!(!self.centroids.is_empty(), "call build() first");

        let mut candidate_set: HashSet<usize> = HashSet::new();
        for q_tok in query {
            for c_idx in top_n_centroids(q_tok, &self.centroids, self.n_probe) {
                for &doc_idx in &self.inv_lists[c_idx] {
                    candidate_set.insert(doc_idx);
                }
            }
        }

        let scores: Vec<(usize, f32)> = candidate_set
            .iter()
            .map(|&di| {
                let (id, doc) = &self.docs[di];
                (*id, maxsim(query, doc))
            })
            .collect();

        topk(scores, k)
    }

    fn name(&self) -> &str {
        "centroid"
    }
}

fn nearest_centroid(v: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, l2sq(v, c)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn top_n_centroids(v: &[f32], centroids: &[Vec<f32>], n: usize) -> Vec<usize> {
    let mut scores: Vec<(usize, f32)> = centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, dot(v, c)))
        .collect();
    scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(n);
    scores.into_iter().map(|(i, _)| i).collect()
}

/// Lloyd's k-means, seeded deterministically. Handles empty clusters by
/// reinitialising to a random existing vector.
fn kmeans(vecs: &[&Vec<f32>], k: usize, max_iters: usize, rng: &mut SmallRng) -> Vec<Vec<f32>> {
    let n = vecs.len();
    let d = vecs[0].len();

    // Spread-initialisation: pick k evenly-spaced indices
    let step = (n / k).max(1);
    let mut centroids: Vec<Vec<f32>> = (0..k).map(|i| vecs[(i * step) % n].clone()).collect();

    let mut assignments = vec![0usize; n];

    for _ in 0..max_iters {
        // Assign
        let mut changed = false;
        for (i, v) in vecs.iter().enumerate() {
            let c = nearest_centroid(v, &centroids);
            if c != assignments[i] {
                assignments[i] = c;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // Update
        let mut new_c = vec![vec![0.0f32; d]; k];
        let mut counts = vec![0usize; k];
        for (i, &a) in assignments.iter().enumerate() {
            for (j, &x) in vecs[i].iter().enumerate() {
                new_c[a][j] += x;
            }
            counts[a] += 1;
        }
        for (c, &cnt) in new_c.iter_mut().zip(&counts) {
            if cnt > 0 {
                c.iter_mut().for_each(|x| *x /= cnt as f32);
            } else {
                // Reinitialise empty cluster to a random existing vector
                *c = vecs[rng.gen_range(0..n)].clone();
            }
        }
        centroids = new_c;
    }

    centroids
}
