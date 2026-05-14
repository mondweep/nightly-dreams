use crate::{dot, maxsim, topk, MaxSimIndex, MultiVec};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// MUVERA-inspired Fixed-Dimensional Encoding (FDE) MaxSim index.
///
/// Each multi-vector document is projected to a single P-dimensional vector
/// (mean-pooled random linear projection).  Candidate retrieval is O(N·P)
/// via dot-product scan; then exact MaxSim re-ranks the top-`n_candidates`.
///
/// Avoids the `rand_distr` crate by using the Box-Muller transform inline.
pub struct FdeIndex {
    proj_dim: usize,
    n_candidates: usize,
    /// proj_dim × input_dim random Gaussian projection matrix
    projection: Vec<Vec<f32>>,
    docs: Vec<(usize, MultiVec)>,
    /// mean-pooled projection for each doc
    doc_projs: Vec<Vec<f32>>,
}

impl FdeIndex {
    pub fn new(proj_dim: usize, n_candidates: usize) -> Self {
        Self {
            proj_dim,
            n_candidates,
            projection: Vec::new(),
            docs: Vec::new(),
            doc_projs: Vec::new(),
        }
    }

    /// Lazy-initialise the projection matrix on first insert.
    fn ensure_projection(&mut self, input_dim: usize) {
        if !self.projection.is_empty() {
            return;
        }
        let mut rng = SmallRng::seed_from_u64(99);
        let scale = (1.0 / self.proj_dim as f32).sqrt();
        self.projection = (0..self.proj_dim)
            .map(|_| {
                (0..input_dim)
                    .map(|_| gaussian_sample(&mut rng) * scale)
                    .collect()
            })
            .collect();
    }

    fn project(&self, mv: &MultiVec) -> Vec<f32> {
        let mut out = vec![0.0f32; self.proj_dim];
        for v in mv {
            for (r, row) in self.projection.iter().enumerate() {
                out[r] += dot(v, row);
            }
        }
        let inv_n = 1.0 / mv.len() as f32;
        out.iter_mut().for_each(|x| *x *= inv_n);
        out
    }
}

impl MaxSimIndex for FdeIndex {
    fn insert(&mut self, id: usize, doc: MultiVec) {
        if let Some(tok) = doc.first() {
            self.ensure_projection(tok.len());
        }
        let proj = self.project(&doc);
        self.doc_projs.push(proj);
        self.docs.push((id, doc));
    }

    fn build(&mut self) {}

    fn search(&self, query: &MultiVec, k: usize) -> Vec<(usize, f32)> {
        let q_proj = self.project(query);
        let n_cand = self.n_candidates.min(self.docs.len());

        // Fast candidate retrieval by dot product in projection space
        let mut cand: Vec<(usize, f32)> = self
            .doc_projs
            .iter()
            .enumerate()
            .map(|(i, dp)| (i, dot(&q_proj, dp)))
            .collect();
        cand.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        cand.truncate(n_cand);

        // Exact MaxSim re-rank on candidates
        let scores: Vec<(usize, f32)> = cand
            .iter()
            .map(|(di, _)| {
                let (id, doc) = &self.docs[*di];
                (*id, maxsim(query, doc))
            })
            .collect();

        topk(scores, k)
    }

    fn name(&self) -> &str {
        "fde"
    }
}

/// Box-Muller transform: generates a standard-normal sample from two U(0,1) draws.
#[inline]
fn gaussian_sample(rng: &mut SmallRng) -> f32 {
    let u1: f32 = rng.gen::<f32>().max(1e-10);
    let u2: f32 = rng.gen::<f32>();
    (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * std::f32::consts::PI * u2).cos()
}
