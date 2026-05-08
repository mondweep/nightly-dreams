pub mod graph;
pub mod strategies;
pub mod planner;

/// A dense f32 vector.
#[derive(Debug, Clone)]
pub struct Vector(pub Vec<f32>);

impl Vector {
    pub fn new(data: Vec<f32>) -> Self { Self(data) }
    pub fn dim(&self) -> usize { self.0.len() }

    /// Squared Euclidean distance (no sqrt — monotone for ranking).
    pub fn l2_sq(&self, other: &Self) -> f32 {
        self.0.iter().zip(&other.0).map(|(a, b)| (a - b) * (a - b)).sum()
    }
}

/// One result from a search call.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub id: usize,
    pub score: f32, // lower = more similar
}

/// Simple equality predicate on an integer category attribute.
#[derive(Debug, Clone)]
pub struct CategoryFilter(pub u32);

impl CategoryFilter {
    pub fn matches(&self, cat: u32) -> bool { cat == self.0 }
}

/// All index implementations must satisfy this contract.
pub trait FilteredIndex {
    fn build(&mut self, vectors: Vec<Vector>, categories: Vec<u32>);
    fn search(&self, query: &Vector, k: usize, filter: &CategoryFilter) -> Vec<SearchResult>;
    fn name(&self) -> &'static str;
}

/// Recall@k: fraction of ground-truth top-k IDs found in `results`.
pub fn recall_at_k(results: &[SearchResult], ground_truth: &[usize]) -> f32 {
    if ground_truth.is_empty() { return 1.0; }
    let gt: std::collections::HashSet<usize> = ground_truth.iter().cloned().collect();
    let hits = results.iter().filter(|r| gt.contains(&r.id)).count();
    hits as f32 / ground_truth.len().min(results.len() + 1) as f32
}

/// Sort and truncate a result list in-place.
pub fn sort_top_k(mut v: Vec<SearchResult>, k: usize) -> Vec<SearchResult> {
    v.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
    v.truncate(k);
    v
}
