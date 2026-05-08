use std::collections::HashMap;
use crate::{CategoryFilter, FilteredIndex, SearchResult, Vector};
use crate::strategies::{AcornExpand, PostFilterFlat, PreFilterBrute};

/// Tracks per-category counts to estimate query selectivity in O(1).
pub struct SelectivityEstimator {
    counts: HashMap<u32, usize>,
    total: usize,
}

impl SelectivityEstimator {
    pub fn new() -> Self { Self { counts: HashMap::new(), total: 0 } }

    pub fn record(&mut self, cat: u32) {
        *self.counts.entry(cat).or_insert(0) += 1;
        self.total += 1;
    }

    /// Returns the fraction of corpus vectors that match `filter` (0.0–1.0).
    pub fn estimate(&self, filter: &CategoryFilter) -> f32 {
        if self.total == 0 { return 0.0; }
        let n = self.counts.get(&filter.0).copied().unwrap_or(0);
        n as f32 / self.total as f32
    }
}

/// Routing thresholds (fraction of corpus matching the predicate).
///
/// For flat (non-graph) indexes PreFilterBrute is exact AND faster than
/// PostFilterFlat at all selectivities (it visits N*sel vectors instead of N).
/// AcornExpand belongs in a graph-aware planner variant where the backing store
/// is HNSW — leave it in the planner tree so callers can activate it by
/// setting MID_SEL_THRESHOLD above LOW_SEL_THRESHOLD.
///
/// Current policy: PreFilterBrute for sel < 50%, PostFilterFlat otherwise.
const LOW_SEL_THRESHOLD: f32 = 0.50;   // < 50% → PreFilterBrute (exact + fast)
const MID_SEL_THRESHOLD: f32 = 0.50;   // = LOW → AcornExpand never selected here
                                         // ≥ 50% → PostFilterFlat

/// Chooses the execution strategy based on estimated predicate selectivity.
pub struct AdaptivePlanner {
    post_filter: PostFilterFlat,
    acorn: AcornExpand,
    pre_filter: PreFilterBrute,
    estimator: SelectivityEstimator,
    last_strategy: std::cell::Cell<&'static str>,
}

impl AdaptivePlanner {
    pub fn new(ef: usize) -> Self {
        Self {
            post_filter: PostFilterFlat::new(),
            acorn: AcornExpand::new(ef),
            pre_filter: PreFilterBrute::new(),
            estimator: SelectivityEstimator::new(),
            last_strategy: std::cell::Cell::new("none"),
        }
    }

    /// Which strategy was chosen for the most recent query (for reporting).
    pub fn last_strategy(&self) -> &'static str { self.last_strategy.get() }
}

impl FilteredIndex for AdaptivePlanner {
    fn build(&mut self, vectors: Vec<Vector>, categories: Vec<u32>) {
        for &c in &categories { self.estimator.record(c); }

        // All three strategies share the same corpus — clone is unavoidable here.
        self.post_filter.build(vectors.clone(), categories.clone());
        self.acorn.build(vectors.clone(), categories.clone());
        self.pre_filter.build(vectors, categories);
    }

    fn search(&self, query: &Vector, k: usize, filter: &CategoryFilter) -> Vec<SearchResult> {
        let sel = self.estimator.estimate(filter);
        if sel < LOW_SEL_THRESHOLD {
            self.last_strategy.set("PreFilterBrute");
            self.pre_filter.search(query, k, filter)
        } else if sel < MID_SEL_THRESHOLD {
            self.last_strategy.set("AcornExpand");
            self.acorn.search(query, k, filter)
        } else {
            self.last_strategy.set("PostFilterFlat");
            self.post_filter.search(query, k, filter)
        }
    }

    fn name(&self) -> &'static str { "AdaptivePlanner" }
}
