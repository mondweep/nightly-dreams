use std::collections::{BinaryHeap, HashSet};
use crate::{CategoryFilter, FilteredIndex, SearchResult, Vector, sort_top_k};
use crate::graph::KnnGraph;

// ---------------------------------------------------------------------------
// Min-heap wrapper so BinaryHeap pops the CLOSEST candidate first.
// ---------------------------------------------------------------------------
#[derive(PartialEq)]
struct MinNode {
    dist: f32,
    id: usize,
}
impl Eq for MinNode {}
impl PartialOrd for MinNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Reverse: smaller dist → higher priority.
        other.dist.partial_cmp(&self.dist)
    }
}
impl Ord for MinNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// Strategy 1 — PostFilterFlat
// Compute L2 to EVERY corpus vector; apply predicate to the result list.
// Simulates naive post-filter HNSW: all N distance computations happen
// regardless of selectivity.  Recall = 1.0; QPS ∝ 1/N at low selectivity.
// ---------------------------------------------------------------------------
pub struct PostFilterFlat {
    vectors: Vec<Vector>,
    categories: Vec<u32>,
}

impl PostFilterFlat {
    pub fn new() -> Self { Self { vectors: Vec::new(), categories: Vec::new() } }
}

impl FilteredIndex for PostFilterFlat {
    fn build(&mut self, vectors: Vec<Vector>, categories: Vec<u32>) {
        self.vectors = vectors;
        self.categories = categories;
    }

    fn search(&self, query: &Vector, k: usize, filter: &CategoryFilter) -> Vec<SearchResult> {
        let results: Vec<SearchResult> = self.vectors
            .iter()
            .enumerate()
            .map(|(id, v)| SearchResult { id, score: query.l2_sq(v) })
            .filter(|r| filter.matches(self.categories[r.id]))
            .collect();
        sort_top_k(results, k)
    }

    fn name(&self) -> &'static str { "PostFilterFlat" }
}

// ---------------------------------------------------------------------------
// Strategy 2 — AcornExpand (ACORN-1)
// Graph beam-search that ALWAYS expands a node's neighbours, even when the
// node itself fails the predicate.  At low selectivity the filtered graph
// would otherwise starve; ACORN-1 uses each filtered node as a routing hop.
// Reference: Patel et al. "ACORN: Performant and Predicate-Agnostic Search"
// SIGMOD 2024 — https://arxiv.org/abs/2403.04871
// ---------------------------------------------------------------------------
pub struct AcornExpand {
    vectors: Vec<Vector>,
    categories: Vec<u32>,
    graph: Option<KnnGraph>,
    ef: usize, // beam width
}

impl AcornExpand {
    pub fn new(ef: usize) -> Self {
        Self { vectors: Vec::new(), categories: Vec::new(), graph: None, ef }
    }
}

impl FilteredIndex for AcornExpand {
    fn build(&mut self, vectors: Vec<Vector>, categories: Vec<u32>) {
        self.graph = Some(KnnGraph::build(&vectors, 16));
        self.vectors = vectors;
        self.categories = categories;
    }

    fn search(&self, query: &Vector, k: usize, filter: &CategoryFilter) -> Vec<SearchResult> {
        let graph = self.graph.as_ref().expect("call build first");
        let n = self.vectors.len();
        if n == 0 { return Vec::new(); }

        let mut candidates: BinaryHeap<MinNode> = BinaryHeap::new();
        let mut visited: HashSet<usize> = HashSet::new();
        let mut results: Vec<SearchResult> = Vec::new();

        // Seed beam with evenly-spaced entry points for better graph coverage.
        // At low selectivity the graph needs multiple seeds to avoid cold starts.
        let n_seeds = 10usize.min(n);
        let step = n / n_seeds;
        for s in 0..n_seeds {
            let entry = s * step;
            if visited.insert(entry) {
                candidates.push(MinNode {
                    dist: query.l2_sq(&self.vectors[entry]),
                    id: entry,
                });
            }
        }

        // Visit up to ef*30 nodes, or until we collect k*3 matching results.
        let beam_limit = (self.ef * 30).max(3000);

        while let Some(MinNode { dist: score, id: node }) = candidates.pop() {
            if filter.matches(self.categories[node]) {
                results.push(SearchResult { id: node, score });
                if results.len() >= k * 3 { break; }
            }

            // ACORN-1: always expand neighbours even from filtered-out nodes,
            // so filtered nodes act as routing hops toward valid candidates.
            for &nb in &graph.adj[node] {
                if !visited.contains(&nb) {
                    visited.insert(nb);
                    let d = query.l2_sq(&self.vectors[nb]);
                    candidates.push(MinNode { dist: d, id: nb });
                }
            }

            if visited.len() >= beam_limit { break; }
        }

        sort_top_k(results, k)
    }

    fn name(&self) -> &'static str { "AcornExpand" }
}

// ---------------------------------------------------------------------------
// Strategy 3 — PreFilterBrute
// Collect matching vector IDs via the attribute index first; then compute
// exact L2 only over that subset.  Recall = 1.0; cost ∝ selectivity * N.
// Fastest at low selectivity; as fast as PostFilterFlat at 100% selectivity.
// ---------------------------------------------------------------------------
pub struct PreFilterBrute {
    vectors: Vec<Vector>,
    categories: Vec<u32>,
    // Inverted index: category → sorted list of vector IDs.
    cat_index: std::collections::HashMap<u32, Vec<usize>>,
}

impl PreFilterBrute {
    pub fn new() -> Self {
        Self {
            vectors: Vec::new(),
            categories: Vec::new(),
            cat_index: std::collections::HashMap::new(),
        }
    }
}

impl FilteredIndex for PreFilterBrute {
    fn build(&mut self, vectors: Vec<Vector>, categories: Vec<u32>) {
        let mut idx: std::collections::HashMap<u32, Vec<usize>> =
            std::collections::HashMap::new();
        for (id, &cat) in categories.iter().enumerate() {
            idx.entry(cat).or_default().push(id);
        }
        self.cat_index = idx;
        self.vectors = vectors;
        self.categories = categories;
    }

    fn search(&self, query: &Vector, k: usize, filter: &CategoryFilter) -> Vec<SearchResult> {
        let ids = match self.cat_index.get(&filter.0) {
            Some(v) => v,
            None => return Vec::new(),
        };
        let results: Vec<SearchResult> = ids
            .iter()
            .map(|&id| SearchResult { id, score: query.l2_sq(&self.vectors[id]) })
            .collect();
        sort_top_k(results, k)
    }

    fn name(&self) -> &'static str { "PreFilterBrute" }
}
