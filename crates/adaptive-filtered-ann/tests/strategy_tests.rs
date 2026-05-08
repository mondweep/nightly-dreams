use adaptive_filtered_ann::{
    CategoryFilter, FilteredIndex, Vector, recall_at_k,
    strategies::{AcornExpand, PostFilterFlat, PreFilterBrute},
    planner::AdaptivePlanner,
};
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;

fn make_corpus(n: usize, dim: usize, n_cats: u32, seed: u64) -> (Vec<Vector>, Vec<u32>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let vectors = (0..n)
        .map(|_| Vector::new((0..dim).map(|_| rng.gen::<f32>()).collect()))
        .collect();
    let categories = (0..n).map(|_| rng.gen_range(0..n_cats)).collect();
    (vectors, categories)
}

fn make_queries(n: usize, dim: usize, seed: u64) -> Vec<Vector> {
    let mut rng = SmallRng::seed_from_u64(seed + 99);
    (0..n)
        .map(|_| Vector::new((0..dim).map(|_| rng.gen::<f32>()).collect()))
        .collect()
}

/// Build ground truth using PreFilterBrute (exact).
fn ground_truth(
    vectors: &[Vector],
    categories: &[u32],
    queries: &[Vector],
    k: usize,
    cat: u32,
) -> Vec<Vec<usize>> {
    let mut idx = PreFilterBrute::new();
    idx.build(vectors.to_vec(), categories.to_vec());
    let filter = CategoryFilter(cat);
    queries
        .iter()
        .map(|q| idx.search(q, k, &filter).into_iter().map(|r| r.id).collect())
        .collect()
}

// ── PostFilterFlat ────────────────────────────────────────────────────────────

#[test]
fn post_filter_flat_recall_one() {
    let (vecs, cats) = make_corpus(500, 32, 10, 1);
    let queries = make_queries(20, 32, 1);
    let gt = ground_truth(&vecs, &cats, &queries, 5, 0);

    let mut idx = PostFilterFlat::new();
    idx.build(vecs, cats);
    let filter = CategoryFilter(0);

    for (q, g) in queries.iter().zip(&gt) {
        let results = idx.search(q, 5, &filter);
        // Exact search: recall must be 1.0.
        assert_eq!(
            recall_at_k(&results, g),
            1.0,
            "PostFilterFlat must be exact"
        );
    }
}

#[test]
fn post_filter_flat_returns_at_most_k() {
    let (vecs, cats) = make_corpus(200, 16, 5, 2);
    let mut idx = PostFilterFlat::new();
    idx.build(vecs, cats);
    let q = Vector::new(vec![0.5_f32; 16]);
    let results = idx.search(&q, 10, &CategoryFilter(0));
    assert!(results.len() <= 10);
}

#[test]
fn post_filter_flat_empty_result_when_no_match() {
    let (vecs, cats) = make_corpus(100, 16, 5, 3);
    let mut idx = PostFilterFlat::new();
    idx.build(vecs, cats);
    let q = Vector::new(vec![0.0_f32; 16]);
    // Category 99 doesn't exist → empty result.
    let results = idx.search(&q, 10, &CategoryFilter(99));
    assert!(results.is_empty());
}

// ── PreFilterBrute ────────────────────────────────────────────────────────────

#[test]
fn pre_filter_brute_recall_one() {
    let (vecs, cats) = make_corpus(500, 32, 10, 4);
    let queries = make_queries(20, 32, 4);

    let mut ref_idx = PostFilterFlat::new();
    ref_idx.build(vecs.clone(), cats.clone());
    let filter = CategoryFilter(0);
    let gt: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| ref_idx.search(q, 5, &filter).into_iter().map(|r| r.id).collect())
        .collect();

    let mut idx = PreFilterBrute::new();
    idx.build(vecs, cats);

    for (q, g) in queries.iter().zip(&gt) {
        let results = idx.search(q, 5, &filter);
        assert_eq!(recall_at_k(&results, g), 1.0, "PreFilterBrute must be exact");
    }
}

#[test]
fn pre_filter_brute_results_sorted_ascending() {
    let (vecs, cats) = make_corpus(200, 16, 3, 5);
    let mut idx = PreFilterBrute::new();
    idx.build(vecs, cats);
    let q = Vector::new(vec![0.5_f32; 16]);
    let results = idx.search(&q, 8, &CategoryFilter(1));
    for w in results.windows(2) {
        assert!(w[0].score <= w[1].score, "results must be sorted by ascending L2²");
    }
}

// ── AdaptivePlanner ───────────────────────────────────────────────────────────

#[test]
fn adaptive_planner_recall_one_at_low_selectivity() {
    // 1% selectivity → should route to PreFilterBrute.
    let (vecs, cats) = make_corpus(500, 32, 100, 6);
    let queries = make_queries(10, 32, 6);
    let gt = ground_truth(&vecs, &cats, &queries, 5, 0);

    let mut planner = AdaptivePlanner::new(200);
    planner.build(vecs, cats);
    let filter = CategoryFilter(0);

    for (q, g) in queries.iter().zip(&gt) {
        let results = planner.search(q, 5, &filter);
        assert_eq!(
            recall_at_k(&results, g),
            1.0,
            "planner at 1% selectivity must return exact results"
        );
    }
}

#[test]
fn adaptive_planner_recall_one_at_high_selectivity() {
    // 50% selectivity → should route to PostFilterFlat.
    let (vecs, cats) = make_corpus(400, 32, 2, 7);
    let queries = make_queries(10, 32, 7);
    let gt = ground_truth(&vecs, &cats, &queries, 5, 0);

    let mut planner = AdaptivePlanner::new(200);
    planner.build(vecs, cats);
    let filter = CategoryFilter(0);

    for (q, g) in queries.iter().zip(&gt) {
        let results = planner.search(q, 5, &filter);
        assert_eq!(
            recall_at_k(&results, g),
            1.0,
            "planner at 50% selectivity must return exact results"
        );
    }
}

// ── AcornExpand ───────────────────────────────────────────────────────────────

#[test]
fn acorn_expand_returns_at_most_k() {
    let (vecs, cats) = make_corpus(300, 16, 5, 8);
    let mut idx = AcornExpand::new(100);
    idx.build(vecs, cats);
    let q = Vector::new(vec![0.5_f32; 16]);
    let results = idx.search(&q, 10, &CategoryFilter(0));
    assert!(results.len() <= 10);
}

#[test]
fn acorn_expand_recall_at_high_selectivity() {
    // At 50% selectivity the graph has enough matching neighbours that
    // recall should be non-trivially above 0.
    let (vecs, cats) = make_corpus(300, 16, 2, 9);
    let queries = make_queries(10, 16, 9);
    let gt = ground_truth(&vecs, &cats, &queries, 5, 0);

    let mut idx = AcornExpand::new(150);
    idx.build(vecs, cats);
    let filter = CategoryFilter(0);

    let recall: f32 = queries
        .iter()
        .zip(&gt)
        .map(|(q, g)| recall_at_k(&idx.search(q, 5, &filter), g))
        .sum::<f32>()
        / queries.len() as f32;

    assert!(recall > 0.3, "AcornExpand at 50% selectivity should find some results (got {recall})");
}

// ── Vector internals ──────────────────────────────────────────────────────────

#[test]
fn l2_sq_self_is_zero() {
    let v = Vector::new(vec![1.0, 2.0, 3.0]);
    assert!((v.l2_sq(&v)).abs() < 1e-6);
}

#[test]
fn l2_sq_known_distance() {
    let a = Vector::new(vec![0.0, 0.0]);
    let b = Vector::new(vec![3.0, 4.0]);
    // L2² = 9 + 16 = 25
    assert!((a.l2_sq(&b) - 25.0).abs() < 1e-5);
}

#[test]
fn recall_at_k_perfect() {
    use adaptive_filtered_ann::SearchResult;
    let results = vec![
        SearchResult { id: 1, score: 0.1 },
        SearchResult { id: 2, score: 0.2 },
    ];
    assert_eq!(recall_at_k(&results, &[1, 2]), 1.0);
}

#[test]
fn recall_at_k_zero() {
    use adaptive_filtered_ann::SearchResult;
    let results = vec![SearchResult { id: 5, score: 0.1 }];
    assert_eq!(recall_at_k(&results, &[1, 2, 3]), 0.0);
}
