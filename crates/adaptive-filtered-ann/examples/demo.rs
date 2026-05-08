use adaptive_filtered_ann::{
    CategoryFilter, FilteredIndex, Vector, recall_at_k,
    strategies::{AcornExpand, PostFilterFlat, PreFilterBrute},
    planner::AdaptivePlanner,
};
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use std::time::Instant;

const N: usize = 10_000;    // corpus size
const DIM: usize = 128;     // vector dimension
const N_QUERIES: usize = 200;
const K: usize = 10;
const WARMUP: usize = 20;
const EF: usize = 200;      // ACORN beam width

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║   ruvector Adaptive Filtered-ANN Benchmark  (2026-05-08)       ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!("  Corpus : {N} vectors × {DIM} dimensions (f32)");
    println!("  Queries: {N_QUERIES}  k={K}  ef={EF}");
    println!();

    let mut rng = SmallRng::seed_from_u64(0xdeadbeef);

    // ── Generate corpus ───────────────────────────────────────────────────────
    let vectors: Vec<Vector> = (0..N)
        .map(|_| {
            let data: Vec<f32> = (0..DIM).map(|_| rng.gen::<f32>()).collect();
            Vector::new(data)
        })
        .collect();

    // ── Three selectivity scenarios ───────────────────────────────────────────
    // categories uniform in [0, C) → selectivity ≈ 1/C
    let scenarios: &[(&str, u32)] = &[
        ("1%  (C=100)", 100),
        ("10% (C=10) ", 10),
        ("50% (C=2)  ", 2),
    ];

    for &(label, n_cats) in scenarios {
        println!("────────────────────────────────────────────────────────────────────");
        println!("  Selectivity {label}");
        println!("────────────────────────────────────────────────────────────────────");

        let categories: Vec<u32> = (0..N).map(|_| rng.gen_range(0..n_cats)).collect();
        let query_category: u32 = 0; // always query for category 0

        // Count how many corpus vectors match
        let actual_matches = categories.iter().filter(|&&c| c == query_category).count();
        let actual_sel = actual_matches as f32 / N as f32;
        println!("  Actual matches for cat=0: {actual_matches}/{N}  ({:.1}%)", actual_sel * 100.0);

        // Generate queries
        let queries: Vec<Vector> = (0..N_QUERIES)
            .map(|_| {
                let data: Vec<f32> = (0..DIM).map(|_| rng.gen::<f32>()).collect();
                Vector::new(data)
            })
            .collect();
        let filter = CategoryFilter(query_category);

        // ── Ground truth via exact pre-filter brute-force ─────────────────────
        let mut gt_ref = PreFilterBrute::new();
        gt_ref.build(vectors.clone(), categories.clone());
        let ground_truths: Vec<Vec<usize>> = queries
            .iter()
            .map(|q| {
                gt_ref
                    .search(q, K, &filter)
                    .into_iter()
                    .map(|r| r.id)
                    .collect()
            })
            .collect();

        // ── Build all strategies ──────────────────────────────────────────────
        let mut post_flat = PostFilterFlat::new();
        let mut acorn = AcornExpand::new(EF);
        let mut pre_brute = PreFilterBrute::new();
        let mut adaptive = AdaptivePlanner::new(EF);

        let t0 = Instant::now();
        post_flat.build(vectors.clone(), categories.clone());
        let post_build_ms = t0.elapsed().as_millis();

        let t0 = Instant::now();
        acorn.build(vectors.clone(), categories.clone());
        let acorn_build_ms = t0.elapsed().as_millis();

        let t0 = Instant::now();
        pre_brute.build(vectors.clone(), categories.clone());
        let pre_build_ms = t0.elapsed().as_millis();

        let t0 = Instant::now();
        adaptive.build(vectors.clone(), categories.clone());
        let adap_build_ms = t0.elapsed().as_millis();

        // ── Benchmark helper ──────────────────────────────────────────────────
        fn bench<F: FilteredIndex>(
            idx: &F,
            queries: &[Vector],
            filter: &CategoryFilter,
            k: usize,
            warmup: usize,
            gts: &[Vec<usize>],
            build_ms: u128,
        ) {
            // Warm-up
            for q in queries.iter().take(warmup) {
                let _ = idx.search(q, k, filter);
            }

            // Timed run
            let mut latencies: Vec<u128> = Vec::with_capacity(queries.len());
            let t0 = Instant::now();
            let mut recalls: Vec<f32> = Vec::with_capacity(queries.len());
            for (q, gt) in queries.iter().zip(gts.iter()) {
                let ts = Instant::now();
                let results = idx.search(q, k, filter);
                latencies.push(ts.elapsed().as_micros());
                recalls.push(recall_at_k(&results, gt));
            }
            let total_s = t0.elapsed().as_secs_f64();
            let qps = queries.len() as f64 / total_s;

            latencies.sort_unstable();
            let p50 = latencies[latencies.len() / 2];
            let p99_idx = ((latencies.len() as f64 * 0.99) as usize).min(latencies.len() - 1);
            let p99 = latencies[p99_idx];
            let mean_recall: f32 = recalls.iter().sum::<f32>() / recalls.len() as f32;

            println!(
                "  {:15}  build={:4}ms  QPS={:6.0}  recall@10={:.3}  p50={:5}µs  p99={:5}µs",
                idx.name(), build_ms, qps, mean_recall, p50, p99
            );
        }

        bench(&post_flat, &queries, &filter, K, WARMUP, &ground_truths, post_build_ms);
        bench(&acorn,     &queries, &filter, K, WARMUP, &ground_truths, acorn_build_ms);
        bench(&pre_brute, &queries, &filter, K, WARMUP, &ground_truths, pre_build_ms);
        bench(&adaptive,  &queries, &filter, K, WARMUP, &ground_truths, adap_build_ms);
        println!();
    }

    // ── Memory estimate ────────────────────────────────────────────────────────
    let vec_bytes = N * DIM * 4;
    let graph_bytes = N * 16 * 8; // 16 neighbours × 8-byte usize
    println!("  Memory estimates (N={N}, D={DIM}):");
    println!("    Corpus vectors : {:7} KB  ({} B/vec)", vec_bytes / 1024, DIM * 4);
    println!("    KNN graph      : {:7} KB  (16 neighbours × 8 B)", graph_bytes / 1024);
    println!("    Inverted index : ≤{:6} KB  (cat→id lists, depends on n_cats)", N * 4 / 1024);
    println!();
    println!("Run with: cargo run --release --example demo -p adaptive-filtered-ann");
}
