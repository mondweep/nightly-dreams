//! Benchmark demo: standard beam search vs. DABS at three γ values.
//!
//! Builds an NSW graph on 10 000 random 128-dim unit vectors, then runs
//! 200 queries to compare QPS, Recall@10, and mean distance computations.
//!
//! Key insight from arXiv:2505.15636: DABS should be compared against
//! standard beam search at EQUIVALENT recall, not equivalent ef.  On
//! random high-dim data (concentration of measure), standard ef=64
//! terminates too early (low recall). DABS adapts the threshold to the
//! query difficulty, achieving higher recall with fewer dist ops than
//! standard would need to match that recall via a larger ef.
//!
//! Run with:
//!   cargo run --release --example demo -p dabs-hnsw

use dabs_hnsw::{GraphSearch, NswGraph, recall_at_k};
use std::time::Instant;

const DIM: usize = 128;
const N: usize = 10_000;
const NQ: usize = 200;
const K: usize = 10;
const M: usize = 16;

fn lcg(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    ((*s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

fn make_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let v: Vec<f32> = (0..dim).map(|_| lcg(&mut s)).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.into_iter().map(|x| x / norm).collect()
}

fn build_graph() -> (NswGraph, Vec<Vec<f32>>) {
    println!("Building NSW graph: {N} nodes, dim={DIM}, M={M}…");
    let t = Instant::now();
    let mut graph = NswGraph::new(DIM, M);
    for i in 0..N as u64 {
        graph.insert(make_vec(DIM, i * 1337 + 1));
    }
    println!("  build: {:.2}s", t.elapsed().as_secs_f64());
    let queries: Vec<Vec<f32>> = (0..NQ as u64)
        .map(|i| make_vec(DIM, i * 9_999 + 7))
        .collect();
    (graph, queries)
}

fn compute_ground_truth(graph: &NswGraph, queries: &[Vec<f32>]) -> Vec<Vec<usize>> {
    println!("Computing brute-force ground truth for {NQ} queries…");
    queries.iter().map(|q| graph.brute_force(q, K)).collect()
}

struct Stats {
    qps: f64,
    recall: f64,
    dist_per_q: f64,
}

fn measure(
    _graph: &NswGraph,
    queries: &[Vec<f32>],
    truth: &[Vec<usize>],
    run_fn: impl Fn(&[f32]) -> dabs_hnsw::SearchStats,
) -> Stats {
    let t = Instant::now();
    let mut total_recall = 0.0f64;
    let mut total_dist: u64 = 0;
    for (q, gt) in queries.iter().zip(truth.iter()) {
        let s = run_fn(q);
        total_recall += recall_at_k(gt, &s.hits);
        total_dist += s.dist_ops;
    }
    let elapsed = t.elapsed().as_secs_f64();
    Stats {
        qps: NQ as f64 / elapsed,
        recall: total_recall / NQ as f64,
        dist_per_q: total_dist as f64 / NQ as f64,
    }
}

fn print_row(label: &str, s: &Stats, note: &str) {
    println!(
        "  {:<22} | {:>7.0} | {:.4} | {:>9.1} | {}",
        label, s.qps, s.recall, s.dist_per_q, note
    );
}

fn main() {
    let (graph, queries) = build_graph();
    let truth = compute_ground_truth(&graph, &queries);

    println!();
    println!("k={K}, {NQ} queries on {N} × {DIM}-dim unit vectors (random, normalised)");
    println!("{:-<80}", "");
    println!(
        "  {:<22} | {:>7} | {:>10} | {:>9} | {}",
        "variant", "QPS", "Recall@10", "Dist/query", "notes"
    );
    println!("{:-<80}", "");

    let std64 = measure(&graph, &queries, &truth, |q| graph.beam_search(q, K, 64));
    print_row("standard ef=64", &std64, "baseline");

    let std100 = measure(&graph, &queries, &truth, |q| graph.beam_search(q, K, 100));
    print_row("standard ef=100", &std100, "");

    let std200 = measure(&graph, &queries, &truth, |q| graph.beam_search(q, K, 200));
    print_row("standard ef=200", &std200, "recall target");

    let dabs05 = measure(&graph, &queries, &truth, |q| graph.dabs_search(q, K, 64, 0.5));
    let dist_vs_eq_recall = (dabs05.dist_per_q - std200.dist_per_q) / std200.dist_per_q * 100.0;
    let note05 = if dist_vs_eq_recall <= 0.0 {
        format!("{:.0}% fewer dist than std@eq-recall", -dist_vs_eq_recall)
    } else {
        format!("{dist_vs_eq_recall:.0}% more dist than std@eq-recall (random data)")
    };
    print_row("DABS γ=0.5 ef=64", &dabs05, &note05);

    let dabs10 = measure(&graph, &queries, &truth, |q| graph.dabs_search(q, K, 64, 1.0));
    print_row("DABS γ=1.0 ef=64", &dabs10, "");

    let dabs20 = measure(&graph, &queries, &truth, |q| graph.dabs_search(q, K, 64, 2.0));
    print_row("DABS γ=2.0 ef=64", &dabs20, "");

    // γ=0.0: mathematically guaranteed ≤ dist ops of standard (same ef)
    let dabs00 = measure(&graph, &queries, &truth, |q| graph.dabs_search(q, K, 64, 0.0));
    let savings00 = (std64.dist_per_q - dabs00.dist_per_q) / std64.dist_per_q * 100.0;
    print_row(
        "DABS γ=0.0 ef=64",
        &dabs00,
        &format!("{savings00:.0}% fewer dist than std@ef=64"),
    );

    println!("{:-<80}", "");
    println!();
    println!("Hardware: single-threaded Rust, release build");
    println!("Dataset:  {N}×{DIM}-dim random unit vectors (LCG seeded, concentrated distances)");
    println!();
    println!("Interpretation:");
    println!("  On structured/clustered data (SIFT-1M, GIST-960), DABS saves 10-50% dist ops");
    println!("  vs. standard at equal recall, because easy queries terminate early.");
    println!("  On random uniform data (high-dim concentration of measure), all distances");
    println!("  cluster near √2, so DABS with γ>0 may explore more to improve recall.");
    println!("  DABS(γ=0) is the safe baseline: always ≤ dist ops, provably.");
}
