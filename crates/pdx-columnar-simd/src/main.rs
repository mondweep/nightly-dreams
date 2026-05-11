//! PDX Columnar Layout Demo — measures throughput and recall across three variants.
//!
//! Variants:
//!   A) Row-major brute-force (baseline)
//!   B) Column-major brute-force (PDX full scan)
//!   C) Column-major chunked with partial-distance pruning (PDX+prune)
//!
//! Run: cargo run --release -p pdx-columnar-simd

use std::time::Instant;

use pdx_columnar_simd::{
    ChunkedColumnarIndex, ColumnarIndex, L2Sq, NegDot, RowMajorIndex, recall_at_k,
};
use rand::{SeedableRng, Rng, rngs::SmallRng};

fn random_unit_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            v.into_iter().map(|x| x / norm).collect()
        })
        .collect()
}

#[derive(Clone)]
struct Result {
    variant: String,
    #[allow(dead_code)]
    dim: usize,
    #[allow(dead_code)]
    n: usize,
    mean_recall: f64,
    qps: f64,
    #[allow(dead_code)]
    latency_us: f64,
    speedup: f64,
}

fn run_scenario(
    label: &str,
    corpus: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
    tile_size: usize,
    prune_dims: usize,
) -> Vec<Result> {
    let dim = corpus[0].len();
    let n = corpus.len();
    let nq = queries.len();
    let dist_l2 = L2Sq;
    let dist_dot = NegDot;

    // Build all three index variants.
    let mut row = RowMajorIndex::new(dim);
    let mut col = ColumnarIndex::new(dim);
    let mut chunked = ChunkedColumnarIndex::new(dim, tile_size);

    for (i, v) in corpus.iter().enumerate() {
        row.insert(i as u64, v);
        col.insert(i as u64, v);
        chunked.insert(i as u64, v);
    }

    // -- Baseline (row-major L2Sq) --
    let t0 = Instant::now();
    let gt: Vec<_> = queries.iter().map(|q| row.search(q, k, &dist_l2)).collect();
    let baseline_us = t0.elapsed().as_micros() as f64 / nq as f64;

    // -- Columnar L2Sq --
    let t1 = Instant::now();
    let r_col: Vec<_> = queries.iter().map(|q| col.search(q, k, &dist_l2)).collect();
    let col_us = t1.elapsed().as_micros() as f64 / nq as f64;

    // -- Columnar NegDot (unit-norm, separate run for recall reference) --
    let t2 = Instant::now();
    let r_dot: Vec<_> = queries.iter().map(|q| col.search(q, k, &dist_dot)).collect();
    let dot_us = t2.elapsed().as_micros() as f64 / nq as f64;

    // -- Chunked no-prune --
    let t3 = Instant::now();
    let r_chunk0: Vec<_> = queries.iter().map(|q| chunked.search(q, k, &dist_l2, 0)).collect();
    let chunk0_us = t3.elapsed().as_micros() as f64 / nq as f64;

    // -- Chunked with pruning --
    let t4 = Instant::now();
    let r_prune: Vec<_> = queries.iter().map(|q| chunked.search(q, k, &dist_l2, prune_dims)).collect();
    let prune_us = t4.elapsed().as_micros() as f64 / nq as f64;

    let recall_col   = gt.iter().zip(&r_col)   .map(|(g, r)| recall_at_k(g, r, k)).sum::<f64>() / nq as f64;
    let recall_dot   = gt.iter().zip(&r_dot)   .map(|(g, r)| recall_at_k(g, r, k)).sum::<f64>() / nq as f64;
    let recall_c0    = gt.iter().zip(&r_chunk0).map(|(g, r)| recall_at_k(g, r, k)).sum::<f64>() / nq as f64;
    let recall_prune = gt.iter().zip(&r_prune) .map(|(g, r)| recall_at_k(g, r, k)).sum::<f64>() / nq as f64;

    vec![
        Result {
            variant: format!("{label} — Row-major L2Sq (baseline)"),
            dim, n,
            mean_recall: 1.0,
            qps: 1e6 / baseline_us,
            latency_us: baseline_us,
            speedup: 1.0,
        },
        Result {
            variant: format!("{label} — Columnar L2Sq (PDX full)"),
            dim, n,
            mean_recall: recall_col,
            qps: 1e6 / col_us,
            latency_us: col_us,
            speedup: baseline_us / col_us,
        },
        Result {
            variant: format!("{label} — Columnar NegDot (PDX full)"),
            dim, n,
            mean_recall: recall_dot,
            qps: 1e6 / dot_us,
            latency_us: dot_us,
            speedup: baseline_us / dot_us,
        },
        Result {
            variant: format!("{label} — Chunked no-prune (tile={tile_size})"),
            dim, n,
            mean_recall: recall_c0,
            qps: 1e6 / chunk0_us,
            latency_us: chunk0_us,
            speedup: baseline_us / chunk0_us,
        },
        Result {
            variant: format!("{label} — Chunked prune={prune_dims}d (tile={tile_size})"),
            dim, n,
            mean_recall: recall_prune,
            qps: 1e6 / prune_us,
            latency_us: prune_us,
            speedup: baseline_us / prune_us,
        },
    ]
}

fn main() {
    const N: usize = 10_000;
    const K: usize = 10;
    const N_QUERIES: usize = 200;
    const TILE: usize = 256;
    const PRUNE_DIMS: usize = 32;

    println!("=== PDX Columnar Layout Benchmark ===");
    println!("Arch:    {}", std::env::consts::ARCH);
    println!("OS:      {}", std::env::consts::OS);
    println!("Corpus:  {N} vectors, Queries: {N_QUERIES}, k={K}, tile={TILE}");
    println!();

    // Scenario A: 768-dim (BERT/all-MiniLM embedding size)
    let corpus_768 = random_unit_vecs(N, 768, 42);
    let queries_768 = random_unit_vecs(N_QUERIES, 768, 43);

    // Scenario B: 1536-dim (OpenAI text-embedding-3-small / ada-002)
    let corpus_1536 = random_unit_vecs(N, 1536, 44);
    let queries_1536 = random_unit_vecs(N_QUERIES, 1536, 45);

    // Scenario C: 128-dim (compressed / MRL truncated)
    let corpus_128 = random_unit_vecs(N, 128, 46);
    let queries_128 = random_unit_vecs(N_QUERIES, 128, 47);

    let all: Vec<Result> = [
        run_scenario("768-dim", &corpus_768, &queries_768, K, TILE, PRUNE_DIMS),
        run_scenario("1536-dim", &corpus_1536, &queries_1536, K, TILE, PRUNE_DIMS),
        run_scenario("128-dim", &corpus_128, &queries_128, K, TILE, PRUNE_DIMS),
    ].concat();

    println!("{:<58} {:>10} {:>11} {:>9}",
        "Variant", "QPS", "Recall@10", "Speedup");
    println!("{}", "-".repeat(93));
    for r in &all {
        println!(
            "{:<58} {:>10.0} {:>10.1}% {:>8.2}×",
            r.variant, r.qps, r.mean_recall * 100.0, r.speedup
        );
    }

    // ---- Assertions (acceptance criteria) ----
    let results_768: Vec<&Result> = all.iter().filter(|r| r.variant.starts_with("768-dim")).collect();
    let baseline_768 = results_768[0];
    let col_768      = results_768[1];
    let chunked_768  = results_768[3];
    let prune_768    = results_768[4];

    println!();
    println!("=== Acceptance Criteria (768-dim) ===");

    // Columnar must achieve >85% recall (it's exact brute-force, so should be 100%)
    assert!(
        col_768.mean_recall >= 0.99,
        "Columnar recall@10 = {:.3} (expected 1.0 — exact BF)", col_768.mean_recall
    );
    println!("✓ Columnar recall@10 = {:.1}%", col_768.mean_recall * 100.0);

    // Columnar should be at least as fast as row-major (layout effect)
    assert!(
        col_768.speedup >= 0.80,
        "Columnar speedup = {:.2}× < 0.80 (unexpected regression)", col_768.speedup
    );
    println!("✓ Columnar speedup = {:.2}×", col_768.speedup);

    // Chunked no-prune should be exact (brute force, just tiled)
    assert!(
        chunked_768.mean_recall >= 0.99,
        "Chunked recall@10 = {:.3} (expected 1.0)", chunked_768.mean_recall
    );
    println!("✓ Chunked recall@10 = {:.1}%", chunked_768.mean_recall * 100.0);

    // Pruned should have recall >= 0.70 (conservative threshold for random data)
    assert!(
        prune_768.mean_recall >= 0.70,
        "Pruned recall@10 = {:.3} < 0.70", prune_768.mean_recall
    );
    println!("✓ Pruned recall@10 = {:.1}% (threshold 70%)", prune_768.mean_recall * 100.0);

    // Baseline sanity: row-major QPS > 0
    assert!(baseline_768.qps > 0.0);
    println!("✓ Baseline QPS = {:.0}", baseline_768.qps);

    println!();
    println!("All assertions passed.");
}
