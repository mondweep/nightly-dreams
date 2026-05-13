// CoDEQ Streaming Quantization — benchmark demo
//
// Measures three variants under a synthetic streaming workload:
//   1. EagerCoDEQ   — immediate reassignment on every insert
//   2. LazyCoDEQ    — batch reassignment triggered at 5% stale fraction
//   3. ApproxCoDEQ  — KS-drift-gated lazy reassignment
//
// Workload: 2000 base vectors (dim=128) + 5 batches of 100 inserts/deletes
//           with progressive mean-shift drift (+0.15 per batch).

use codeq_streaming_quantization::*;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use std::time::Instant;

const DIM: usize = M * SUB_DIM;   // 128
const BASE_N: usize = 2_000;
const BATCHES: usize = 5;
const BATCH_SIZE: usize = 100;
const KNN_K: usize = 10;
const QUERIES: usize = 50;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     CoDEQ: Consistent Dynamic Quantization Benchmark            ║");
    println!("║     ruvector nightly research 2026-05-13                        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Config: base_n={BASE_N}, dim={DIM}, M={M}, K={K}, batches={BATCHES}×{BATCH_SIZE}, k={KNN_K}");
    println!();

    let mut rng = SmallRng::seed_from_u64(42);

    // Build base dataset
    let base: Vec<Vector> = (0..BASE_N)
        .map(|_| (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
        .collect();

    // Pre-build query set (drawn before any mutations so GT is deterministic)
    let queries: Vec<Vector> = (0..QUERIES)
        .map(|_| (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
        .collect();

    run_variant("Variant 1 — EagerCoDEQ",   &base, &queries, &mut SmallRng::seed_from_u64(1));
    run_variant("Variant 2 — LazyCoDEQ",    &base, &queries, &mut SmallRng::seed_from_u64(2));
    run_variant_approx("Variant 3 — ApproxCoDEQ", &base, &queries, &mut SmallRng::seed_from_u64(3));

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("Legend:");
    println!("  Insert μs   = average per-insert wall-clock time across all batches");
    println!("  Query μs    = average per-query wall-clock time (LUT-scan over codes)");
    println!("  Recall@10   = fraction of true top-10 neighbours found (averaged over 50 queries)");
    println!("  Stale frac  = fraction of codes potentially misassigned after final batch");
    println!("  Reassigns   = cumulative codeword reassignment operations");
    println!("  Drift gates = times the KS sketch triggered a flush (Variant 3 only)");
}

fn run_variant(name: &str, base: &[Vector], queries: &[Vector], rng: &mut SmallRng) {
    println!("──────────────────────────────────────────────────────────────────");
    println!("{name}");

    let t0 = Instant::now();

    // Build
    let mut eager = EagerCoDEQ::new(base, rng);
    let build_ms = t0.elapsed().as_millis();

    // Streaming batches with progressive drift
    let mut total_insert_ns = 0u128;
    let mut total_inserts = 0usize;
    for batch in 0..BATCHES {
        let drift = 0.15 * (batch + 1) as f32;
        // Inserts
        for _ in 0..BATCH_SIZE {
            let v: Vector = (0..DIM)
                .map(|_| rng.gen_range(-1.0f32..1.0) + drift)
                .collect();
            let t = Instant::now();
            eager.insert(v);
            total_insert_ns += t.elapsed().as_nanos();
            total_inserts += 1;
        }
        // Deletes (remove oldest BATCH_SIZE/2 vectors)
        let del_count = BATCH_SIZE / 2;
        for _ in 0..del_count {
            if eager.vectors.len() > 1 {
                eager.delete(0);
            }
        }
    }

    // Query benchmark
    let mut total_query_ns = 0u128;
    let mut total_recall = 0.0f64;
    let current_vecs = eager.vectors.clone();
    for q in queries {
        let t = Instant::now();
        let pred = eager.knn(q, KNN_K);
        total_query_ns += t.elapsed().as_nanos();
        let gt = brute_force_knn(&current_vecs, q, KNN_K);
        total_recall += recall_at_k(&pred, &gt);
    }

    let avg_insert_us = (total_insert_ns as f64 / total_inserts as f64) / 1_000.0;
    let avg_query_us = (total_query_ns as f64 / queries.len() as f64) / 1_000.0;
    let avg_recall = total_recall / queries.len() as f64;

    println!("  Build time:     {build_ms} ms");
    println!("  Insert μs:      {avg_insert_us:.3}");
    println!("  Query μs:       {avg_query_us:.3}");
    println!("  Recall@10:      {:.4}", avg_recall);
    println!("  Final N:        {}", eager.stats.total_vectors);
    println!("  Reassigns:      {}", eager.stats.reassignments);
    println!("  Stale frac:     {:.4}", eager.stats.stale_fraction());
}

fn run_variant_approx(name: &str, base: &[Vector], queries: &[Vector], rng: &mut SmallRng) {
    println!("──────────────────────────────────────────────────────────────────");
    println!("{name}");

    let t0 = Instant::now();
    let mut approx = ApproxCoDEQ::new(base, rng, 0.05);
    let build_ms = t0.elapsed().as_millis();

    let mut total_insert_ns = 0u128;
    let mut total_inserts = 0usize;
    for batch in 0..BATCHES {
        let drift = 0.15 * (batch + 1) as f32;
        for _ in 0..BATCH_SIZE {
            let v: Vector = (0..DIM)
                .map(|_| rng.gen_range(-1.0f32..1.0) + drift)
                .collect();
            let t = Instant::now();
            approx.insert(v);
            total_insert_ns += t.elapsed().as_nanos();
            total_inserts += 1;
        }
        let del_count = BATCH_SIZE / 2;
        for _ in 0..del_count {
            if approx.vectors.len() > 1 {
                approx.delete(0);
            }
        }
    }

    let mut total_query_ns = 0u128;
    let mut total_recall = 0.0f64;
    let current_vecs = approx.vectors.clone();
    for q in queries {
        let t = Instant::now();
        let pred = approx.knn(q, KNN_K);
        total_query_ns += t.elapsed().as_nanos();
        let gt = brute_force_knn(&current_vecs, q, KNN_K);
        total_recall += recall_at_k(&pred, &gt);
    }

    let avg_insert_us = (total_insert_ns as f64 / total_inserts as f64) / 1_000.0;
    let avg_query_us = (total_query_ns as f64 / queries.len() as f64) / 1_000.0;
    let avg_recall = total_recall / queries.len() as f64;

    println!("  Build time:     {build_ms} ms");
    println!("  Insert μs:      {avg_insert_us:.3}");
    println!("  Query μs:       {avg_query_us:.3}");
    println!("  Recall@10:      {:.4}", avg_recall);
    println!("  Final N:        {}", approx.stats.total_vectors);
    println!("  Reassigns:      {}", approx.stats.reassignments);
    println!("  Stale frac:     {:.4}", approx.stats.stale_fraction());
    println!("  Drift triggers: {}", approx.stats.drift_triggers);
}
