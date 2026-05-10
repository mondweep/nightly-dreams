//! MRL cascaded search demo: measures recall and throughput across three variants
//! on two corpora — random vectors (worst case) and MRL-structured vectors
//! (cluster-centre + noise, simulating prefix-discriminative embeddings such as
//! OpenAI text-embedding-3 or Nomic embed v1.5).
//!
//! Usage: cargo run --release -p mrl-cascaded-search

use std::time::Instant;

use mrl_cascaded_search::{
    CascadeConfig, DotProduct, FlatIndex, MrlCascadedIndex, recall_at_k,
};

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::Rng;

fn make_unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            v.into_iter().map(|x| x / norm).collect()
        })
        .collect()
}

/// Prefix-predictive data model for MRL-structured embeddings.
///
/// The first `coarse_dim` dimensions carry the full semantic identity of the
/// vector (a low-dimensional unit embedding), while the remaining dimensions
/// add fine-grained noise.  This matches how MRL training works: d' leading
/// dimensions form an already-useful d'-dimensional embedding.
///
/// `fine_noise` controls how much noise is added to the suffix dimensions.
/// Lower values → prefix is a near-exact predictor; higher → more uncertainty.
fn make_mrl_vectors(
    n: usize,
    dim: usize,
    coarse_dim: usize,
    fine_noise: f32,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            // Semantic "key" lives in the first coarse_dim dimensions.
            let semantics: Vec<f32> = (0..coarse_dim)
                .map(|_| rng.gen::<f32>() - 0.5)
                .collect();
            let s_norm = semantics.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);

            // Build full vector: [semantics/norm | fine_noise dims]
            let mut v: Vec<f32> = semantics.iter().map(|x| x / s_norm).collect();
            for _ in coarse_dim..dim {
                v.push((rng.gen::<f32>() - 0.5) * fine_noise);
            }
            // Normalise full vector so inner-product and L2 are consistent.
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            v.iter_mut().for_each(|x| *x /= norm);
            v
        })
        .collect()
}

#[derive(Clone)]
struct BenchResult {
    label: String,
    coarse_dim: usize,
    fine_dim: usize,
    mean_recall: f64,
    qps: f64,
    latency_us: f64,
}

fn run_scenario(
    label: &str,
    corpus: &[Vec<f32>],
    queries: &[Vec<f32>],
    full_dim: usize,
    k: usize,
) -> Vec<BenchResult> {
    let n_queries = queries.len();
    let dist = DotProduct;

    // ---- Build baseline flat index ----
    let mut baseline_idx = FlatIndex::new(full_dim);
    for (i, v) in corpus.iter().enumerate() {
        baseline_idx.insert(i as u64, v.clone());
    }

    // ---- Build MRL-128 → 1024 (oversample 10×) ----
    let mut mrl128 = MrlCascadedIndex::new(
        full_dim,
        CascadeConfig { coarse_dim: 128, oversample: 10, fine_dim: full_dim },
    );
    for (i, v) in corpus.iter().enumerate() {
        mrl128.insert(i as u64, v.clone());
    }

    // ---- Build MRL-64 → 1024 (oversample 20×) ----
    let mut mrl64 = MrlCascadedIndex::new(
        full_dim,
        CascadeConfig { coarse_dim: 64, oversample: 20, fine_dim: full_dim },
    );
    for (i, v) in corpus.iter().enumerate() {
        mrl64.insert(i as u64, v.clone());
    }

    // ---- Time each variant ----
    let t0 = Instant::now();
    let gt: Vec<_> = queries.iter().map(|q| baseline_idx.search_at_dim(q, full_dim, k, &dist)).collect();
    let baseline_us = t0.elapsed().as_micros() as f64 / n_queries as f64;

    let t1 = Instant::now();
    let res128: Vec<_> = queries.iter().map(|q| mrl128.search(q, k, &dist)).collect();
    let us128 = t1.elapsed().as_micros() as f64 / n_queries as f64;

    let t2 = Instant::now();
    let res64: Vec<_> = queries.iter().map(|q| mrl64.search(q, k, &dist)).collect();
    let us64 = t2.elapsed().as_micros() as f64 / n_queries as f64;

    let recall128 = gt.iter().zip(&res128).map(|(g, r)| recall_at_k(g, r, k)).sum::<f64>() / n_queries as f64;
    let recall64  = gt.iter().zip(&res64) .map(|(g, r)| recall_at_k(g, r, k)).sum::<f64>() / n_queries as f64;

    vec![
        BenchResult {
            label: format!("{label} — Baseline 1024-dim"),
            coarse_dim: full_dim, fine_dim: full_dim,
            mean_recall: 1.0,
            qps: 1_000_000.0 / baseline_us,
            latency_us: baseline_us,
        },
        BenchResult {
            label: format!("{label} — MRL-128 → 1024 (10× oversample)"),
            coarse_dim: 128, fine_dim: full_dim,
            mean_recall: recall128,
            qps: 1_000_000.0 / us128,
            latency_us: us128,
        },
        BenchResult {
            label: format!("{label} — MRL-64 → 1024 (20× oversample)"),
            coarse_dim: 64, fine_dim: full_dim,
            mean_recall: recall64,
            qps: 1_000_000.0 / us64,
            latency_us: us64,
        },
    ]
}

fn main() {
    const N: usize = 10_000;
    const FULL_DIM: usize = 1024;
    const N_QUERIES: usize = 200;
    const K: usize = 10;

    println!("=== MRL Cascaded Search Benchmark ===");
    println!("Hardware: {}", std::env::consts::ARCH);
    println!("Corpus:   {N} vectors × {FULL_DIM} dims");
    println!("Queries:  {N_QUERIES}  k={K}");
    println!();

    // Scenario A: uniform random unit vectors (worst-case, no MRL structure)
    let rnd_corpus  = make_unit_vectors(N, FULL_DIM, 42);
    let rnd_queries = make_unit_vectors(N_QUERIES, FULL_DIM, 43);

    // Scenario B: MRL-structured (128-dim semantic key + fine noise in dims 129-1024)
    // fine_noise=0.05 → suffix adds little distance noise relative to prefix.
    let mrl_corpus  = make_mrl_vectors(N, FULL_DIM, 128, 0.05, 42);
    let mrl_queries = make_mrl_vectors(N_QUERIES, FULL_DIM, 128, 0.05, 43);

    let all_results: Vec<BenchResult> = [
        run_scenario("Random", &rnd_corpus, &rnd_queries, FULL_DIM, K),
        run_scenario("MRL-structured", &mrl_corpus, &mrl_queries, FULL_DIM, K),
    ]
    .concat();

    println!(
        "{:<55} {:>6} {:>6} {:>11} {:>10}",
        "Variant", "Coarse", "Fine", "Recall@10", "QPS"
    );
    println!("{}", "-".repeat(95));

    let mut mrl_128_recall = 0.0f64;
    let mut mrl_64_recall  = 0.0f64;
    let mut baseline_lat   = 0.0f64;
    let mut mrl128_lat     = 0.0f64;
    let mut mrl64_lat      = 0.0f64;

    for v in &all_results {
        println!(
            "{:<55} {:>6} {:>6} {:>10.1}% {:>10.0}",
            v.label, v.coarse_dim, v.fine_dim,
            v.mean_recall * 100.0,
            v.qps,
        );
        // collect MRL-structured scenario numbers for assertions
        if v.label.contains("MRL-structured") {
            if v.coarse_dim == 128 {
                mrl_128_recall = v.mean_recall;
                mrl128_lat = v.latency_us;
            } else if v.coarse_dim == 64 {
                mrl_64_recall = v.mean_recall;
                mrl64_lat = v.latency_us;
            } else {
                baseline_lat = v.latency_us;
            }
        }
    }

    println!();
    println!("Speedup on MRL-structured corpus:");
    println!("  MRL-128: {:.2}× faster than baseline", baseline_lat / mrl128_lat);
    println!("  MRL-64:  {:.2}× faster than baseline", baseline_lat / mrl64_lat);
    println!();
    println!("Dimension savings (coarse pass):");
    println!("  MRL-128: {:.0}% fewer FLOPs in coarse pass", (1.0 - 128.0 / FULL_DIM as f64) * 100.0);
    println!("  MRL-64:  {:.0}% fewer FLOPs in coarse pass", (1.0 - 64.0  / FULL_DIM as f64) * 100.0);
    println!();

    assert!(
        mrl_128_recall >= 0.70,
        "MRL-128 recall@{K} = {mrl_128_recall:.3} fell below 0.70 on MRL-structured data"
    );
    assert!(
        mrl_64_recall >= 0.60,
        "MRL-64 recall@{K} = {mrl_64_recall:.3} fell below 0.60 on MRL-structured data"
    );

    println!("All assertions passed. Results are valid.");
}
