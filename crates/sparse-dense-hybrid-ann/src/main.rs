use std::collections::BTreeMap;
use std::time::Instant;

use sparse_dense_hybrid_ann::{
    BruteForce, HybridNSW, HybridVector, SearchMode, calibrate_alpha, recall_at_k,
};

// ─────────────────────── Deterministic PRNG ──────────────────────────────────

fn xorshift64(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f32) / (u64::MAX as f32)
}

fn make_vec(dense_dim: usize, vocab: u32, nnz: usize, seed: u64) -> HybridVector {
    let mut s = seed ^ 0x9e3779b97f4a7c15;

    let mut dense: Vec<f32> = (0..dense_dim).map(|_| xorshift64(&mut s) * 2.0 - 1.0).collect();
    let norm = dense.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    for x in &mut dense {
        *x /= norm;
    }

    let mut sparse: BTreeMap<u32, f32> = BTreeMap::new();
    for i in 0..nnz {
        let k = ((s ^ (i as u64).wrapping_mul(2862933555777941757)) % vocab as u64) as u32;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let w = xorshift64(&mut s).abs() + 1e-4;
        sparse.entry(k).and_modify(|e| *e += w).or_insert(w);
    }
    // L2-normalise sparse weights
    let snorm = sparse.values().map(|v| v * v).sum::<f32>().sqrt().max(1e-9);
    for v in sparse.values_mut() {
        *v /= snorm;
    }

    HybridVector::new(dense, sparse)
}

// ─────────────────────── Benchmark helpers ───────────────────────────────────

struct BenchResult {
    label: String,
    qps: f64,
    recall: f32,
}

fn run_bench(
    label: &str,
    nsw: &HybridNSW,
    bf: &BruteForce,
    queries: &[HybridVector],
    k: usize,
    mode: SearchMode,
    gt_alpha: f32,
) -> BenchResult {
    let mut total_recall = 0.0f32;
    let t0 = Instant::now();
    for q in queries {
        let gt = bf.search(q, k, gt_alpha);
        let res = nsw.search(q, k, mode);
        total_recall += recall_at_k(&gt, &res);
    }
    let elapsed = t0.elapsed();
    BenchResult {
        label: label.to_string(),
        qps: queries.len() as f64 / elapsed.as_secs_f64(),
        recall: total_recall / queries.len() as f32,
    }
}

fn print_row(r: &BenchResult, k: usize) {
    println!("  {:<38} | QPS: {:>9.1} | Recall@{k}: {:.4}", r.label, r.qps, r.recall);
}

// ─────────────────────── Main ────────────────────────────────────────────────

fn main() {
    const N: usize = 2_000;
    const DIM: usize = 128;
    const VOCAB: u32 = 1_024;
    const NNZ: usize = 20;
    const N_Q: usize = 200;
    const K: usize = 10;
    const M: usize = 16;

    println!("=== Sparse-Dense Hybrid ANNS Benchmark ===");
    println!(
        "Corpus N={N}, dense_dim={DIM}, sparse_vocab={VOCAB}, nnz_per_vec={NNZ}, queries={N_Q}, k={K}, M={M}"
    );
    println!();

    // ── Dataset ────────────────────────────────────────────────────────────────
    let data: Vec<HybridVector> = (0..N)
        .map(|i| make_vec(DIM, VOCAB, NNZ, (i as u64).wrapping_mul(6364136223846793005).wrapping_add(1)))
        .collect();

    let queries: Vec<HybridVector> = (0..N_Q)
        .map(|i| make_vec(DIM, VOCAB, NNZ, (i as u64).wrapping_mul(2862933555777941757).wrapping_add(3)))
        .collect();

    // ── Distribution-aligned alpha ─────────────────────────────────────────────
    let alpha = calibrate_alpha(&data);
    println!("Distribution-aligned alpha = {alpha:.4}");

    // ── Ground truth (brute-force) ─────────────────────────────────────────────
    let bf = BruteForce::new(data.clone());

    // ── Build Variant 1: Dense-only NSW (alpha=1.0) ────────────────────────────
    let t_build = Instant::now();
    let mut nsw_dense = HybridNSW::new(M, 1.0);
    for v in &data {
        nsw_dense.insert(v.clone());
    }
    let build_ms_dense = t_build.elapsed().as_millis();

    // ── Build Variant 2: Hybrid NSW (calibrated alpha) ─────────────────────────
    let t_build = Instant::now();
    let mut nsw_hybrid = HybridNSW::new(M, alpha);
    for v in &data {
        nsw_hybrid.insert(v.clone());
    }
    let build_ms_hybrid = t_build.elapsed().as_millis();

    println!("Build times — DenseOnly: {build_ms_dense}ms, Hybrid(α={alpha:.4}): {build_ms_hybrid}ms");
    println!();

    // ── Brute-force QPS ────────────────────────────────────────────────────────
    let t0 = Instant::now();
    let mut _sink = 0usize;
    for q in &queries {
        _sink += bf.search(q, K, alpha).len();
    }
    let bf_qps = N_Q as f64 / t0.elapsed().as_secs_f64();
    println!("  {:<38} | QPS: {:>9.1} | Recall@{K}: 1.0000 (ground truth)", "BruteForce", bf_qps);

    // ── Variant 1: Dense-only NSW ──────────────────────────────────────────────
    let r1 = run_bench(
        "Variant-1: DenseOnly-NSW",
        &nsw_dense,
        &bf,
        &queries,
        K,
        SearchMode::HybridFull,
        1.0,
    );
    print_row(&r1, K);

    // ── Variant 2: Hybrid-Full NSW ─────────────────────────────────────────────
    let r2 = run_bench(
        "Variant-2: HybridFull-NSW (α-align)",
        &nsw_hybrid,
        &bf,
        &queries,
        K,
        SearchMode::HybridFull,
        alpha,
    );
    print_row(&r2, K);

    // ── Variant 3a: Two-Stage NSW (small pool — fast but lower recall) ────────
    let r3a = run_bench(
        "Variant-3a: TwoStage-NSW (cf=8)",
        &nsw_hybrid,
        &bf,
        &queries,
        K,
        SearchMode::TwoStage { candidates_factor: 8 },
        alpha,
    );
    print_row(&r3a, K);

    // ── Variant 3b: Two-Stage NSW (large pool — higher recall) ────────────────
    let r3b = run_bench(
        "Variant-3b: TwoStage-NSW (cf=30)",
        &nsw_hybrid,
        &bf,
        &queries,
        K,
        SearchMode::TwoStage { candidates_factor: 30 },
        alpha,
    );
    print_row(&r3b, K);

    println!();
    println!("--- Alpha sensitivity (HybridFull-NSW) ---");
    for &a in &[0.2f32, 0.35, 0.5, 0.65, 0.8] {
        let mut nsw_a = HybridNSW::new(M, a);
        for v in &data {
            nsw_a.insert(v.clone());
        }
        let r = run_bench(
            &format!("  HybridFull α={a:.2}"),
            &nsw_a,
            &bf,
            &queries,
            K,
            SearchMode::HybridFull,
            a,
        );
        print_row(&r, K);
    }

    println!();

    // ── Larger-N crossover test (N=8000, no alpha sweep) ──────────────────────
    println!("--- Crossover: N=8000 (where graph search amortises traversal overhead) ---");
    const N2: usize = 8_000;
    let data2: Vec<HybridVector> = (0..N2)
        .map(|i| make_vec(DIM, VOCAB, NNZ, (i as u64).wrapping_mul(1442695040888963407).wrapping_add(5)))
        .collect();
    let queries2: Vec<HybridVector> = (0..50)
        .map(|i| make_vec(DIM, VOCAB, NNZ, (i as u64).wrapping_mul(6364136223846793005).wrapping_add(7)))
        .collect();
    let bf2 = BruteForce::new(data2.clone());

    let t_bf2 = Instant::now();
    let mut _s2 = 0usize;
    for q in &queries2 {
        _s2 += bf2.search(q, K, alpha).len();
    }
    let bf2_qps = 50.0f64 / t_bf2.elapsed().as_secs_f64();

    let mut nsw2 = HybridNSW::new(M, alpha);
    for v in &data2 { nsw2.insert(v.clone()); }

    let r_n2 = run_bench(
        "HybridFull-NSW N=8000",
        &nsw2, &bf2, &queries2, K,
        SearchMode::HybridFull, alpha,
    );
    println!("  {:<38} | QPS: {:>9.1} | Recall@{K}: 1.0000 (ground truth)", "BruteForce N=8000", bf2_qps);
    print_row(&r_n2, K);
    println!("  Speedup at N=8000: {:.2}x", r_n2.qps / bf2_qps);

    println!();

    // ── Memory estimate ────────────────────────────────────────────────────────
    let bytes_dense = N * DIM * 4; // f32 per dimension
    let bytes_sparse = N * NNZ * (4 + 4); // u32 key + f32 val
    let bytes_graph = N * M * 8; // usize edge × 2 directions approx
    let total_mb = (bytes_dense + bytes_sparse + bytes_graph) as f64 / 1_048_576.0;
    println!(
        "Memory estimate: dense={:.1}MB sparse={:.1}MB graph={:.1}MB total≈{total_mb:.1}MB",
        bytes_dense as f64 / 1_048_576.0,
        bytes_sparse as f64 / 1_048_576.0,
        bytes_graph as f64 / 1_048_576.0
    );

    println!();
    println!("Speedup Variant-2 vs BruteForce: {:.2}x", r2.qps / bf_qps);
    println!("Speedup Variant-3a vs BruteForce: {:.2}x  (recall={:.4})", r3a.qps / bf_qps, r3a.recall);
    println!("Speedup Variant-3b vs BruteForce: {:.2}x  (recall={:.4})", r3b.qps / bf_qps, r3b.recall);
    println!("Done.");
}
