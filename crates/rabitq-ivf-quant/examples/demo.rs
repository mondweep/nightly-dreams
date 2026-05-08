/// RaBitQ + IVF live benchmark — run with:
///   cargo run --example demo --release -p rabitq-ivf-quant
///
/// Measures build time, query throughput (QPS), memory, and recall@10
/// for four variants on 10 000 synthetic 128-d vectors.
use rabitq_ivf_quant::{BitFlatStore, FlatStore, IvfBitStore, VectorIndex};
use std::collections::HashSet;
use std::time::Instant;

const N: usize = 10_000;
const DIM: usize = 128;
const N_QUERIES: usize = 500;
const K: usize = 10;
const RESCORE_K: usize = 64;
const IVF_NLIST: usize = 32;
const IVF_NPROBE: usize = 6;

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    // >> 32 keeps full 32-bit range → uniform in [-1, 1)
    ((*state >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn make_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..dim).map(|_| lcg(&mut s)).collect()
}

fn recall(pred: &[(usize, f32)], gt: &[(usize, f32)]) -> f32 {
    let gt_ids: HashSet<usize> = gt.iter().take(K).map(|(id, _)| *id).collect();
    let pred_ids: HashSet<usize> = pred.iter().take(K).map(|(id, _)| *id).collect();
    gt_ids.intersection(&pred_ids).count() as f32 / K as f32
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   ruvector RaBitQ + IVF Quantization Benchmark (2026)   ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("  Corpus : {N} vectors × {DIM} dimensions (f32)");
    println!("  Queries: {N_QUERIES}  k={K}  rescore_k={RESCORE_K}");
    println!(
        "  IVF    : nlist={IVF_NLIST}  nprobe={IVF_NPROBE}\n"
    );

    let corpus: Vec<Vec<f32>> = (0..N).map(|i| make_vec(DIM, i as u64 * 1337 + 1)).collect();
    let queries: Vec<Vec<f32>> =
        (0..N_QUERIES).map(|i| make_vec(DIM, i as u64 * 9_999 + 7)).collect();

    // ── Baseline: exact f32 FlatStore ──────────────────────────────────────
    let mut flat = FlatStore::new(DIM);
    let t0 = Instant::now();
    for (i, v) in corpus.iter().enumerate() {
        flat.insert(i, v);
    }
    let flat_build = t0.elapsed();

    let t0 = Instant::now();
    let gt: Vec<Vec<(usize, f32)>> = queries.iter().map(|q| flat.search(q, K)).collect();
    let flat_query = t0.elapsed();

    print_row(
        "Exact f32 (baseline)",
        flat_build.as_millis(),
        N_QUERIES as f64 / flat_query.as_secs_f64(),
        flat.memory_bytes(),
        N,
        1.0,
        None,
    );

    // ── Variant 1: 1-bit RaBitQ, no rescore ────────────────────────────────
    let mut bit1 = BitFlatStore::new(DIM, false, 0);
    let t0 = Instant::now();
    for (i, v) in corpus.iter().enumerate() {
        bit1.insert(i, v);
    }
    let bit1_build = t0.elapsed();

    let t0 = Instant::now();
    let res1: Vec<Vec<(usize, f32)>> = queries.iter().map(|q| bit1.search(q, K)).collect();
    let bit1_query = t0.elapsed();
    let recall1 = avg_recall(&res1, &gt);

    print_row(
        "RaBitQ 1-bit (no rescore)",
        bit1_build.as_millis(),
        N_QUERIES as f64 / bit1_query.as_secs_f64(),
        bit1.memory_bytes(),
        N,
        recall1,
        Some(flat.memory_bytes() as f32 / bit1.memory_bytes() as f32),
    );

    // ── Variant 2: 1-bit RaBitQ + exact rescore top-64 ─────────────────────
    let mut bit2 = BitFlatStore::new(DIM, true, RESCORE_K);
    let t0 = Instant::now();
    for (i, v) in corpus.iter().enumerate() {
        bit2.insert(i, v);
    }
    let bit2_build = t0.elapsed();

    let t0 = Instant::now();
    let res2: Vec<Vec<(usize, f32)>> = queries.iter().map(|q| bit2.search(q, K)).collect();
    let bit2_query = t0.elapsed();
    let recall2 = avg_recall(&res2, &gt);

    print_row(
        "RaBitQ + rescore-64",
        bit2_build.as_millis(),
        N_QUERIES as f64 / bit2_query.as_secs_f64(),
        bit2.memory_bytes(),
        N,
        recall2,
        Some(flat.memory_bytes() as f32 / bit2.memory_bytes() as f32),
    );

    // ── Variant 3: IVF-32 + RaBitQ + rescore ──────────────────────────────
    let corpus_pairs: Vec<(usize, Vec<f32>)> = corpus.iter().cloned().enumerate().collect();
    let mut ivf = IvfBitStore::new(DIM, IVF_NLIST, IVF_NPROBE, RESCORE_K);
    let t0 = Instant::now();
    ivf.build(&corpus_pairs);
    let ivf_build = t0.elapsed();

    let t0 = Instant::now();
    let res3: Vec<Vec<(usize, f32)>> = queries.iter().map(|q| ivf.search(q, K)).collect();
    let ivf_query = t0.elapsed();
    let recall3 = avg_recall(&res3, &gt);

    print_row(
        &format!("IVF-{IVF_NLIST}+RaBitQ (nprobe={IVF_NPROBE})"),
        ivf_build.as_millis(),
        N_QUERIES as f64 / ivf_query.as_secs_f64(),
        ivf.memory_bytes(),
        N,
        recall3,
        Some(flat.memory_bytes() as f32 / ivf.memory_bytes() as f32),
    );

    println!("\n  * Memory excludes Rust Vec overhead; measures stored payload bytes.");
    println!("  * Recall@{K} = fraction of true top-{K} neighbors found on average.");
    println!("  * Compression ratio = baseline_bytes / index_bytes (bits only).");
}

fn avg_recall(results: &[Vec<(usize, f32)>], gt: &[Vec<(usize, f32)>]) -> f32 {
    results
        .iter()
        .zip(gt.iter())
        .map(|(r, g)| recall(r, g))
        .sum::<f32>()
        / results.len() as f32
}

fn print_row(
    name: &str,
    build_ms: u128,
    qps: f64,
    mem: usize,
    n: usize,
    r: f32,
    compress: Option<f32>,
) {
    let bytes_per_vec = mem / n;
    let compress_str = compress
        .map(|c| format!("{c:.1}x"))
        .unwrap_or_else(|| "—".to_string());
    println!(
        "  ┌─ {name}");
    println!(
        "  │  build={build_ms}ms  QPS={qps:.0}  mem={}KB ({bytes_per_vec}B/vec)  recall@{K}={r:.3}  compress={compress_str}",
        mem / 1024
    );
    println!("  └──");
}
