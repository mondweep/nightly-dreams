use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use speculative_ann::{
    recall_at_k, LinearF32Index, LinearI8Index, SpecAnnIndex, SpecSearch,
};
use std::time::Instant;

const N_DOCS: usize = 10_000;
const DIM: usize = 128;
const K: usize = 10;
const N_QUERIES: usize = 100;
const OVERDRAFT: usize = 4;
const SEED: u64 = 42;

fn random_unit_vec(rng: &mut SmallRng, dim: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    speculative_ann::normalize(&mut v);
    v
}

fn bench_index(
    name: &str,
    index: &dyn SpecAnnIndex,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<(usize, f32)>],
) {
    let t0 = Instant::now();
    let mut total_recall = 0.0f32;
    for (q, gt) in queries.iter().zip(ground_truth.iter()) {
        let results = index.search(q, K);
        total_recall += recall_at_k(&results, gt);
    }
    let elapsed = t0.elapsed();
    let qps = N_QUERIES as f64 / elapsed.as_secs_f64();
    let avg_recall = total_recall / N_QUERIES as f32;
    println!(
        "  {:<20} recall@{K}: {:.3}  QPS: {:.0}  latency: {:.3} ms/q",
        name,
        avg_recall,
        qps,
        elapsed.as_secs_f64() * 1000.0 / N_QUERIES as f64,
    );
}

fn main() {
    println!("=== Speculative ANN Demo ===");
    println!("corpus: {N_DOCS} docs × {DIM} dims | k={K} | {N_QUERIES} queries\n");

    let mut rng = SmallRng::seed_from_u64(SEED);

    let corpus: Vec<Vec<f32>> = (0..N_DOCS).map(|_| random_unit_vec(&mut rng, DIM)).collect();
    let queries: Vec<Vec<f32>> = (0..N_QUERIES)
        .map(|_| random_unit_vec(&mut rng, DIM))
        .collect();

    // Build all three indexes
    let build_indexes = |name: &str, mut idx: Box<dyn SpecAnnIndex>| -> Box<dyn SpecAnnIndex> {
        let t0 = Instant::now();
        for (id, v) in corpus.iter().enumerate() {
            idx.insert(id, v);
        }
        idx.build();
        println!("  {name:<20} build: {:.2} ms", t0.elapsed().as_secs_f64() * 1000.0);
        idx
    };

    println!("Build times:");
    let f32_idx = build_indexes("linear-f32", Box::new(LinearF32Index::new(DIM)));
    let i8_idx = build_indexes("linear-i8", Box::new(LinearI8Index::new(DIM)));
    let spec_idx = build_indexes(
        &format!("spec-search (od={})", OVERDRAFT),
        Box::new(SpecSearch::new(DIM, OVERDRAFT)),
    );

    // Compute ground truth from f32 index
    let ground_truth: Vec<Vec<(usize, f32)>> =
        queries.iter().map(|q| f32_idx.search(q, K)).collect();

    println!("\nSearch performance:");
    bench_index("linear-f32 (exact)", f32_idx.as_ref(), &queries, &ground_truth);
    bench_index("linear-i8 (draft)", i8_idx.as_ref(), &queries, &ground_truth);
    bench_index(
        &format!("spec-search od={}", OVERDRAFT),
        spec_idx.as_ref(),
        &queries,
        &ground_truth,
    );

    // Extra: show how recall scales with overdraft factor
    println!("\nRecall@{K} vs overdraft factor (spec-search, {N_DOCS} docs):");
    for od in [1, 2, 4, 8, 16] {
        let mut spec = SpecSearch::new(DIM, od);
        for (id, v) in corpus.iter().enumerate() {
            spec.insert(id, v);
        }
        spec.build();
        let total: f32 = queries
            .iter()
            .zip(ground_truth.iter())
            .map(|(q, gt)| recall_at_k(&spec.search(q, K), gt))
            .sum();
        let avg = total / N_QUERIES as f32;
        // re-measure latency
        let t0 = Instant::now();
        for q in &queries {
            let _ = spec.search(q, K);
        }
        let ms_per_q = t0.elapsed().as_secs_f64() * 1000.0 / N_QUERIES as f64;
        println!("  overdraft={od:>2}  recall={avg:.3}  {ms_per_q:.3} ms/q");
    }

    println!("\nDone.");
}
