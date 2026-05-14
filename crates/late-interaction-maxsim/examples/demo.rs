use late_interaction_maxsim::{
    CentroidIndex, FdeIndex, LinearIndex, MaxSimIndex, MultiVec, normalize,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use std::time::Instant;

const N_DOCS: usize = 2_000;
const K_DOC: usize = 16; // tokens per document
const K_QUERY: usize = 8; // tokens per query
const DIM: usize = 128;
const N_QUERIES: usize = 200;
const TOP_K: usize = 10;

fn rand_multivec(rng: &mut SmallRng, n_tokens: usize, dim: usize) -> MultiVec {
    (0..n_tokens)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
            normalize(&mut v);
            v
        })
        .collect()
}

fn recall_at_k(gt: &[usize], pred: &[usize]) -> f64 {
    let gt_set: HashSet<usize> = gt.iter().cloned().collect();
    pred.iter().filter(|id| gt_set.contains(id)).count() as f64 / gt.len() as f64
}

fn benchmark<I: MaxSimIndex>(
    idx: &mut I,
    docs: &[MultiVec],
    queries: &[MultiVec],
    ground_truth: &[Vec<usize>],
) -> (f64, f64) {
    for (i, doc) in docs.iter().enumerate() {
        idx.insert(i, doc.clone());
    }
    idx.build();

    let start = Instant::now();
    let results: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| {
            idx.search(q, TOP_K)
                .into_iter()
                .map(|(id, _)| id)
                .collect()
        })
        .collect();
    let elapsed = start.elapsed().as_secs_f64();

    let qps = N_QUERIES as f64 / elapsed;
    let recall = ground_truth
        .iter()
        .zip(&results)
        .map(|(gt, pred)| recall_at_k(gt, pred))
        .sum::<f64>()
        / N_QUERIES as f64;

    (qps, recall)
}

fn main() {
    let mut rng = SmallRng::seed_from_u64(42);

    let docs: Vec<MultiVec> = (0..N_DOCS)
        .map(|_| rand_multivec(&mut rng, K_DOC, DIM))
        .collect();
    let queries: Vec<MultiVec> = (0..N_QUERIES)
        .map(|_| rand_multivec(&mut rng, K_QUERY, DIM))
        .collect();

    println!("=== Late-Interaction MaxSim Benchmark ===");
    println!(
        "N_DOCS={N_DOCS}  tokens/doc={K_DOC}  tokens/query={K_QUERY}  dim={DIM}  queries={N_QUERIES}  top_k={TOP_K}"
    );
    println!(
        "{:<12}  {:>10}  {:>12}  Notes",
        "Variant", "QPS", "Recall@10"
    );
    println!("{}", "-".repeat(60));

    // --- Variant 1: Linear (ground truth) ---
    let mut linear = LinearIndex::new();
    let (linear_qps, _) = benchmark(&mut linear, &docs, &queries, &[]);
    // collect ground truth
    let ground_truth: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| {
            linear
                .search(q, TOP_K)
                .into_iter()
                .map(|(id, _)| id)
                .collect()
        })
        .collect();
    println!(
        "{:<12}  {:>10.0}  {:>12}  brute-force baseline",
        "linear", linear_qps, "1.000"
    );

    // --- Variant 2: Centroid (PLAID-inspired) ---
    let n_centroids = 100;
    let n_probe = 8;
    let mut centroid = CentroidIndex::new(n_centroids, n_probe);
    let (c_qps, c_recall) = benchmark(&mut centroid, &docs, &queries, &ground_truth);
    println!(
        "{:<12}  {:>10.0}  {:>12.3}  n_centroids={n_centroids} n_probe={n_probe}",
        "centroid", c_qps, c_recall
    );

    // --- Variant 3: FDE (MUVERA-inspired) ---
    let proj_dim = 64;
    let n_cand = 200;
    let mut fde = FdeIndex::new(proj_dim, n_cand);
    let (f_qps, f_recall) = benchmark(&mut fde, &docs, &queries, &ground_truth);
    println!(
        "{:<12}  {:>10.0}  {:>12.3}  proj_dim={proj_dim} candidates={n_cand}",
        "fde", f_qps, f_recall
    );

    // --- Variant 3b: FDE wider projection ---
    let proj_dim2 = 128;
    let n_cand2 = 400;
    let mut fde2 = FdeIndex::new(proj_dim2, n_cand2);
    let (f2_qps, f2_recall) = benchmark(&mut fde2, &docs, &queries, &ground_truth);
    println!(
        "{:<12}  {:>10.0}  {:>12.3}  proj_dim={proj_dim2} candidates={n_cand2}",
        "fde-wide", f2_qps, f2_recall
    );

    println!("{}", "-".repeat(60));
    println!("Hardware: synthetic dataset, single-threaded, --release build");
    println!("All scores are Chamfer/MaxSim (sum of per-token max cosine).");
}
