use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mrl_cascaded_search::{CascadeConfig, DotProduct, FlatIndex, MrlCascadedIndex};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::Rng;

fn make_unit_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            v.into_iter().map(|x| x / norm).collect()
        })
        .collect()
}

fn bench_variants(c: &mut Criterion) {
    const N: usize = 5_000;
    const FULL_DIM: usize = 1024;
    const K: usize = 10;

    let corpus = make_unit_vecs(N, FULL_DIM, 42);
    let query = make_unit_vecs(1, FULL_DIM, 99)[0].clone();
    let dist = DotProduct;

    // Build baseline
    let mut baseline = FlatIndex::new(FULL_DIM);
    for (i, v) in corpus.iter().enumerate() {
        baseline.insert(i as u64, v.clone());
    }

    // Build MRL-128
    let mut mrl128 = MrlCascadedIndex::new(
        FULL_DIM,
        CascadeConfig { coarse_dim: 128, oversample: 10, fine_dim: FULL_DIM },
    );
    for (i, v) in corpus.iter().enumerate() {
        mrl128.insert(i as u64, v.clone());
    }

    // Build MRL-64
    let mut mrl64 = MrlCascadedIndex::new(
        FULL_DIM,
        CascadeConfig { coarse_dim: 64, oversample: 20, fine_dim: FULL_DIM },
    );
    for (i, v) in corpus.iter().enumerate() {
        mrl64.insert(i as u64, v.clone());
    }

    let mut group = c.benchmark_group("cascade_search_k10");

    group.bench_function(BenchmarkId::new("baseline_1024dim", N), |b| {
        b.iter(|| baseline.search_at_dim(&query, FULL_DIM, K, &dist))
    });

    group.bench_function(BenchmarkId::new("mrl128_cascade", N), |b| {
        b.iter(|| mrl128.search(&query, K, &dist))
    });

    group.bench_function(BenchmarkId::new("mrl64_cascade", N), |b| {
        b.iter(|| mrl64.search(&query, K, &dist))
    });

    group.finish();
}

criterion_group!(benches, bench_variants);
criterion_main!(benches);
