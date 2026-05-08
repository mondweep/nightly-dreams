use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rabitq_ivf_quant::{BitFlatStore, FlatStore, IvfBitStore, VectorIndex};

const DIM: usize = 128;
const N: usize = 5_000;
const RESCORE_K: usize = 64;

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn make_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..dim).map(|_| lcg(&mut s)).collect()
}

fn build_flat(n: usize) -> (FlatStore, Vec<Vec<f32>>) {
    let mut store = FlatStore::new(DIM);
    let queries: Vec<Vec<f32>> = (0..50).map(|i| make_vec(DIM, i as u64 * 9_999 + 7)).collect();
    for i in 0..n {
        store.insert(i, &make_vec(DIM, i as u64 * 1337 + 1));
    }
    (store, queries)
}

fn build_bit(n: usize, rescore: bool) -> (BitFlatStore, Vec<Vec<f32>>) {
    let mut store = BitFlatStore::new(DIM, rescore, RESCORE_K);
    let queries: Vec<Vec<f32>> = (0..50).map(|i| make_vec(DIM, i as u64 * 9_999 + 7)).collect();
    for i in 0..n {
        store.insert(i, &make_vec(DIM, i as u64 * 1337 + 1));
    }
    (store, queries)
}

fn build_ivf(n: usize) -> (IvfBitStore, Vec<Vec<f32>>) {
    let corpus: Vec<(usize, Vec<f32>)> =
        (0..n).map(|i| (i, make_vec(DIM, i as u64 * 1337 + 1))).collect();
    let queries: Vec<Vec<f32>> = (0..50).map(|i| make_vec(DIM, i as u64 * 9_999 + 7)).collect();
    let mut store = IvfBitStore::new(DIM, 20, 4, RESCORE_K);
    store.build(&corpus);
    (store, queries)
}

fn bench_query(c: &mut Criterion) {
    let (flat, fq) = build_flat(N);
    let (bit_raw, bq) = build_bit(N, false);
    let (bit_rsc, rq) = build_bit(N, true);
    let (ivf, iq) = build_ivf(N);

    let mut group = c.benchmark_group("search_k10");

    group.bench_function(BenchmarkId::new("flat_exact", N), |b| {
        b.iter(|| flat.search(black_box(&fq[0]), black_box(10)))
    });

    group.bench_function(BenchmarkId::new("rabitq_raw", N), |b| {
        b.iter(|| bit_raw.search(black_box(&bq[0]), black_box(10)))
    });

    group.bench_function(BenchmarkId::new("rabitq_rescore64", N), |b| {
        b.iter(|| bit_rsc.search(black_box(&rq[0]), black_box(10)))
    });

    group.bench_function(BenchmarkId::new("ivf_rabitq_nprobe4", N), |b| {
        b.iter(|| ivf.search(black_box(&iq[0]), black_box(10)))
    });

    group.finish();
}

criterion_group!(benches, bench_query);
criterion_main!(benches);
