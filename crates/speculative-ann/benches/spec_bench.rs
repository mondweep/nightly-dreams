use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use speculative_ann::{LinearF32Index, LinearI8Index, SpecAnnIndex, SpecSearch};

const DIM: usize = 128;
const K: usize = 10;

fn make_corpus(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let mut v: Vec<f32> = (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
            speculative_ann::normalize(&mut v);
            v
        })
        .collect()
}

fn make_query(seed: u64) -> Vec<f32> {
    let mut rng = SmallRng::seed_from_u64(seed + 9999);
    let mut v: Vec<f32> = (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    speculative_ann::normalize(&mut v);
    v
}

fn build_f32(corpus: &[Vec<f32>]) -> LinearF32Index {
    let mut idx = LinearF32Index::new(DIM);
    for (id, v) in corpus.iter().enumerate() {
        idx.insert(id, v);
    }
    idx.build();
    idx
}

fn build_i8(corpus: &[Vec<f32>]) -> LinearI8Index {
    let mut idx = LinearI8Index::new(DIM);
    for (id, v) in corpus.iter().enumerate() {
        idx.insert(id, v);
    }
    idx.build();
    idx
}

fn build_spec(corpus: &[Vec<f32>], overdraft: usize) -> SpecSearch {
    let mut idx = SpecSearch::new(DIM, overdraft);
    for (id, v) in corpus.iter().enumerate() {
        idx.insert(id, v);
    }
    idx.build();
    idx
}

pub fn bench_search(c: &mut Criterion) {
    let n = 10_000usize;
    let corpus = make_corpus(n, 42);
    let query = make_query(42);

    let f32_idx = build_f32(&corpus);
    let i8_idx = build_i8(&corpus);
    let spec4_idx = build_spec(&corpus, 4);
    let spec8_idx = build_spec(&corpus, 8);

    let mut group = c.benchmark_group("ann_search");
    group.throughput(Throughput::Elements(1));

    group.bench_with_input(
        BenchmarkId::new("linear-f32", n),
        &query,
        |b, q| b.iter(|| f32_idx.search(q, K)),
    );
    group.bench_with_input(
        BenchmarkId::new("linear-i8", n),
        &query,
        |b, q| b.iter(|| i8_idx.search(q, K)),
    );
    group.bench_with_input(
        BenchmarkId::new("spec-search-od4", n),
        &query,
        |b, q| b.iter(|| spec4_idx.search(q, K)),
    );
    group.bench_with_input(
        BenchmarkId::new("spec-search-od8", n),
        &query,
        |b, q| b.iter(|| spec8_idx.search(q, K)),
    );

    group.finish();
}

pub fn bench_corpus_sizes(c: &mut Criterion) {
    let query = make_query(77);
    let mut group = c.benchmark_group("corpus_size_scaling");
    group.throughput(Throughput::Elements(1));

    for &n in &[1_000usize, 5_000, 10_000, 50_000] {
        let corpus = make_corpus(n, 42);

        let f32_idx = build_f32(&corpus);
        let spec_idx = build_spec(&corpus, 4);

        group.bench_with_input(
            BenchmarkId::new("linear-f32", n),
            &query,
            |b, q| b.iter(|| f32_idx.search(q, K)),
        );
        group.bench_with_input(
            BenchmarkId::new("spec-od4", n),
            &query,
            |b, q| b.iter(|| spec_idx.search(q, K)),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_search, bench_corpus_sizes);
criterion_main!(benches);
