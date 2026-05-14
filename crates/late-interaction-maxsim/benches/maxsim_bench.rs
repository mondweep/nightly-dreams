use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use late_interaction_maxsim::{
    CentroidIndex, FdeIndex, LinearIndex, MaxSimIndex, MultiVec, normalize,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

fn rand_multivec(rng: &mut SmallRng, n_tokens: usize, dim: usize) -> MultiVec {
    (0..n_tokens)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
            normalize(&mut v);
            v
        })
        .collect()
}

fn build_indexes(
    n_docs: usize,
) -> (LinearIndex, CentroidIndex, FdeIndex, Vec<MultiVec>) {
    let mut rng = SmallRng::seed_from_u64(7);
    let docs: Vec<MultiVec> = (0..n_docs)
        .map(|_| rand_multivec(&mut rng, 16, 128))
        .collect();

    let mut linear = LinearIndex::new();
    let mut centroid = CentroidIndex::new(50, 5);
    let mut fde = FdeIndex::new(64, n_docs / 5);

    for (i, doc) in docs.iter().enumerate() {
        linear.insert(i, doc.clone());
        centroid.insert(i, doc.clone());
        fde.insert(i, doc.clone());
    }
    linear.build();
    centroid.build();
    fde.build();

    (linear, centroid, fde, docs)
}

fn bench_search(c: &mut Criterion) {
    let n_docs = 1_000;
    let (linear, centroid, fde, docs) = build_indexes(n_docs);

    let mut rng = SmallRng::seed_from_u64(123);
    let query = rand_multivec(&mut rng, 8, 128);

    let mut group = c.benchmark_group(format!("maxsim_search_n{n_docs}"));

    group.bench_function("linear", |b| {
        b.iter(|| linear.search(black_box(&query), 10))
    });

    group.bench_function("centroid", |b| {
        b.iter(|| centroid.search(black_box(&query), 10))
    });

    group.bench_function("fde", |b| {
        b.iter(|| fde.search(black_box(&query), 10))
    });

    group.finish();
}

fn bench_insert_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("maxsim_build");

    for n_docs in [100usize, 500, 1000] {
        let mut rng = SmallRng::seed_from_u64(55);
        let docs: Vec<MultiVec> = (0..n_docs)
            .map(|_| rand_multivec(&mut rng, 16, 128))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("centroid", n_docs),
            &n_docs,
            |b, &_| {
                b.iter(|| {
                    let mut idx = CentroidIndex::new(50, 5);
                    for (i, doc) in docs.iter().enumerate() {
                        idx.insert(i, doc.clone());
                    }
                    idx.build();
                    black_box(idx)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fde", n_docs),
            &n_docs,
            |b, &_| {
                b.iter(|| {
                    let mut idx = FdeIndex::new(64, n_docs / 5);
                    for (i, doc) in docs.iter().enumerate() {
                        idx.insert(i, doc.clone());
                    }
                    idx.build();
                    black_box(idx)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_search, bench_insert_build);
criterion_main!(benches);
