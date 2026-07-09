use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pdx_columnar_simd::{ChunkedColumnarIndex, ColumnarIndex, L2Sq, RowMajorIndex};
use rand::{rngs::SmallRng, Rng, SeedableRng};

fn make_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n).map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect()).collect()
}

fn bench_layouts(c: &mut Criterion) {
    let dim = 768usize;
    let n = 10_000usize;
    let k = 10usize;
    let corpus = make_vecs(n, dim, 42);
    let queries = make_vecs(100, dim, 99);
    let dist = L2Sq;

    let mut row = RowMajorIndex::new(dim);
    let mut col = ColumnarIndex::new(dim);
    let mut chunked = ChunkedColumnarIndex::new(dim, 256);

    for (i, v) in corpus.iter().enumerate() {
        row.insert(i as u64, v);
        col.insert(i as u64, v);
        chunked.insert(i as u64, v);
    }

    let mut group = c.benchmark_group("layout_comparison");
    group.sample_size(20);

    group.bench_function(BenchmarkId::new("row_major", dim), |b| {
        b.iter(|| {
            for q in &queries {
                criterion::black_box(row.search(q, k, &dist));
            }
        });
    });

    group.bench_function(BenchmarkId::new("columnar", dim), |b| {
        b.iter(|| {
            for q in &queries {
                criterion::black_box(col.search(q, k, &dist));
            }
        });
    });

    group.bench_function(BenchmarkId::new("chunked_no_prune", dim), |b| {
        b.iter(|| {
            for q in &queries {
                criterion::black_box(chunked.search(q, k, &dist, 0));
            }
        });
    });

    group.bench_function(BenchmarkId::new("chunked_prune16", dim), |b| {
        b.iter(|| {
            for q in &queries {
                criterion::black_box(chunked.search(q, k, &dist, 16));
            }
        });
    });

    group.finish();
}

criterion_group!(benches, bench_layouts);
criterion_main!(benches);
