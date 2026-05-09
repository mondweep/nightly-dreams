use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use dabs_hnsw::{GraphSearch, NswGraph};

const DIM: usize = 128;
const N: usize = 5_000;
const K: usize = 10;
const EF: usize = 64;

fn lcg(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    ((*s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

fn make_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let v: Vec<f32> = (0..dim).map(|_| lcg(&mut s)).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.into_iter().map(|x| x / norm).collect()
}

fn build_bench_graph() -> (NswGraph, Vec<Vec<f32>>) {
    let mut g = NswGraph::new(DIM, 16);
    for i in 0..N as u64 {
        g.insert(make_vec(DIM, i * 1337 + 1));
    }
    let queries: Vec<Vec<f32>> = (0..50u64).map(|i| make_vec(DIM, i * 9_999 + 7)).collect();
    (g, queries)
}

fn bench_search(c: &mut Criterion) {
    let (graph, queries) = build_bench_graph();
    let mut group = c.benchmark_group("hnsw_search");

    group.bench_function("standard_ef64", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(graph.beam_search(q, K, EF));
            }
        })
    });

    for gamma in [0.5f32, 1.0, 2.0] {
        group.bench_with_input(BenchmarkId::new("dabs", gamma), &gamma, |b, &g| {
            b.iter(|| {
                for q in &queries {
                    black_box(graph.dabs_search(q, K, EF, g));
                }
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
