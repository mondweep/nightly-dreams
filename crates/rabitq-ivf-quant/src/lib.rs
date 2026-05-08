pub mod encoder;
pub mod flat;
pub mod ivf;

pub use encoder::{BitVector, RaBitQEncoder};
pub use flat::{BitFlatStore, FlatStore};
pub use ivf::IvfBitStore;

/// Common trait for index types that support incremental insertion.
pub trait VectorIndex {
    fn insert(&mut self, id: usize, vector: &[f32]);
    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)>;
    fn memory_bytes(&self) -> usize;
}

#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum()
}
