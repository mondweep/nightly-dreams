use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Bit-packed 1-bit quantized vector with norm correction factor.
#[derive(Clone)]
pub struct BitVector {
    /// Packed bits: bit i is set if (rotated_v[i] > 0).
    pub bits: Vec<u64>,
    /// L2 norm of the original vector, used for distance rescaling.
    pub norm: f32,
    pub dim: usize,
}

/// RaBitQ encoder: random sign-flip rotation + 1-bit quantization.
///
/// Implements the core idea from "RaBitQ: Quantizing High-Dimensional Vectors
/// with a Theoretical Error Bound for Approximate Nearest Neighbor Search"
/// (SIGMOD 2024). The random sign-flip decorrelates dimensions before binarisation,
/// preserving angle information in the Hamming distance.
pub struct RaBitQEncoder {
    pub dim: usize,
    /// Per-dimension random sign flip: ±1.0
    signs: Vec<f32>,
}

impl RaBitQEncoder {
    pub fn new(dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let signs: Vec<f32> = (0..dim)
            .map(|_| if rng.gen_bool(0.5) { 1.0_f32 } else { -1.0_f32 })
            .collect();
        RaBitQEncoder { dim, signs }
    }

    /// Encode a float vector to a bit-packed RaBitQ code.
    pub fn encode(&self, v: &[f32]) -> BitVector {
        assert_eq!(v.len(), self.dim, "vector dimension mismatch");
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n_words = (self.dim + 63) / 64;
        let mut bits = vec![0u64; n_words];
        for (i, (&val, &sign)) in v.iter().zip(self.signs.iter()).enumerate() {
            if val * sign > 0.0 {
                bits[i / 64] |= 1u64 << (i % 64);
            }
        }
        BitVector { bits, norm, dim: self.dim }
    }

    /// Hamming distance between two bit vectors (number of differing bits).
    #[inline]
    pub fn hamming(a: &BitVector, b: &BitVector) -> u32 {
        a.bits
            .iter()
            .zip(b.bits.iter())
            .map(|(x, y)| (x ^ y).count_ones())
            .sum()
    }

    /// Estimate the inner product ⟨q, v⟩ from bit codes.
    ///
    /// Derivation: after random sign flip, each bit captures sign(v_i * s_i).
    /// Two vectors with small angle θ share many sign bits:
    ///   cos(θ) ≈ (D - 2H) / D  where H = Hamming distance.
    /// Scaling by norms recovers ⟨q, v⟩ ≈ ||q|| * ||v|| * (D - 2H) / D.
    #[inline]
    pub fn estimated_inner_product(a: &BitVector, b: &BitVector) -> f32 {
        let h = Self::hamming(a, b) as f32;
        let d = a.dim as f32;
        a.norm * b.norm * (d - 2.0 * h) / d
    }

    /// Estimate squared L2 distance: ||a - b||² = ||a||² + ||b||² - 2⟨a,b⟩.
    #[inline]
    pub fn estimated_l2_sq(a: &BitVector, b: &BitVector) -> f32 {
        let ip = Self::estimated_inner_product(a, b);
        a.norm * a.norm + b.norm * b.norm - 2.0 * ip
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_have_zero_hamming() {
        let enc = RaBitQEncoder::new(64, 1);
        let v: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let a = enc.encode(&v);
        let b = enc.encode(&v);
        assert_eq!(RaBitQEncoder::hamming(&a, &b), 0);
    }

    #[test]
    fn opposite_vectors_have_max_hamming() {
        let enc = RaBitQEncoder::new(64, 1);
        let v: Vec<f32> = (0..64).map(|i| (i as f32) + 1.0).collect();
        let neg: Vec<f32> = v.iter().map(|x| -x).collect();
        let a = enc.encode(&v);
        let b = enc.encode(&neg);
        // After random sign flip, all bits should differ → hamming = dim
        assert_eq!(RaBitQEncoder::hamming(&a, &b), 64);
    }

    #[test]
    fn estimated_ip_same_direction_is_positive() {
        let enc = RaBitQEncoder::new(128, 42);
        let v: Vec<f32> = (0..128).map(|i| (i as f32).sin()).collect();
        let a = enc.encode(&v);
        let b = enc.encode(&v);
        let ip = RaBitQEncoder::estimated_inner_product(&a, &b);
        assert!(ip > 0.0, "same-direction inner product must be positive, got {ip}");
    }
}
