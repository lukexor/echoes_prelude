//! Math utilities.

#[cfg(feature = "rand")]
pub mod rand {
    //! Random number generators.

    use std::ops::Range;

    /// Returns a random number within a given range.
    #[inline]
    #[must_use]
    pub fn random_rng(val: Range<f32>, seed: impl Into<Option<u64>>) -> f32 {
        use rand::{rngs::StdRng, Rng, SeedableRng};

        let seed = seed.into();
        match seed {
            Some(seed) => StdRng::seed_from_u64(seed).gen_range(val),
            None => rand::thread_rng().gen_range(val),
        }
    }

    /// Returns a random number between zero and a given value.
    #[inline]
    #[must_use]
    pub fn random(val: f32, seed: impl Into<Option<u64>>) -> f32 {
        if val > 0.0 {
            random_rng(0.0..val, seed)
        } else {
            random_rng(val..0.0, seed)
        }
    }
}

/// Re-map a value from one range to another range.
#[inline]
#[must_use]
pub fn map(value: f32, from_min: f32, from_max: f32, to_min: f32, to_max: f32) -> f32 {
    (((value - from_min) * (to_max - to_min)) / (from_max - from_min)) + to_min
}

/// Linear interpolates between two values by a given amount.
#[inline]
#[must_use]
pub fn lerp(from: f32, to: f32, amount: f32) -> f32 {
    (1.0 - amount) * from + amount * to
}
