//! Math utilities.

/// Generic range mapping trait for numbers.
pub trait Map {
    /// Re-map a value from one range to another range.
    fn map(&self, from_min: Self, from_max: Self, to_min: Self, to_max: Self) -> Self;
}

macro_rules! impl_map {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl Map for $ty {
                /// Re-map a value from one range to another range.
                #[inline]
                #[must_use]
                fn map(&self, from_min: $ty, from_max: $ty, to_min: $ty, to_max: $ty) -> $ty {
                    (((self - from_min) * (to_max - to_min)) / (from_max - from_min)) + to_min
                }
            }
        )+
    }
}

impl_map!(i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, isize, usize);

/// Generic linear interpolation trait for numbers.
pub trait Lerp {
    /// Linear interpolates between two values by a given amount.
    fn lerp(&self, to: Self, amount: Self) -> Self;
}

macro_rules! impl_lerp {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl Lerp for $ty {
                /// Re-map a value from one range to another range.
                #[inline]
                #[must_use]
                fn lerp(&self, to: $ty, amount: $ty) -> $ty {
                    (1.0 - amount) * self + amount * to
                }
            }
        )+
    }
}

impl_lerp!(f32, f64);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map() {
        assert_eq!(50.map(0, 100, 0, 10), 5);
        assert_eq!(3.map(0, 10, 0, 100), 30);
        assert_eq!(9f32.map(0.0, 90.0, 0.0, 45.0), 4.5);
    }

    #[test]
    fn lerp() {
        assert_eq!(0f32.lerp(100.0, 0.5), 50.0);
        assert_eq!(0f32.lerp(1.0, 0.2), 0.2);
    }
}
