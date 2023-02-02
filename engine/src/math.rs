//! Math utilities.

use std::ops::{Add, Div, Mul, Sub};

/// Represents a generic number.
pub trait Num:
    Sized
    + Default
    + Copy
    + Clone
    + PartialEq
    + PartialOrd
    + 'static
    + Add<Self, Output = Self>
    + Div<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Sub<Self, Output = Self>
{
    /// A `zero` value. e.g. 0 or 0.0
    fn zero() -> Self;

    /// A `one` value. e.g. 1 or 1.0
    fn one() -> Self;
}

macro_rules! impl_num {
    ($($ty:ty => $zero:expr, $one:expr),* $(,)?) => {$(
        impl Num for $ty {
            /// A `zero` value. e.g. 0 or 0.0
            #[inline]
            fn zero() -> $ty {
                $zero
            }

            /// A `one` value. e.g. 1 or 1.0
            #[inline]
            fn one() -> $ty {
                $one
            }
        }
    )*}
}

impl_num!(
    u8 => 0u8, 1u8,
    i8 => 0i8, 1i8,
    u16 => 0u16, 1u16,
    i16 => 0i16, 1i16,
    u32 => 0u32, 1u32,
    i32 => 0i32, 1i32,
    u64 => 0u64, 1u64,
    i64 => 0i64, 1i64,
    f32 => 0f32, 1f32,
    f64 => 0f64, 1f64,
    usize => 0usize, 1usize,
    isize => 0isize, 1isize,
);

#[cfg(feature = "rand")]
pub mod rand {
    //! Random number generators.

    use super::Num;
    use rand::distributions::uniform::SampleUniform;

    /// Returns a random number within a given range.
    #[inline]
    #[must_use]
    pub fn random_rng<T, R>(val: R, seed: impl Into<Option<u64>>) -> T
    where
        T: Num + SampleUniform,
        R: Into<std::ops::Range<T>>,
    {
        use rand::{rngs::StdRng, Rng, SeedableRng};

        let val = val.into();
        let seed = seed.into();
        match seed {
            Some(seed) => StdRng::seed_from_u64(seed).gen_range(val),
            None => rand::thread_rng().gen_range(val),
        }
    }

    /// Returns a random number between zero and a given value.
    #[inline]
    #[must_use]
    pub fn random<T>(val: T, seed: impl Into<Option<u64>>) -> T
    where
        T: Num + SampleUniform,
    {
        if val > T::zero() {
            random_rng(T::zero()..val, seed)
        } else {
            random_rng(val..T::zero(), seed)
        }
    }
}

/// Re-map a value from one range to another range.
#[inline]
#[must_use]
pub fn map<T: Num>(value: T, from_min: T, from_max: T, to_min: T, to_max: T) -> T {
    (((value - from_min) * (to_max - to_min)) / (from_max - from_min)) + to_min
}

/// Linear interpolates between two values by a given amount.
#[inline]
#[must_use]
pub fn lerp<T: Num>(from: T, to: T, amount: T) -> T {
    (T::one() - amount) * from + amount * to
}
