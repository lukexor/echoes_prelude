use std::f32::consts::{FRAC_PI_2, PI};

pub trait ApproxEq {
    type Type;

    fn is_approx_eq(&self, rhs: Self::Type, epsilon: Self::Type) -> bool;
}

impl ApproxEq for f32 {
    type Type = f32;

    fn is_approx_eq(&self, rhs: f32, epsilon: f32) -> bool {
        (self - rhs).abs() <= epsilon
    }
}

impl ApproxEq for f64 {
    type Type = f64;

    fn is_approx_eq(&self, rhs: f64, epsilon: f64) -> bool {
        (self - rhs).abs() <= epsilon
    }
}

/// An angle in radians.
#[derive(
    Default,
    Debug,
    Copy,
    Clone,
    PartialEq,
    PartialOrd,
    derive_more::From,
    derive_more::Into,
    derive_more::Deref,
    derive_more::DerefMut,
    derive_more::Add,
    derive_more::AddAssign,
    derive_more::Mul,
    derive_more::MulAssign,
    derive_more::Div,
    derive_more::DivAssign,
    derive_more::Sub,
    derive_more::SubAssign,
    derive_more::Neg,
)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
#[must_use]
#[repr(transparent)]
pub struct Radians(f32);

impl Radians {
    /// Creates `Radians` without normalizing the value.
    #[inline]
    pub const fn new_unchecked(value: f32) -> Self {
        Self(value)
    }

    /// Creates `Radians` normalized to -π..=π.
    #[inline]
    pub fn new(value: f32) -> Self {
        let mut value = (value + FRAC_PI_2) % PI;
        if value < 0.0 {
            value += PI;
        }
        Self(value - FRAC_PI_2)
    }

    /// Returns the value as a primitive type.
    #[inline]
    #[must_use]
    pub fn get(self) -> f32 {
        self.0
    }
}

impl From<Degrees> for Radians {
    /// Converts `Degrees` into `Radians`.
    fn from(degrees: Degrees) -> Self {
        Radians(degrees.to_radians())
    }
}

impl From<&Degrees> for Radians {
    /// Converts `&Degrees` into `Radians`.
    fn from(degrees: &Degrees) -> Self {
        Radians(degrees.to_radians())
    }
}

#[derive(
    Default,
    Debug,
    Copy,
    Clone,
    PartialEq,
    PartialOrd,
    derive_more::From,
    derive_more::Into,
    derive_more::Deref,
    derive_more::DerefMut,
    derive_more::Add,
    derive_more::AddAssign,
    derive_more::Mul,
    derive_more::MulAssign,
    derive_more::Div,
    derive_more::DivAssign,
    derive_more::Sub,
    derive_more::SubAssign,
    derive_more::Neg,
)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
#[must_use]
#[repr(transparent)]
pub struct Degrees(f32);

impl Degrees {
    /// Creates `Degrees` without normalizing the value.
    #[inline]
    pub const fn new_unchecked(value: f32) -> Self {
        Self(value)
    }

    /// Creates `Degrees` normalized to -180.0..=180.0.
    #[inline]
    pub fn new(value: f32) -> Self {
        let mut value = (value + 180.0) % 360.0;
        if value < 0.0 {
            value += 360.0;
        }
        Self(value - 180.0)
    }

    /// Returns the value as a primitive type.
    #[inline]
    #[must_use]
    pub fn get(self) -> f32 {
        self.0
    }
}

impl From<Radians> for Degrees {
    /// Converts `Radians` into `Degrees`.
    fn from(radians: Radians) -> Self {
        Degrees(radians.to_degrees())
    }
}

impl From<&Radians> for Degrees {
    /// Converts `&Radians` into `Degrees`.
    fn from(radians: &Radians) -> Self {
        Degrees(radians.to_degrees())
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, derive_more::From)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(untagged))]
#[must_use]
pub enum Angle {
    Radians(Radians),
    Degrees(Degrees),
}

impl Default for Angle {
    fn default() -> Self {
        Self::Degrees(Degrees(0.0))
    }
}

impl Angle {
    /// Convert `Angle` to `Radians`.
    #[inline]
    pub fn to_radians(self) -> Radians {
        match self {
            Angle::Radians(radians) => radians,
            Angle::Degrees(degrees) => degrees.into(),
        }
    }

    /// Convert `Angle` to `Degrees`.
    #[inline]
    pub fn to_degrees(self) -> Degrees {
        match self {
            Angle::Radians(radians) => radians.into(),
            Angle::Degrees(degrees) => degrees,
        }
    }

    /// Returns the value as a primitive type.
    #[inline]
    #[must_use]
    pub fn get(self) -> f32 {
        match self {
            Angle::Radians(radians) => radians.get(),
            Angle::Degrees(degrees) => degrees.get(),
        }
    }
}

impl From<f32> for Angle {
    fn from(value: f32) -> Self {
        Self::Degrees(Degrees(value))
    }
}
