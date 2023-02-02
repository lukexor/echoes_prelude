/// An angle in radians.
#[derive(
    Default,
    Debug,
    Copy,
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
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
#[must_use]
#[repr(transparent)]
pub struct Radians<T>(pub T);

impl From<Degrees<f32>> for Radians<f32> {
    fn from(degrees: Degrees<f32>) -> Self {
        Radians(degrees.to_radians())
    }
}

impl From<Degrees<f64>> for Radians<f64> {
    fn from(degrees: Degrees<f64>) -> Self {
        Radians(degrees.to_radians())
    }
}

impl From<&Degrees<f32>> for Radians<f32> {
    fn from(degrees: &Degrees<f32>) -> Self {
        Radians(degrees.to_radians())
    }
}

impl From<&Degrees<f64>> for Radians<f64> {
    fn from(degrees: &Degrees<f64>) -> Self {
        Radians(degrees.to_radians())
    }
}

#[derive(
    Default,
    Debug,
    Copy,
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
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
#[must_use]
#[repr(transparent)]
pub struct Degrees<T>(pub T);

impl From<Radians<f32>> for Degrees<f32> {
    fn from(radians: Radians<f32>) -> Self {
        Degrees(radians.to_degrees())
    }
}

impl From<Radians<f64>> for Degrees<f64> {
    fn from(radians: Radians<f64>) -> Self {
        Degrees(radians.to_degrees())
    }
}

impl From<&Radians<f32>> for Degrees<f32> {
    fn from(radians: &Radians<f32>) -> Self {
        Degrees(radians.to_degrees())
    }
}

impl From<&Radians<f64>> for Degrees<f64> {
    fn from(radians: &Radians<f64>) -> Self {
        Degrees(radians.to_degrees())
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[must_use]
pub enum Angle<T> {
    Radians(Radians<T>),
    Degrees(Degrees<T>),
}

impl<T: Default> Default for Angle<T> {
    fn default() -> Self {
        Self::Degrees(Degrees(T::default()))
    }
}

impl<T> From<T> for Angle<T> {
    fn from(value: T) -> Self {
        Self::Degrees(Degrees(value))
    }
}

impl Angle<f32> {
    /// Convert this angle to radians.
    #[inline]
    pub fn to_radians(&self) -> Radians<f32> {
        match self {
            Angle::Radians(radians) => *radians,
            Angle::Degrees(degrees) => degrees.into(),
        }
    }
}

impl Angle<f64> {
    /// Convert this angle to radians.
    #[inline]
    pub fn to_radians(&self) -> Radians<f64> {
        match self {
            Angle::Radians(radians) => *radians,
            Angle::Degrees(degrees) => degrees.into(),
        }
    }
}
