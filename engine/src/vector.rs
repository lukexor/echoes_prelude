use crate::matrix::Mat4;
use std::{
    mem,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

macro_rules! impl_vector {
    ($({
        $Vec:ident, $dim:expr => $($field:ident),+
    }),+ $(,)?) => {
        $(
            #[doc = concat!("A ", stringify!($dim), "-dimensional vector.")]
            #[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
            #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
            #[must_use]
            pub struct $Vec {
                $(pub $field: f32),+
            }

            impl Default for $Vec {
                fn default() -> Self {
                    Self::origin()
                }
            }

            impl $Vec {
                #[doc = concat!("Create a ", stringify!($dim), "-dimensional vector from given coordinates.")]
                #[inline]
                pub fn new($($field: f32),+) -> Self {
                    Self { $($field),+ }
                }

                #[doc = concat!("Create a ", stringify!($dim), "-dimensional vector at the origin.")]
                #[inline]
                pub fn origin() -> Self {
                    Self { $($field: 0.0),+ }
                }

                #[doc = concat!("Create a ", stringify!($dim), "-dimensional unit vector.")]
                #[inline]
                pub fn unit() -> Self {
                    Self { $($field: 1.0),+ }
                }

                #[cfg(feature = "rand")]
                #[doc = concat!("Create a ", stringify!($dim), "-dimensional random vector.")]
                #[inline]
                pub fn rand<R: rand::Rng>(rng: &mut R) -> Self {
                    Self { $($field: rng.gen()),+ }
                }

                #[doc = concat!("Create a ", stringify!($dim), "-dimensional vector reference from given array of values.")]
                #[inline]
                pub fn from_array(array: &[f32; $dim]) -> &Self {
                    let array: &Self = unsafe { mem::transmute(array) };
                    array
                }

                #[doc = concat!("Create a ", stringify!($dim), "-dimensional mutable vector reference from given array of values.")]
                #[inline]
                pub fn from_array_mut(array: &mut [f32; $dim]) -> &mut Self {
                    let array: &mut Self = unsafe { mem::transmute(array) };
                    array
                }

                /// Converts the vector into an array reference of `f32`.
                #[must_use]
                #[inline]
                pub fn as_array(&self) -> &[f32; $dim] {
                    let array: &[f32; $dim] = unsafe { mem::transmute(self) };
                    array
                }

                /// Converts the vector into a mutable array reference of `f32`.
                #[must_use]
                #[inline]
                pub fn as_array_mut(&mut self) -> &mut [f32; $dim] {
                    let array: &mut [f32; $dim] = unsafe { mem::transmute(self) };
                    array
                }

                /// Converts the vector into an array of `f32`.
                #[must_use]
                #[inline]
                pub fn to_array(self) -> [f32; $dim] {
                    let array: [f32; $dim] = unsafe { mem::transmute(self) };
                    array
                }

                /// Returns whether two vectors are equal given an epsilon.
                #[must_use]
                #[inline]
                pub fn compare(&self, rhs: Self, epsilon: f32) -> bool {
                    self.iter()
                        .zip(rhs.iter())
                        .all(|(a, b)| (a - b).abs() <= epsilon)
                }

                /// Calculate the squared magnitude of the vector.
                #[must_use]
                #[inline]
                pub fn magnitude_squared(&self) -> f32 {
                    self.iter().map(|val| val * val).sum()
                }

                /// Calculate the magnitude of the vector.
                #[must_use]
                #[inline]
                pub fn magnitude(&self) -> f32 {
                    self.magnitude_squared().sqrt()
                }

                /// Normalize the vector into a unit vector.
                #[inline]
                pub fn normalize(&mut self) {
                    let magnitude = self.magnitude();
                    if magnitude != 0.0 {
                        self.iter_mut().for_each(|val| *val *= magnitude.recip());
                    }
                }

                /// Create a normalized copy of the vector.
                #[inline]
                pub fn normalized(&self) -> Self {
                    let mut vector = *self;
                    vector.normalize();
                    vector
                }

                /// Create the Euclidean distance between two vectors.
                #[must_use]
                #[inline]
                pub fn distance(&self, vector: Self) -> f32 {
                    (*self - vector).magnitude()
                }

                /// Create an iterator over the vector dimensions.
                #[inline]
                fn iter(&self) ->  std::slice::Iter<'_, f32> {
                    self.as_array().iter()
                }

                /// Create a mutable iterator over the vector dimensions.
                #[inline]
                fn iter_mut(&mut self) -> std::slice::IterMut<'_, f32> {
                    self.as_array_mut().iter_mut()
                }
            }

            impl IntoIterator for $Vec {
                type Item = f32;
                type IntoIter = std::array::IntoIter<Self::Item, $dim>;

                fn into_iter(self) -> Self::IntoIter {
                    self.to_array().into_iter()
                }
            }

            impl IntoIterator for &$Vec {
                type Item = f32;
                type IntoIter = std::array::IntoIter<Self::Item, $dim>;

                fn into_iter(self) -> Self::IntoIter {
                    self.to_array().into_iter()
                }
            }

            impl Index<usize> for $Vec {
                type Output = f32;

                fn index(&self, index: usize) -> &f32 {
                    self.as_array().index(index)
                }
            }

            impl IndexMut<usize> for $Vec {
                fn index_mut(&mut self, index: usize) -> &mut f32 {
                    self.as_array_mut().index_mut(index)
                }
            }

            impl From<[f32; $dim]> for $Vec {
                fn from(array: [f32; $dim]) -> Self {
                    let v: Self = unsafe { mem::transmute(array) };
                    v
                }
            }

            impl From<&[f32; $dim]> for &$Vec {
                fn from(array: &[f32; $dim]) -> Self {
                    let v: &Self = unsafe { mem::transmute(array) };
                    v
                }
            }

            impl Add for $Vec {
                type Output = $Vec;

                fn add(self, rhs: Self) -> Self::Output {
                    $Vec::new($(self.$field + rhs.$field),+)
                }
            }

            impl Add for &$Vec {
                type Output = $Vec;

                fn add(self, rhs: Self) -> Self::Output {
                    $Vec::new($(self.$field + rhs.$field),+)
                }
            }

            impl AddAssign for $Vec {
                fn add_assign(&mut self, rhs: Self) {
                    self.iter_mut()
                        .zip(rhs.iter())
                        .for_each(|(val, rhs)| *val += rhs);
                }
            }

            impl Sub for $Vec {
                type Output = $Vec;

                fn sub(self, rhs: Self) -> Self::Output {
                    $Vec::new($(self.$field - rhs.$field),+)
                }
            }

            impl Sub for &$Vec {
                type Output = $Vec;

                fn sub(self, rhs: Self) -> Self::Output {
                    $Vec::new($(self.$field - rhs.$field),+)
                }
            }

            impl SubAssign for $Vec {
                fn sub_assign(&mut self, rhs: Self) {
                    self.iter_mut()
                        .zip(rhs.iter())
                        .for_each(|(val, rhs)| *val -= rhs);
                }
            }

            impl Mul for $Vec {
                type Output = $Vec;

                fn mul(self, rhs: Self) -> Self::Output {
                    $Vec::new($(self.$field * rhs.$field),+)
                }
            }

            impl Mul<&$Vec> for $Vec {
                type Output = $Vec;

                fn mul(self, rhs: &$Vec) -> Self::Output {
                    $Vec::new($(self.$field * rhs.$field),+)
                }
            }

            impl Mul<$Vec> for &$Vec {
                type Output = $Vec;

                fn mul(self, rhs: $Vec) -> Self::Output {
                    $Vec::new($(self.$field * rhs.$field),+)
                }
            }

            impl Mul for &$Vec {
                type Output = $Vec;

                fn mul(self, rhs: &$Vec) -> Self::Output {
                    $Vec::new($(self.$field * rhs.$field),+)
                }
            }

            impl Mul<f32> for $Vec {
                type Output = $Vec;

                fn mul(self, rhs: f32) -> Self::Output {
                    $Vec::new($(self.$field * rhs),+)
                }
            }

            impl Mul<&f32> for $Vec {
                type Output = $Vec;

                fn mul(self, rhs: &f32) -> Self::Output {
                    $Vec::new($(self.$field * rhs),+)
                }
            }

            impl Mul<f32> for &$Vec {
                type Output = $Vec;

                fn mul(self, rhs: f32) -> Self::Output {
                    $Vec::new($(self.$field * rhs),+)
                }
            }

            impl Mul<&f32> for &$Vec {
                type Output = $Vec;

                fn mul(self, rhs: &f32) -> Self::Output {
                    $Vec::new($(self.$field * rhs),+)
                }
            }

            impl Mul<$Vec> for f32 {
                type Output = $Vec;

                fn mul(self, rhs: $Vec) -> Self::Output {
                    $Vec::new($(rhs.$field * self),+)
                }
            }

            impl Mul<&$Vec> for f32 {
                type Output = $Vec;

                fn mul(self, rhs: &$Vec) -> Self::Output {
                    $Vec::new($(rhs.$field * self),+)
                }
            }

            impl Mul<$Vec> for &f32 {
                type Output = $Vec;

                fn mul(self, rhs: $Vec) -> Self::Output {
                    $Vec::new($(rhs.$field * self),+)
                }
            }

            impl Mul<&$Vec> for &f32 {
                type Output = $Vec;

                fn mul(self, rhs: &$Vec) -> Self::Output {
                    $Vec::new($(rhs.$field * self),+)
                }
            }

            impl MulAssign for $Vec {
                fn mul_assign(&mut self, rhs: Self) {
                    self.iter_mut()
                        .zip(rhs.iter())
                        .for_each(|(val, rhs)| *val *= rhs);
                }
            }

            impl MulAssign<f32> for $Vec {
                fn mul_assign(&mut self, rhs: f32) {
                    self.iter_mut().for_each(|val| *val *= rhs);
                }
            }

            impl MulAssign<&f32> for $Vec {
                fn mul_assign(&mut self, rhs: &f32) {
                    self.iter_mut().for_each(|val| *val *= rhs);
                }
            }

            impl Div for $Vec {
                type Output = $Vec;

                fn div(self, rhs: $Vec) -> Self::Output {
                    $Vec::new($(self.$field / rhs.$field),+)
                }
            }

            impl Div<&$Vec> for $Vec {
                type Output = $Vec;

                fn div(self, rhs: &$Vec) -> Self::Output {
                    $Vec::new($(self.$field / rhs.$field),+)
                }
            }

            impl Div<$Vec> for &$Vec {
                type Output = $Vec;

                fn div(self, rhs: $Vec) -> Self::Output {
                    $Vec::new($(self.$field / rhs.$field),+)
                }
            }

            impl Div for &$Vec {
                type Output = $Vec;

                fn div(self, rhs: &$Vec) -> Self::Output {
                    $Vec::new($(self.$field / rhs.$field),+)
                }
            }

            impl Div<f32> for $Vec {
                type Output = $Vec;

                fn div(self, rhs: f32) -> Self::Output {
                    $Vec::new($(self.$field / rhs),+)
                }
            }

            impl Div<&f32> for $Vec {
                type Output = $Vec;

                fn div(self, rhs: &f32) -> Self::Output {
                    $Vec::new($(self.$field / rhs),+)
                }
            }

            impl Div<f32> for &$Vec {
                type Output = $Vec;

                fn div(self, rhs: f32) -> Self::Output {
                    $Vec::new($(self.$field / rhs),+)
                }
            }

            impl Div<&f32> for &$Vec {
                type Output = $Vec;

                fn div(self, rhs: &f32) -> Self::Output {
                    $Vec::new($(self.$field / rhs),+)
                }
            }

            impl Div<$Vec> for f32 {
                type Output = $Vec;

                fn div(self, rhs: $Vec) -> Self::Output {
                    $Vec::new($(rhs.$field / self),+)
                }
            }

            impl Div<&$Vec> for f32 {
                type Output = $Vec;

                fn div(self, rhs: &$Vec) -> Self::Output {
                    $Vec::new($(rhs.$field / self),+)
                }
            }

            impl Div<$Vec> for &f32 {
                type Output = $Vec;

                fn div(self, rhs: $Vec) -> Self::Output {
                    $Vec::new($(rhs.$field / self),+)
                }
            }

            impl Div<&$Vec> for &f32 {
                type Output = $Vec;

                fn div(self, rhs: &$Vec) -> Self::Output {
                    $Vec::new($(rhs.$field / self),+)
                }
            }

            impl DivAssign for $Vec {
                fn div_assign(&mut self, rhs: Self) {
                    self.iter_mut()
                        .zip(rhs.iter())
                        .for_each(|(val, rhs)| *val /= rhs);
                }
            }

            impl DivAssign<f32> for $Vec {
                fn div_assign(&mut self, rhs: f32) {
                    self.iter_mut().for_each(|val| *val /= rhs);
                }
            }

            impl DivAssign<&f32> for $Vec {
                fn div_assign(&mut self, rhs: &f32) {
                    self.iter_mut().for_each(|val| *val /= rhs);
                }
            }

            impl Neg for $Vec {
                type Output = $Vec;

                fn neg(self) -> Self::Output {
                    $Vec::new($(self.$field.neg()),+)
                }
            }

            impl Neg for &$Vec {
                type Output = $Vec;

                fn neg(self) -> Self::Output {
                    $Vec::new($(self.$field.neg()),+)
                }
            }
        )+
    };
}

impl_vector! {
    { Vec1, 1 => x },
    { Vec2, 2 => x, y },
    { Vec3, 3 => x, y, z },
    { Vec4, 4 => x, y, z, w },
}

/// Constructs a new 1D [Vector].
#[macro_export]
macro_rules! vec1 {
    () => {
        $crate::vector::Vec1::origin()
    };
    ($val:expr) => {
        $crate::vector::Vec1::from($val)
    };
}

/// Constructs a new 2D [Vector].
#[macro_export]
macro_rules! vec2 {
    () => {
        $crate::vector::Vec2::origin()
    };
    ($val:expr) => {
        $crate::vector::Vec2::from($val)
    };
    ($val:expr, $y:expr $(,)?) => {{
        let v = $crate::vec1!($val);
        $crate::vector::Vec2::new(v.x, $y)
    }};
}

/// Constructs a new 3D [Vector].
#[macro_export]
macro_rules! vec3 {
    () => {
        $crate::vector::Vec3::origin()
    };
    ($val:expr) => {
        $crate::vector::Vec3::from($val)
    };
    ($val:expr, $z:expr $(,)?) => {{
        let v = $crate::vec2!($val);
        $crate::vector::Vec3::new(v.x, v.y, $z)
    }};
    ($val:expr, $y:expr, $z:expr $(,)?) => {{
        let v = $crate::vec1!($val);
        $crate::vector::Vec3::new(v.x, $y, $z)
    }};
}

/// Constructs a new 4D [Vector].
#[macro_export]
macro_rules! vec4 {
    () => {
        $crate::vector::Vec4::origin()
    };
    ($val:expr) => {
        $crate::vector::Vec4::from($val)
    };
    ($val:expr, $w:expr $(,)?) => {{
        let v = $crate::vec3!($val);
        $crate::vector::Vec4::new(v.x, v.y, v.z, $w)
    }};
    ($val:expr, $z:expr, $w:expr $(,)?) => {{
        let v = $crate::vec2!($val);
        $crate::vector::Vec4::new(v.x, v.y, $z, $w)
    }};
    ($val:expr, $y:expr, $z:expr, $w:expr $(,)?) => {{
        let v = $crate::vec1!($val);
        $crate::vector::Vec4::new(v.x, $y, $z, $w)
    }};
}

impl From<f32> for Vec1 {
    fn from(x: f32) -> Self {
        Self { x }
    }
}

impl Vec2 {
    /// Return the `x` component as an array.
    #[inline]
    #[must_use]
    pub fn x(&self) -> [f32; 1] {
        [self.x]
    }

    /// Return the `x` and `y` components as an array.
    #[inline]
    #[must_use]
    pub fn xy(&self) -> [f32; 2] {
        [self.x, self.y]
    }

    /// Create a 2D vector from a 3D vector.
    #[inline]
    pub fn from_vec3(vector: Vec3) -> Self {
        vec2!(vector.xy())
    }

    /// Create a 2D vector from a 4D vector.
    #[inline]
    pub fn from_vec4(vector: Vec4) -> Self {
        vec2!(vector.xy())
    }

    /// Create a 3D vector from a 2D vector.
    #[inline]
    pub fn to_vec3(self, z: f32) -> Vec3 {
        vec3!(self.xy(), z)
    }

    /// Create a 4D vector from a 2D vector.
    #[inline]
    pub fn to_vec4(self, z: f32, w: f32) -> Vec4 {
        vec4!(self.xy(), z, w)
    }

    /// Create a 2D unit vector pointing up.
    #[inline]
    pub fn up() -> Self {
        vec2!(0.0, 1.0)
    }

    /// Create a 2D unit vector pointing down.
    #[inline]
    pub fn down() -> Self {
        vec2!(0.0, -1.0)
    }

    /// Create a 2D unit vector pointing left.
    #[inline]
    pub fn left() -> Self {
        vec2!(-1.0, 0.0)
    }

    /// Create a 2D unit vector pointing right.
    #[inline]
    pub fn right() -> Self {
        vec2!(1.0, 0.0)
    }
}

impl From<[f32; 1]> for Vec2 {
    fn from(array: [f32; 1]) -> Self {
        let v: Vec1 = unsafe { mem::transmute(array) };
        Self::new(v.x, 0.0)
    }
}

impl Vec3 {
    /// Return the `x` component as an array.
    #[inline]
    #[must_use]
    pub fn x(&self) -> [f32; 1] {
        [self.x]
    }

    /// Return the `x` and `y` components as an array.
    #[inline]
    #[must_use]
    pub fn xy(&self) -> [f32; 2] {
        [self.x, self.y]
    }

    /// Return the `x`, `y`, and `z` components as an array.
    #[inline]
    #[must_use]
    pub fn xyz(&self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    /// Create a 3D vector from a 2D vector.
    #[inline]
    pub fn from_vec2(vector: Vec2, z: f32) -> Self {
        vec3!(vector.xy(), z)
    }

    /// Create a 3D vector from a 4D vector.
    #[inline]
    pub fn from_vec4(vector: Vec4) -> Self {
        vec3!(vector.xy(), vector.z)
    }

    /// Create a 2D vector from a 3D vector.
    #[inline]
    pub fn to_vec2(self) -> Vec2 {
        vec2!(self.x, self.y)
    }

    /// Create a 4D vector from a 3D vector.
    #[inline]
    pub fn to_vec4(self, w: f32) -> Vec4 {
        vec4!(self.xyz(), w)
    }

    /// Create a 3D vector from separate RGB values.
    #[inline]
    pub fn from_rgb(r: u32, g: u32, b: u32) -> Self {
        vec3!(r as f32, g as f32, b as f32) / 255.0
    }

    /// Create separate RGB values from the vector.
    #[inline]
    #[must_use]
    pub fn to_rgb(&self) -> [u32; 3] {
        let rgb = *self * 255.0;
        [rgb.x as u32, rgb.y as u32, rgb.z as u32]
    }

    /// Truncates to a [Vec2] by removing the `i`th element, indexed from `0`.
    ///
    /// # Panic
    ///
    /// Panics if `i` is larger than `2`.
    pub fn truncate(&self, i: usize) -> Vec2 {
        match i {
            0 => vec2!(self.y, self.z),
            1 => vec2!(self.x, self.z),
            2 => vec2!(self.x, self.y),
            _ => panic!("index out of bounds. the len is 3 but the index is {i}"),
        }
    }

    /// Create a 3D unit vector pointing up.
    #[inline]
    pub fn up() -> Self {
        vec3!(0.0, 1.0, 0.0)
    }

    /// Create a 3D unit vector pointing down.
    #[inline]
    pub fn down() -> Self {
        vec3!(0.0, -1.0, 0.0)
    }

    /// Create a 3D unit vector pointing left.
    #[inline]
    pub fn left() -> Self {
        vec3!(-1.0, 0.0, 0.0)
    }

    /// Create a 3D unit vector pointing right.
    #[inline]
    pub fn right() -> Self {
        vec3!(1.0, 0.0, 0.0)
    }

    /// Create a 3D unit vector pointing forward.
    #[inline]
    pub fn forward() -> Self {
        vec3!(0.0, 0.0, -1.0)
    }

    /// Create a 3D unit vector pointing backward.
    #[inline]
    pub fn backward() -> Self {
        vec3!(0.0, 0.0, 1.0)
    }

    /// Calculate the dot-product between two vectors.
    #[must_use]
    #[inline]
    pub fn dot(&self, rhs: Self) -> f32 {
        self.iter()
            .zip(rhs.iter())
            .map(|(val, rhs)| val * rhs)
            .sum()
    }

    /// Calculate the cross-product between two vectors.
    #[inline]
    pub fn cross(&self, rhs: Self) -> Self {
        vec3!(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x
        )
    }

    /// Create a transformed vector by applying a matrix.
    #[inline]
    pub fn transformed(&self, matrix: Mat4) -> Self {
        vec3!(
            self.x * matrix[(0, 0)]
                + self.y * matrix[(1, 0)]
                + self.z * matrix[(2, 0)]
                + 1.0 * matrix[(3, 0)],
            self.x * matrix[(0, 1)]
                + self.y * matrix[(1, 1)]
                + self.z * matrix[(2, 1)]
                + 1.0 * matrix[(3, 1)],
            self.x * matrix[(0, 2)]
                + self.y * matrix[(1, 2)]
                + self.z * matrix[(2, 2)]
                + 1.0 * matrix[(3, 2)],
        )
    }
}

impl From<[f32; 1]> for Vec3 {
    fn from(array: [f32; 1]) -> Self {
        let v: Vec1 = unsafe { mem::transmute(array) };
        Self::new(v.x, 0.0, 0.0)
    }
}

impl From<[f32; 2]> for Vec3 {
    fn from(array: [f32; 2]) -> Self {
        let v: Vec2 = unsafe { mem::transmute(array) };
        Self::new(v.x, v.y, 0.0)
    }
}

impl Vec4 {
    /// Return the `x` component as an array.
    #[inline]
    #[must_use]
    pub fn x(&self) -> [f32; 1] {
        [self.x]
    }

    /// Return the `x` and `y` components as an array.
    #[inline]
    #[must_use]
    pub fn xy(&self) -> [f32; 2] {
        [self.x, self.y]
    }

    /// Return the `x`, `y`, and `z` components as an array.
    #[inline]
    #[must_use]
    pub fn xyz(&self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    /// Return the `x`, `y`, `z`, and `w` components as an array.
    #[inline]
    #[must_use]
    pub fn xyzw(&self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }

    /// Create a 4D vector from a 2D vector.
    #[inline]
    pub fn from_vec2(vector: Vec2, z: f32, w: f32) -> Self {
        vec4!(vector.xy(), z, w)
    }

    /// Create a 4D vector from a 3D vector.
    #[inline]
    pub fn from_vec3(vector: Vec3, w: f32) -> Self {
        vec4!(vector.xy(), vector.z, w)
    }

    /// Create a 2D vector from a 4D vector.
    #[inline]
    pub fn to_vec2(self) -> Vec2 {
        vec2!(self.xy())
    }

    /// Create a 3D vector from a 4D vector.
    #[inline]
    pub fn to_vec3(self) -> Vec3 {
        vec3!(self.xyz())
    }

    /// Truncates to a [Vec3] by removing the `i`th element, indexed from `0`.
    ///
    /// # Panic
    ///
    /// Panics if `i` is larger than `3`.
    pub fn truncate(&self, i: usize) -> Vec3 {
        match i {
            0 => vec3!(self.y, self.z, self.w),
            1 => vec3!(self.x, self.z, self.w),
            2 => vec3!(self.x, self.y, self.w),
            3 => vec3!(self.x, self.y, self.z),
            _ => panic!("index out of bounds. the len is 4 but the index is {i}"),
        }
    }

    /// Calculate the dot-product between two vectors, pairwise.
    #[must_use]
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn dot_pairwise(a: [f32; 4], b: [f32; 4]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    }
}

impl From<[f32; 1]> for Vec4 {
    fn from(array: [f32; 1]) -> Self {
        let v: Vec1 = unsafe { mem::transmute(array) };
        Self::new(v.x, 0.0, 0.0, 0.0)
    }
}

impl From<[f32; 2]> for Vec4 {
    fn from(array: [f32; 2]) -> Self {
        let v: Vec2 = unsafe { mem::transmute(array) };
        Self::new(v.x, v.y, 0.0, 0.0)
    }
}

impl From<[f32; 3]> for Vec4 {
    fn from(array: [f32; 3]) -> Self {
        let v: Vec3 = unsafe { mem::transmute(array) };
        Self::new(v.x, v.y, v.z, 0.0)
    }
}

/// A `Quaternion`.
#[derive(Debug, Copy, Clone)]
#[must_use]
pub struct Quaternion {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Default for Quaternion {
    fn default() -> Self {
        Self::identity()
    }
}

/// Constructs a new 4D [Quaternion].
#[macro_export]
macro_rules! quaternion {
    () => {
        $crate::math::Quaternion::identity()
    };
    ($val:expr) => {
        $crate::math::Quaternion::from($val)
    };
    ($x:expr, $y:expr $(,)?) => {
        $crate::math::Quaternion::new($x, $y, 0.0, 0.0)
    };
    ($x:expr, $y:expr, $z:expr $(,)?) => {
        $crate::math::Quaternion::new($x, $y, $z, 0.0)
    };
    ($x:expr, $y:expr, $z:expr, $w:expr $(,)?) => {
        $crate::math::Quaternion::new($x, $y, $z, $w)
    };
}

impl Quaternion {
    /// Create an identity `Quaternion`.
    #[inline]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Create an identity `Quaternion`.
    #[inline]
    pub fn identity() -> Self {
        Self::new(0.0, 0.0, 0.0, 1.0)
    }

    /// Create a `Quaternion` from an axis and angle.
    #[inline]
    pub fn from_axis_angle(axis: impl Into<Vec3>, angle: f32, normalize: bool) -> Self {
        let axis = axis.into();
        let frac_angle_two = 0.5 * angle;
        let (sin, cos) = frac_angle_two.sin_cos();
        let quaternion = Self::new(sin * axis.x, sin * axis.y, sin * axis.z, cos);
        normalize
            .then(|| quaternion.normalized())
            .unwrap_or(quaternion)
    }

    /// Create an identity `Quaternion`.
    #[must_use]
    #[inline]
    pub fn normal(&self) -> f32 {
        self.iter().map(|val| val * val).sum::<f32>().sqrt()
    }

    /// Normalize the `Quaternion` into a unit `Quaternion`.
    #[inline]
    pub fn normalize(&mut self) {
        let normal = self.normal();
        self.iter_mut().for_each(|val| *val /= normal);
    }

    /// Create a normalized copy of the `Quaternion`.
    #[inline]
    pub fn normalized(&self) -> Self {
        let mut quaternion = *self;
        quaternion.normalize();
        quaternion
    }

    /// Create a conjugate of the `Quaternion`.
    #[inline]
    pub fn conjugate(&self) -> Self {
        Self::new(-self.x, -self.y, -self.z, self.w)
    }

    /// Create an inversed copy of the `Quaternion`.
    #[inline]
    pub fn inverse(&self) -> Self {
        self.conjugate().normalized()
    }

    /// Create the dot-product beween two `Quaternion`s.
    #[must_use]
    #[inline]
    pub fn dot(&self, rhs: Self) -> f32 {
        self.iter()
            .zip(rhs.iter())
            .map(|(val, rhs)| val * rhs)
            .sum()
    }

    /// Caclculate a spherical linear interpolation of the `Quaternion` with a given percentage.
    pub fn slerp(&self, rhs: Self, percentage: f32) -> Self {
        // https://en.wikipedia.org/wiki/Slerp
        let q0 = self.normalized();
        let mut q1 = rhs.normalized();

        let mut dot = q1.dot(q1);
        if dot < 0.0 {
            q1 = -q1;
            dot = -dot;
        }

        let dot_threshold = 0.9995;
        if dot > dot_threshold {
            return (q0 + (q1 - q0) * percentage).normalized();
        }

        let theta0 = dot.acos();
        let theta = theta0 * percentage;
        let sin_theta0 = theta0.sin();
        let (sin_theta, cos_theta) = theta.sin_cos();

        let s0 = cos_theta - dot * sin_theta / sin_theta0;
        let s1 = sin_theta / sin_theta0;

        q0 * s0 + q1 * s1
    }

    /// Converts the `Quaternion` into an array reference of `f32`.
    #[must_use]
    #[inline]
    pub fn as_array(&self) -> &[f32; 4] {
        let array: &[f32; 4] = unsafe { mem::transmute(self) };
        array
    }

    /// Converts the `Quaternion` into a mutable array reference of `f32`.
    #[must_use]
    #[inline]
    pub fn as_array_mut(&mut self) -> &mut [f32; 4] {
        let array: &mut [f32; 4] = unsafe { mem::transmute(self) };
        array
    }

    /// Converts the `Quaternion` into an array of `f32`.
    #[must_use]
    #[inline]
    pub fn to_array(self) -> [f32; 4] {
        let array: [f32; 4] = unsafe { mem::transmute(self) };
        array
    }

    /// Create an iterator over the `Quaternion` dimensions.
    #[inline]
    fn iter(&self) -> std::slice::Iter<'_, f32> {
        self.as_array().iter()
    }

    /// Create a mutable iterator over the `Quaternion` dimensions.
    #[inline]
    fn iter_mut(&mut self) -> std::slice::IterMut<'_, f32> {
        self.as_array_mut().iter_mut()
    }
}

impl IntoIterator for Quaternion {
    type Item = f32;
    type IntoIter = std::array::IntoIter<Self::Item, 4>;

    fn into_iter(self) -> Self::IntoIter {
        self.to_array().into_iter()
    }
}

impl IntoIterator for &Quaternion {
    type Item = f32;
    type IntoIter = std::array::IntoIter<Self::Item, 4>;

    fn into_iter(self) -> Self::IntoIter {
        self.to_array().into_iter()
    }
}

impl Index<usize> for Quaternion {
    type Output = f32;

    fn index(&self, index: usize) -> &f32 {
        self.as_array().index(index)
    }
}

impl IndexMut<usize> for Quaternion {
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        self.as_array_mut().index_mut(index)
    }
}

impl From<[f32; 4]> for Quaternion {
    fn from(array: [f32; 4]) -> Self {
        let v: Self = unsafe { mem::transmute(array) };
        v
    }
}

impl From<&[f32; 4]> for &Quaternion {
    fn from(array: &[f32; 4]) -> Self {
        let v: &Self = unsafe { mem::transmute(array) };
        v
    }
}

impl From<&mut [f32; 4]> for &mut Quaternion {
    fn from(array: &mut [f32; 4]) -> Self {
        let v: &mut Self = unsafe { mem::transmute(array) };
        v
    }
}

impl Add for Quaternion {
    type Output = Quaternion;

    fn add(self, rhs: Self) -> Self::Output {
        let mut quaternion = self;
        quaternion
            .iter_mut()
            .zip(rhs.iter())
            .for_each(|(val, rhs)| *val += rhs);
        quaternion
    }
}

impl Add for &Quaternion {
    type Output = Quaternion;

    fn add(self, rhs: Self) -> Self::Output {
        let mut quaternion = *self;
        quaternion
            .iter_mut()
            .zip(rhs.iter())
            .for_each(|(val, rhs)| *val += rhs);
        quaternion
    }
}

impl Sub for Quaternion {
    type Output = Quaternion;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut quaternion = self;
        quaternion
            .iter_mut()
            .zip(rhs.iter())
            .for_each(|(val, rhs)| *val -= rhs);
        quaternion
    }
}

impl Sub for &Quaternion {
    type Output = Quaternion;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut quaternion = *self;
        quaternion
            .iter_mut()
            .zip(rhs.iter())
            .for_each(|(val, rhs)| *val -= rhs);
        quaternion
    }
}

impl Mul for Quaternion {
    type Output = Quaternion;

    fn mul(self, rhs: Self) -> Self::Output {
        let x = self.x * rhs.w + self.y * rhs.z - self.z * rhs.y + self.w * rhs.x;
        let y = -self.x * rhs.z + self.y * rhs.w + self.z * rhs.x + self.w * rhs.y;
        let z = self.x * rhs.y - self.y * rhs.x + self.z * rhs.w + self.w * rhs.z;
        let w = -self.x * rhs.x - self.y * rhs.y - self.z * rhs.z + self.w * rhs.w;
        Self::new(x, y, z, w)
    }
}

impl Mul for &Quaternion {
    type Output = Quaternion;

    fn mul(self, rhs: Self) -> Self::Output {
        let x = self.x * rhs.w + self.y * rhs.z - self.z * rhs.y + self.w * rhs.x;
        let y = -self.x * rhs.z + self.y * rhs.w + self.z * rhs.x + self.w * rhs.y;
        let z = self.x * rhs.y - self.y * rhs.x + self.z * rhs.w + self.w * rhs.z;
        let w = -self.x * rhs.x - self.y * rhs.y - self.z * rhs.z + self.w * rhs.w;
        Quaternion::new(x, y, z, w)
    }
}

impl Mul<f32> for Quaternion {
    type Output = Quaternion;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut quaternion = self;
        quaternion.iter_mut().for_each(|val| *val *= rhs);
        quaternion
    }
}

impl Mul<f32> for &Quaternion {
    type Output = Quaternion;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut quaternion = *self;
        quaternion.iter_mut().for_each(|val| *val *= rhs);
        quaternion
    }
}

impl Neg for Quaternion {
    type Output = Quaternion;

    fn neg(self) -> Self::Output {
        let mut quaternion = self;
        quaternion.iter_mut().for_each(|val| *val = val.neg());
        quaternion
    }
}

impl Neg for &Quaternion {
    type Output = Quaternion;

    fn neg(self) -> Self::Output {
        let mut quaternion = *self;
        quaternion.iter_mut().for_each(|val| *val = val.neg());
        quaternion
    }
}
