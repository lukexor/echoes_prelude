//! Math types and utilities.

use std::{
    mem,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

macro_rules! impl_vector {
    ($VecDim:ident, $dim:expr => $($field:ident),+) => {
        #[doc = concat!("A ", stringify!($dim), "-dimensional vector")]
        #[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[must_use]
        pub struct $VecDim {
            $(pub $field: f32),+
        }

        impl Default for $VecDim {
            fn default() -> Self {
                Self::origin()
            }
        }

        impl $VecDim {
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

        impl IntoIterator for $VecDim {
            type Item = f32;
            type IntoIter = std::array::IntoIter<Self::Item, $dim>;

            fn into_iter(self) -> Self::IntoIter {
                self.to_array().into_iter()
            }
        }

        impl IntoIterator for &$VecDim {
            type Item = f32;
            type IntoIter = std::array::IntoIter<Self::Item, $dim>;

            fn into_iter(self) -> Self::IntoIter {
                self.to_array().into_iter()
            }
        }

        impl Index<usize> for $VecDim {
            type Output = f32;

            fn index(&self, index: usize) -> &f32 {
                self.as_array().index(index)
            }
        }

        impl IndexMut<usize> for $VecDim {
            fn index_mut(&mut self, index: usize) -> &mut f32 {
                self.as_array_mut().index_mut(index)
            }
        }

        impl From<[f32; $dim]> for $VecDim {
            fn from(array: [f32; $dim]) -> Self {
                let v: Self = unsafe { mem::transmute(array) };
                v
            }
        }

        impl From<&[f32; $dim]> for &$VecDim {
            fn from(array: &[f32; $dim]) -> Self {
                let v: &Self = unsafe { mem::transmute(array) };
                v
            }
        }

        impl From<&mut [f32; $dim]> for &mut $VecDim {
            fn from(array: &mut [f32; $dim]) -> Self {
                let v: &mut Self = unsafe { mem::transmute(array) };
                v
            }
        }

        impl Add for $VecDim {
            type Output = $VecDim;

            fn add(self, rhs: Self) -> Self::Output {
                let mut vector = self;
                vector
                    .iter_mut()
                    .zip(rhs.iter())
                    .for_each(|(val, rhs)| *val += rhs);
                vector
            }
        }

        impl Add for &$VecDim {
            type Output = $VecDim;

            fn add(self, rhs: Self) -> Self::Output {
                let mut vector = *self;
                vector
                    .iter_mut()
                    .zip(rhs.iter())
                    .for_each(|(val, rhs)| *val += rhs);
                vector
            }
        }

        impl AddAssign for $VecDim {
            fn add_assign(&mut self, rhs: Self) {
                self.iter_mut()
                    .zip(rhs.iter())
                    .for_each(|(val, rhs)| *val += rhs);
            }
        }

        impl Sub for $VecDim {
            type Output = $VecDim;

            fn sub(self, rhs: Self) -> Self::Output {
                let mut vector = self;
                vector
                    .iter_mut()
                    .zip(rhs.iter())
                    .for_each(|(val, rhs)| *val -= rhs);
                vector
            }
        }

        impl Sub for &$VecDim {
            type Output = $VecDim;

            fn sub(self, rhs: Self) -> Self::Output {
                let mut vector = *self;
                vector
                    .iter_mut()
                    .zip(rhs.iter())
                    .for_each(|(val, rhs)| *val -= rhs);
                vector
            }
        }

        impl SubAssign for $VecDim {
            fn sub_assign(&mut self, rhs: Self) {
                self.iter_mut()
                    .zip(rhs.iter())
                    .for_each(|(val, rhs)| *val -= rhs);
            }
        }

        impl Mul for $VecDim {
            type Output = $VecDim;

            fn mul(self, rhs: Self) -> Self::Output {
                let mut vector = self;
                vector
                    .iter_mut()
                    .zip(rhs.iter())
                    .for_each(|(val, rhs)| *val *= rhs);
                vector
            }
        }

        impl Mul for &$VecDim {
            type Output = $VecDim;

            fn mul(self, rhs: Self) -> Self::Output {
                let mut vector = *self;
                vector
                    .iter_mut()
                    .zip(rhs.iter())
                    .for_each(|(val, rhs)| *val *= rhs);
                vector
            }
        }

        impl Mul<f32> for $VecDim {
            type Output = $VecDim;

            fn mul(self, rhs: f32) -> Self::Output {
                let mut vector = self;
                vector.iter_mut().for_each(|val| *val *= rhs);
                vector
            }
        }

        impl Mul<f32> for &$VecDim {
            type Output = $VecDim;

            fn mul(self, rhs: f32) -> Self::Output {
                let mut vector = *self;
                vector.iter_mut().for_each(|val| *val *= rhs);
                vector
            }
        }

        impl Mul<$VecDim> for f32 {
            type Output = $VecDim;

            fn mul(self, rhs: $VecDim) -> Self::Output {
                let mut vector = rhs;
                vector.iter_mut().for_each(|val| *val *= self);
                vector
            }
        }

        impl Mul<&$VecDim> for f32 {
            type Output = $VecDim;

            fn mul(self, rhs: &$VecDim) -> Self::Output {
                let mut vector = *rhs;
                vector.iter_mut().for_each(|val| *val *= self);
                vector
            }
        }

        impl MulAssign for $VecDim {
            fn mul_assign(&mut self, rhs: Self) {
                self.iter_mut()
                    .zip(rhs.iter())
                    .for_each(|(val, rhs)| *val *= rhs);
            }
        }

        impl MulAssign<f32> for $VecDim {
            fn mul_assign(&mut self, rhs: f32) {
                self.iter_mut().for_each(|val| *val *= rhs);
            }
        }

        impl Div for $VecDim {
            type Output = $VecDim;

            fn div(self, rhs: Self) -> Self::Output {
                let mut vector = self;
                vector
                    .iter_mut()
                    .zip(rhs.iter())
                    .for_each(|(val, rhs)| *val /= rhs);
                vector
            }
        }

        impl Div for &$VecDim {
            type Output = $VecDim;

            fn div(self, rhs: Self) -> Self::Output {
                let mut vector = *self;
                vector
                    .iter_mut()
                    .zip(rhs.iter())
                    .for_each(|(val, rhs)| *val /= rhs);
                vector
            }
        }

        impl Div<f32> for $VecDim {
            type Output = $VecDim;

            fn div(self, rhs: f32) -> Self::Output {
                let mut vector = self;
                vector.iter_mut().for_each(|val| *val /= rhs);
                vector
            }
        }

        impl Div<f32> for &$VecDim {
            type Output = $VecDim;

            fn div(self, rhs: f32) -> Self::Output {
                let mut vector = *self;
                vector.iter_mut().for_each(|val| *val /= rhs);
                vector
            }
        }

        impl Div<$VecDim> for f32 {
            type Output = $VecDim;

            fn div(self, rhs: $VecDim) -> Self::Output {
                let mut vector = rhs;
                vector.iter_mut().for_each(|val| *val /= self);
                vector
            }
        }

        impl Div<&$VecDim> for f32 {
            type Output = $VecDim;

            fn div(self, rhs: &$VecDim) -> Self::Output {
                let mut vector = *rhs;
                vector.iter_mut().for_each(|val| *val /= self);
                vector
            }
        }

        impl DivAssign for $VecDim {
            fn div_assign(&mut self, rhs: Self) {
                self.iter_mut()
                    .zip(rhs.iter())
                    .for_each(|(val, rhs)| *val /= rhs);
            }
        }

        impl DivAssign<f32> for $VecDim {
            fn div_assign(&mut self, rhs: f32) {
                self.iter_mut().for_each(|val| *val /= rhs);
            }
        }

        impl Neg for $VecDim {
            type Output = $VecDim;

            fn neg(self) -> Self::Output {
                let mut vector = self;
                vector.iter_mut().for_each(|val| *val = val.neg());
                vector
            }
        }

        impl Neg for &$VecDim {
            type Output = $VecDim;

            fn neg(self) -> Self::Output {
                let mut vector = *self;
                vector.iter_mut().for_each(|val| *val = val.neg());
                vector
            }
        }
    };
}

impl_vector!(Vec1, 1 => x);
impl_vector!(Vec2, 2 => x, y);
impl_vector!(Vec3, 3 => x, y, z);
impl_vector!(Vec4, 4 => x, y, z, w);

/// Constructs a new 1D [Vector].
#[macro_export]
macro_rules! vec1 {
    () => {
        $crate::math::Vec1::origin()
    };
    ($val:expr) => {
        $crate::math::Vec1::from($val)
    };
}

/// Constructs a new 2D [Vector].
#[macro_export]
macro_rules! vec2 {
    () => {
        $crate::math::Vec2::origin()
    };
    ($val:expr) => {
        $crate::math::Vec2::from($val)
    };
    ($val:expr, $y:expr $(,)?) => {{
        let v = $crate::vec1!($val);
        $crate::math::Vec2::new(v.x, $y)
    }};
}

/// Constructs a new 3D [Vector].
#[macro_export]
macro_rules! vec3 {
    () => {
        $crate::math::Vec3::origin()
    };
    ($val:expr) => {
        $crate::math::Vec3::from($val)
    };
    ($val:expr, $z:expr $(,)?) => {{
        let v = $crate::vec2!($val);
        $crate::math::Vec3::new(v.x, v.y, $z)
    }};
    ($val:expr, $y:expr, $z:expr $(,)?) => {{
        let v = $crate::vec1!($val);
        $crate::math::Vec3::new(v.x, $y, $z)
    }};
}

/// Constructs a new 4D [Vector].
#[macro_export]
macro_rules! vec4 {
    () => {
        $crate::math::Vec4::origin()
    };
    ($val:expr) => {
        $crate::math::Vec4::from($val)
    };
    ($val:expr, $w:expr $(,)?) => {{
        let v = $crate::vec3!($val);
        $crate::math::Vec4::new(v.x, v.y, v.z, $w)
    }};
    ($val:expr, $z:expr, $w:expr $(,)?) => {{
        let v = $crate::vec2!($val);
        $crate::math::Vec4::new(v.x, v.y, $z, $w)
    }};
    ($val:expr, $y:expr, $z:expr, $w:expr $(,)?) => {{
        let v = $crate::vec1!($val);
        $crate::math::Vec4::new(v.x, $y, $z, $w)
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

    /// Calculate the dot-product between two vectors, pairwise.
    #[must_use]
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn dot_pairwise(a: [f32; 4], b: [f32; 4]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    }
}

/// A 4x4 matrix.
#[derive(Debug, Copy, Clone, PartialEq)]
#[must_use]
pub struct Mat4 {
    col0: Vec4,
    col1: Vec4,
    col2: Vec4,
    col3: Vec4,
}

impl Default for Mat4 {
    fn default() -> Self {
        Self {
            col0: vec4!(1.0, 0.0, 0.0, 0.0),
            col1: vec4!(0.0, 1.0, 0.0, 0.0),
            col2: vec4!(0.0, 0.0, 1.0, 0.0),
            col3: vec4!(0.0, 0.0, 0.0, 1.0),
        }
    }
}

/// Constructs a new [Mat4].
#[macro_export]
macro_rules! mat4 {
    () => {
        $crate::math::Mat4::identity()
    };
    ($val:expr) => {
        $crate::math::Mat4::from($val)
    };
    ($c0:expr, $c1:expr, $(,)?) => {
        $crate::math::Mat4::new(
            $c0,
            $c1,
            vec4!(0.0, 0.0, 1.0, 0.0),
            vec4!(0.0, 0.0, 0.0, 1.0),
        )
    };
    ($c0:expr, $c1:expr, $c2:expr $(,)?) => {
        $crate::math::Mat4::new($c0, $c1, $c2, vec4!(0.0, 0.0, 0.0, 1.0))
    };
    ($c0:expr, $c1:expr, $c2:expr, $c3:expr $(,)?) => {
        $crate::math::Mat4::new($c0, $c1, $c2, $c3)
    };
}

impl Mat4 {
    /// Create a 4x4 matrix from given columns.
    #[inline]
    pub fn new(
        col0: impl Into<Vec4>,
        col1: impl Into<Vec4>,
        col2: impl Into<Vec4>,
        col3: impl Into<Vec4>,
    ) -> Self {
        Self {
            col0: col0.into(),
            col1: col1.into(),
            col2: col2.into(),
            col3: col3.into(),
        }
    }

    /// Create an orthographic projection matrix.
    #[inline]
    pub fn orthographic(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near_clip: f32,
        far_clip: f32,
    ) -> Self {
        let lr = 1.0 / (left - right);
        let bt = 1.0 / (bottom - top);
        let nf = 1.0 / (near_clip - far_clip);
        let mut matrix = Self::identity();

        matrix[(0, 0)] = -2.0 * lr;
        matrix[(1, 1)] = -2.0 * bt;
        matrix[(2, 2)] = -2.0 * nf;

        matrix[(3, 0)] = (left + right) * lr;
        matrix[(3, 1)] = (bottom + top) * bt;
        matrix[(3, 2)] = (near_clip + far_clip) * nf;

        matrix
    }

    /// Create a perspective projection matrix.
    #[inline]
    pub fn perspective(
        fov: Radians<f32>,
        aspect_ratio: f32,
        near_clip: f32,
        far_clip: f32,
    ) -> Self {
        let tan_frac_fov_two = (fov * 0.5).tan();
        let nf = far_clip - near_clip;
        let mut matrix = Self::identity();

        matrix[(0, 0)] = 1.0 / (aspect_ratio * tan_frac_fov_two);
        matrix[(1, 1)] = 1.0 / tan_frac_fov_two;
        matrix[(2, 2)] = -((near_clip + far_clip) / nf);

        matrix[(2, 3)] = -1.0;
        matrix[(3, 2)] = -((2.0 * near_clip * far_clip) / nf);

        matrix
    }

    /// Create a look-at matrix.
    #[inline]
    pub fn look_at(
        position: impl Into<Vec3>,
        target: impl Into<Vec3>,
        up: impl Into<Vec3>,
    ) -> Self {
        let position = position.into();
        let target = target.into();
        let up = up.into();

        let z_axis = (target - position).normalized();
        let x_axis = z_axis.cross(up).normalized();
        let y_axis = x_axis.cross(z_axis);
        mat4!(
            [x_axis.x, y_axis.x, -z_axis.x, 0.0],
            [x_axis.y, y_axis.y, -z_axis.y, 0.0],
            [x_axis.z, y_axis.z, -z_axis.z, 0.0],
            [
                -x_axis.dot(position),
                -y_axis.dot(position),
                z_axis.dot(position),
                1.0,
            ],
        )
    }

    /// Create a translation matrix for the given position.
    #[inline]
    pub fn translation(position: impl Into<Vec3>) -> Self {
        let position = position.into();
        let mut matrix = Self::identity();
        matrix[(3, 0)] = position.x;
        matrix[(3, 1)] = position.y;
        matrix[(3, 2)] = position.z;
        matrix
    }

    /// Create a scale matrix for the given scale.
    #[inline]
    pub fn scale(scale: impl Into<Vec3>) -> Self {
        let scale = scale.into();
        let mut matrix = Self::identity();
        matrix[(0, 0)] = scale.x;
        matrix[(1, 1)] = scale.y;
        matrix[(2, 2)] = scale.z;
        matrix
    }

    /// Create a rotation matrix for all axes for the given angles in degrees.
    #[inline]
    pub fn rotation(rotation: impl Into<Vec3>) -> Self {
        let rotation = rotation.into();
        let rotate_x = Self::rotation_x(rotation.x.into());
        let rotate_y = Self::rotation_y(rotation.y.into());
        let rotate_z = Self::rotation_z(rotation.z.into());
        rotate_x * rotate_y * rotate_z
    }

    /// Create a rotation matrix for about the x-axis for the given angle in degrees.
    #[inline]
    pub fn rotation_x(angle: Angle<f32>) -> Self {
        let mut matrix = Self::identity();
        let (sin, cos) = angle.to_radians().sin_cos();
        matrix[(1, 1)] = cos;
        matrix[(1, 2)] = sin;
        matrix[(2, 1)] = -sin;
        matrix[(2, 2)] = cos;
        matrix
    }

    /// Create a rotation matrix for about the y-axis for the given angle in degrees.
    #[inline]
    pub fn rotation_y(angle: Angle<f32>) -> Self {
        let mut matrix = Self::identity();
        let (sin, cos) = angle.to_radians().sin_cos();
        matrix[(0, 0)] = cos;
        matrix[(0, 2)] = -sin;
        matrix[(2, 0)] = sin;
        matrix[(2, 2)] = cos;
        matrix
    }

    /// Create a rotation matrix for about the z-axis for the given angle in degrees.
    #[inline]
    pub fn rotation_z(angle: Angle<f32>) -> Self {
        let mut matrix = Self::identity();
        let (sin, cos) = angle.to_radians().sin_cos();
        matrix[(0, 0)] = cos;
        matrix[(0, 1)] = sin;
        matrix[(1, 0)] = -sin;
        matrix[(1, 1)] = cos;
        matrix
    }

    /// Create a rotation matrix from a `Quaternion`.
    pub fn from_quaternion(&self, quaternion: Quaternion) -> Self {
        let q = quaternion.normalized();
        let mut matrix = Self::identity();

        matrix[(0, 0)] = 1.0 - 2.0 * q.y * q.y - 2.0 * q.z * q.z;
        matrix[(0, 1)] = 2.0 * q.x * q.y - 2.0 * q.z * q.w;
        matrix[(0, 2)] = 2.0 * q.x * q.z - 2.0 * q.y * q.w;

        matrix[(1, 0)] = 2.0 * q.x * q.y + 2.0 * q.z * q.w;
        matrix[(1, 1)] = 1.0 - 2.0 * q.x * q.x - 2.0 * q.z * q.z;
        matrix[(1, 2)] = 2.0 * q.y * q.z - 2.0 * q.x * q.w;

        matrix[(2, 0)] = 2.0 * q.x * q.z - 2.0 * q.y * q.w;
        matrix[(0, 0)] = 2.0 * q.y * q.z + 2.0 * q.x * q.w;
        matrix[(2, 2)] = 1.0 - 2.0 * q.x * q.x - 2.0 * q.y * q.y;

        matrix
    }

    /// Create a rotation matrix from a `Quaternion` around a center point.
    pub fn from_quaternion_center(&self, quaternion: Quaternion, center: Vec3) -> Self {
        let q = quaternion;
        let c = center;
        let mut matrix = Self::default();

        matrix[(0, 0)] = (q.x * q.x) - (q.y * q.y) - (q.z * q.z) + (q.w * q.w);
        matrix[(0, 1)] = 2.0 * ((q.x * q.y) + (q.z * q.w));
        matrix[(0, 2)] = 2.0 * ((q.x * q.z) - (q.y * q.w));
        matrix[(0, 3)] = c.x - c.x * matrix[(0, 0)] - c.y * matrix[(0, 1)] - c.z * matrix[(0, 2)];

        matrix[(1, 0)] = 2.0 * ((q.x * q.y) - (q.z * q.w));
        matrix[(1, 1)] = -(q.x * q.x) + (q.y * q.y) - (q.z * q.z) + (q.w * q.w);
        matrix[(1, 2)] = 2.0 * ((q.y * q.z) + (q.x * q.w));
        matrix[(1, 3)] = c.y - c.x * matrix[(1, 0)] - c.y * matrix[(1, 1)] - c.z * matrix[(1, 2)];

        matrix[(2, 0)] = 2.0 * ((q.x * q.z) + (q.y * q.w));
        matrix[(2, 1)] = 2.0 * ((q.y * q.z) - (q.x * q.w));
        matrix[(2, 2)] = -(q.x * q.x) - (q.y * q.y) + (q.z * q.z) + (q.w * q.w);
        matrix[(2, 3)] = c.z - c.x * matrix[(2, 0)] - c.y * matrix[(2, 1)] - c.z * matrix[(2, 2)];

        matrix[(3, 0)] = 0.0;
        matrix[(3, 1)] = 0.0;
        matrix[(3, 2)] = 0.0;
        matrix[(3, 3)] = 1.0;

        matrix
    }

    /// Return matrix columns as a single array reference of `Vec4`.
    #[inline]
    pub fn as_array(&self) -> &[Vec4; 4] {
        let array: &[Vec4; 4] = unsafe { mem::transmute(self) };
        array
    }

    /// Return matrix columns as a single mutable array reference of `Vec4`.
    #[inline]
    pub fn as_array_mut(&mut self) -> &mut [Vec4; 4] {
        let array: &mut [Vec4; 4] = unsafe { mem::transmute(self) };
        array
    }

    /// Create an iterator over the matrix columns.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, Vec4> {
        self.as_array().iter()
    }

    /// Create a mutable iterator over the matrix columns.
    #[inline]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Vec4> {
        self.as_array_mut().iter_mut()
    }

    /// Create an up unit vector relative to the matrix.
    #[inline]
    pub fn up(&self) -> Vec3 {
        vec3!(self[(0, 1)], self[(1, 1)], self[(2, 1)]).normalized()
    }

    /// Create a down unit vector relative to the matrix.
    #[inline]
    pub fn down(&self) -> Vec3 {
        vec3!(-self[(0, 1)], -self[(1, 1)], -self[(2, 1)]).normalized()
    }

    /// Create a left unit vector relative to the matrix.
    #[inline]
    pub fn left(&self) -> Vec3 {
        vec3!(-self[(0, 0)], -self[(1, 0)], -self[(2, 0)]).normalized()
    }

    /// Create a right unit vector relative to the matrix.
    #[inline]
    pub fn right(&self) -> Vec3 {
        vec3!(self[(0, 0)], self[(1, 0)], self[(2, 0)]).normalized()
    }

    /// Create a forward unit vector relative to the matrix.
    #[inline]
    pub fn forward(&self) -> Vec3 {
        vec3!(-self[(0, 2)], -self[(1, 2)], -self[(2, 2)]).normalized()
    }

    /// Create a backward unit vector relative to the matrix.
    #[inline]
    pub fn backward(&self) -> Vec3 {
        vec3!(self[(0, 2)], self[(1, 2)], self[(2, 2)]).normalized()
    }

    /// Create a copy of the matrix inverted about the x-axis.
    #[inline]
    pub fn inverted_x(&self) -> Self {
        let mut matrix = *self;
        matrix[(0, 0)] *= -1.0;
        matrix
    }

    /// Create a copy of the matrix inverted about the y-axis.
    #[inline]
    pub fn inverted_y(&self) -> Self {
        let mut matrix = *self;
        matrix[(1, 1)] *= -1.0;
        matrix
    }

    /// Create a copy of the matrix inverted about the z-axis.
    #[inline]
    pub fn inverted_z(&self) -> Self {
        let mut matrix = *self;
        matrix[(2, 2)] *= -1.0;
        matrix
    }

    /// Create a transposed copy of the matrix.
    pub fn transposed(&self) -> Self {
        let mut matrix = Self::identity();
        matrix[(0, 0)] = self[(0, 0)];
        matrix[(0, 1)] = self[(1, 0)];
        matrix[(0, 2)] = self[(2, 0)];
        matrix[(0, 3)] = self[(3, 0)];
        matrix[(1, 0)] = self[(0, 1)];
        matrix[(1, 1)] = self[(1, 1)];
        matrix[(1, 2)] = self[(2, 1)];
        matrix[(1, 3)] = self[(3, 1)];
        matrix[(2, 0)] = self[(0, 2)];
        matrix[(2, 1)] = self[(1, 2)];
        matrix[(2, 2)] = self[(2, 2)];
        matrix[(2, 3)] = self[(3, 2)];
        matrix[(3, 0)] = self[(0, 3)];
        matrix[(3, 1)] = self[(1, 3)];
        matrix[(3, 2)] = self[(2, 3)];
        matrix[(3, 3)] = self[(3, 3)];
        matrix
    }

    /// Create an inversed copy of the matrix.
    pub fn inverse(&self) -> Self {
        let m = self;

        let t0 = m[(2, 2)] * m[(3, 3)];
        let t1 = m[(3, 2)] * m[(2, 3)];
        let t2 = m[(1, 2)] * m[(3, 3)];
        let t3 = m[(3, 2)] * m[(1, 3)];
        let t4 = m[(1, 2)] * m[(2, 3)];
        let t5 = m[(2, 2)] * m[(1, 3)];
        let t6 = m[(0, 2)] * m[(3, 3)];
        let t7 = m[(3, 2)] * m[(0, 3)];
        let t8 = m[(0, 2)] * m[(2, 3)];
        let t9 = m[(2, 2)] * m[(0, 3)];
        let t10 = m[(0, 2)] * m[(1, 3)];
        let t11 = m[(1, 2)] * m[(0, 3)];
        let t12 = m[(2, 0)] * m[(3, 1)];
        let t13 = m[(3, 0)] * m[(2, 1)];
        let t14 = m[(1, 0)] * m[(3, 1)];
        let t15 = m[(3, 0)] * m[(1, 1)];
        let t16 = m[(1, 0)] * m[(2, 1)];
        let t17 = m[(2, 0)] * m[(1, 1)];
        let t18 = m[(0, 0)] * m[(3, 1)];
        let t19 = m[(3, 0)] * m[(0, 1)];
        let t20 = m[(0, 0)] * m[(2, 1)];
        let t21 = m[(2, 0)] * m[(0, 1)];
        let t22 = m[(0, 0)] * m[(1, 1)];
        let t23 = m[(1, 0)] * m[(0, 1)];

        let mut matrix = Mat4::default();

        matrix[(0, 0)] = (t0 * m[(1, 1)] + t3 * m[(2, 1)] + t4 * m[(3, 1)])
            - (t1 * m[(1, 1)] + t2 * m[(2, 1)] + t5 * m[(3, 1)]);
        matrix[(0, 1)] = (t1 * m[(0, 1)] + t6 * m[(2, 1)] + t9 * m[(3, 1)])
            - (t0 * m[(0, 1)] + t7 * m[(2, 1)] + t8 * m[(3, 1)]);
        matrix[(0, 2)] = (t2 * m[(0, 1)] + t7 * m[(1, 1)] + t10 * m[(3, 1)])
            - (t3 * m[(0, 1)] + t6 * m[(1, 1)] + t11 * m[(3, 1)]);
        matrix[(0, 3)] = (t5 * m[(0, 1)] + t8 * m[(1, 1)] + t11 * m[(2, 1)])
            - (t4 * m[(0, 1)] + t9 * m[(1, 1)] + t10 * m[(2, 1)]);

        let d = 1.0
            / (m[(0, 0)] * matrix[(0, 0)]
                + m[(1, 0)] * matrix[(0, 1)]
                + m[(2, 0)] * matrix[(0, 2)]
                + m[(3, 0)] * matrix[(0, 3)]);

        matrix[(0, 0)] *= d;
        matrix[(0, 1)] *= d;
        matrix[(0, 2)] *= d;
        matrix[(0, 3)] *= d;
        matrix[(1, 0)] = d
            * ((t1 * m[(1, 0)] + t2 * m[(2, 0)] + t5 * m[(3, 0)])
                - (t0 * m[(1, 0)] + t3 * m[(2, 0)] + t4 * m[(3, 0)]));
        matrix[(1, 1)] = d
            * ((t0 * m[(0, 0)] + t7 * m[(2, 0)] + t8 * m[(3, 0)])
                - (t1 * m[(0, 0)] + t6 * m[(2, 0)] + t9 * m[(3, 0)]));
        matrix[(1, 2)] = d
            * ((t3 * m[(0, 0)] + t6 * m[(1, 0)] + t11 * m[(3, 0)])
                - (t2 * m[(0, 0)] + t7 * m[(1, 0)] + t10 * m[(3, 0)]));
        matrix[(1, 3)] = d
            * ((t4 * m[(0, 0)] + t9 * m[(1, 0)] + t10 * m[(2, 0)])
                - (t5 * m[(0, 0)] + t8 * m[(1, 0)] + t11 * m[(2, 0)]));
        matrix[(2, 0)] = d
            * ((t12 * m[(1, 3)] + t15 * m[(2, 3)] + t16 * m[(3, 3)])
                - (t13 * m[(1, 3)] + t14 * m[(2, 3)] + t17 * m[(3, 3)]));
        matrix[(2, 1)] = d
            * ((t13 * m[(0, 3)] + t18 * m[(2, 3)] + t21 * m[(3, 3)])
                - (t12 * m[(0, 3)] + t19 * m[(2, 3)] + t20 * m[(3, 3)]));
        matrix[(2, 2)] = d
            * ((t14 * m[(0, 3)] + t19 * m[(1, 3)] + t22 * m[(3, 3)])
                - (t15 * m[(0, 3)] + t18 * m[(1, 3)] + t23 * m[(3, 3)]));
        matrix[(2, 3)] = d
            * ((t17 * m[(0, 3)] + t20 * m[(1, 3)] + t23 * m[(2, 3)])
                - (t16 * m[(0, 3)] + t21 * m[(1, 3)] + t22 * m[(2, 3)]));
        matrix[(3, 0)] = d
            * ((t14 * m[(2, 2)] + t17 * m[(3, 2)] + t13 * m[(1, 2)])
                - (t16 * m[(3, 2)] + t12 * m[(1, 2)] + t15 * m[(2, 2)]));
        matrix[(3, 1)] = d
            * ((t20 * m[(3, 2)] + t12 * m[(0, 2)] + t19 * m[(2, 2)])
                - (t18 * m[(2, 2)] + t21 * m[(3, 2)] + t13 * m[(0, 2)]));
        matrix[(3, 2)] = d
            * ((t18 * m[(1, 2)] + t23 * m[(3, 2)] + t15 * m[(0, 2)])
                - (t22 * m[(3, 2)] + t14 * m[(0, 2)] + t19 * m[(1, 2)]));
        matrix[(3, 3)] = d
            * ((t22 * m[(2, 2)] + t16 * m[(0, 2)] + t21 * m[(1, 2)])
                - (t20 * m[(1, 2)] + t23 * m[(2, 2)] + t17 * m[(0, 2)]));

        matrix
    }
}

impl Mul for Mat4 {
    type Output = Mat4;

    fn mul(self, rhs: Self) -> Self::Output {
        let mat4_mul = |m: Mat4, v: Vec4| {
            vec4!(
                v.x * m[(0, 0)] + v.y * m[(1, 0)] + v.z * m[(2, 0)] + v.w * m[(3, 0)],
                v.x * m[(0, 1)] + v.y * m[(1, 1)] + v.z * m[(2, 1)] + v.w * m[(3, 1)],
                v.x * m[(0, 2)] + v.y * m[(1, 2)] + v.z * m[(2, 2)] + v.w * m[(3, 2)],
                v.x * m[(0, 3)] + v.y * m[(1, 3)] + v.z * m[(2, 3)] + v.w * m[(3, 3)],
            )
        };
        mat4!(
            mat4_mul(
                self,
                vec4!(rhs[(0, 0)], rhs[(0, 1)], rhs[(0, 2)], rhs[(0, 3)])
            ),
            mat4_mul(
                self,
                vec4!(rhs[(1, 0)], rhs[(1, 1)], rhs[(1, 2)], rhs[(1, 3)])
            ),
            mat4_mul(
                self,
                vec4!(rhs[(2, 0)], rhs[(2, 1)], rhs[(2, 2)], rhs[(2, 3)])
            ),
            mat4_mul(
                self,
                vec4!(rhs[(3, 0)], rhs[(3, 1)], rhs[(3, 2)], rhs[(3, 3)])
            ),
        )
    }
}

impl Mul<Vec3> for Mat4 {
    type Output = Vec3;

    fn mul(self, v: Vec3) -> Self::Output {
        let m = self;
        vec3!(
            v.x * m[(0, 0)] + v.y * m[(0, 1)] + v.z * m[(0, 2)] + m[(0, 3)],
            v.x * m[(1, 0)] + v.y * m[(1, 1)] + v.z * m[(1, 2)] + m[(1, 3)],
            v.x * m[(2, 0)] + v.y * m[(2, 1)] + v.z * m[(2, 2)] + m[(2, 3)],
        )
    }
}

impl Mul<Mat4> for Vec3 {
    type Output = Vec3;

    fn mul(self, m: Mat4) -> Self::Output {
        let v = self;
        vec3!(
            v.x * m[(0, 0)] + v.y * m[(0, 1)] + v.z * m[(0, 2)] + m[(0, 3)],
            v.x * m[(1, 0)] + v.y * m[(1, 1)] + v.z * m[(1, 2)] + m[(1, 3)],
            v.x * m[(2, 0)] + v.y * m[(2, 1)] + v.z * m[(2, 2)] + m[(2, 3)],
        )
    }
}

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;

    fn mul(self, v: Vec4) -> Self::Output {
        let m = self;
        vec4!(
            v.x * m[(0, 0)] + v.y * m[(0, 1)] + v.z * m[(0, 2)] + v.w * m[(0, 3)],
            v.x * m[(1, 0)] + v.y * m[(1, 1)] + v.z * m[(1, 2)] + v.w * m[(1, 3)],
            v.x * m[(2, 0)] + v.y * m[(2, 1)] + v.z * m[(2, 2)] + v.w * m[(2, 3)],
            v.x * m[(3, 0)] + v.y * m[(3, 1)] + v.z * m[(3, 2)] + v.w * m[(3, 3)],
        )
    }
}

impl Mul<Mat4> for Vec4 {
    type Output = Vec4;

    fn mul(self, m: Mat4) -> Self::Output {
        let v = self;
        vec4!(
            v.x * m[(0, 0)] + v.y * m[(0, 1)] + v.z * m[(0, 2)] + v.w * m[(0, 3)],
            v.x * m[(1, 0)] + v.y * m[(1, 1)] + v.z * m[(1, 2)] + v.w * m[(1, 3)],
            v.x * m[(2, 0)] + v.y * m[(2, 1)] + v.z * m[(2, 2)] + v.w * m[(2, 3)],
            v.x * m[(3, 0)] + v.y * m[(3, 1)] + v.z * m[(3, 2)] + v.w * m[(3, 3)],
        )
    }
}

impl Mat4 {
    /// Construct an identity matrix.
    #[inline]
    pub fn identity() -> Self {
        Self::default()
    }
}

impl Index<usize> for Mat4 {
    type Output = Vec4;

    fn index(&self, index: usize) -> &Self::Output {
        self.as_array().index(index)
    }
}

impl IndexMut<usize> for Mat4 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.as_array_mut().index_mut(index)
    }
}

impl Index<(usize, usize)> for Mat4 {
    type Output = f32;

    fn index(&self, (n, m): (usize, usize)) -> &Self::Output {
        self.index(n).index(m)
    }
}

impl IndexMut<(usize, usize)> for Mat4 {
    fn index_mut(&mut self, (n, m): (usize, usize)) -> &mut Self::Output {
        self.index_mut(n).index_mut(m)
    }
}

impl From<[f32; 16]> for Mat4 {
    fn from(array: [f32; 16]) -> Self {
        let v: Self = unsafe { mem::transmute(array) };
        v
    }
}

impl From<&[f32; 16]> for &Mat4 {
    fn from(array: &[f32; 16]) -> Self {
        let v: &Self = unsafe { mem::transmute(array) };
        v
    }
}

impl From<&mut [f32; 16]> for &mut Mat4 {
    fn from(array: &mut [f32; 16]) -> Self {
        let v: &mut Self = unsafe { mem::transmute(array) };
        v
    }
}

impl From<[[f32; 4]; 4]> for Mat4 {
    fn from(array: [[f32; 4]; 4]) -> Self {
        let v: Self = unsafe { mem::transmute(array) };
        v
    }
}

impl From<&[[f32; 4]; 4]> for &Mat4 {
    fn from(array: &[[f32; 4]; 4]) -> Self {
        let v: &Self = unsafe { mem::transmute(array) };
        v
    }
}

impl From<&mut [[f32; 4]; 4]> for &mut Mat4 {
    fn from(array: &mut [[f32; 4]; 4]) -> Self {
        let v: &mut Self = unsafe { mem::transmute(array) };
        v
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

/// Convert a value from one range to another range.
#[inline]
#[must_use]
pub fn convert_range(value: f32, from_min: f32, from_max: f32, to_min: f32, to_max: f32) -> f32 {
    (((value - from_min) * (to_max - to_min)) / (from_max - from_min)) + to_min
}

/// Convert separate RGB values to a single u32.
#[inline]
#[must_use]
pub fn rgb_to_u32(r: u32, g: u32, b: u32) -> u32 {
    ((r * 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF)
}

/// Convert a single u32 into separate RGB values.
#[inline]
#[must_use]
pub fn u32_to_rgb(value: u32) -> [u32; 3] {
    [(value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF]
}

#[cfg(test)]
mod tests {
    #[test]
    fn matrix_multiply_matrix() {
        let m1 = mat4!(
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0]
        );
        let m2 = mat4!(
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0]
        );
        assert_eq!(
            m1 * m2,
            mat4!(
                [34.0, 44.0, 54.0, 64.0],
                [82.0, 108.0, 134.0, 160.0],
                [34.0, 44.0, 54.0, 64.0],
                [82.0, 108.0, 134.0, 160.0]
            )
        );
    }

    #[test]
    fn matrix_multiply_vector() {
        let m = mat4!(
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0]
        );
        let v = vec4!(1.0, 2.0, 3.0, 4.0);
        assert_eq!(m * v, vec4!(30.0, 70.0, 30.0, 70.0));
    }

    #[test]
    fn matrix_forward() {
        let m = mat4!(
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0]
        );
        assert!(m
            .forward()
            .compare(vec3!(-0.366_508, -0.855_186, -0.366_508), 0.0002));
    }
}
