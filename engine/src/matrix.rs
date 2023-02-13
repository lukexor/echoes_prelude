//! Matrix types and methods.

use crate::{
    num::{Angle, ApproxEq},
    vec3, vec4,
    vector::{Quaternion, Vec2, Vec3, Vec4},
};
use serde::{Deserialize, Serialize};
use std::{
    mem,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// Implements matrix multiplication with a vector with a given dimensionality.
macro_rules! mul_vector {
    ($Mat:ident, $Vec:ident, Vec3) => {
        Vec3::new(
            $Mat[0].x * $Vec.x + $Mat[1].x * $Vec.y + $Mat[2].x * $Vec.z,
            $Mat[0].y * $Vec.x + $Mat[1].y * $Vec.y + $Mat[2].y * $Vec.z,
            $Mat[0].z * $Vec.x + $Mat[1].z * $Vec.y + $Mat[2].z * $Vec.z,
        )
    };
    ($Mat:ident, $Vec:ident, Vec4) => {
        Vec4::new(
            $Mat[0].x * $Vec.x + $Mat[1].x * $Vec.y + $Mat[2].x * $Vec.z + $Mat[3].x * $Vec.w,
            $Mat[0].y * $Vec.x + $Mat[1].y * $Vec.y + $Mat[2].y * $Vec.z + $Mat[3].y * $Vec.w,
            $Mat[0].z * $Vec.x + $Mat[1].z * $Vec.y + $Mat[2].z * $Vec.z + $Mat[3].z * $Vec.w,
            $Mat[0].w * $Vec.x + $Mat[1].w * $Vec.y + $Mat[2].w * $Vec.z + $Mat[3].w * $Vec.w,
        )
    };
}

/// Implements matrix multiplication with given size.
macro_rules! mul_matrix {
    ($Mat:ident, $Rhs:ident, Mat3) => {
        Mat3::new($Mat * $Rhs.col0, $Mat * $Rhs.col1, $Mat * $Rhs.col2)
    };
    ($Mat:ident, $Rhs:ident, Mat4) => {
        Mat4::new(
            $Mat * $Rhs.col0,
            $Mat * $Rhs.col1,
            $Mat * $Rhs.col2,
            $Mat * $Rhs.col3,
        )
    };
}

/// Implements matrix type and methods for a given size.
macro_rules! impl_matrix {
    ($({
        $Mat:ident, $Vec:ident, $col_dim:expr, $row_dim:expr => $($field:ident),+
    }),+ $(,)?) => {
        $(
            #[doc = concat!("A ", stringify!($col_dim), "x", stringify!($row_dim), " matrix.")]
            #[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
            #[must_use]
            pub struct $Mat {
                $(pub $field: $Vec),+
            }

            impl Default for $Mat {
                fn default() -> Self {
                    Self::identity()
                }
            }

            impl $Mat {
                #[doc = concat!("Create a ", stringify!($col_dim), "x", stringify!($row_dim), " matrix from given columns.")]
                #[inline]
                pub fn new($($field: impl Into<$Vec>),+) -> Self {
                    Self {
                        $($field: $field.into()),+
                    }
                }

                #[doc = concat!("Create a ", stringify!($col_dim), "x", stringify!($row_dim), " matrix reference from given array of columns.")]
                #[inline]
                pub fn from_array(array: &[$Vec; $col_dim]) -> &Self {
                    let array: &Self = unsafe { mem::transmute(array) };
                    array
                }

                #[doc = concat!("Create a ", stringify!($col_dim), "x", stringify!($row_dim), " mutable matrix reference from given array of columns.")]
                #[inline]
                pub fn from_array_mut(array: &mut [$Vec; $col_dim]) -> &mut Self {
                    let array: &mut Self = unsafe { mem::transmute(array) };
                    array
                }

                #[doc = concat!("Return matrix columns as a single array reference of [", stringify!($Vec), "].")]
                #[inline]
                pub fn as_array(&self) -> &[$Vec; $col_dim] {
                    let array: &[$Vec; $col_dim] = unsafe { mem::transmute(self) };
                    array
                }

                #[doc = concat!("Return matrix columns as a single mutable array reference of [", stringify!($Vec), "].")]
                #[inline]
                pub fn as_array_mut(&mut self) -> &mut [$Vec; $col_dim] {
                    let array: &mut [$Vec; $col_dim] = unsafe { mem::transmute(self) };
                    array
                }

                /// Converts the matrix into an array of column vectors.
                #[inline]
                pub fn to_array(self) -> [$Vec; $col_dim] {
                    let array: [$Vec; $col_dim] = unsafe { mem::transmute(self) };
                    array
                }

                /// Returns whether two matrices are equal given an epsilon.
                #[must_use]
                #[inline]
                pub fn compare(&self, rhs: &Self, epsilon: f32) -> bool {
                    self.iter()
                        .zip(rhs.iter())
                        .all(|(a, b)| a.compare(b, epsilon))
                }

                /// Create an iterator over the matrix columns.
                #[inline]
                pub fn iter(&self) -> std::slice::Iter<'_, $Vec> {
                    self.as_array().iter()
                }

                /// Create a mutable iterator over the matrix columns.
                #[inline]
                pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, $Vec> {
                    self.as_array_mut().iter_mut()
                }
            }

            impl IntoIterator for $Mat {
                type Item = $Vec;
                type IntoIter = std::array::IntoIter<Self::Item, $col_dim>;

                fn into_iter(self) -> Self::IntoIter {
                    self.to_array().into_iter()
                }
            }

            impl IntoIterator for &$Mat {
                type Item = $Vec;
                type IntoIter = std::array::IntoIter<Self::Item, $col_dim>;

                fn into_iter(self) -> Self::IntoIter {
                    self.to_array().into_iter()
                }
            }

            impl Index<usize> for $Mat {
                type Output = $Vec;

                fn index(&self, index: usize) -> &Self::Output {
                    self.as_array().index(index)
                }
            }

            impl IndexMut<usize> for $Mat {
                fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                    self.as_array_mut().index_mut(index)
                }
            }

            impl Index<(usize, usize)> for $Mat {
                type Output = f32;

                fn index(&self, (n, m): (usize, usize)) -> &Self::Output {
                    self.index(n).index(m)
                }
            }

            impl IndexMut<(usize, usize)> for $Mat {
                fn index_mut(&mut self, (n, m): (usize, usize)) -> &mut Self::Output {
                    self.index_mut(n).index_mut(m)
                }
            }

            impl From<[f32; $col_dim * $row_dim]> for $Mat {
                fn from(array: [f32; $col_dim * $row_dim]) -> Self {
                    let v: Self = unsafe { mem::transmute(array) };
                    v
                }
            }

            impl From<&[f32; $col_dim * $row_dim]> for &$Mat {
                fn from(array: &[f32; $col_dim * $row_dim]) -> Self {
                    let v: &Self = unsafe { mem::transmute(array) };
                    v
                }
            }

            impl From<&mut [f32; $col_dim * $row_dim]> for &mut $Mat {
                fn from(array: &mut [f32; $col_dim * $row_dim]) -> Self {
                    let v: &mut Self = unsafe { mem::transmute(array) };
                    v
                }
            }

            impl From<[[f32; $row_dim]; $col_dim]> for $Mat {
                fn from(array: [[f32; $row_dim]; $col_dim]) -> Self {
                    let v: Self = unsafe { mem::transmute(array) };
                    v
                }
            }

            impl From<&[[f32; $row_dim]; $col_dim]> for &$Mat {
                fn from(array: &[[f32; $row_dim]; $col_dim]) -> Self {
                    let v: &Self = unsafe { mem::transmute(array) };
                    v
                }
            }

            impl From<&mut [[f32; $row_dim]; $col_dim]> for &mut $Mat {
                fn from(array: &mut [[f32; $row_dim]; $col_dim]) -> Self {
                    let v: &mut Self = unsafe { mem::transmute(array) };
                    v
                }
            }

            impl Add for $Mat {
                type Output = $Mat;

                fn add(self, rhs: Self) -> Self::Output {
                    $Mat::new($(self.$field + rhs.$field),+)
                }
            }

            impl Add for &$Mat {
                type Output = $Mat;

                fn add(self, rhs: Self) -> Self::Output {
                    $Mat::new($(self.$field + rhs.$field),+)
                }
            }

            impl AddAssign for $Mat {
                fn add_assign(&mut self, rhs: Self) {
                    self.iter_mut()
                        .zip(rhs.iter())
                        .for_each(|(val, rhs)| *val += *rhs);
                }
            }

            impl Sub for $Mat {
                type Output = $Mat;

                fn sub(self, rhs: Self) -> Self::Output {
                    $Mat::new($(self.$field - rhs.$field),+)
                }
            }

            impl Sub for &$Mat {
                type Output = $Mat;

                fn sub(self, rhs: Self) -> Self::Output {
                    $Mat::new($(self.$field - rhs.$field),+)
                }
            }

            impl SubAssign for $Mat {
                fn sub_assign(&mut self, rhs: Self) {
                    self.iter_mut()
                        .zip(rhs.iter())
                        .for_each(|(val, rhs)| *val -= *rhs);
                }
            }

            impl Mul for $Mat {
                type Output = $Mat;

                fn mul(self, rhs: $Mat) -> Self::Output {
                    mul_matrix!(self, rhs, $Mat)
                }
            }

            impl Mul<&$Mat> for $Mat {
                type Output = $Mat;

                fn mul(self, rhs: &$Mat) -> Self::Output {
                    mul_matrix!(self, rhs, $Mat)
                }
            }

            impl Mul<$Mat> for &$Mat {
                type Output = $Mat;

                fn mul(self, rhs: $Mat) -> Self::Output {
                    mul_matrix!(self, rhs, $Mat)
                }
            }

            impl Mul for &$Mat {
                type Output = $Mat;

                fn mul(self, rhs: &$Mat) -> Self::Output {
                    mul_matrix!(self, rhs, $Mat)
                }
            }

            impl Mul<$Vec> for $Mat {
                type Output = $Vec;

                fn mul(self, rhs: $Vec) -> Self::Output {
                    mul_vector!(self, rhs, $Vec)
                }
            }

            impl Mul<$Vec> for &$Mat {
                type Output = $Vec;

                fn mul(self, rhs: $Vec) -> Self::Output {
                    mul_vector!(self, rhs, $Vec)
                }
            }

            impl Mul<&$Vec> for $Mat {
                type Output = $Vec;

                fn mul(self, rhs: &$Vec) -> Self::Output {
                    mul_vector!(self, rhs, $Vec)
                }
            }

            impl Mul<&$Vec> for &$Mat {
                type Output = $Vec;

                fn mul(self, rhs: &$Vec) -> Self::Output {
                    mul_vector!(self, rhs, $Vec)
                }
            }

            impl Mul<f32> for $Mat {
                type Output = $Mat;

                fn mul(self, rhs: f32) -> Self::Output {
                    $Mat::new($(self.$field * rhs),+)
                }
            }

            impl Mul<f32> for &$Mat {
                type Output = $Mat;

                fn mul(self, rhs: f32) -> Self::Output {
                    $Mat::new($(self.$field * rhs),+)
                }
            }

            impl Mul<$Mat> for f32 {
                type Output = $Mat;

                fn mul(self, rhs: $Mat) -> Self::Output {
                    $Mat::new($(rhs.$field * self),+)
                }
            }

            impl Mul<&$Mat> for f32 {
                type Output = $Mat;

                fn mul(self, rhs: &$Mat) -> Self::Output {
                    $Mat::new($(rhs.$field * self),+)
                }
            }

            impl Mul<$Mat> for &f32 {
                type Output = $Mat;

                fn mul(self, rhs: $Mat) -> Self::Output {
                    $Mat::new($(rhs.$field * self),+)
                }
            }

            impl Mul<&$Mat> for &f32 {
                type Output = $Mat;

                fn mul(self, rhs: &$Mat) -> Self::Output {
                    $Mat::new($(rhs.$field * self),+)
                }
            }

            impl MulAssign for $Mat {
                fn mul_assign(&mut self, rhs: Self) {
                    self.iter_mut()
                        .zip(rhs.iter())
                        .for_each(|(val, rhs)| *val *= *rhs);
                }
            }

            impl MulAssign<f32> for $Mat {
                fn mul_assign(&mut self, rhs: f32) {
                    self.iter_mut().for_each(|val| *val *= rhs);
                }
            }

            impl MulAssign<&f32> for $Mat {
                fn mul_assign(&mut self, rhs: &f32) {
                    self.iter_mut().for_each(|val| *val *= rhs);
                }
            }

            impl Div for $Mat {
                type Output = $Mat;

                fn div(self, rhs: $Mat) -> Self::Output {
                    $Mat::new($(self.$field / rhs.$field),+)
                }
            }

            impl Div<&$Mat> for $Mat {
                type Output = $Mat;

                fn div(self, rhs: &$Mat) -> Self::Output {
                    $Mat::new($(self.$field / rhs.$field),+)
                }
            }

            impl Div<$Mat> for &$Mat {
                type Output = $Mat;

                fn div(self, rhs: $Mat) -> Self::Output {
                    $Mat::new($(self.$field / rhs.$field),+)
                }
            }

            impl Div for &$Mat {
                type Output = $Mat;

                fn div(self, rhs: &$Mat) -> Self::Output {
                    $Mat::new($(self.$field / rhs.$field),+)
                }
            }

            impl Div<f32> for $Mat {
                type Output = $Mat;

                fn div(self, rhs: f32) -> Self::Output {
                    $Mat::new($(self.$field / rhs),+)
                }
            }

            impl Div<&f32> for $Mat {
                type Output = $Mat;

                fn div(self, rhs: &f32) -> Self::Output {
                    $Mat::new($(self.$field / rhs),+)
                }
            }

            impl Div<f32> for &$Mat {
                type Output = $Mat;

                fn div(self, rhs: f32) -> Self::Output {
                    $Mat::new($(self.$field / rhs),+)
                }
            }

            impl Div<&f32> for &$Mat {
                type Output = $Mat;

                fn div(self, rhs: &f32) -> Self::Output {
                    $Mat::new($(self.$field / rhs),+)
                }
            }

            impl Div<$Mat> for f32 {
                type Output = $Mat;

                fn div(self, rhs: $Mat) -> Self::Output {
                    $Mat::new($(rhs.$field / self),+)
                }
            }

            impl Div<&$Mat> for f32 {
                type Output = $Mat;

                fn div(self, rhs: &$Mat) -> Self::Output {
                    $Mat::new($(rhs.$field / self),+)
                }
            }

            impl Div<$Mat> for &f32 {
                type Output = $Mat;

                fn div(self, rhs: $Mat) -> Self::Output {
                    $Mat::new($(rhs.$field / self),+)
                }
            }

            impl Div<&$Mat> for &f32 {
                type Output = $Mat;

                fn div(self, rhs: &$Mat) -> Self::Output {
                    $Mat::new($(rhs.$field / self),+)
                }
            }

            impl DivAssign for $Mat {
                fn div_assign(&mut self, rhs: Self) {
                    self.iter_mut()
                        .zip(rhs.iter())
                        .for_each(|(val, rhs)| *val /= *rhs);
                }
            }

            impl DivAssign<f32> for $Mat {
                fn div_assign(&mut self, rhs: f32) {
                    self.iter_mut().for_each(|val| *val /= rhs);
                }
            }

            impl DivAssign<&f32> for $Mat {
                fn div_assign(&mut self, rhs: &f32) {
                    self.iter_mut().for_each(|val| *val /= rhs);
                }
            }

            impl Neg for $Mat {
                type Output = $Mat;

                fn neg(self) -> Self::Output {
                    $Mat::new($(self.$field.neg()),+)
                }
            }

            impl Neg for &$Mat {
                type Output = $Mat;

                fn neg(self) -> Self::Output {
                    $Mat::new($(self.$field.neg()),+)
                }
            }
        )+
    }
}
impl_matrix! {
    { Mat3, Vec3, 3, 3 => col0, col1, col2 },
    { Mat4, Vec4, 4, 4 => col0, col1, col2, col3 },
}

/// Constructs a new [Mat3] (3x3 matrix).
#[macro_export]
macro_rules! mat3 {
    () => {
        $crate::matrix::Mat3::identity()
    };
    ($val:expr) => {
        $crate::matrix::Mat3::from($val)
    };
    ($c0:expr, $c1:expr, $(,)?) => {
        $crate::matrix::Mat3::new($c0, $c1, vec3!(0.0, 0.0, 1.0, 0.0))
    };
    ($c0:expr, $c1:expr, $c2:expr $(,)?) => {
        $crate::matrix::Mat3::new($c0, $c1, $c2)
    };
}

/// Constructs a new [Mat4] (4x4 matrix).
#[macro_export]
macro_rules! mat4 {
    () => {
        $crate::matrix::Mat4::identity()
    };
    ($val:expr) => {
        $crate::matrix::Mat4::from($val)
    };
    ($c0:expr, $c1:expr, $(,)?) => {
        $crate::matrix::Mat4::new(
            $c0,
            $c1,
            vec4!(0.0, 0.0, 1.0, 0.0),
            vec4!(0.0, 0.0, 0.0, 1.0),
        )
    };
    ($c0:expr, $c1:expr, $c2:expr $(,)?) => {
        $crate::matrix::Mat4::new($c0, $c1, $c2, vec4!(0.0, 0.0, 0.0, 1.0))
    };
    ($c0:expr, $c1:expr, $c2:expr, $c3:expr $(,)?) => {
        $crate::matrix::Mat4::new($c0, $c1, $c2, $c3)
    };
}

impl Mat3 {
    /// Construct an identity matrix.
    #[inline]
    pub fn identity() -> Self {
        Self {
            col0: vec3!(1.0, 0.0, 0.0),
            col1: vec3!(0.0, 1.0, 0.0),
            col2: vec3!(0.0, 0.0, 1.0),
        }
    }

    /// Create a 3x3 translation matrix for the given position.
    #[inline]
    pub fn translation(position: impl Into<Vec2>) -> Self {
        let position = position.into();
        mat3!(
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [position.x, position.y, 1.0],
        )
    }

    /// Create a 3x3 translation matrix for the given position combining with an existing matrix.
    #[inline]
    pub fn translate(mat: &Mat3, position: impl Into<Vec2>) -> Self {
        mat * Self::translation(position)
    }

    /// Create a 3x3 scale matrix for the given scale.
    #[inline]
    pub fn scaling(scale: impl Into<Vec2>) -> Self {
        let scale = scale.into();
        mat3!([scale.x, 0.0, 0.0], [0.0, scale.y, 0.0], [0.0, 0.0, 1.0])
    }

    /// Create a 3x3 scale matrix for the given scale combining with an existing matrix.
    #[inline]
    pub fn scale(mat: &Mat3, scale: impl Into<Vec2>) -> Self {
        mat * Self::scaling(scale)
    }

    /// Create a 3x3 rotation matrix from a rotation vector in degrees.
    #[inline]
    pub fn rotation(rotation: impl Into<Vec2>) -> Self {
        let rotation = rotation.into();
        let (sin_x, cos_x) = rotation.x.to_radians().sin_cos();
        let (sin_y, cos_y) = rotation.y.to_radians().sin_cos();

        let rotate_x = mat3!([1.0, 0.0, 0.0], [0.0, cos_x, -sin_x], [0.0, sin_x, cos_x]);
        let rotate_y = mat3!([cos_y, 0.0, sin_y], [0.0, 1.0, 0.0], [-sin_y, 0.0, cos_y]);

        let rotate = rotate_x * rotate_y;
        mat3!(rotate[0], rotate[1], [0.0, 0.0, 1.0])
    }

    /// Create a 3x3 rotation matrix from a rotation vector in degrees combining with an existing
    /// matrix.
    #[inline]
    pub fn rotate(mat: &Mat3, rotation: impl Into<Vec2>) -> Self {
        mat * Self::rotation(rotation)
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

    /// Create a transposed copy of the matrix.
    #[inline]
    pub fn transposed(&self) -> Self {
        mat3!(
            [self[(0, 0)], self[(1, 0)], self[(2, 0)]],
            [self[(0, 1)], self[(1, 1)], self[(2, 1)]],
            [self[(0, 2)], self[(1, 2)], self[(2, 2)]],
        )
    }

    /// Calculate the matrix determinant.
    #[must_use]
    #[inline]
    pub fn determinant(&self) -> f32 {
        self[(0, 0)] * (self[(1, 1)] * self[(2, 2)] - self[(2, 1)] * self[(1, 2)])
            - self[(1, 0)] * (self[(0, 1)] * self[(2, 2)] - self[(2, 1)] * self[(0, 2)])
            + self[(2, 0)] * (self[(0, 1)] * self[(1, 2)] - self[(1, 1)] * self[(0, 2)])
    }

    /// Create an inversed copy of the matrix.
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.is_approx_eq(0.0, f32::EPSILON) {
            None
        } else {
            let inv_det = det.recip();
            let r11 = self[(1, 1)] * self[(2, 2)] - self[(2, 1)] * self[(1, 2)];
            let r12 = self[(2, 0)] * self[(1, 2)] - self[(1, 0)] * self[(2, 2)];
            let r13 = self[(1, 0)] * self[(2, 1)] - self[(2, 0)] * self[(1, 1)];
            let r21 = self[(2, 1)] * self[(0, 2)] - self[(0, 1)] * self[(2, 2)];
            let r22 = self[(0, 0)] * self[(2, 2)] - self[(2, 0)] * self[(0, 2)];
            let r23 = self[(2, 0)] * self[(0, 1)] - self[(0, 0)] * self[(2, 1)];
            let r31 = self[(0, 1)] * self[(1, 2)] - self[(1, 1)] * self[(0, 2)];
            let r32 = self[(1, 0)] * self[(0, 2)] - self[(0, 0)] * self[(1, 2)];
            let r33 = self[(0, 0)] * self[(1, 1)] - self[(1, 0)] * self[(0, 1)];
            Some(mat3!(
                [r11 * inv_det, r21 * inv_det, r31 * inv_det],
                [r12 * inv_det, r22 * inv_det, r32 * inv_det],
                [r13 * inv_det, r23 * inv_det, r33 * inv_det],
            ))
        }
    }
}

impl Mat4 {
    /// Construct an identity matrix.
    #[inline]
    pub fn identity() -> Self {
        Self {
            col0: vec4!(1.0, 0.0, 0.0, 0.0),
            col1: vec4!(0.0, 1.0, 0.0, 0.0),
            col2: vec4!(0.0, 0.0, 1.0, 0.0),
            col3: vec4!(0.0, 0.0, 0.0, 1.0),
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
        fov: impl Into<Angle>,
        aspect_ratio: f32,
        near_clip: f32,
        far_clip: f32,
    ) -> Self {
        let tan_frac_fov_two = (fov.into().to_radians() / 2.0).tan();
        let nf = near_clip - far_clip;

        let q = 1.0 / tan_frac_fov_two;
        let a = q / aspect_ratio;
        let b = (near_clip + far_clip) / nf;
        let c = (2.0 * near_clip * far_clip) / nf;

        mat4!(
            [a, 0.0, 0.0, 0.0],
            [0.0, q, 0.0, 0.0],
            [0.0, 0.0, b, -1.0],
            [0.0, 0.0, c, 0.0],
        )
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

    /// Create a 4x4 translation matrix for the given position.
    #[inline]
    pub fn translation(position: impl Into<Vec3>) -> Self {
        let position = position.into();
        mat4!(
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [position.x, position.y, position.z, 1.0],
        )
    }

    /// Create a 4x4 translation matrix for the given position combining with an existing matrix.
    #[inline]
    pub fn translate(mat: &Mat4, position: impl Into<Vec3>) -> Self {
        mat * Self::translation(position)
    }

    /// Create a 4x4 scale matrix for the given scale.
    #[inline]
    pub fn scaling(scale: impl Into<Vec3>) -> Self {
        let scale = scale.into();
        mat4!(
            [scale.x, 0.0, 0.0, 0.0],
            [0.0, scale.y, 0.0, 0.0],
            [0.0, 0.0, scale.z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        )
    }

    /// Create a 4x4 scale matrix for the given scale combining with an existing matrix.
    #[inline]
    pub fn scale(mat: &Mat4, scale: impl Into<Vec3>) -> Self {
        mat * Self::scaling(scale)
    }

    /// Create a 4x4 rotation matrix from a rotation vector in degrees.
    #[inline]
    pub fn rotation(rotation: impl Into<Vec3>) -> Self {
        let rotation = rotation.into();
        let (sin_x, cos_x) = rotation.x.to_radians().sin_cos();
        let (sin_y, cos_y) = rotation.y.to_radians().sin_cos();
        let (sin_z, cos_z) = rotation.z.to_radians().sin_cos();

        let rotate_x = mat3!([1.0, 0.0, 0.0], [0.0, cos_x, -sin_x], [0.0, sin_x, cos_x]);
        let rotate_y = mat3!([cos_y, 0.0, sin_y], [0.0, 1.0, 0.0], [-sin_y, 0.0, cos_y]);
        let rotate_z = mat3!([cos_z, -sin_z, 0.0], [sin_z, cos_z, 0.0], [0.0, 0.0, 1.0]);

        let rotate = rotate_x * rotate_y * rotate_z;
        mat4!(
            rotate[0].to_vec4(0.0),
            rotate[1].to_vec4(0.0),
            rotate[2].to_vec4(0.0),
            [0.0, 0.0, 0.0, 1.0]
        )
    }

    /// Create a 4x4 rotation matrix from a rotation vector in degrees combining with an existing
    /// matrix.
    #[inline]
    pub fn rotate(mat: &Mat4, rotation: impl Into<Vec3>) -> Self {
        mat * Self::rotation(rotation)
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
        mat4!(
            [self[(0, 0)], self[(1, 0)], self[(2, 0)], self[(3, 0)]],
            [self[(0, 1)], self[(1, 1)], self[(2, 1)], self[(3, 1)]],
            [self[(0, 2)], self[(1, 2)], self[(2, 2)], self[(3, 2)]],
            [self[(0, 3)], self[(1, 3)], self[(2, 3)], self[(3, 3)]],
        )
    }

    /// Calculate the matrix determinant.
    #[must_use]
    #[inline]
    pub fn determinant(&self) -> f32 {
        self[(0, 0)]
            * (self[(1, 1)] * self[(2, 2)] * self[(3, 3)]
                + self[(2, 1)] * self[(3, 2)] * self[(1, 3)]
                + self[(3, 1)] * self[(1, 2)] * self[(2, 3)]
                - self[(3, 1)] * self[(2, 2)] * self[(1, 3)]
                - self[(1, 1)] * self[(3, 2)] * self[(2, 3)]
                - self[(2, 1)] * self[(1, 2)] * self[(3, 3)])
            - self[(1, 0)]
                * (self[(0, 1)] * self[(2, 2)] * self[(3, 3)]
                    + self[(2, 1)] * self[(3, 2)] * self[(0, 3)]
                    + self[(3, 1)] * self[(0, 2)] * self[(2, 3)]
                    - self[(3, 1)] * self[(2, 2)] * self[(0, 3)]
                    - self[(0, 1)] * self[(3, 2)] * self[(2, 3)]
                    - self[(2, 1)] * self[(0, 2)] * self[(3, 3)])
            + self[(2, 0)]
                * (self[(0, 1)] * self[(1, 2)] * self[(3, 3)]
                    + self[(1, 1)] * self[(3, 2)] * self[(0, 3)]
                    + self[(3, 1)] * self[(0, 2)] * self[(1, 3)]
                    - self[(3, 1)] * self[(1, 2)] * self[(0, 3)]
                    - self[(0, 1)] * self[(3, 2)] * self[(1, 3)]
                    - self[(1, 1)] * self[(0, 2)] * self[(3, 3)])
            - self[(3, 0)]
                * (self[(0, 1)] * self[(1, 2)] * self[(2, 3)]
                    + self[(1, 1)] * self[(2, 2)] * self[(0, 3)]
                    + self[(2, 1)] * self[(0, 2)] * self[(1, 3)]
                    - self[(2, 1)] * self[(1, 2)] * self[(0, 3)]
                    - self[(0, 1)] * self[(2, 2)] * self[(1, 3)]
                    - self[(1, 1)] * self[(0, 2)] * self[(2, 3)])
    }

    /// Create an inversed copy of the matrix.
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.is_approx_eq(0.0, f32::EPSILON) {
            None
        } else {
            let inv_det = det.recip();
            let tr = self.transposed();
            let cf = |i, j| -> f32 {
                let mat = match i {
                    0 => mat3!(
                        tr.col1.truncate(j),
                        tr.col2.truncate(j),
                        tr.col3.truncate(j)
                    ),
                    1 => mat3!(
                        tr.col0.truncate(j),
                        tr.col2.truncate(j),
                        tr.col3.truncate(j)
                    ),
                    2 => mat3!(
                        tr.col0.truncate(j),
                        tr.col1.truncate(j),
                        tr.col3.truncate(j)
                    ),
                    3 => mat3!(
                        tr.col0.truncate(j),
                        tr.col1.truncate(j),
                        tr.col2.truncate(j)
                    ),
                    _ => unreachable!("invalid matrix index"),
                };
                let d = mat.determinant() * inv_det;
                if (i + j) & 1 == 1 {
                    -d
                } else {
                    d
                }
            };
            Some(mat4!(
                [cf(0, 0), cf(0, 1), cf(0, 2), cf(0, 3)],
                [cf(1, 0), cf(1, 1), cf(1, 2), cf(1, 3)],
                [cf(2, 0), cf(2, 1), cf(2, 2), cf(2, 3)],
                [cf(3, 0), cf(3, 1), cf(3, 2), cf(3, 3)],
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{matrix::Mat4, prelude::Radians, vec3, vec4};

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
        assert_eq!(m * v, vec4!(34.0, 44.0, 54.0, 64.0));
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
            .compare(&vec3!(-0.366_508, -0.855_186, -0.366_508), 0.0002));
    }

    #[test]
    fn translate() {
        let v = vec3!(1.0, 3.0, 2.0);
        let t = Mat4::translation(v);
        assert_eq!(t[0], vec4!(1.0, 0.0, 0.0, 0.0));
        assert_eq!(t[1], vec4!(0.0, 1.0, 0.0, 0.0));
        assert_eq!(t[2], vec4!(0.0, 0.0, 1.0, 0.0));
        assert_eq!(t[3], vec4!(1.0, 3.0, 2.0, 1.0));
    }

    #[test]
    fn perspective() {
        let p = Mat4::perspective(
            Radians::new(std::f32::consts::PI * 2.0 * 45.0 / 360.0),
            1920.0 / 1080.0,
            0.1,
            100.0,
        );
        assert!((p[0].x - 1.357_995_2).abs() < f32::EPSILON);
        assert_eq!(p[0].y, 0.0);
        assert_eq!(p[0].z, 0.0);
        assert_eq!(p[0].w, 0.0);
        assert_eq!(p[1].x, 0.0);
        assert!((p[1].y - 2.414_213_7).abs() < f32::EPSILON);
        assert_eq!(p[1].z, 0.0);
        assert_eq!(p[1].w, 0.0);
        assert_eq!(p[2].x, 0.0);
        assert_eq!(p[2].y, 0.0);
        assert!((p[2].z + 1.002_002).abs() < f32::EPSILON);
        assert_eq!(p[2].w, -1.0);
        assert_eq!(p[3].x, 0.0);
        assert_eq!(p[3].y, 0.0);
        assert!((p[3].z + 0.2002_002).abs() < f32::EPSILON);
        assert_eq!(p[3].w, 0.0);
    }
}
