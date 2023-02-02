use crate::{
    num::{Angle, Radians},
    vec3, vec4,
    vector::{Quaternion, Vec3, Vec4},
};
use std::{
    mem,
    ops::{Index, IndexMut, Mul},
};

/// A 3x3 matrix.
#[derive(Debug, Copy, Clone, PartialEq)]
#[must_use]
pub struct Mat3 {
    col0: Vec3,
    col1: Vec3,
    col2: Vec3,
}

impl Default for Mat3 {
    fn default() -> Self {
        Self {
            col0: vec3!(1.0, 0.0, 0.0),
            col1: vec3!(0.0, 1.0, 0.0),
            col2: vec3!(0.0, 0.0, 1.0),
        }
    }
}

/// Constructs a new [Mat3].
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
        $crate::matrix::Mat4::new($c0, $c1, $c2)
    };
}

impl Mat3 {
    /// Create a 4x4 matrix from given columns.
    #[inline]
    pub fn new(col0: impl Into<Vec3>, col1: impl Into<Vec3>, col2: impl Into<Vec3>) -> Self {
        Self {
            col0: col0.into(),
            col1: col1.into(),
            col2: col2.into(),
        }
    }
}

impl Mat3 {
    /// Construct an identity matrix.
    #[inline]
    pub fn identity() -> Self {
        Self::default()
    }

    /// Return matrix columns as a single array reference of `Vec3`.
    #[inline]
    pub fn as_array(&self) -> &[Vec3; 3] {
        let array: &[Vec3; 3] = unsafe { mem::transmute(self) };
        array
    }

    /// Return matrix columns as a single mutable array reference of `Vec3`.
    #[inline]
    pub fn as_array_mut(&mut self) -> &mut [Vec3; 3] {
        let array: &mut [Vec3; 3] = unsafe { mem::transmute(self) };
        array
    }

    /// Create an iterator over the matrix columns.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, Vec3> {
        self.as_array().iter()
    }

    /// Create a mutable iterator over the matrix columns.
    #[inline]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Vec3> {
        self.as_array_mut().iter_mut()
    }
}

impl Index<usize> for Mat3 {
    type Output = Vec3;

    fn index(&self, index: usize) -> &Self::Output {
        self.as_array().index(index)
    }
}

impl IndexMut<usize> for Mat3 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.as_array_mut().index_mut(index)
    }
}

impl Index<(usize, usize)> for Mat3 {
    type Output = f32;

    fn index(&self, (n, m): (usize, usize)) -> &Self::Output {
        self.index(n).index(m)
    }
}

impl IndexMut<(usize, usize)> for Mat3 {
    fn index_mut(&mut self, (n, m): (usize, usize)) -> &mut Self::Output {
        self.index_mut(n).index_mut(m)
    }
}

impl From<[f32; 9]> for Mat3 {
    fn from(array: [f32; 9]) -> Self {
        let v: Self = unsafe { mem::transmute(array) };
        v
    }
}

impl From<&[f32; 9]> for &Mat3 {
    fn from(array: &[f32; 9]) -> Self {
        let v: &Self = unsafe { mem::transmute(array) };
        v
    }
}

impl From<&mut [f32; 9]> for &mut Mat3 {
    fn from(array: &mut [f32; 9]) -> Self {
        let v: &mut Self = unsafe { mem::transmute(array) };
        v
    }
}

impl From<[[f32; 3]; 3]> for Mat3 {
    fn from(array: [[f32; 3]; 3]) -> Self {
        let v: Self = unsafe { mem::transmute(array) };
        v
    }
}

impl From<&[[f32; 3]; 3]> for &Mat3 {
    fn from(array: &[[f32; 3]; 3]) -> Self {
        let v: &Self = unsafe { mem::transmute(array) };
        v
    }
}

impl From<&mut [[f32; 3]; 3]> for &mut Mat3 {
    fn from(array: &mut [[f32; 3]; 3]) -> Self {
        let v: &mut Self = unsafe { mem::transmute(array) };
        v
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

impl Mat4 {
    /// Construct an identity matrix.
    #[inline]
    pub fn identity() -> Self {
        Self::default()
    }

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

#[cfg(test)]
mod tests {
    use crate::{vec3, vec4};

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
