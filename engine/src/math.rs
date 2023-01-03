use std::ops::{
    Add, AddAssign, Deref, DerefMut, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub,
    SubAssign,
};

#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(C)]
#[must_use]
pub struct Vertex {
    pub position: Vec3,
    pub color: Vec3,
    pub texcoord: Vec2,
}

impl Vertex {
    /// Create a new `Vertex` instance.
    pub fn new(position: Vec3, color: Vec3, texcoord: Vec2) -> Self {
        Self {
            position,
            color,
            texcoord,
        }
    }
}

#[derive(Default, Debug, Copy, Clone)]
#[repr(C)]
#[must_use]
pub struct UniformBufferObject {
    // pub model: Mat4,
    pub view: Mat4,
    pub projection: Mat4,
}

/// Constructs a new [Vector].
#[macro_export]
macro_rules! vector {
    () => {
        $crate::math::Vector::origin()
    };
    ($($v:expr),* $(,)?) => {
        $crate::math::Vector::new([$($v,)*])
    };
}

/// A N-dimensional `Vector`.
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
#[must_use]
#[repr(transparent)]
pub struct Vector<const N: usize = 2>([f32; N]);

pub type Vec1 = Vector<1>;
pub type Vec2 = Vector<2>;
pub type Vec3 = Vector<3>;
pub type Vec4 = Vector<4>;

impl<const N: usize> Default for Vector<N> {
    fn default() -> Self {
        Self::origin()
    }
}

impl Vector<2> {
    /// Create a 2D `Vector` from a 3D `Vector`.
    #[inline]
    pub fn from_3d(vector: Vector<3>) -> Self {
        Self([vector.x(), vector.y()])
    }

    /// Create a 2D `Vector` from a 4D `Vector`.
    #[inline]
    pub fn from_4d(vector: Vector<4>) -> Self {
        Self([vector.x(), vector.y()])
    }

    /// Create a 2D unit `Vector` pointing up.
    #[inline]
    pub fn up() -> Self {
        Self([0.0, 1.0])
    }

    /// Create a 2D unit `Vector` pointing down.
    #[inline]
    pub fn down() -> Self {
        Self([0.0, -1.0])
    }

    /// Create a 2D unit `Vector` pointing left.
    #[inline]
    pub fn left() -> Self {
        Self([-1.0, 0.0])
    }

    /// Create a 2D unit `Vector` pointing right.
    #[inline]
    pub fn right() -> Self {
        Self([1.0, 0.0])
    }

    /// Return the `x` coordinate.
    #[must_use]
    #[inline]
    pub fn x(&self) -> f32 {
        self[0]
    }

    /// Return the `y` coordinate.
    #[must_use]
    #[inline]
    pub fn y(&self) -> f32 {
        self[1]
    }
}

impl Vector<3> {
    /// Create a 3D `Vector` from a 2D `Vector`.
    #[inline]
    pub fn from_2d(vector: Vector<2>, z: f32) -> Self {
        Self([vector.x(), vector.y(), z])
    }

    /// Create a 3D `Vector` from a 4D `Vector`.
    #[inline]
    pub fn from_4d(vector: Vector<4>) -> Self {
        Self([vector.x(), vector.y(), vector.z()])
    }

    /// Create a 3D `Vector` from separate RGB values.
    #[inline]
    pub fn from_rgb(r: u32, g: u32, b: u32) -> Self {
        Self([r as f32, g as f32, b as f32]) / 255.0
    }

    /// Create separate RGB values from the `Vector`.
    #[inline]
    #[must_use]
    pub fn to_rgb(&self) -> [u32; 3] {
        let rgb = *self * 255.0;
        [rgb[0] as u32, rgb[1] as u32, rgb[2] as u32]
    }

    /// Create a 3D unit `Vector` pointing up.
    #[inline]
    pub fn up() -> Self {
        Self([0.0, 1.0, 0.0])
    }

    /// Create a 3D unit `Vector` pointing down.
    #[inline]
    pub fn down() -> Self {
        Self([0.0, -1.0, 0.0])
    }

    /// Create a 3D unit `Vector` pointing left.
    #[inline]
    pub fn left() -> Self {
        Self([-1.0, 0.0, 0.0])
    }

    /// Create a 3D unit `Vector` pointing right.
    #[inline]
    pub fn right() -> Self {
        Self([1.0, 0.0, 0.0])
    }

    /// Create a 3D unit `Vector` pointing forward.
    #[inline]
    pub fn forward() -> Self {
        Self([0.0, 0.0, -1.0])
    }

    /// Create a 3D unit `Vector` pointing backward.
    #[inline]
    pub fn backward() -> Self {
        Self([0.0, 0.0, 1.0])
    }

    /// Return the `x` coordinate.
    #[must_use]
    #[inline]
    pub fn x(&self) -> f32 {
        self[0]
    }

    /// Return the `y` coordinate.
    #[must_use]
    #[inline]
    pub fn y(&self) -> f32 {
        self[1]
    }

    /// Return the `z` coordinate.
    #[must_use]
    #[inline]
    pub fn z(&self) -> f32 {
        self[2]
    }

    /// Calculate the dot-product between two `Vector`s.
    #[must_use]
    #[inline]
    pub fn dot(&self, rhs: Self) -> f32 {
        self.iter()
            .zip(rhs.iter())
            .map(|(val, rhs)| val * rhs)
            .sum()
    }

    /// Calculate the cross-product between two `Vector`s.
    #[inline]
    pub fn cross(&self, rhs: Self) -> Self {
        let [x, y, z] = self.0;
        let [ox, oy, oz] = rhs.0;
        Self([y * oz - z * oy, z * ox - x * oz, x * oy - y * ox])
    }

    /// Create a transformed `Vector` by applying a `Matrix`.
    pub fn transformed(&self, matrix: Matrix<4, 4>) -> Self {
        let [x, y, z] = self.0;
        Self([
            x * matrix[0] + y * matrix[4] + z * matrix[8] + 1.0 * matrix[12],
            x * matrix[1] + y * matrix[5] + z * matrix[9] + 1.0 * matrix[13],
            x * matrix[2] + y * matrix[6] + z * matrix[10] + 1.0 * matrix[14],
        ])
    }
}

impl Vector<4> {
    /// Create a 4D `Vector` from a 2D `Vector`.
    #[inline]
    pub fn from_2d(vector: Vector<2>, z: f32, w: f32) -> Self {
        Self([vector.x(), vector.y(), z, w])
    }

    /// Create a 4D `Vector` from a 3D `Vector`.
    #[inline]
    pub fn from_3d(vector: Vector<3>, w: f32) -> Self {
        Self([vector.x(), vector.y(), vector.z(), w])
    }

    /// Return the `x` coordinate.
    #[must_use]
    #[inline]
    pub fn x(&self) -> f32 {
        self[0]
    }

    /// Return the `y` coordinate.
    #[must_use]
    #[inline]
    pub fn y(&self) -> f32 {
        self[1]
    }

    /// Return the `z` coordinate.
    #[must_use]
    #[inline]
    pub fn z(&self) -> f32 {
        self[2]
    }

    /// Return the `w` coordinate.
    #[must_use]
    #[inline]
    pub fn w(&self) -> f32 {
        self[3]
    }

    /// Calculate the dot-product between two `Vector`s, pairwise.
    #[must_use]
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn dot_pairwise(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
        a.0 * b.0 + a.1 * b.1 + a.2 * b.2 + a.3 * b.3
    }
}

impl<const N: usize> Vector<N> {
    /// Create a N-dimentional `Vector` from given coordinates.
    #[inline]
    pub fn new(coordinates: [f32; N]) -> Self {
        Self(coordinates)
    }

    /// Create a N-dimensional `Vector` at the origin.
    pub fn origin() -> Self {
        Self([0.0; N])
    }

    /// Create a N-dimensional unit `Vector`.
    pub fn unit() -> Self {
        Self([1.0; N])
    }

    /// Calculate the squared magnitude of the `Vector`.
    #[must_use]
    pub fn magnitude_squared(&self) -> f32 {
        self.iter().map(|val| val * val).sum()
    }

    /// Calculate the magnitude of the `Vector`.
    #[must_use]
    pub fn magnitude(&self) -> f32 {
        self.magnitude_squared().sqrt()
    }

    /// Normalize the `Vector` into a unit `Vector`.
    pub fn normalize(&mut self) {
        let magnitude = self.magnitude();
        if magnitude != 0.0 {
            self.iter_mut().for_each(|val| *val *= magnitude.recip());
        }
    }

    /// Create a normalized copy of the `Vector`.
    pub fn normalized(&self) -> Self {
        let mut vector = *self;
        vector.normalize();
        vector
    }

    /// Create the Euclidean distance between two `Vector`s.
    #[must_use]
    pub fn distance(&self, vector: Self) -> f32 {
        (*self - vector).magnitude()
    }
}

impl<const N: usize> Deref for Vector<N> {
    type Target = [f32; N];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const N: usize> DerefMut for Vector<N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<const N: usize> IntoIterator for &Vector<N> {
    type Item = f32;
    type IntoIter = std::array::IntoIter<Self::Item, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const N: usize> Add for Vector<N> {
    type Output = Vector<N>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut vector = Self::Output::default();
        vector
            .iter_mut()
            .zip(rhs.iter())
            .for_each(|(val, rhs)| *val += rhs);
        vector
    }
}

impl<const N: usize> Add for &Vector<N> {
    type Output = Vector<N>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut vector = Self::Output::default();
        vector
            .iter_mut()
            .zip(rhs.iter())
            .for_each(|(val, rhs)| *val += rhs);
        vector
    }
}

impl<const N: usize> AddAssign for Vector<N> {
    fn add_assign(&mut self, rhs: Self) {
        self.iter_mut()
            .zip(rhs.iter())
            .for_each(|(val, rhs)| *val += rhs);
    }
}

impl<const N: usize> Sub for Vector<N> {
    type Output = Vector<N>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut vector = Self::Output::default();
        vector
            .iter_mut()
            .zip(rhs.iter())
            .for_each(|(val, rhs)| *val -= rhs);
        vector
    }
}

impl<const N: usize> Sub for &Vector<N> {
    type Output = Vector<N>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut vector = Self::Output::default();
        vector
            .iter_mut()
            .zip(rhs.iter())
            .for_each(|(val, rhs)| *val -= rhs);
        vector
    }
}

impl<const N: usize> SubAssign for Vector<N> {
    fn sub_assign(&mut self, rhs: Self) {
        self.iter_mut()
            .zip(rhs.iter())
            .for_each(|(val, rhs)| *val -= rhs);
    }
}

impl<const N: usize> Mul for Vector<N> {
    type Output = Vector<N>;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut vector = Self::Output::default();
        vector
            .iter_mut()
            .zip(rhs.iter())
            .for_each(|(val, rhs)| *val *= rhs);
        vector
    }
}

impl<const N: usize> Mul for &Vector<N> {
    type Output = Vector<N>;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut vector = Self::Output::default();
        vector
            .iter_mut()
            .zip(rhs.iter())
            .for_each(|(val, rhs)| *val *= rhs);
        vector
    }
}

impl<const N: usize> Mul<f32> for Vector<N> {
    type Output = Vector<N>;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut vector = Self::Output::default();
        vector.iter_mut().for_each(|val| *val *= rhs);
        vector
    }
}

impl<const N: usize> MulAssign for Vector<N> {
    fn mul_assign(&mut self, rhs: Self) {
        self.iter_mut()
            .zip(rhs.iter())
            .for_each(|(val, rhs)| *val *= rhs);
    }
}

impl<const N: usize> Mul<f32> for &Vector<N> {
    type Output = Vector<N>;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut vector = Self::Output::default();
        vector.iter_mut().for_each(|val| *val *= rhs);
        vector
    }
}

impl<const N: usize> MulAssign<f32> for Vector<N> {
    fn mul_assign(&mut self, rhs: f32) {
        self.iter_mut().for_each(|val| *val *= rhs);
    }
}

impl<const N: usize> Div for Vector<N> {
    type Output = Vector<N>;

    fn div(self, rhs: Self) -> Self::Output {
        let mut vector = Self::Output::default();
        vector
            .iter_mut()
            .zip(rhs.iter())
            .for_each(|(val, rhs)| *val /= rhs);
        vector
    }
}

impl<const N: usize> Div for &Vector<N> {
    type Output = Vector<N>;

    fn div(self, rhs: Self) -> Self::Output {
        let mut vector = Self::Output::default();
        vector
            .iter_mut()
            .zip(rhs.iter())
            .for_each(|(val, rhs)| *val /= rhs);
        vector
    }
}

impl<const N: usize> Div<f32> for Vector<N> {
    type Output = Vector<N>;

    fn div(self, rhs: f32) -> Self::Output {
        let mut vector = Self::Output::default();
        vector.iter_mut().for_each(|val| *val /= rhs);
        vector
    }
}

impl<const N: usize> Div<f32> for &Vector<N> {
    type Output = Vector<N>;

    fn div(self, rhs: f32) -> Self::Output {
        let mut vector = Self::Output::default();
        vector.iter_mut().for_each(|val| *val /= rhs);
        vector
    }
}

impl<const N: usize> DivAssign for Vector<N> {
    fn div_assign(&mut self, rhs: Self) {
        self.iter_mut()
            .zip(rhs.iter())
            .for_each(|(val, rhs)| *val /= rhs);
    }
}

impl<const N: usize> Neg for Vector<N> {
    type Output = Vector<N>;

    fn neg(self) -> Self::Output {
        let mut vector = Self::Output::default();
        vector.iter_mut().for_each(|val| *val = val.neg());
        vector
    }
}

impl<const N: usize> Neg for &Vector<N> {
    type Output = Vector<N>;

    fn neg(self) -> Self::Output {
        let mut vector = Self::Output::default();
        vector.iter_mut().for_each(|val| *val = val.neg());
        vector
    }
}

/// A NxM `Matrix`.
#[derive(Debug, Copy, Clone)]
#[must_use]
#[repr(transparent)]
pub struct Matrix<const N: usize = 4, const M: usize = 4>([[f32; M]; N]);

pub type Mat4 = Matrix<4, 4>;

impl<const N: usize, const M: usize> Default for Matrix<N, M> {
    fn default() -> Self {
        Self([[0.0; M]; N])
    }
}

impl Matrix<4, 4> {
    /// Create an orthographic projection `Matrix`.
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

        matrix[0] = -2.0 * lr;
        matrix[5] = -2.0 * bt;
        matrix[10] = -2.0 * nf;

        matrix[12] = (left + right) * lr;
        matrix[13] = (bottom + top) * bt;
        matrix[14] = (near_clip + far_clip) * nf;

        matrix
    }

    /// Create a perspective projection `Matrix`.
    #[inline]
    pub fn perspective(fov: Radians, aspect_ratio: f32, near_clip: f32, far_clip: f32) -> Self {
        let tan_frac_fov_two = (fov * 0.5).tan();
        let nf = far_clip - near_clip;
        let mut matrix = Self::identity();

        matrix[0] = 1.0 / (aspect_ratio * tan_frac_fov_two);
        matrix[5] = 1.0 / tan_frac_fov_two;
        matrix[10] = -((near_clip + far_clip) / nf);

        matrix[11] = -1.0;
        matrix[14] = -((2.0 * near_clip * far_clip) / nf);

        matrix
    }

    /// Create a look-at `Matrix`.
    #[inline]
    pub fn look_at(position: Vector<3>, target: Vector<3>, up: Vector<3>) -> Self {
        let z_axis = (target - position).normalized();
        let x_axis = z_axis.cross(up).normalized();
        let y_axis = x_axis.cross(z_axis);
        Self([
            [x_axis.x(), y_axis.x(), -z_axis.x(), 0.0],
            [x_axis.y(), y_axis.y(), -z_axis.y(), 0.0],
            [x_axis.z(), y_axis.z(), -z_axis.z(), 0.0],
            [
                -x_axis.dot(position),
                -y_axis.dot(position),
                z_axis.dot(position),
                1.0,
            ],
        ])
    }

    /// Create a translation `Matrix` for the given position.
    #[inline]
    pub fn translation(position: Vector<3>) -> Self {
        let mut matrix = Self::identity();
        matrix[12] = position.x();
        matrix[13] = position.y();
        matrix[14] = position.z();
        matrix
    }

    /// Create a scale `Matrix` for the given scale.
    #[inline]
    pub fn scale(scale: Vector<3>) -> Self {
        let mut matrix = Self::identity();
        matrix[0] = scale.x();
        matrix[5] = scale.y();
        matrix[10] = scale.z();
        matrix
    }

    /// Create a rotation `Matrix` from a `Quaternion`.
    #[inline]
    pub fn from_quaternion(&self, quaternion: Quaternion) -> Self {
        let q = quaternion.normalized();
        let mut matrix = Self::identity();

        matrix[0] = 1.0 - 2.0 * q.y() * q.y() - 2.0 * q.z() * q.z();
        matrix[1] = 2.0 * q.x() * q.y() - 2.0 * q.z() * q.w();
        matrix[2] = 2.0 * q.x() * q.z() - 2.0 * q.y() * q.w();

        matrix[4] = 2.0 * q.x() * q.y() + 2.0 * q.z() * q.w();
        matrix[5] = 1.0 - 2.0 * q.x() * q.x() - 2.0 * q.z() * q.z();
        matrix[6] = 2.0 * q.y() * q.z() - 2.0 * q.x() * q.w();

        matrix[8] = 2.0 * q.x() * q.z() - 2.0 * q.y() * q.w();
        matrix[0] = 2.0 * q.y() * q.z() + 2.0 * q.x() * q.w();
        matrix[10] = 1.0 - 2.0 * q.x() * q.x() - 2.0 * q.y() * q.y();

        matrix
    }

    /// Create a rotation `Matrix` from a `Quaternion` around a center point.
    #[inline]
    pub fn from_quaternion_center(&self, quaternion: Quaternion, center: Vector<3>) -> Self {
        let q = quaternion;
        let c = center;
        let mut matrix = Self::default();

        matrix[0] = (q.x() * q.x()) - (q.y() * q.y()) - (q.z() * q.z()) + (q.w() * q.w());
        matrix[1] = 2.0 * ((q.x() * q.y()) + (q.z() * q.w()));
        matrix[2] = 2.0 * ((q.x() * q.z()) - (q.y() * q.w()));
        matrix[3] = c.x() - c.x() * matrix[0] - c.y() * matrix[1] - c.z() * matrix[2];

        matrix[4] = 2.0 * ((q.x() * q.y()) - (q.z() * q.w()));
        matrix[5] = -(q.x() * q.x()) + (q.y() * q.y()) - (q.z() * q.z()) + (q.w() * q.w());
        matrix[6] = 2.0 * ((q.y() * q.z()) + (q.x() * q.w()));
        matrix[7] = c.y() - c.x() * matrix[4] - c.y() * matrix[5] - c.z() * matrix[6];

        matrix[8] = 2.0 * ((q.x() * q.z()) + (q.y() * q.w()));
        matrix[9] = 2.0 * ((q.y() * q.z()) - (q.x() * q.w()));
        matrix[10] = -(q.x() * q.x()) - (q.y() * q.y()) + (q.z() * q.z()) + (q.w() * q.w());
        matrix[11] = c.z() - c.x() * matrix[8] - c.y() * matrix[9] - c.z() * matrix[10];

        matrix[12] = 0.0;
        matrix[13] = 0.0;
        matrix[14] = 0.0;
        matrix[15] = 1.0;

        matrix
    }

    /// Create a rotation `Matrix` for all axes for the given angles.
    #[inline]
    pub fn rotation(x_angle: Radians, y_angle: Radians, z_angle: Radians) -> Self {
        let rotate_x = Self::rotation_x(x_angle);
        let rotate_y = Self::rotation_y(y_angle);
        let rotate_z = Self::rotation_z(z_angle);
        rotate_x * rotate_y * rotate_z
    }

    /// Create a rotation `Matrix` for about the x-axis for the given angle.
    #[inline]
    pub fn rotation_x(angle: Radians) -> Self {
        let mut matrix = Self::identity();
        let (sin, cos) = angle.sin_cos();
        matrix[5] = cos;
        matrix[6] = sin;
        matrix[9] = -sin;
        matrix[10] = cos;
        matrix
    }

    /// Create a rotation `Matrix` for about the y-axis for the given angle.
    #[inline]
    pub fn rotation_y(angle: Radians) -> Self {
        let mut matrix = Self::identity();
        let (sin, cos) = angle.sin_cos();
        matrix[0] = cos;
        matrix[2] = -sin;
        matrix[8] = sin;
        matrix[10] = cos;
        matrix
    }

    /// Create a rotation `Matrix` for about the z-axis for the given angle.
    #[inline]
    pub fn rotation_z(angle: Radians) -> Self {
        let mut matrix = Self::identity();
        let (sin, cos) = angle.sin_cos();
        matrix[0] = cos;
        matrix[1] = sin;
        matrix[4] = -sin;
        matrix[5] = cos;
        matrix
    }

    /// Create an up unit `Vector` relative to the `Matrix`.
    #[inline]
    pub fn up(&self) -> Vector<3> {
        Vector::new([self[1], self[5], self[9]]).normalized()
    }

    /// Create a down unit `Vector` relative to the `Matrix`.
    #[inline]
    pub fn down(&self) -> Vector<3> {
        Vector::new([-self[1], -self[5], -self[9]]).normalized()
    }

    /// Create a left unit `Vector` relative to the `Matrix`.
    #[inline]
    pub fn left(&self) -> Vector<3> {
        Vector::new([-self[0], -self[4], -self[8]]).normalized()
    }

    /// Create a right unit `Vector` relative to the `Matrix`.
    #[inline]
    pub fn right(&self) -> Vector<3> {
        Vector::new([self[0], self[4], self[8]]).normalized()
    }

    /// Create a forward unit `Vector` relative to the `Matrix`.
    #[inline]
    pub fn forward(&self) -> Vector<3> {
        Vector::new([-self[2], -self[6], -self[10]]).normalized()
    }

    /// Create a backward unit `Vector` relative to the `Matrix`.
    #[inline]
    pub fn backward(&self) -> Vector<3> {
        Vector::new([self[2], self[6], self[10]]).normalized()
    }

    /// Create a copy of the `Matrix` inverted about the x-axis.
    #[inline]
    pub fn inverted_x(&self) -> Self {
        let mut matrix = *self;
        matrix[0] *= -1.0;
        matrix
    }

    /// Create a copy of the `Matrix` inverted about the y-axis.
    #[inline]
    pub fn inverted_y(&self) -> Self {
        let mut matrix = *self;
        matrix[5] *= -1.0;
        matrix
    }

    /// Create a copy of the `Matrix` inverted about the z-axis.
    #[inline]
    pub fn inverted_z(&self) -> Self {
        let mut matrix = *self;
        matrix[10] *= -1.0;
        matrix
    }

    /// Create a transposed copy of the `Matrix`.
    #[inline]
    pub fn transposed(&self) -> Self {
        let mut matrix = Self::identity();
        matrix[0] = self[0];
        matrix[1] = self[4];
        matrix[2] = self[8];
        matrix[3] = self[12];
        matrix[4] = self[1];
        matrix[5] = self[5];
        matrix[6] = self[9];
        matrix[7] = self[13];
        matrix[8] = self[2];
        matrix[9] = self[6];
        matrix[10] = self[10];
        matrix[11] = self[14];
        matrix[12] = self[3];
        matrix[13] = self[7];
        matrix[14] = self[11];
        matrix[15] = self[15];
        matrix
    }

    /// Create an inversed copy of the `Matrix`.
    #[inline]
    pub fn inverse(&self) -> Self {
        let m = self;

        let t0 = m[10] * m[15];
        let t1 = m[14] * m[11];
        let t2 = m[6] * m[15];
        let t3 = m[14] * m[7];
        let t4 = m[6] * m[11];
        let t5 = m[10] * m[7];
        let t6 = m[2] * m[15];
        let t7 = m[14] * m[3];
        let t8 = m[2] * m[11];
        let t9 = m[10] * m[3];
        let t10 = m[2] * m[7];
        let t11 = m[6] * m[3];
        let t12 = m[8] * m[13];
        let t13 = m[12] * m[9];
        let t14 = m[4] * m[13];
        let t15 = m[12] * m[5];
        let t16 = m[4] * m[9];
        let t17 = m[8] * m[5];
        let t18 = m[0] * m[13];
        let t19 = m[12] * m[1];
        let t20 = m[0] * m[9];
        let t21 = m[8] * m[1];
        let t22 = m[0] * m[5];
        let t23 = m[4] * m[1];

        let mut matrix = Matrix::default();

        matrix[0] = (t0 * m[5] + t3 * m[9] + t4 * m[13]) - (t1 * m[5] + t2 * m[9] + t5 * m[13]);
        matrix[1] = (t1 * m[1] + t6 * m[9] + t9 * m[13]) - (t0 * m[1] + t7 * m[9] + t8 * m[13]);
        matrix[2] = (t2 * m[1] + t7 * m[5] + t10 * m[13]) - (t3 * m[1] + t6 * m[5] + t11 * m[13]);
        matrix[3] = (t5 * m[1] + t8 * m[5] + t11 * m[9]) - (t4 * m[1] + t9 * m[5] + t10 * m[9]);

        let d = 1.0 / (m[0] * matrix[0] + m[4] * matrix[1] + m[8] * matrix[2] + m[12] * matrix[3]);

        matrix[0] *= d;
        matrix[1] *= d;
        matrix[2] *= d;
        matrix[3] *= d;
        matrix[4] =
            d * ((t1 * m[4] + t2 * m[8] + t5 * m[12]) - (t0 * m[4] + t3 * m[8] + t4 * m[12]));
        matrix[5] =
            d * ((t0 * m[0] + t7 * m[8] + t8 * m[12]) - (t1 * m[0] + t6 * m[8] + t9 * m[12]));
        matrix[6] =
            d * ((t3 * m[0] + t6 * m[4] + t11 * m[12]) - (t2 * m[0] + t7 * m[4] + t10 * m[12]));
        matrix[7] =
            d * ((t4 * m[0] + t9 * m[4] + t10 * m[8]) - (t5 * m[0] + t8 * m[4] + t11 * m[8]));
        matrix[8] = d
            * ((t12 * m[7] + t15 * m[11] + t16 * m[15]) - (t13 * m[7] + t14 * m[11] + t17 * m[15]));
        matrix[9] = d
            * ((t13 * m[3] + t18 * m[11] + t21 * m[15]) - (t12 * m[3] + t19 * m[11] + t20 * m[15]));
        matrix[10] =
            d * ((t14 * m[3] + t19 * m[7] + t22 * m[15]) - (t15 * m[3] + t18 * m[7] + t23 * m[15]));
        matrix[11] =
            d * ((t17 * m[3] + t20 * m[7] + t23 * m[11]) - (t16 * m[3] + t21 * m[7] + t22 * m[11]));
        matrix[12] = d
            * ((t14 * m[10] + t17 * m[14] + t13 * m[6]) - (t16 * m[14] + t12 * m[6] + t15 * m[10]));
        matrix[13] = d
            * ((t20 * m[14] + t12 * m[2] + t19 * m[10]) - (t18 * m[10] + t21 * m[14] + t13 * m[2]));
        matrix[14] =
            d * ((t18 * m[6] + t23 * m[14] + t15 * m[2]) - (t22 * m[14] + t14 * m[2] + t19 * m[6]));
        matrix[15] =
            d * ((t22 * m[10] + t16 * m[2] + t21 * m[6]) - (t20 * m[6] + t23 * m[10] + t17 * m[2]));

        matrix
    }
}

impl Mul for Matrix<4, 4> {
    type Output = Matrix;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut matrix = Self::Output::identity();
        for i in 0..4 {
            for j in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += self[i * k] * rhs[k * j];
                }
                matrix[i * j] = sum;
            }
        }
        matrix
    }
}

impl Mul<Vector<3>> for Matrix<4, 4> {
    type Output = Vector<3>;

    fn mul(self, v: Vector<3>) -> Self::Output {
        let m = self;
        Vector::new([
            v.x() * m[0] + v.y() * m[1] + v.z() * m[2] + m[3],
            v.x() * m[4] + v.y() * m[5] + v.z() * m[6] + m[7],
            v.x() * m[8] + v.y() * m[9] + v.z() * m[10] + m[11],
        ])
    }
}

impl Mul<Matrix<4, 4>> for Vector<3> {
    type Output = Vector<3>;

    fn mul(self, m: Matrix<4, 4>) -> Self::Output {
        let v = self;
        Vector::new([
            v.x() * m[0] + v.y() * m[4] + v.z() * m[8] + m[12],
            v.x() * m[1] + v.y() * m[5] + v.z() * m[9] + m[13],
            v.x() * m[2] + v.y() * m[6] + v.z() * m[10] + m[14],
        ])
    }
}

impl Mul<Vector<4>> for Matrix<4, 4> {
    type Output = Vector<4>;

    fn mul(self, v: Vector<4>) -> Self::Output {
        let m = self;
        Vector::new([
            v.x() * m[0] + v.y() * m[1] + v.z() * m[2] + m[3],
            v.x() * m[4] + v.y() * m[5] + v.z() * m[6] + m[7],
            v.x() * m[8] + v.y() * m[9] + v.z() * m[10] + m[11],
            v.x() * m[12] + v.y() * m[13] + v.z() * m[14] + m[15],
        ])
    }
}

impl Mul<Matrix<4, 4>> for Vector<4> {
    type Output = Vector<4>;

    fn mul(self, m: Matrix<4, 4>) -> Self::Output {
        let v = self;
        Vector::new([
            v.x() * m[0] + v.y() * m[4] + v.z() * m[8] + m[12],
            v.x() * m[1] + v.y() * m[5] + v.z() * m[9] + m[13],
            v.x() * m[2] + v.y() * m[6] + v.z() * m[10] + m[14],
            v.x() * m[3] + v.y() * m[7] + v.z() * m[11] + m[15],
        ])
    }
}

impl<const N: usize, const M: usize> Matrix<N, M> {
    /// Construct an identity `Matrix`.
    #[inline]
    pub fn identity() -> Self {
        assert_eq!(N, M, "matrix is not symmetrical");
        let mut matrix = Self::default();
        for i in (0..(N * M)).step_by(N + 1) {
            matrix[i] = 1.0;
        }
        matrix
    }
}

impl<const N: usize, const M: usize> Deref for Matrix<N, M> {
    type Target = [[f32; M]; N];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const N: usize, const M: usize> DerefMut for Matrix<N, M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<const N: usize, const M: usize> Index<usize> for Matrix<N, M> {
    type Output = f32;

    // TODO: Make this a two-dimensional index
    fn index(&self, index: usize) -> &Self::Output {
        let n = index / M;
        let m = index % M;
        &self.0[n][m]
    }
}

impl<const N: usize, const M: usize> IndexMut<usize> for Matrix<N, M> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let n = index / M;
        let m = index % M;
        &mut self.0[n][m]
    }
}

/// A `Quaternion`.
#[derive(Debug, Copy, Clone)]
#[must_use]
#[repr(transparent)]
pub struct Quaternion([f32; 4]);

impl Default for Quaternion {
    fn default() -> Self {
        Self::identity()
    }
}

impl Quaternion {
    /// Create an identity `Quaternion`.
    #[inline]
    pub fn identity() -> Self {
        Self([0.0, 0.0, 0.0, 1.0])
    }

    /// Create a `Quaternion` from an axis and angle.
    #[inline]
    pub fn from_axis_angle(axis: Vector<3>, angle: Radians, normalize: bool) -> Self {
        let frac_angle_two = 0.5 * *angle;
        let (sin, cos) = frac_angle_two.sin_cos();
        let quaternion = Self([sin * axis.x(), sin * axis.y(), sin * axis.z(), cos]);
        normalize
            .then(|| quaternion.normalized())
            .unwrap_or(quaternion)
    }

    /// Return the `x` coordinate.
    #[must_use]
    #[inline]
    pub fn x(&self) -> f32 {
        self[0]
    }

    /// Return the `y` coordinate.
    #[must_use]
    #[inline]
    pub fn y(&self) -> f32 {
        self[1]
    }

    /// Return the `z` coordinate.
    #[must_use]
    #[inline]
    pub fn z(&self) -> f32 {
        self[2]
    }

    /// Return the `w` coordinate.
    #[must_use]
    #[inline]
    pub fn w(&self) -> f32 {
        self[3]
    }

    /// Create an identity `Quaternion`.
    #[must_use]
    #[inline]
    pub fn normal(&self) -> f32 {
        self.0.iter().map(|val| val * val).sum::<f32>().sqrt()
    }

    /// Normalize the `Quaternion` into a unit `Quaternion`.
    #[inline]
    pub fn normalize(&mut self) {
        let normal = self.normal();
        self.0.iter_mut().for_each(|val| *val /= normal);
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
        Self([-self.x(), -self.y(), -self.z(), self.w()])
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
        self.0
            .iter()
            .zip(rhs.0.iter())
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
}

impl Deref for Quaternion {
    type Target = [f32; 4];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Quaternion {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Add for Quaternion {
    type Output = Quaternion;

    fn add(self, rhs: Self) -> Self::Output {
        let mut quaternion = Self::Output::default();
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
        let mut quaternion = Self::Output::default();
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
        let mut quaternion = Self::Output::default();
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
        let mut quaternion = Self::Output::default();
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
        let x = self.x() * rhs.w() + self.y() * rhs.z() - self.z() * rhs.y() + self.w() * rhs.x();
        let y = -self.x() * rhs.z() + self.y() * rhs.w() + self.z() * rhs.x() + self.w() * rhs.y();
        let z = self.x() * rhs.y() - self.y() * rhs.x() + self.z() * rhs.w() + self.w() * rhs.z();
        let w = -self.x() * rhs.x() - self.y() * rhs.y() - self.z() * rhs.z() + self.w() * rhs.w();
        Self([x, y, z, w])
    }
}

impl Mul for &Quaternion {
    type Output = Quaternion;

    fn mul(self, rhs: Self) -> Self::Output {
        let x = self.x() * rhs.w() + self.y() * rhs.z() - self.z() * rhs.y() + self.w() * rhs.x();
        let y = -self.x() * rhs.z() + self.y() * rhs.w() + self.z() * rhs.x() + self.w() * rhs.y();
        let z = self.x() * rhs.y() - self.y() * rhs.x() + self.z() * rhs.w() + self.w() * rhs.z();
        let w = -self.x() * rhs.x() - self.y() * rhs.y() - self.z() * rhs.z() + self.w() * rhs.w();
        Quaternion([x, y, z, w])
    }
}

impl Mul<f32> for Quaternion {
    type Output = Quaternion;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut quaternion = Self::Output::default();
        quaternion.iter_mut().for_each(|val| *val *= rhs);
        quaternion
    }
}

impl Mul<f32> for &Quaternion {
    type Output = Quaternion;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut quaternion = Self::Output::default();
        quaternion.iter_mut().for_each(|val| *val *= rhs);
        quaternion
    }
}

impl Neg for Quaternion {
    type Output = Quaternion;

    fn neg(self) -> Self::Output {
        let mut vector = Self::Output::default();
        vector.iter_mut().for_each(|val| *val = val.neg());
        vector
    }
}

impl Neg for &Quaternion {
    type Output = Quaternion;

    fn neg(self) -> Self::Output {
        let mut vector = Self::Output::default();
        vector.iter_mut().for_each(|val| *val = val.neg());
        vector
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
    derive_more::Add,
    derive_more::AddAssign,
    derive_more::From,
    derive_more::Into,
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
pub struct Radians(pub f32);

impl Deref for Radians {
    type Target = f32;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Radians {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// An angle in degrees.
#[derive(
    Default,
    Debug,
    Copy,
    Clone,
    PartialEq,
    PartialOrd,
    derive_more::Add,
    derive_more::AddAssign,
    derive_more::From,
    derive_more::Into,
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
pub struct Degrees(f32);

impl Deref for Degrees {
    type Target = f32;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Degrees {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
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
