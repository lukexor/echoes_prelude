use crate::vector::{Vec2, Vec3};

#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use]
pub struct Tri {
    pub p0: Vec3,
    pub p1: Vec3,
    pub p2: Vec3,
}

impl From<[[f32; 2]; 3]> for Tri {
    fn from(array: [[f32; 2]; 3]) -> Self {
        Self {
            p0: Vec2::from(array[0]).to_vec3(0.0),
            p1: Vec2::from(array[1]).to_vec3(0.0),
            p2: Vec2::from(array[2]).to_vec3(0.0),
        }
    }
}

impl From<[[f32; 3]; 3]> for Tri {
    fn from(array: [[f32; 3]; 3]) -> Self {
        Self {
            p0: Vec3::from(array[0]),
            p1: Vec3::from(array[1]),
            p2: Vec3::from(array[2]),
        }
    }
}
