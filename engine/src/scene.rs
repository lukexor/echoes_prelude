use crate::vector::Vec4;

#[derive(Default, Debug, Copy, Clone)]
#[repr(C)]
#[must_use]
pub struct SceneData {
    pub fog_color: Vec4,     // w = exponent
    pub fog_distances: Vec4, // x = min, y = max, zw = unused
    pub ambient_color: Vec4,
    pub sunlight_direction: Vec4, // w = sun power
    pub sunlight_color: Vec4,
}
