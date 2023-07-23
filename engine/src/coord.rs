//! Coordinates.

use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
#[must_use]
pub enum Pos {
    #[default]
    Center,
    CenterX(i32),
    CenterY(i32),
    Screen([i32; 2]),
    World([f64; 3]),
}

impl From<[i32; 2]> for Pos {
    fn from([x, y]: [i32; 2]) -> Self {
        Self::Screen([x, y])
    }
}
impl From<(i32, i32)> for Pos {
    fn from((x, y): (i32, i32)) -> Self {
        Self::Screen([x, y])
    }
}

impl From<[f64; 3]> for Pos {
    fn from([x, y, z]: [f64; 3]) -> Self {
        Self::World([x, y, z])
    }
}
impl From<(f64, f64, f64)> for Pos {
    fn from((x, y, z): (f64, f64, f64)) -> Self {
        Self::World([x, y, z])
    }
}
