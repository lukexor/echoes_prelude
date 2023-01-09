//! Traits and types for window backends.

pub(crate) mod winit;

pub type Window = ::winit::window::Window;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[must_use]
pub struct WindowId(pub(crate) ::winit::window::WindowId);

#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use]
pub enum Size {
    Physical(PhysicalSize),
    Logical(LogicalSize),
}

impl From<PhysicalSize> for Size {
    fn from(size: PhysicalSize) -> Self {
        Self::Physical(size)
    }
}

impl From<LogicalSize> for Size {
    fn from(size: LogicalSize) -> Self {
        Self::Logical(size)
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use]
pub struct PhysicalSize {
    pub width: u32,
    pub height: u32,
}

impl PhysicalSize {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use]
pub struct LogicalSize {
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use]
pub enum Position {
    Physical(PhysicalPosition<i32>),
    Logical(LogicalPosition<f64>),
}

impl From<PhysicalPosition<i32>> for Position {
    fn from(size: PhysicalPosition<i32>) -> Self {
        Self::Physical(size)
    }
}

impl From<LogicalPosition<f64>> for Position {
    fn from(size: LogicalPosition<f64>) -> Self {
        Self::Logical(size)
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use]
pub struct PhysicalPosition<T> {
    pub x: T,
    pub y: T,
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use]
pub struct LogicalPosition<T> {
    pub x: T,
    pub y: T,
}
