//! Window types.

pub type Window = winit::window::Window;

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

impl From<winit::dpi::Size> for Size {
    fn from(size: winit::dpi::Size) -> Self {
        match size {
            winit::dpi::Size::Physical(size) => Self::Physical(size.into()),
            winit::dpi::Size::Logical(size) => Self::Logical(size.into()),
        }
    }
}

impl From<Size> for winit::dpi::Size {
    fn from(size: Size) -> Self {
        match size {
            Size::Physical(size) => Self::Physical(size.into()),
            Size::Logical(size) => Self::Logical(size.into()),
        }
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

impl From<winit::dpi::PhysicalSize<u32>> for PhysicalSize {
    fn from(size: winit::dpi::PhysicalSize<u32>) -> Self {
        Self {
            width: size.width,
            height: size.height,
        }
    }
}

impl From<PhysicalSize> for winit::dpi::PhysicalSize<u32> {
    fn from(size: PhysicalSize) -> Self {
        Self {
            width: size.width,
            height: size.height,
        }
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use]
pub struct LogicalSize {
    pub width: f64,
    pub height: f64,
}

impl From<winit::dpi::LogicalSize<f64>> for LogicalSize {
    fn from(size: winit::dpi::LogicalSize<f64>) -> Self {
        Self {
            width: size.width,
            height: size.height,
        }
    }
}

impl From<LogicalSize> for winit::dpi::LogicalSize<f64> {
    fn from(size: LogicalSize) -> Self {
        Self {
            width: size.width,
            height: size.height,
        }
    }
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

impl From<winit::dpi::Position> for Position {
    fn from(position: winit::dpi::Position) -> Self {
        match position {
            winit::dpi::Position::Physical(position) => Self::Physical(position.into()),
            winit::dpi::Position::Logical(position) => Self::Logical(position.into()),
        }
    }
}

impl From<Position> for winit::dpi::Position {
    fn from(position: Position) -> Self {
        match position {
            Position::Physical(position) => Self::Physical(position.into()),
            Position::Logical(position) => Self::Logical(position.into()),
        }
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use]
pub struct PhysicalPosition<T> {
    pub x: T,
    pub y: T,
}

impl<T> From<winit::dpi::PhysicalPosition<T>> for PhysicalPosition<T> {
    fn from(position: winit::dpi::PhysicalPosition<T>) -> Self {
        Self {
            x: position.x,
            y: position.y,
        }
    }
}

impl<T> From<PhysicalPosition<T>> for winit::dpi::PhysicalPosition<T> {
    fn from(position: PhysicalPosition<T>) -> Self {
        Self {
            x: position.x,
            y: position.y,
        }
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use]
pub struct LogicalPosition<T> {
    pub x: T,
    pub y: T,
}

impl<T> From<winit::dpi::LogicalPosition<T>> for LogicalPosition<T> {
    fn from(position: winit::dpi::LogicalPosition<T>) -> Self {
        Self {
            x: position.x,
            y: position.y,
        }
    }
}

impl<T> From<LogicalPosition<T>> for winit::dpi::LogicalPosition<T> {
    fn from(position: LogicalPosition<T>) -> Self {
        Self {
            x: position.x,
            y: position.y,
        }
    }
}
