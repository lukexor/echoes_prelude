//! Traits and types for window backends.

use crate::config::Fullscreen;
use serde::{Deserialize, Serialize};

pub(crate) mod winit;

pub type Window = ::winit::window::Window;
pub type EventLoopProxy<T> = ::winit::event_loop::EventLoopProxy<T>;

#[derive(Debug, Copy, Clone, PartialEq)]
#[must_use]
pub(crate) struct WindowCreateInfo {
    pub(crate) size: Size,
    pub(crate) positioned: Positioned,
    pub(crate) fullscreen: Option<Fullscreen>,
    pub(crate) cursor_grab: bool,
    pub(crate) resizable: bool,
}

impl Default for WindowCreateInfo {
    fn default() -> Self {
        Self {
            size: PhysicalSize::new(1024, 768).into(),
            positioned: Positioned::default(),
            fullscreen: None,
            cursor_grab: false,
            resizable: true,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[must_use]
pub struct WindowId(pub(crate) ::winit::window::WindowId);

#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
#[must_use]
pub enum Size {
    Physical(PhysicalSize<u32>),
    Logical(LogicalSize<f64>),
}

impl From<PhysicalSize<u32>> for Size {
    fn from(size: PhysicalSize<u32>) -> Self {
        Self::Physical(size)
    }
}

impl From<LogicalSize<f64>> for Size {
    fn from(size: LogicalSize<f64>) -> Self {
        Self::Logical(size)
    }
}

impl Size {
    #[inline]
    pub fn to_logical(&self, scale_factor: f64) -> LogicalSize<f64> {
        match *self {
            Size::Physical(size) => size.to_logical(scale_factor),
            Size::Logical(size) => size,
        }
    }

    #[inline]
    pub fn to_physical(&self, scale_factor: f64) -> PhysicalSize<u32> {
        match *self {
            Size::Physical(size) => size,
            Size::Logical(size) => size.to_physical(scale_factor),
        }
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[must_use]
pub struct PhysicalSize<T> {
    pub width: T,
    pub height: T,
}

impl PhysicalSize<u32> {
    #[inline]
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    #[inline]
    pub fn from_logical(logical: LogicalSize<f64>, scale_factor: f64) -> Self {
        logical.to_physical(scale_factor)
    }

    #[inline]
    pub fn to_logical(&self, scale_factor: f64) -> LogicalSize<f64> {
        let width = self.width as f64 / scale_factor;
        let height = self.height as f64 / scale_factor;
        LogicalSize::new(width, height)
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[must_use]
pub struct LogicalSize<T> {
    pub width: T,
    pub height: T,
}

impl LogicalSize<f64> {
    #[inline]
    pub fn new(width: f64, height: f64) -> Self {
        Self { width, height }
    }

    #[inline]
    pub fn from_physical(physical: PhysicalSize<u32>, scale_factor: f64) -> Self {
        physical.to_logical(scale_factor)
    }

    #[inline]
    pub fn to_physical(&self, scale_factor: f64) -> PhysicalSize<u32> {
        let width = (self.width * scale_factor).round() as u32;
        let height = (self.height * scale_factor).round() as u32;
        PhysicalSize::new(width, height)
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
#[must_use]
pub enum Positioned {
    #[default]
    Center,
    Position(Position),
}

impl From<Position> for Positioned {
    fn from(position: Position) -> Self {
        Self::Position(position)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
#[must_use]
pub enum Position {
    Physical(PhysicalPosition<i32>),
    Logical(LogicalPosition<f64>),
}

impl Default for Position {
    fn default() -> Self {
        Self::Physical(PhysicalPosition::default())
    }
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

impl Position {
    #[inline]
    pub fn to_logical(&self, scale_factor: f64) -> LogicalPosition<f64> {
        match *self {
            Position::Physical(position) => position.to_logical(scale_factor),
            Position::Logical(position) => position,
        }
    }

    #[inline]
    pub fn to_physical(&self, scale_factor: f64) -> PhysicalPosition<i32> {
        match *self {
            Position::Physical(position) => position,
            Position::Logical(position) => position.to_physical(scale_factor),
        }
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[must_use]
pub struct PhysicalPosition<T> {
    pub x: T,
    pub y: T,
}

impl PhysicalPosition<i32> {
    #[inline]
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn from_logical(logical: LogicalPosition<f64>, scale_factor: f64) -> Self {
        logical.to_physical(scale_factor)
    }

    #[inline]
    pub fn to_logical(&self, scale_factor: f64) -> LogicalPosition<f64> {
        let x = self.x as f64 / scale_factor;
        let y = self.y as f64 / scale_factor;
        LogicalPosition::new(x, y)
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[must_use]
pub struct LogicalPosition<T> {
    pub x: T,
    pub y: T,
}

impl LogicalPosition<f64> {
    #[inline]
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn from_physical(physical: PhysicalPosition<i32>, scale_factor: f64) -> Self {
        physical.to_logical(scale_factor)
    }

    #[inline]
    pub fn to_physical(&self, scale_factor: f64) -> PhysicalPosition<i32> {
        let x = (self.x * scale_factor).round() as i32;
        let y = (self.y * scale_factor).round() as i32;
        PhysicalPosition::new(x, y)
    }
}
