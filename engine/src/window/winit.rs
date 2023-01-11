//! Winit window backend.

use super::Positioned;
use crate::config::Fullscreen;
use crate::event::{
    DeviceEvent, DeviceId, Event, InputState, KeyCode, ModifierKeys, MouseButton, MouseScrollDelta,
    TouchPhase, WindowEvent,
};
use crate::window::{
    LogicalPosition, LogicalSize, PhysicalPosition, PhysicalSize, Position, Size, WindowId,
};

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

impl From<winit::dpi::PhysicalSize<u32>> for PhysicalSize<u32> {
    fn from(size: winit::dpi::PhysicalSize<u32>) -> Self {
        Self {
            width: size.width,
            height: size.height,
        }
    }
}

impl From<PhysicalSize<u32>> for winit::dpi::PhysicalSize<u32> {
    fn from(size: PhysicalSize<u32>) -> Self {
        Self {
            width: size.width,
            height: size.height,
        }
    }
}

impl From<winit::dpi::LogicalSize<f64>> for LogicalSize<f64> {
    fn from(size: winit::dpi::LogicalSize<f64>) -> Self {
        Self {
            width: size.width,
            height: size.height,
        }
    }
}

impl From<LogicalSize<f64>> for winit::dpi::LogicalSize<f64> {
    fn from(size: LogicalSize<f64>) -> Self {
        Self {
            width: size.width,
            height: size.height,
        }
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

impl From<winit::window::WindowId> for WindowId {
    fn from(window_id: winit::window::WindowId) -> Self {
        Self(window_id)
    }
}

impl From<winit::event::DeviceId> for DeviceId {
    fn from(device_id: winit::event::DeviceId) -> Self {
        Self(device_id)
    }
}

impl<'a, T> From<winit::event::Event<'a, T>> for Event<T> {
    fn from(event: winit::event::Event<'a, T>) -> Self {
        match event {
            winit::event::Event::NewEvents(_) => Self::Unhandled,
            winit::event::Event::WindowEvent { window_id, event } => Self::WindowEvent {
                window_id: window_id.into(),
                event: event.into(),
            },
            winit::event::Event::DeviceEvent { device_id, event } => Self::DeviceEvent {
                device_id: device_id.into(),
                event: event.into(),
            },
            winit::event::Event::UserEvent(event) => Self::UserEvent(event),
            winit::event::Event::Suspended => Self::Suspended,
            winit::event::Event::Resumed => Self::Resumed,
            winit::event::Event::MainEventsCleared => Self::Unhandled,
            winit::event::Event::RedrawRequested(_) => Self::Unhandled,
            winit::event::Event::RedrawEventsCleared => Self::Unhandled,
            winit::event::Event::LoopDestroyed => Self::Unhandled,
        }
    }
}

impl<'a> From<winit::event::WindowEvent<'a>> for WindowEvent {
    fn from(event: winit::event::WindowEvent<'a>) -> Self {
        match event {
            winit::event::WindowEvent::Resized(size) => Self::Resized(size.into()),
            winit::event::WindowEvent::Moved(position) => Self::Moved(position.into()),
            winit::event::WindowEvent::CloseRequested => Self::CloseRequested,
            winit::event::WindowEvent::Destroyed => Self::Destroyed,
            winit::event::WindowEvent::Focused(focused) => Self::Focused(focused),
            winit::event::WindowEvent::KeyboardInput {
                device_id,
                input,
                is_synthetic,
            } => Self::KeyboardInput {
                device_id: device_id.into(),
                scancode: input.scancode,
                keycode: input.virtual_keycode.map(Into::into),
                state: input.state.into(),
                is_synthetic,
            },
            winit::event::WindowEvent::ModifiersChanged(modifiers) => {
                Self::ModifiersChanged(modifiers.into())
            }
            winit::event::WindowEvent::CursorMoved {
                device_id,
                position,
                ..
            } => Self::CursorMoved {
                device_id: device_id.into(),
                position: position.into(),
            },
            winit::event::WindowEvent::CursorEntered { device_id } => Self::CursorEntered {
                device_id: device_id.into(),
            },
            winit::event::WindowEvent::CursorLeft { device_id } => Self::CursorLeft {
                device_id: device_id.into(),
            },
            winit::event::WindowEvent::MouseWheel {
                device_id,
                delta,
                phase,
                ..
            } => Self::MouseWheel {
                device_id: device_id.into(),
                delta: delta.into(),
                phase: phase.into(),
            },
            winit::event::WindowEvent::MouseInput {
                device_id,
                button,
                state,
                ..
            } => Self::MouseInput {
                device_id: device_id.into(),
                button: button.into(),
                state: state.into(),
            },
            winit::event::WindowEvent::AxisMotion {
                device_id,
                axis,
                value,
            } => Self::AxisMotion {
                device_id: device_id.into(),
                axis,
                value,
            },
            winit::event::WindowEvent::Touch(winit::event::Touch {
                device_id,
                phase,
                location,
                id,
                ..
            }) => Self::Touch {
                device_id: device_id.into(),
                phase: phase.into(),
                position: location.into(),
                id,
            },
            winit::event::WindowEvent::ScaleFactorChanged {
                scale_factor,
                new_inner_size,
            } => Self::ScaleFactorChanged {
                scale_factor,
                new_inner_size: (*new_inner_size).into(),
            },
            winit::event::WindowEvent::Occluded(occluded) => Self::Occluded(occluded),
            _ => Self::Unhandled,
        }
    }
}

impl From<winit::event::DeviceEvent> for DeviceEvent {
    fn from(event: winit::event::DeviceEvent) -> Self {
        match event {
            winit::event::DeviceEvent::Added => Self::Unhandled,
            winit::event::DeviceEvent::Removed => Self::Unhandled,
            winit::event::DeviceEvent::MouseMotion { delta } => Self::MouseMotion { delta },
            winit::event::DeviceEvent::MouseWheel { delta } => Self::MouseWheel {
                delta: delta.into(),
            },
            winit::event::DeviceEvent::Motion { axis, value } => Self::Motion { axis, value },
            winit::event::DeviceEvent::Button { button, state } => Self::Button {
                button,
                state: state.into(),
            },
            winit::event::DeviceEvent::Key(input) => Self::Key {
                scancode: input.scancode,
                keycode: input.virtual_keycode.map(Into::into),
                state: input.state.into(),
            },
            winit::event::DeviceEvent::Text { codepoint } => Self::Text { codepoint },
        }
    }
}

impl From<winit::event::VirtualKeyCode> for KeyCode {
    fn from(keycode: winit::event::VirtualKeyCode) -> Self {
        match keycode {
            winit::event::VirtualKeyCode::Key1 => Self::Key1,
            winit::event::VirtualKeyCode::Key2 => Self::Key2,
            winit::event::VirtualKeyCode::Key3 => Self::Key3,
            winit::event::VirtualKeyCode::Key4 => Self::Key4,
            winit::event::VirtualKeyCode::Key5 => Self::Key5,
            winit::event::VirtualKeyCode::Key6 => Self::Key6,
            winit::event::VirtualKeyCode::Key7 => Self::Key7,
            winit::event::VirtualKeyCode::Key8 => Self::Key8,
            winit::event::VirtualKeyCode::Key9 => Self::Key9,
            winit::event::VirtualKeyCode::Key0 => Self::Key0,
            winit::event::VirtualKeyCode::A => Self::A,
            winit::event::VirtualKeyCode::B => Self::B,
            winit::event::VirtualKeyCode::C => Self::C,
            winit::event::VirtualKeyCode::D => Self::D,
            winit::event::VirtualKeyCode::E => Self::E,
            winit::event::VirtualKeyCode::F => Self::F,
            winit::event::VirtualKeyCode::G => Self::G,
            winit::event::VirtualKeyCode::H => Self::H,
            winit::event::VirtualKeyCode::I => Self::I,
            winit::event::VirtualKeyCode::J => Self::J,
            winit::event::VirtualKeyCode::K => Self::K,
            winit::event::VirtualKeyCode::L => Self::L,
            winit::event::VirtualKeyCode::M => Self::M,
            winit::event::VirtualKeyCode::N => Self::N,
            winit::event::VirtualKeyCode::O => Self::O,
            winit::event::VirtualKeyCode::P => Self::P,
            winit::event::VirtualKeyCode::Q => Self::Q,
            winit::event::VirtualKeyCode::R => Self::R,
            winit::event::VirtualKeyCode::S => Self::S,
            winit::event::VirtualKeyCode::T => Self::T,
            winit::event::VirtualKeyCode::U => Self::U,
            winit::event::VirtualKeyCode::V => Self::V,
            winit::event::VirtualKeyCode::W => Self::W,
            winit::event::VirtualKeyCode::X => Self::X,
            winit::event::VirtualKeyCode::Y => Self::Y,
            winit::event::VirtualKeyCode::Z => Self::Z,
            winit::event::VirtualKeyCode::Escape => Self::Escape,
            winit::event::VirtualKeyCode::F1 => Self::F1,
            winit::event::VirtualKeyCode::F2 => Self::F2,
            winit::event::VirtualKeyCode::F3 => Self::F3,
            winit::event::VirtualKeyCode::F4 => Self::F4,
            winit::event::VirtualKeyCode::F5 => Self::F5,
            winit::event::VirtualKeyCode::F6 => Self::F6,
            winit::event::VirtualKeyCode::F7 => Self::F7,
            winit::event::VirtualKeyCode::F8 => Self::F8,
            winit::event::VirtualKeyCode::F9 => Self::F9,
            winit::event::VirtualKeyCode::F10 => Self::F10,
            winit::event::VirtualKeyCode::F11 => Self::F11,
            winit::event::VirtualKeyCode::F12 => Self::F12,
            winit::event::VirtualKeyCode::F13 => Self::F13,
            winit::event::VirtualKeyCode::F14 => Self::F14,
            winit::event::VirtualKeyCode::F15 => Self::F15,
            winit::event::VirtualKeyCode::F16 => Self::F16,
            winit::event::VirtualKeyCode::F17 => Self::F17,
            winit::event::VirtualKeyCode::F18 => Self::F18,
            winit::event::VirtualKeyCode::F19 => Self::F19,
            winit::event::VirtualKeyCode::F20 => Self::F20,
            winit::event::VirtualKeyCode::F21 => Self::F21,
            winit::event::VirtualKeyCode::F22 => Self::F22,
            winit::event::VirtualKeyCode::F23 => Self::F23,
            winit::event::VirtualKeyCode::F24 => Self::F24,
            winit::event::VirtualKeyCode::Snapshot => Self::Snapshot,
            winit::event::VirtualKeyCode::Insert => Self::Insert,
            winit::event::VirtualKeyCode::Home => Self::Home,
            winit::event::VirtualKeyCode::Delete => Self::Delete,
            winit::event::VirtualKeyCode::End => Self::End,
            winit::event::VirtualKeyCode::PageDown => Self::PageDown,
            winit::event::VirtualKeyCode::PageUp => Self::PageUp,
            winit::event::VirtualKeyCode::Left => Self::Left,
            winit::event::VirtualKeyCode::Up => Self::Up,
            winit::event::VirtualKeyCode::Right => Self::Right,
            winit::event::VirtualKeyCode::Down => Self::Down,
            winit::event::VirtualKeyCode::Back => Self::Back,
            winit::event::VirtualKeyCode::Return => Self::Return,
            winit::event::VirtualKeyCode::Space => Self::Space,
            winit::event::VirtualKeyCode::Caret => Self::Caret,
            winit::event::VirtualKeyCode::Numpad0 => Self::Numpad0,
            winit::event::VirtualKeyCode::Numpad1 => Self::Numpad1,
            winit::event::VirtualKeyCode::Numpad2 => Self::Numpad2,
            winit::event::VirtualKeyCode::Numpad3 => Self::Numpad3,
            winit::event::VirtualKeyCode::Numpad4 => Self::Numpad4,
            winit::event::VirtualKeyCode::Numpad5 => Self::Numpad5,
            winit::event::VirtualKeyCode::Numpad6 => Self::Numpad6,
            winit::event::VirtualKeyCode::Numpad7 => Self::Numpad7,
            winit::event::VirtualKeyCode::Numpad8 => Self::Numpad8,
            winit::event::VirtualKeyCode::Numpad9 => Self::Numpad9,
            winit::event::VirtualKeyCode::NumpadAdd => Self::NumpadAdd,
            winit::event::VirtualKeyCode::NumpadDivide => Self::NumpadDivide,
            winit::event::VirtualKeyCode::NumpadDecimal => Self::NumpadDecimal,
            winit::event::VirtualKeyCode::NumpadComma => Self::NumpadComma,
            winit::event::VirtualKeyCode::NumpadEnter => Self::NumpadEnter,
            winit::event::VirtualKeyCode::NumpadEquals => Self::NumpadEquals,
            winit::event::VirtualKeyCode::NumpadMultiply => Self::NumpadMultiply,
            winit::event::VirtualKeyCode::NumpadSubtract => Self::NumpadSubtract,
            winit::event::VirtualKeyCode::Apostrophe => Self::Apostrophe,
            winit::event::VirtualKeyCode::Asterisk => Self::Asterisk,
            winit::event::VirtualKeyCode::At => Self::At,
            winit::event::VirtualKeyCode::Backslash => Self::Backslash,
            winit::event::VirtualKeyCode::Colon => Self::Colon,
            winit::event::VirtualKeyCode::Comma => Self::Comma,
            winit::event::VirtualKeyCode::Equals => Self::Equals,
            winit::event::VirtualKeyCode::Grave => Self::Grave,
            winit::event::VirtualKeyCode::LAlt => Self::LAlt,
            winit::event::VirtualKeyCode::LBracket => Self::LBracket,
            winit::event::VirtualKeyCode::LControl => Self::LControl,
            winit::event::VirtualKeyCode::LShift => Self::LShift,
            winit::event::VirtualKeyCode::LWin => Self::LWin,
            winit::event::VirtualKeyCode::Minus => Self::Minus,
            winit::event::VirtualKeyCode::Period => Self::Period,
            winit::event::VirtualKeyCode::Plus => Self::Plus,
            winit::event::VirtualKeyCode::RAlt => Self::RAlt,
            winit::event::VirtualKeyCode::RBracket => Self::RBracket,
            winit::event::VirtualKeyCode::RControl => Self::RControl,
            winit::event::VirtualKeyCode::RShift => Self::RShift,
            winit::event::VirtualKeyCode::RWin => Self::RWin,
            winit::event::VirtualKeyCode::Semicolon => Self::Semicolon,
            winit::event::VirtualKeyCode::Slash => Self::Slash,
            _ => Self::Unhandled,
        }
    }
}

impl From<winit::event::MouseButton> for MouseButton {
    fn from(button: winit::event::MouseButton) -> Self {
        match button {
            winit::event::MouseButton::Left => Self::Left,
            winit::event::MouseButton::Right => Self::Right,
            winit::event::MouseButton::Middle => Self::Middle,
            winit::event::MouseButton::Other(button) => Self::Other(button),
        }
    }
}

impl From<winit::event::MouseScrollDelta> for MouseScrollDelta {
    fn from(delta: winit::event::MouseScrollDelta) -> Self {
        match delta {
            winit::event::MouseScrollDelta::LineDelta(x, y) => Self::LineDelta(x, y),
            winit::event::MouseScrollDelta::PixelDelta(delta) => Self::PixelDelta(delta.into()),
        }
    }
}

impl From<winit::event::TouchPhase> for TouchPhase {
    fn from(phase: winit::event::TouchPhase) -> Self {
        match phase {
            winit::event::TouchPhase::Started => Self::Started,
            winit::event::TouchPhase::Moved => Self::Moved,
            winit::event::TouchPhase::Ended => Self::Ended,
            winit::event::TouchPhase::Cancelled => Self::Cancelled,
        }
    }
}

impl From<winit::event::ElementState> for InputState {
    fn from(state: winit::event::ElementState) -> Self {
        match state {
            winit::event::ElementState::Pressed => Self::Pressed,
            winit::event::ElementState::Released => Self::Released,
        }
    }
}

impl From<winit::event::ModifiersState> for ModifierKeys {
    fn from(state: winit::event::ModifiersState) -> Self {
        use winit::event::ModifiersState;
        let mut keys = Self::NONE;
        keys.set(Self::SHIFT, state.contains(ModifiersState::SHIFT));
        keys.set(Self::CTRL, state.contains(ModifiersState::CTRL));
        keys.set(Self::ALT, state.contains(ModifiersState::ALT));
        keys.set(Self::GUI, state.contains(ModifiersState::LOGO));
        keys
    }
}

pub(crate) trait FullscreenModeExt {
    fn for_monitor(
        &self,
        monitor: Option<winit::monitor::MonitorHandle>,
    ) -> Option<winit::window::Fullscreen>;
}

impl FullscreenModeExt for Fullscreen {
    fn for_monitor(
        &self,
        monitor: Option<winit::monitor::MonitorHandle>,
    ) -> Option<winit::window::Fullscreen> {
        match self {
            Self::Exclusive => monitor
                .and_then(|monitor| monitor.video_modes().next())
                .map(winit::window::Fullscreen::Exclusive),
            Self::Borderless => Some(winit::window::Fullscreen::Borderless(monitor)),
        }
    }
}

pub(crate) trait PositionedExt {
    fn for_monitor(
        &self,
        monitor: Option<winit::monitor::MonitorHandle>,
        window_size: Size,
    ) -> Position;
}

impl PositionedExt for Positioned {
    fn for_monitor(
        &self,
        monitor: Option<winit::monitor::MonitorHandle>,
        window_size: Size,
    ) -> Position {
        match self {
            Positioned::Center => {
                if let Some(monitor) = monitor {
                    let ::winit::dpi::PhysicalSize {
                        width: monitor_width,
                        height: monitor_height,
                    } = monitor.size();
                    let window_size = window_size.to_physical(monitor.scale_factor());
                    let PhysicalSize {
                        width: window_width,
                        height: window_height,
                    } = window_size;
                    Position::Physical(PhysicalPosition {
                        x: (monitor_width as i32) / 2 - (window_width as i32 / 2),
                        y: (monitor_height as i32) / 2 - (window_height as i32 / 2),
                    })
                } else {
                    Position::default()
                }
            }
            Positioned::Position(position) => *position,
        }
    }
}
