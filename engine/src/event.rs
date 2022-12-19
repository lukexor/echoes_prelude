//! Engine events

use bitflags::bitflags;
use winit::event::{Event as WinitEvent, WindowEvent};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[must_use]
#[non_exhaustive]
pub enum Event {
    Quit,
    WindowClose,
    Resized(u32, u32),
    Moved(i32, i32),
    KeyInput {
        keycode: KeyCode,
        state: InputState,
        modifiers: ModifierKeys,
    },
    MouseInput {
        button: MouseButton,
        state: InputState,
        modifiers: ModifierKeys,
    },
    MouseMotion {
        x: i32,
        y: i32,
    },
    MouseWheel {
        delta: i32,
    },
    ControllerInput {
        button: ControllerButton,
        state: InputState,
    },
    Unknown,
}

impl<'a, T> From<WinitEvent<'a, T>> for Event {
    fn from(event: WinitEvent<'a, T>) -> Self {
        match event {
            // WinitEvent::NewEvents(_) => todo!(),
            WinitEvent::WindowEvent {
                window_id: _,
                event,
            } => match event {
                WindowEvent::Resized(size) => Self::Resized(size.width, size.height),
                WindowEvent::Moved(position) => Self::Moved(position.x, position.y),
                WindowEvent::CloseRequested | WindowEvent::Destroyed => Self::WindowClose,
                // WindowEvent::DroppedFile(_) => todo!(),
                // WindowEvent::HoveredFile(_) => todo!(),
                // WindowEvent::HoveredFileCancelled => todo!(),
                // WindowEvent::ReceivedCharacter(_) => todo!(),
                // WindowEvent::Focused(_) => todo!(),
                // WindowEvent::KeyboardInput {
                //     device_id,
                //     input,
                //     is_synthetic,
                // } => todo!(),
                // WindowEvent::ModifiersChanged(_) => todo!(),
                // WindowEvent::Ime(_) => todo!(),
                // WindowEvent::CursorMoved {
                //     device_id,
                //     position,
                //     ..
                // } => todo!(),
                // WindowEvent::CursorEntered { device_id } => todo!(),
                // WindowEvent::CursorLeft { device_id } => todo!(),
                // WindowEvent::MouseWheel {
                //     device_id,
                //     delta,
                //     phase,
                //     ..
                // } => todo!(),
                // WindowEvent::MouseInput {
                //     device_id,
                //     state,
                //     button,
                //     ..
                // } => todo!(),
                // WindowEvent::TouchpadPressure {
                //     device_id,
                //     pressure,
                //     stage,
                // } => todo!(),
                // WindowEvent::AxisMotion {
                //     device_id,
                //     axis,
                //     value,
                // } => todo!(),
                // WindowEvent::Touch(_) => todo!(),
                // WindowEvent::ScaleFactorChanged {
                //     scale_factor,
                //     new_inner_size,
                // } => todo!(),
                // WindowEvent::ThemeChanged(_) => todo!(),
                // WindowEvent::Occluded(_) => todo!(),
                _ => Self::Unknown,
            },
            // WinitEvent::DeviceEvent { device_id, event } => match event {
            // DeviceEvent::Added => todo!(),
            // DeviceEvent::Removed => todo!(),
            // DeviceEvent::MouseMotion { delta } => todo!(),
            // DeviceEvent::MouseWheel { delta } => todo!(),
            // DeviceEvent::Motion { axis, value } => todo!(),
            // DeviceEvent::Button { button, state } => todo!(),
            // DeviceEvent::Key(_) => todo!(),
            // DeviceEvent::Text { codepoint } => todo!(),
            // },
            // WinitEvent::UserEvent(_) => todo!(),
            // WinitEvent::Suspended => todo!(),
            // WinitEvent::Resumed => todo!(),
            // WinitEvent::MainEventsCleared => todo!(),
            // WinitEvent::RedrawRequested(window_id) => todo!(),
            // WinitEvent::RedrawEventsCleared => todo!(),
            // WinitEvent::LoopDestroyed => todo!(),
            _ => Self::Unknown,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[must_use]
pub enum InputState {
    Pressed,
    Released,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[must_use]
#[non_exhaustive]
pub enum KeyCode {
    W,
    A,
    S,
    D,
    Escape,
    Up,
    Right,
    Down,
    Left,
    Space,
    LAlt,
    LControl,
    LShift,
    RAlt,
    RControl,
    RShift,
}

bitflags! {
    #[must_use]
    pub struct ModifierKeys: u16 {
        /// No key modifier.
        const NONE = 0x0000;
        /// Left Shift or Right Shift.
        const SHIFT = 0x0001;
        /// Left Control or Right Control.
        const CTRL = 0x0040;
        /// Left Alt/Option or Right Alt/Option.
        const ALT = 0x0100;
        /// Left GUI or Right GUI (e.g. Windows or Command keys).
        const GUI = 0x0400;
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[must_use]
#[non_exhaustive]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[must_use]
#[non_exhaustive]
pub enum ControllerButton {}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[must_use]
#[non_exhaustive]
pub struct ControllerAxis {}
