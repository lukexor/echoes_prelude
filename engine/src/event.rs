//! Engine events

use bitflags::bitflags;
use winit::event::{
    DeviceEvent, ElementState, Event as WinitEvent, ModifiersState, MouseScrollDelta,
    VirtualKeyCode, WindowEvent,
};

#[derive(Debug, Copy, Clone, PartialEq)]
#[must_use]
#[non_exhaustive]
pub enum Event {
    Quit,
    WindowClose,
    Resized(u32, u32),
    Moved(i32, i32),
    Focused(bool),
    KeyInput {
        keycode: KeyCode,
        state: InputState,
    },
    MouseInput {
        button: MouseButton,
        state: InputState,
    },
    ModifiersChanged(ModifierKeys),
    MouseMotion {
        x: f64,
        y: f64,
    },
    MouseWheel {
        x: f64,
        y: f64,
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
                WindowEvent::Focused(focused) => Self::Focused(focused),
                WindowEvent::KeyboardInput { input, .. } => match input.virtual_keycode {
                    Some(keycode) => Self::KeyInput {
                        keycode: keycode.into(),
                        state: input.state.into(),
                    },
                    None => Self::Unknown,
                },
                WindowEvent::ModifiersChanged(state) => Self::ModifiersChanged(state.into()),
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
            WinitEvent::DeviceEvent {
                device_id: _,
                event,
            } => match event {
                // DeviceEvent::Added => todo!(),
                // DeviceEvent::Removed => todo!(),
                DeviceEvent::MouseMotion { delta: (x, y) } => Self::MouseMotion { x, y },
                DeviceEvent::MouseWheel { delta } => match delta {
                    MouseScrollDelta::LineDelta(x, y) => Self::MouseWheel {
                        x: x as f64,
                        y: y as f64,
                    },
                    MouseScrollDelta::PixelDelta(delta) => Self::MouseWheel {
                        x: delta.x,
                        y: delta.y,
                    },
                },
                // DeviceEvent::Motion { axis, value } => todo!(),
                // DeviceEvent::Button { button, state } => todo!(),
                // DeviceEvent::Key(_) => todo!(),
                // DeviceEvent::Text { codepoint } => todo!(),
                _ => Self::Unknown,
            },
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

impl From<ElementState> for InputState {
    fn from(state: ElementState) -> Self {
        match state {
            ElementState::Pressed => Self::Pressed,
            ElementState::Released => Self::Released,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[must_use]
#[non_exhaustive]
pub enum KeyCode {
    Key1,
    Key2,
    Key3,
    Key4,
    Key5,
    Key6,
    Key7,
    Key8,
    Key9,
    Key0,
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,
    Escape,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    F13,
    F14,
    F15,
    F16,
    F17,
    F18,
    F19,
    F20,
    F21,
    F22,
    F23,
    F24,
    Snapshot,
    Insert,
    Home,
    Delete,
    End,
    PageDown,
    PageUp,
    Left,
    Up,
    Right,
    Down,
    Back,
    Return,
    Space,
    Caret,
    Numpad0,
    Numpad1,
    Numpad2,
    Numpad3,
    Numpad4,
    Numpad5,
    Numpad6,
    Numpad7,
    Numpad8,
    Numpad9,
    NumpadAdd,
    NumpadDivide,
    NumpadDecimal,
    NumpadComma,
    NumpadEnter,
    NumpadEquals,
    NumpadMultiply,
    NumpadSubtract,
    Apostrophe,
    Asterisk,
    At,
    Backslash,
    Colon,
    Comma,
    Equals,
    Grave,
    LAlt,
    LBracket,
    LControl,
    LShift,
    LWin,
    Minus,
    Period,
    Plus,
    RAlt,
    RBracket,
    RControl,
    RShift,
    RWin,
    Semicolon,
    Slash,
    Tab,
    Unknown,
}

impl From<VirtualKeyCode> for KeyCode {
    fn from(keycode: VirtualKeyCode) -> Self {
        match keycode {
            VirtualKeyCode::Key1 => Self::Key1,
            VirtualKeyCode::Key2 => Self::Key2,
            VirtualKeyCode::Key3 => Self::Key3,
            VirtualKeyCode::Key4 => Self::Key4,
            VirtualKeyCode::Key5 => Self::Key5,
            VirtualKeyCode::Key6 => Self::Key6,
            VirtualKeyCode::Key7 => Self::Key7,
            VirtualKeyCode::Key8 => Self::Key8,
            VirtualKeyCode::Key9 => Self::Key9,
            VirtualKeyCode::Key0 => Self::Key0,
            VirtualKeyCode::A => Self::A,
            VirtualKeyCode::B => Self::B,
            VirtualKeyCode::C => Self::C,
            VirtualKeyCode::D => Self::D,
            VirtualKeyCode::E => Self::E,
            VirtualKeyCode::F => Self::F,
            VirtualKeyCode::G => Self::G,
            VirtualKeyCode::H => Self::H,
            VirtualKeyCode::I => Self::I,
            VirtualKeyCode::J => Self::J,
            VirtualKeyCode::K => Self::K,
            VirtualKeyCode::L => Self::L,
            VirtualKeyCode::M => Self::M,
            VirtualKeyCode::N => Self::N,
            VirtualKeyCode::O => Self::O,
            VirtualKeyCode::P => Self::P,
            VirtualKeyCode::Q => Self::Q,
            VirtualKeyCode::R => Self::R,
            VirtualKeyCode::S => Self::S,
            VirtualKeyCode::T => Self::T,
            VirtualKeyCode::U => Self::U,
            VirtualKeyCode::V => Self::V,
            VirtualKeyCode::W => Self::W,
            VirtualKeyCode::X => Self::X,
            VirtualKeyCode::Y => Self::Y,
            VirtualKeyCode::Z => Self::Z,
            VirtualKeyCode::Escape => Self::Escape,
            VirtualKeyCode::F1 => Self::F1,
            VirtualKeyCode::F2 => Self::F2,
            VirtualKeyCode::F3 => Self::F3,
            VirtualKeyCode::F4 => Self::F4,
            VirtualKeyCode::F5 => Self::F5,
            VirtualKeyCode::F6 => Self::F6,
            VirtualKeyCode::F7 => Self::F7,
            VirtualKeyCode::F8 => Self::F8,
            VirtualKeyCode::F9 => Self::F9,
            VirtualKeyCode::F10 => Self::F10,
            VirtualKeyCode::F11 => Self::F11,
            VirtualKeyCode::F12 => Self::F12,
            VirtualKeyCode::F13 => Self::F13,
            VirtualKeyCode::F14 => Self::F14,
            VirtualKeyCode::F15 => Self::F15,
            VirtualKeyCode::F16 => Self::F16,
            VirtualKeyCode::F17 => Self::F17,
            VirtualKeyCode::F18 => Self::F18,
            VirtualKeyCode::F19 => Self::F19,
            VirtualKeyCode::F20 => Self::F20,
            VirtualKeyCode::F21 => Self::F21,
            VirtualKeyCode::F22 => Self::F22,
            VirtualKeyCode::F23 => Self::F23,
            VirtualKeyCode::F24 => Self::F24,
            VirtualKeyCode::Snapshot => Self::Snapshot,
            VirtualKeyCode::Insert => Self::Insert,
            VirtualKeyCode::Home => Self::Home,
            VirtualKeyCode::Delete => Self::Delete,
            VirtualKeyCode::End => Self::End,
            VirtualKeyCode::PageDown => Self::PageDown,
            VirtualKeyCode::PageUp => Self::PageUp,
            VirtualKeyCode::Left => Self::Left,
            VirtualKeyCode::Up => Self::Up,
            VirtualKeyCode::Right => Self::Right,
            VirtualKeyCode::Down => Self::Down,
            VirtualKeyCode::Back => Self::Back,
            VirtualKeyCode::Return => Self::Return,
            VirtualKeyCode::Space => Self::Space,
            VirtualKeyCode::Caret => Self::Caret,
            VirtualKeyCode::Numpad0 => Self::Numpad0,
            VirtualKeyCode::Numpad1 => Self::Numpad1,
            VirtualKeyCode::Numpad2 => Self::Numpad2,
            VirtualKeyCode::Numpad3 => Self::Numpad3,
            VirtualKeyCode::Numpad4 => Self::Numpad4,
            VirtualKeyCode::Numpad5 => Self::Numpad5,
            VirtualKeyCode::Numpad6 => Self::Numpad6,
            VirtualKeyCode::Numpad7 => Self::Numpad7,
            VirtualKeyCode::Numpad8 => Self::Numpad8,
            VirtualKeyCode::Numpad9 => Self::Numpad9,
            VirtualKeyCode::NumpadAdd => Self::NumpadAdd,
            VirtualKeyCode::NumpadDivide => Self::NumpadDivide,
            VirtualKeyCode::NumpadDecimal => Self::NumpadDecimal,
            VirtualKeyCode::NumpadComma => Self::NumpadComma,
            VirtualKeyCode::NumpadEnter => Self::NumpadEnter,
            VirtualKeyCode::NumpadEquals => Self::NumpadEquals,
            VirtualKeyCode::NumpadMultiply => Self::NumpadMultiply,
            VirtualKeyCode::NumpadSubtract => Self::NumpadSubtract,
            VirtualKeyCode::Apostrophe => Self::Apostrophe,
            VirtualKeyCode::Asterisk => Self::Asterisk,
            VirtualKeyCode::At => Self::At,
            VirtualKeyCode::Backslash => Self::Backslash,
            VirtualKeyCode::Colon => Self::Colon,
            VirtualKeyCode::Comma => Self::Comma,
            VirtualKeyCode::Equals => Self::Equals,
            VirtualKeyCode::Grave => Self::Grave,
            VirtualKeyCode::LAlt => Self::LAlt,
            VirtualKeyCode::LBracket => Self::LBracket,
            VirtualKeyCode::LControl => Self::LControl,
            VirtualKeyCode::LShift => Self::LShift,
            VirtualKeyCode::LWin => Self::LWin,
            VirtualKeyCode::Minus => Self::Minus,
            VirtualKeyCode::Period => Self::Period,
            VirtualKeyCode::Plus => Self::Plus,
            VirtualKeyCode::RAlt => Self::RAlt,
            VirtualKeyCode::RBracket => Self::RBracket,
            VirtualKeyCode::RControl => Self::RControl,
            VirtualKeyCode::RShift => Self::RShift,
            VirtualKeyCode::RWin => Self::RWin,
            VirtualKeyCode::Semicolon => Self::Semicolon,
            VirtualKeyCode::Slash => Self::Slash,
            VirtualKeyCode::Tab => Self::Tab,
            _ => Self::Unknown,
        }
    }
}

bitflags! {
    #[derive(Default)]
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

impl From<ModifiersState> for ModifierKeys {
    fn from(state: ModifiersState) -> Self {
        let mut keys = ModifierKeys::NONE;
        keys.set(ModifierKeys::SHIFT, state.contains(ModifiersState::SHIFT));
        keys.set(ModifierKeys::CTRL, state.contains(ModifiersState::CTRL));
        keys.set(ModifierKeys::ALT, state.contains(ModifiersState::ALT));
        keys.set(ModifierKeys::GUI, state.contains(ModifiersState::LOGO));
        keys
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
