//! Event types.

use crate::window::{PhysicalPosition, PhysicalSize, WindowId};
use bitflags::bitflags;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[must_use]
pub struct DeviceId(pub(crate) ::winit::event::DeviceId);

pub type Scancode = u32;

#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub enum Event<T> {
    WindowEvent {
        window_id: WindowId,
        event: WindowEvent,
    },
    DeviceEvent {
        device_id: DeviceId,
        event: DeviceEvent,
    },
    UserEvent(T),
    Suspended,
    Resumed,
    Unhandled,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub enum WindowEvent {
    Resized(PhysicalSize),
    Moved(PhysicalPosition<i32>),
    CloseRequested,
    Destroyed,
    Focused(bool),
    KeyboardInput {
        device_id: DeviceId,
        scancode: Scancode,
        keycode: Option<KeyCode>,
        state: InputState,
        is_synthetic: bool,
    },
    ModifiersChanged(ModifierKeys),
    CursorMoved {
        device_id: DeviceId,
        position: PhysicalPosition<f64>,
    },
    CursorEntered {
        device_id: DeviceId,
    },
    CursorLeft {
        device_id: DeviceId,
    },
    MouseWheel {
        device_id: DeviceId,
        delta: MouseScrollDelta,
        phase: TouchPhase,
    },
    MouseInput {
        device_id: DeviceId,
        button: MouseButton,
        state: InputState,
    },
    AxisMotion {
        device_id: DeviceId,
        axis: u32,
        value: f64,
    },
    Touch {
        device_id: DeviceId,
        phase: TouchPhase,
        position: PhysicalPosition<f64>,
        id: u64,
    },
    ScaleFactorChanged {
        scale_factor: f64,
        new_inner_size: PhysicalSize,
    },
    Occluded(bool),
    Unhandled,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub enum DeviceEvent {
    MouseMotion {
        delta: (f64, f64),
    },
    MouseWheel {
        delta: MouseScrollDelta,
    },
    Motion {
        axis: u32,
        value: f64,
    },
    Button {
        button: u32,
        state: InputState,
    },
    Key {
        scancode: Scancode,
        keycode: Option<KeyCode>,
        state: InputState,
    },
    Text {
        codepoint: char,
    },
    Unhandled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use]
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
    Unhandled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    Other(u16),
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use]
pub enum MouseScrollDelta {
    LineDelta(f32, f32),
    PixelDelta(PhysicalPosition<f64>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TouchPhase {
    Started,
    Moved,
    Ended,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use]
pub enum InputState {
    Pressed,
    Released,
}

bitflags! {
    #[derive(Default)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
