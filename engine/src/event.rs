//! Event types.

use bitflags::bitflags;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum Event<T> {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum WindowEvent {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum DeviceEvent {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum VirtualKeyCode {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum MouseButton {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum InputState {}

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
