use crate::event::{InputState, KeyCode, MouseButton};
use std::time::Instant;

#[derive(Debug, Copy, Clone)]
#[must_use]
pub(crate) struct KeyState {
    pub(crate) state: [InputState; Self::MAX_SIZE],
    pub(crate) previous_state: [InputState; Self::MAX_SIZE],
}

impl Default for KeyState {
    fn default() -> Self {
        Self {
            state: [InputState::Released; Self::MAX_SIZE],
            previous_state: [InputState::Released; Self::MAX_SIZE],
        }
    }
}

impl KeyState {
    // FIXME: Replace with https://doc.rust-lang.org/std/mem/fn.variant_count.html when stable
    const MAX_SIZE: usize = 256;

    #[inline]
    pub(crate) fn key_input(&mut self, keycode: KeyCode, state: InputState) {
        self.state[Self::keycode_index(keycode)] = state;
    }

    #[inline]
    pub(crate) fn update(&mut self) {
        self.previous_state = self.state;
    }

    #[inline]
    pub(crate) fn state(&self, keycode: KeyCode) -> InputState {
        self.state[Self::keycode_index(keycode)]
    }

    #[inline]
    pub(crate) fn previous_state(&self, keycode: KeyCode) -> InputState {
        self.previous_state[Self::keycode_index(keycode)]
    }

    #[inline]
    fn keycode_index(keycode: KeyCode) -> usize {
        keycode as usize
    }
}

#[derive(Debug, Copy, Clone)]
#[must_use]
pub(crate) struct MouseState {
    pub(crate) previous_state: [InputState; Self::MAX_SIZE],
    pub(crate) state: [InputState; Self::MAX_SIZE],
    pub(crate) last_click: Option<Instant>,
}

impl Default for MouseState {
    fn default() -> Self {
        Self {
            state: [InputState::Released; Self::MAX_SIZE],
            previous_state: [InputState::Released; Self::MAX_SIZE],
            last_click: None,
        }
    }
}

impl MouseState {
    // FIXME: Replace with https://doc.rust-lang.org/std/mem/fn.variant_count.html when stable
    const MAX_SIZE: usize = 16;

    pub(crate) fn mouse_input(&mut self, button: MouseButton, state: InputState) {
        let id = Self::button_index(button);
        self.previous_state[id] = self.state[id];
        self.state[id] = state;
        if self.previous_state[id] == InputState::Pressed && self.state[id] == InputState::Released
        {
            self.last_click = Some(Instant::now());
        }
    }

    #[inline]
    pub(crate) fn update(&mut self) {
        self.previous_state = self.state;
    }

    #[inline]
    pub(crate) fn state(&self, button: MouseButton) -> InputState {
        self.state[Self::button_index(button)]
    }

    #[inline]
    pub(crate) fn previous_state(&self, button: MouseButton) -> InputState {
        self.previous_state[Self::button_index(button)]
    }

    #[inline]
    fn button_index(button: MouseButton) -> usize {
        match button {
            MouseButton::Left => 0,
            MouseButton::Right => 1,
            MouseButton::Middle => 2,
            MouseButton::Other(button) => 3 + button as usize,
        }
    }
}
