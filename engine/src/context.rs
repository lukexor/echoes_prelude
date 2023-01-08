//! Engine context

use crate::{
    config::Config,
    prelude::{ElementState, ModifiersState, MouseButton, PhysicalSize, VirtualKeyCode, Window},
    renderer::{RenderState, Renderer},
    Result,
};
use std::time::{Duration, Instant};

#[derive(Debug)]
#[must_use]
pub struct Context {
    pub(crate) title: String,
    pub(crate) window: Window,
    pub(crate) focused: bool,
    pub(crate) start: Instant,
    pub(crate) last_frame_time: Instant,
    pub(crate) target_frame_rate: Duration,
    pub(crate) fps_counter: usize,
    pub(crate) fps_timer: Duration,
    pub(crate) suspended: bool,
    pub(crate) should_quit: bool,
    pub(crate) key_state: KeyState,
    pub(crate) mouse_state: MouseState,
    pub(crate) modifiers_state: ModifiersState,
    pub(crate) config: Config,
    pub(crate) renderer: Renderer,
}

impl Context {
    /// Create a new engine `Context`.
    pub(crate) fn new(config: Config, window: Window, renderer: Renderer) -> Self {
        Self {
            title: String::new(),
            window,
            focused: false,
            start: Instant::now(),
            last_frame_time: Instant::now(),
            target_frame_rate: Duration::from_secs(1) / config.target_fps,
            fps_counter: 0,
            fps_timer: Duration::default(),
            suspended: false,
            should_quit: false,
            key_state: KeyState::default(),
            mouse_state: MouseState::default(),
            modifiers_state: ModifiersState::empty(),
            config,
            renderer,
        }
    }

    /// Whether the engine should quit.
    #[inline]
    #[must_use]
    pub fn should_quit(&self) -> bool {
        self.should_quit
    }

    /// Whether the engine is actively running.
    #[inline]
    #[must_use]
    pub fn is_running(&self) -> bool {
        !self.suspended
    }

    /// Whether any engine window has focus.
    #[inline]
    #[must_use]
    pub fn focused(&self) -> bool {
        self.focused
    }

    /// Begin engine shutdown.
    #[inline]
    pub fn quit(&mut self) {
        self.should_quit = true;
    }

    /// Handle window resized event.
    #[inline]
    pub fn on_resized(&mut self, size: PhysicalSize<u32>) {
        self.renderer.on_resized(size);
    }

    /// Draw a frame to the screen.
    pub fn draw_frame(&mut self, state: RenderState) -> Result<()> {
        Ok(self.renderer.draw_frame(state)?)
    }

    /// Time since the engine started running.
    #[inline]
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// The current window width.
    #[inline]
    #[must_use]
    pub fn width(&self) -> u32 {
        self.window.inner_size().width
    }

    /// The current window height.
    #[inline]
    #[must_use]
    pub fn height(&self) -> u32 {
        self.window.inner_size().height
    }

    /// Toggle fullscreen.
    #[inline]
    pub fn toggle_fullscreen(&self) {
        let fullscreen = self
            .window
            .fullscreen()
            .is_none()
            .then(|| {
                self.config
                    .fullscreen_mode
                    .as_monitor(self.window.primary_monitor())
            })
            .flatten();
        self.window.set_fullscreen(fullscreen);
    }

    /// Whether a given key is pressed.
    #[inline]
    #[must_use]
    pub fn key_down(&self, keycode: VirtualKeyCode) -> bool {
        self.key_state.state(keycode) == ElementState::Pressed
    }

    /// Whether a given key is released.
    #[inline]
    #[must_use]
    pub fn key_up(&self, keycode: VirtualKeyCode) -> bool {
        self.key_state.state(keycode) == ElementState::Released
    }

    /// Whether a given key was typed this frame.
    #[inline]
    #[must_use]
    pub fn key_typed(&self, keycode: VirtualKeyCode) -> bool {
        self.key_state.previous_state(keycode) == ElementState::Pressed
            && self.key_state.state(keycode) == ElementState::Released
    }

    /// Whether a given key was pressed last frame.
    #[inline]
    #[must_use]
    pub fn key_was_down(&self, keycode: VirtualKeyCode) -> bool {
        self.key_state.previous_state(keycode) == ElementState::Pressed
    }

    /// Whether a given key was released last frame.
    #[inline]
    #[must_use]
    pub fn key_was_up(&self, keycode: VirtualKeyCode) -> bool {
        self.key_state.previous_state(keycode) == ElementState::Released
    }

    /// Whether a given mouse button is pressed.
    #[inline]
    #[must_use]
    pub fn mouse_down(&self, button: MouseButton) -> bool {
        self.mouse_state.state(button) == ElementState::Pressed
    }

    /// Whether a given mouse is released.
    #[inline]
    #[must_use]
    pub fn mouse_up(&self, button: MouseButton) -> bool {
        self.mouse_state.state(button) == ElementState::Released
    }

    /// Whether a given mouse was clicked this frame.
    #[inline]
    #[must_use]
    pub fn mouse_clicked(&self, button: MouseButton) -> bool {
        self.mouse_state.previous_state(button) == ElementState::Pressed
            && self.mouse_state.state(button) == ElementState::Released
    }

    /// Whether a given mouse was double-clicked this frame.
    #[inline]
    #[must_use]
    pub fn mouse_double_clicked(&self, button: MouseButton) -> bool {
        self.mouse_clicked(button)
            && self.mouse_state.last_click.map_or(false, |last_click| {
                last_click.elapsed() < self.config.double_click_speed
            })
    }

    /// Whether a given mouse was pressed last frame.
    #[inline]
    #[must_use]
    pub fn mouse_was_down(&self, button: MouseButton) -> bool {
        self.mouse_state.previous_state(button) == ElementState::Pressed
    }

    /// Whether a given mouse was released last frame.
    #[inline]
    #[must_use]
    pub fn mouse_was_up(&self, button: MouseButton) -> bool {
        self.mouse_state.previous_state(button) == ElementState::Released
    }

    /// Whether a given set of modifier keys are pressed.
    #[inline]
    #[must_use]
    pub fn modifiers_down(&self, modifiers: ModifiersState) -> bool {
        self.modifiers_state.contains(modifiers)
    }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct KeyState {
    state: Vec<ElementState>,
    previous_state: Vec<ElementState>,
}

impl Default for KeyState {
    fn default() -> Self {
        // FIXME: Replace with https://doc.rust-lang.org/std/mem/fn.variant_count.html when stable
        let size = 256;
        Self {
            state: vec![ElementState::Released; size],
            previous_state: vec![ElementState::Released; size],
        }
    }
}

impl KeyState {
    #[inline]
    pub(crate) fn key_input(&mut self, keycode: VirtualKeyCode, state: ElementState) {
        self.state[Self::keycode_index(keycode)] = state;
    }

    #[inline]
    pub(crate) fn update(&mut self) {
        self.previous_state = self.state.clone();
    }

    #[inline]
    pub(crate) fn state(&self, keycode: VirtualKeyCode) -> ElementState {
        self.state[Self::keycode_index(keycode)]
    }

    #[inline]
    pub(crate) fn previous_state(&self, keycode: VirtualKeyCode) -> ElementState {
        self.previous_state[Self::keycode_index(keycode)]
    }

    #[inline]
    fn keycode_index(keycode: VirtualKeyCode) -> usize {
        keycode as usize
    }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct MouseState {
    previous_state: Vec<ElementState>,
    state: Vec<ElementState>,
    last_click: Option<Instant>,
}

impl Default for MouseState {
    fn default() -> Self {
        // FIXME: Replace with https://doc.rust-lang.org/std/mem/fn.variant_count.html when stable
        let size = 16;
        Self {
            state: vec![ElementState::Released; size],
            previous_state: vec![ElementState::Released; size],
            last_click: None,
        }
    }
}

impl MouseState {
    pub(crate) fn mouse_input(&mut self, button: MouseButton, state: ElementState) {
        let id = Self::button_index(button);
        self.previous_state[id] = self.state[id];
        self.state[id] = state;
        if self.previous_state[id] == ElementState::Pressed
            && self.state[id] == ElementState::Released
        {
            self.last_click = Some(Instant::now());
        }
    }

    #[inline]
    pub(crate) fn update(&mut self) {
        self.previous_state = self.state.clone();
    }

    #[inline]
    pub(crate) fn state(&self, button: MouseButton) -> ElementState {
        self.state[Self::button_index(button)]
    }

    #[inline]
    pub(crate) fn previous_state(&self, button: MouseButton) -> ElementState {
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
