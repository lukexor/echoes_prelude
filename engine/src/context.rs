//! Engine context.

use self::input::{KeyState, MouseState};
use crate::{
    config::{Config, Fullscreen},
    prelude::{InputState, KeyCode, ModifierKeys, MouseButton},
    render::Renderer,
    window::{winit::FullscreenModeExt, EventLoopProxy, Window},
};
use std::time::{Duration, Instant};

mod input;

#[derive(Debug)]
#[must_use]
pub struct Context<T: 'static> {
    pub(crate) title: String,
    pub(crate) window: Window,
    pub(crate) focused: bool,
    pub(crate) start: Instant,
    pub(crate) last_frame_time: Instant,
    pub(crate) delta_time: Duration,
    pub(crate) target_frame_rate: Duration,
    pub(crate) fps_counter: usize,
    pub(crate) fps_timer: Duration,
    pub(crate) suspended: bool,
    pub(crate) should_quit: bool,
    pub(crate) is_quitting: bool,
    pub(crate) key_state: KeyState,
    pub(crate) mouse_state: MouseState,
    pub(crate) modifiers_state: ModifierKeys,
    pub(crate) config: Config,
    pub(crate) renderer: Renderer,
    pub(crate) event_proxy: EventLoopProxy<T>,
}

impl<T> Context<T> {
    /// Create a new engine `Context`.
    pub(crate) fn new(
        config: Config,
        window: Window,
        renderer: Renderer,
        event_proxy: EventLoopProxy<T>,
    ) -> Self {
        Self {
            title: String::new(),
            window,
            focused: false,
            start: Instant::now(),
            last_frame_time: Instant::now(),
            delta_time: Duration::default(),
            target_frame_rate: Duration::from_secs(1) / config.target_fps,
            fps_counter: 0,
            fps_timer: Duration::default(),
            suspended: false,
            should_quit: false,
            is_quitting: false,
            key_state: KeyState::default(),
            mouse_state: MouseState::default(),
            modifiers_state: ModifierKeys::empty(),
            config,
            renderer,
            event_proxy,
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

    /// Returns the delta time in seconds since last frame.
    #[inline]
    #[must_use]
    pub fn delta_time(&self) -> f32 {
        self.delta_time.as_secs_f32()
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

    /// Begin engine shutdown.
    #[inline]
    pub fn send_event(&mut self, event: T) {
        // If event loop is closed, it means app is quitting
        let _ = self.event_proxy.send_event(event);
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
    pub fn toggle_fullscreen(&mut self, fullscreen: Fullscreen) {
        let mode = match self.window.fullscreen() {
            Some(_) => {
                self.config.fullscreen = None;
                None
            }
            None => {
                self.config.fullscreen = Some(fullscreen);
                fullscreen.for_monitor(self.window.primary_monitor())
            }
        };
        self.window.set_fullscreen(mode);
    }

    /// Whether a given key is pressed.
    #[inline]
    #[must_use]
    pub fn key_down(&self, keycode: KeyCode) -> bool {
        self.key_state.state(keycode) == InputState::Pressed
    }

    /// Whether a given key is released.
    #[inline]
    #[must_use]
    pub fn key_up(&self, keycode: KeyCode) -> bool {
        self.key_state.state(keycode) == InputState::Released
    }

    /// Whether a given key was typed this frame.
    #[inline]
    #[must_use]
    pub fn key_typed(&self, keycode: KeyCode) -> bool {
        self.key_state.previous_state(keycode) == InputState::Pressed
            && self.key_state.state(keycode) == InputState::Released
    }

    /// Whether a given key was pressed last frame.
    #[inline]
    #[must_use]
    pub fn key_was_down(&self, keycode: KeyCode) -> bool {
        self.key_state.previous_state(keycode) == InputState::Pressed
    }

    /// Whether a given key was released last frame.
    #[inline]
    #[must_use]
    pub fn key_was_up(&self, keycode: KeyCode) -> bool {
        self.key_state.previous_state(keycode) == InputState::Released
    }

    /// Whether a given mouse button is pressed.
    #[inline]
    #[must_use]
    pub fn mouse_down(&self, button: MouseButton) -> bool {
        self.mouse_state.state(button) == InputState::Pressed
    }

    /// Whether a given mouse is released.
    #[inline]
    #[must_use]
    pub fn mouse_up(&self, button: MouseButton) -> bool {
        self.mouse_state.state(button) == InputState::Released
    }

    /// Whether a given mouse was clicked this frame.
    #[inline]
    #[must_use]
    pub fn mouse_clicked(&self, button: MouseButton) -> bool {
        self.mouse_state.previous_state(button) == InputState::Pressed
            && self.mouse_state.state(button) == InputState::Released
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
        self.mouse_state.previous_state(button) == InputState::Pressed
    }

    /// Whether a given mouse was released last frame.
    #[inline]
    #[must_use]
    pub fn mouse_was_up(&self, button: MouseButton) -> bool {
        self.mouse_state.previous_state(button) == InputState::Released
    }

    /// Whether a given set of modifier keys are pressed.
    #[inline]
    #[must_use]
    pub fn modifiers_down(&self, modifiers: ModifierKeys) -> bool {
        self.modifiers_state.contains(modifiers)
    }
}
