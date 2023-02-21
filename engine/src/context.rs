//! Engine context.

use self::input::{KeyState, MouseState};
use crate::{
    config::{Config, Fullscreen},
    prelude::*,
    render::{DrawCmd, DrawData, RenderBackend, Renderer},
    window::{winit::FullscreenModeExt, EventLoopProxy, Window},
    Result,
};
use std::{
    fmt::{Debug, Write},
    time::{Duration, Instant},
};

mod input;

#[derive(Debug)]
#[must_use]
pub struct Context<T: 'static, R> {
    pub(crate) title: String,
    pub(crate) window: Window,
    pub(crate) focused: bool,
    pub(crate) start: Instant,
    pub(crate) frame_start: Option<Instant>,
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
    pub(crate) draw_cmds: Vec<DrawCmd>,
    pub(crate) config: Config,
    pub(crate) event_proxy: Option<EventLoopProxy<T>>,
    pub(crate) renderer: Renderer<R>,
}

impl<T, R> Context<T, R> {
    /// Create a new engine `Context`.
    #[inline]
    pub fn new(config: Config, window: Window, renderer: Renderer<R>) -> Self {
        Self {
            title: String::new(),
            window,
            focused: false,
            start: Instant::now(),
            frame_start: None,
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
            draw_cmds: vec![],
            config,
            event_proxy: None,
            renderer,
        }
    }

    /// Create a new engine `Context` with an event proxy for custom events.
    #[inline]
    pub fn with_user_events(
        config: Config,
        window: Window,
        event_proxy: EventLoopProxy<T>,
        renderer: Renderer<R>,
    ) -> Self {
        let mut cx = Self::new(config, window, renderer);
        cx.event_proxy = Some(event_proxy);
        cx
    }

    /// Returns the delta time in seconds since last frame.
    #[inline]
    #[must_use]
    pub fn delta_time(&self) -> Duration {
        self.delta_time
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

    /// Get the primary window.
    #[inline]
    #[must_use]
    pub fn window(&self) -> &Window {
        &self.window
    }

    /// Get the configured window title.
    #[inline]
    #[must_use]
    pub fn window_title(&self) -> &str {
        &self.config.window_title
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

    /// Send a custom user event.
    #[inline]
    pub fn send_event(&mut self, event: T)
    where
        T: Copy + Debug,
    {
        match &self.event_proxy {
            Some(event_proxy) => {
                if event_proxy.send_event(event).is_err() {
                    tracing::warn!("custom event `{event:?}` sent while engine is quitting");
                }
            }
            None => tracing::warn!("custom events are not enabled"),
        }
    }

    /// Toggle whether the cursor is grabbed by the focused window.
    #[inline]
    pub fn set_cursor_grab(&mut self, cursor_grab: bool) {
        use winit::window::CursorGrabMode;

        self.config.cursor_grab = cursor_grab;
        if cursor_grab {
            if let Err(err) = self
                .window
                .set_cursor_grab(CursorGrabMode::Confined)
                .or_else(|_e| self.window.set_cursor_grab(CursorGrabMode::Locked))
            {
                tracing::error!("failed to grab cursor: {err:?}");
            }
            self.window.set_cursor_visible(false);
        } else {
            if let Err(err) = self.window.set_cursor_grab(CursorGrabMode::None) {
                tracing::error!("failed to ungrab cursor: {err:?}");
            }
            self.window.set_cursor_visible(true);
        }
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

    /// Begin a frame.
    #[inline]
    pub fn begin_frame(&mut self) {
        self.frame_start = Some(Instant::now());
        self.delta_time = self.last_frame_time.elapsed();
    }

    /// Draw a frame.
    #[inline]
    pub fn draw_frame(
        &mut self,
        #[cfg(feature = "imgui")] imgui_draw_data: &::imgui::DrawData,
    ) -> Result<()>
    where
        R: RenderBackend,
    {
        let mut draw_data = DrawData {
            commands: &self.draw_cmds,
            #[cfg(feature = "imgui")]
            imgui: imgui_draw_data,
        };
        self.renderer.draw_frame(&mut draw_data)?;
        self.draw_cmds.clear();
        Ok(())
    }

    /// End a frame.
    #[inline]
    pub fn end_frame(&mut self) {
        self.draw_cmds.clear();
        let Some(frame_start) = self.frame_start else {
            panic!("must call `Context::begin_frame` first before calling `Context::end_frame`");
        };
        let elapsed = frame_start.elapsed();
        self.fps_timer += self.delta_time;
        let remaining = self
            .target_frame_rate
            .checked_sub(elapsed)
            .unwrap_or_default();
        if remaining.as_millis() > 0 && self.config.limit_frame_rate {
            std::thread::sleep(remaining - Duration::from_millis(1));
        }

        self.fps_counter += 1;

        let one_second = Duration::from_secs(1);
        if self.fps_timer > one_second {
            self.title.clear();
            let _ = write!(
                self.title,
                "{} - FPS: {}",
                self.config.window_title, self.fps_counter
            );
            self.window.set_title(&self.title);
            self.fps_timer -= one_second;
            self.fps_counter = 0;
        }

        self.key_state.update();
        self.mouse_state.update();

        self.last_frame_time = frame_start;
    }

    /// Called on every event.
    #[inline]
    pub fn on_event(&mut self, event: Event<T>)
    where
        T: Copy,
        R: RenderBackend,
    {
        if let Event::WindowEvent {
            window_id: _,
            event,
        } = event
        {
            match event {
                WindowEvent::KeyboardInput {
                    keycode: Some(keycode),
                    state,
                    ..
                } => {
                    self.key_state.key_input(keycode, state);
                }
                WindowEvent::MouseInput { button, state, .. } => {
                    self.mouse_state.mouse_input(button, state);
                }
                WindowEvent::Resized(size) => self.renderer.on_resized(size),
                WindowEvent::Focused(focused) => {
                    self.focused = focused;
                    if self.config.cursor_grab && self.focused {
                        self.set_cursor_grab(true);
                    } else {
                        self.set_cursor_grab(false);
                    }
                }
                WindowEvent::CloseRequested | WindowEvent::Destroyed => self.should_quit = true,
                WindowEvent::ModifiersChanged(modifiers) => {
                    self.modifiers_state = modifiers;
                }
                _ => (),
            }
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
}
