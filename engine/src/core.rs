use crate::{
    config::{Config, FullscreenMode},
    context::Context,
    prelude::{
        CursorGrabMode, Event, PhysicalPosition, PhysicalSize, Position, Size, WindowBuilder,
        WindowEvent,
    },
    renderer::{Renderer, Shader},
    Result,
};
use anyhow::Context as _;
use std::{
    borrow::Cow,
    fmt::{self, Write},
    thread,
    time::{Duration, Instant},
};
use winit::{event::StartCause, event_loop::EventLoopBuilder};

pub trait Update {
    type UserEvent: 'static + fmt::Debug;

    /// Called on engine start.
    fn on_start(&mut self, _cx: &mut Context) -> Result<()> {
        Ok(())
    }

    /// Called every frame.
    fn on_update(&mut self, delta_time: f32, cx: &mut Context) -> Result<()>;

    /// Called on engine shutdown.
    fn on_stop(&mut self, _cx: &mut Context) {}

    /// Called on every event.
    fn on_event(
        &mut self,
        _delta_time: f32,
        _event: Event<'_, Self::UserEvent>,
        _cx: &mut Context,
    ) {
    }
}

#[derive(Default, Debug)]
#[must_use]
pub struct EngineBuilder(Engine);

impl EngineBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn title(&mut self, title: impl Into<Cow<'static, str>>) -> &mut Self {
        self.0.title = title.into();
        self
    }

    pub fn version(&mut self, version: impl Into<Cow<'static, str>>) -> &mut Self {
        self.0.version = version.into();
        self
    }

    pub fn size(&mut self, size: impl Into<Size>) -> &mut Self {
        self.0.size = size.into();
        self
    }

    pub fn position(&mut self, position: impl Into<Position>) -> &mut Self {
        self.0.position = position.into();
        self
    }

    pub fn cursor_grab(&mut self, cursor_grab: bool) -> &mut Self {
        self.0.cursor_grab = cursor_grab;
        self
    }

    pub fn fullscreen(&mut self, fullscreen: FullscreenMode) -> &mut Self {
        self.0.config.fullscreen_mode = fullscreen;
        self
    }

    pub fn resizable(&mut self, resizable: bool) -> &mut Self {
        self.0.resizable = resizable;
        self
    }

    pub fn config(&mut self, config: Config) -> &mut Self {
        self.0.config = config;
        self
    }

    pub fn shader(&mut self, shader: Shader) -> &mut Self {
        self.0.shaders.push(shader);
        self
    }

    pub fn shaders(&mut self, shaders: impl IntoIterator<Item = Shader>) -> &mut Self {
        self.0.shaders.extend(shaders);
        self
    }

    pub fn build(&mut self) -> Engine {
        self.0.clone()
    }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct Engine {
    title: Cow<'static, str>,
    version: Cow<'static, str>,
    size: Size,
    position: Position,
    cursor_grab: bool,
    resizable: bool,
    config: Config,
    shaders: Vec<Shader>,
}

impl Default for Engine {
    fn default() -> Self {
        Self {
            title: "".into(),
            version: "1.0.0".into(),
            size: PhysicalSize::new(1024, 768).into(),
            position: PhysicalPosition::<u32>::default().into(),
            cursor_grab: true,
            resizable: true,
            config: Config::default(),
            shaders: vec![],
        }
    }
}

impl Engine {
    pub fn builder() -> EngineBuilder {
        EngineBuilder::default()
    }

    pub fn run<A: Update + 'static>(self, mut app: A) -> Result<()> {
        let event_loop = EventLoopBuilder::<A::UserEvent>::with_user_event().build();
        let mut cx = None;

        let start = Instant::now();

        log::debug!("starting `Engine::on_update` loop.");
        event_loop.run(move |event, event_loop, control_flow| {
            if start.elapsed() > Duration::from_secs(30) {
                control_flow.set_exit();
            }

            if matches!(event, Event::NewEvents(StartCause::Init)) {
                let Ok(window) = WindowBuilder::new()
                    .with_title(self.title.clone())
                    .with_inner_size(self.size)
                    .with_position(self.position)
                    // TODO: Support Exclusive
                    .with_fullscreen(self
                        .config.fullscreen_mode.as_monitor(event_loop.primary_monitor())
                    )
                    .with_resizable(self.resizable)
                    .build(event_loop)
                    .context("failed to create window") else {
                        control_flow.set_exit_with_code(1);
                        return;
                    };
                let Ok(renderer) =
                    Renderer::initialize(&self.title, &self.version, &window, &self.shaders) else {
                        control_flow.set_exit_with_code(2);
                        return;
                    };
                let mut new_cx = Context::new(self.config, window, renderer);

                let on_start = app.on_start(&mut new_cx);
                if on_start.is_err() || new_cx.should_quit() {
                    log::debug!("quitting after on_start with `Engine::on_stop`");
                    if let Err(ref err) = on_start {
                        log::error!("{err}");
                    }
                    app.on_stop(&mut new_cx);
                    control_flow.set_exit_with_code(3);
                }
                cx = Some(new_cx);
            }

            if let Some(cx) = &mut cx {
                let current_time = Instant::now();
                let delta_time = current_time - cx.last_frame_time;

                match event {
                    Event::MainEventsCleared if cx.is_running() => {
                        let on_update = app.on_update(delta_time.as_secs_f32(), cx);
                        if let Err(err) = on_update {
                            log::error!("{err:?}");
                            control_flow.set_exit_with_code(1);
                            return;
                        }

                        let end_time = Instant::now();
                        let elapsed = end_time - current_time;
                        cx.fps_timer += delta_time;
                        let remaining = cx
                            .target_frame_rate
                            .checked_sub(elapsed)
                            .unwrap_or_default();
                        if remaining.as_millis() > 0 {
                            if self.config.limit_frame_rate {
                                thread::sleep(remaining - Duration::from_millis(1));
                            }
                            cx.fps_counter += 1;
                        }

                        let one_second = Duration::from_secs(1);
                        if cx.fps_timer > one_second {
                            cx.title.clear();
                            let _ = write!(cx.title, "{} - FPS: {}", self.title, cx.fps_counter);
                            cx.window.set_title(&cx.title);
                            cx.fps_timer -= one_second;
                            cx.fps_counter = 0;
                        }

                        cx.key_state.update();
                        cx.mouse_state.update();

                        cx.last_frame_time = current_time;
                    }
                    _ => {
                        log::trace!("received event: {event:?}");
                        if let Event::WindowEvent {
                            window_id: _,
                            ref event,
                        } = event
                        {
                            match event {
                                WindowEvent::Resized(size) => cx.on_resized(*size),
                                WindowEvent::KeyboardInput { input, .. } => {
                                    if let Some(keycode) = input.virtual_keycode {
                                        cx.key_state.key_input(keycode, input.state);
                                    }
                                }
                                WindowEvent::MouseInput { button, state, .. } => {
                                    cx.mouse_state.mouse_input(*button, *state);
                                }
                                WindowEvent::Focused(focused) => {
                                    cx.focused = *focused;
                                    if self.cursor_grab && cx.focused {
                                        cx.window.set_cursor_visible(false);
                                        if let Err(err) = cx
                                            .window
                                            .set_cursor_grab(CursorGrabMode::Confined)
                                            .or_else(|_e| {
                                                cx.window.set_cursor_grab(CursorGrabMode::Locked)
                                            })
                                        {
                                            log::error!("failed to grab cursor: {err:?}");
                                        }
                                    }
                                }
                                WindowEvent::CloseRequested | WindowEvent::Destroyed => cx.quit(),
                                WindowEvent::ModifiersChanged(modifiers) => {
                                    cx.modifiers_state = *modifiers;
                                }
                                _ => (),
                            }
                        }
                        app.on_event(delta_time.as_secs_f32(), event, cx);
                    }
                }

                if cx.should_quit() {
                    log::debug!("shutting down with `Engine::on_stop`");
                    app.on_stop(cx);
                    if cx.should_quit() {
                        log::info!("shutting down...");
                        control_flow.set_exit();
                    }
                }
            }
        });
    }
}
