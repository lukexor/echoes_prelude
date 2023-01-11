//! Core engine logic.

use crate::{
    config::{Config, Fullscreen},
    context::Context,
    prelude::{Event, Size, WindowEvent},
    render::{RenderSettings, Renderer},
    window::{
        winit::{FullscreenModeExt, PositionedExt},
        Positioned, WindowCreateInfo,
    },
    Result,
};
use anyhow::Context as _;
use std::{
    borrow::Cow,
    fmt::{self, Write},
    thread,
    time::{Duration, Instant},
};
use winit::{
    event::{Event as WinitEvent, StartCause},
    event_loop::EventLoopBuilder,
    window::{CursorGrabMode, WindowBuilder},
};

pub trait OnUpdate {
    type UserEvent: 'static + fmt::Debug;

    /// Called on engine start.
    fn on_start(&mut self, _cx: &mut Context<Self::UserEvent>) -> Result<()> {
        Ok(())
    }

    /// Called every frame.
    fn on_update(&mut self, cx: &mut Context<Self::UserEvent>) -> Result<()>;

    /// Called on engine shutdown.
    fn on_stop(&mut self, _cx: &mut Context<Self::UserEvent>) {}

    /// Called on every event.
    fn on_event(&mut self, _cx: &mut Context<Self::UserEvent>, _event: Event<Self::UserEvent>) {}
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

    pub fn inner_size(&mut self, size: impl Into<Size>) -> &mut Self {
        self.0.window_create_info.size = size.into();
        self
    }

    pub fn positioned(&mut self, position: impl Into<Positioned>) -> &mut Self {
        self.0.window_create_info.positioned = position.into();
        self
    }

    pub fn cursor_grab(&mut self, cursor_grab: bool) -> &mut Self {
        self.0.config.cursor_grab = cursor_grab;
        self
    }

    pub fn fullscreen(&mut self, fullscreen: impl Into<Option<Fullscreen>>) -> &mut Self {
        self.0.config.fullscreen = fullscreen.into();
        self
    }

    pub fn resizable(&mut self, resizable: bool) -> &mut Self {
        self.0.window_create_info.resizable = resizable;
        self
    }

    pub fn config(&mut self, config: Config) -> &mut Self {
        self.0.config = config;
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
    window_create_info: WindowCreateInfo,
    config: Config,
}

impl Default for Engine {
    fn default() -> Self {
        Self {
            title: "".into(),
            version: "1.0.0".into(),
            window_create_info: WindowCreateInfo::default(),
            config: Config::default(),
        }
    }
}

impl Engine {
    pub fn builder() -> EngineBuilder {
        EngineBuilder::default()
    }

    pub fn run<A: OnUpdate + 'static>(self, mut app: A) -> Result<()> {
        let event_loop = EventLoopBuilder::<A::UserEvent>::with_user_event().build();
        let event_proxy = event_loop.create_proxy();
        let mut cx = None;

        tracing::debug!("starting `Engine::on_update` loop.");
        event_loop.run(move |event, event_loop, control_flow| {
            if matches!(event, WinitEvent::NewEvents(StartCause::Init)) {
                let Ok(window) = WindowBuilder::new()
                    .with_title(self.title.clone())
                    .with_inner_size(self.window_create_info.size)
                    .with_position(self.window_create_info.positioned.for_monitor(event_loop.primary_monitor(), self.window_create_info.size))
                    .with_resizable(self.window_create_info.resizable)
                    // TODO: Support Exclusive
                    .with_fullscreen(self
                        .config.fullscreen.and_then(|fullscreen| fullscreen.for_monitor(event_loop.primary_monitor()))
                    )
                    .build(event_loop)
                    .context("failed to create window")
                    .map_err(|err| tracing::error!("{err}")) else {
                        control_flow.set_exit_with_code(1);
                        return;
                    };

                let settings = RenderSettings::default();
                let Ok(renderer) = Renderer::initialize(&self.title, &self.version, &window, &settings)
                    .map_err(|err| tracing::error!("{err}")) else {
                    control_flow.set_exit_with_code(2);
                    return;
                };
                let mut new_cx = Context::new(self.config, window, renderer, event_proxy.clone());

                let on_start = app.on_start(&mut new_cx);
                if on_start.is_err() || new_cx.should_quit() {
                    tracing::debug!("quitting after on_start with `Engine::on_stop`");
                    if let Err(ref err) = on_start {
                        tracing::error!("{err}");
                    }
                    app.on_stop(&mut new_cx);
                    control_flow.set_exit_with_code(3);
                } else {
                    cx = Some(new_cx);
                }
            }

            if let Some(cx) = &mut cx {
                let current_time = Instant::now();
                cx.delta_time = current_time - cx.last_frame_time;

                match event {
                    WinitEvent::MainEventsCleared if cx.is_running() => {
                        let on_update = app.on_update(cx);
                        if let Err(err) = on_update {
                            tracing::error!("{err}");
                            control_flow.set_exit_with_code(1);
                            return;
                        }

                        let end_time = Instant::now();
                        let elapsed = end_time - current_time;
                        cx.fps_timer += cx.delta_time;
                        let remaining = cx
                            .target_frame_rate
                            .checked_sub(elapsed)
                            .unwrap_or_default();
                        if remaining.as_millis() > 0 && self.config.limit_frame_rate {
                            thread::sleep(remaining - Duration::from_millis(1));
                        }
                        cx.fps_counter += 1;

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
                        tracing::trace!("received event: {event:?}");
                        let event = event.into();
                        if let Event::WindowEvent {
                            window_id: _,
                            event,
                        } = event
                        {
                            match event {
                                WindowEvent::Resized(size) => cx.on_resized(size),
                                WindowEvent::KeyboardInput {
                                    keycode: Some(keycode),
                                    state,
                                    ..
                                } => {
                                    cx.key_state.key_input(keycode, state);
                                }
                                WindowEvent::MouseInput { button, state, .. } => {
                                    cx.mouse_state.mouse_input(button, state);
                                }
                                WindowEvent::Focused(focused) => {
                                    cx.focused = focused;
                                    if self.config.cursor_grab && cx.focused {
                                        cx.window.set_cursor_visible(false);
                                        if let Err(err) = cx
                                            .window
                                            .set_cursor_grab(CursorGrabMode::Confined)
                                            .or_else(|_e| {
                                                cx.window.set_cursor_grab(CursorGrabMode::Locked)
                                            })
                                        {
                                            tracing::error!("failed to grab cursor: {err}");
                                        }
                                    }
                                }
                                WindowEvent::CloseRequested | WindowEvent::Destroyed => cx.quit(),
                                WindowEvent::ModifiersChanged(modifiers) => {
                                    cx.modifiers_state = modifiers;
                                }
                                _ => (),
                            }
                        }
                        app.on_event(cx, event);
                    }
                }

                if cx.should_quit() && !cx.is_quitting {
                    cx.is_quitting = true;
                    tracing::debug!("shutting down with `Engine::on_stop`");
                    app.on_stop(cx);
                    // on_stop allows aborting the quit request
                    if cx.should_quit() {
                        tracing::debug!("shutting down...");
                        control_flow.set_exit();
                    } else {
                        cx.is_quitting = false;
                    }
                }
            }
        });
    }
}
