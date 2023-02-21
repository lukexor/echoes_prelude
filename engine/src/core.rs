//! Core engine traits and functionality.

use crate::{
    config::{Config, Fullscreen},
    context::Context,
    prelude::*,
    render::{RenderBackend, RenderSettings, Renderer},
    window::{
        winit::{FullscreenModeExt, PositionedExt},
        Positioned, WindowCreateInfo,
    },
    Error, Result,
};
use anyhow::Context as _;
use std::fmt::Debug;
use winit::{
    event::{Event as WinitEvent, StartCause},
    event_loop::{ControlFlow, EventLoopBuilder},
    window::WindowBuilder,
};

pub trait OnUpdate {
    type UserEvent: Copy + Debug + 'static;
    type Renderer: RenderBackend;

    /// Called on engine start for initializing resources and state.
    fn on_start(&mut self, _cx: &mut Context<Self::UserEvent, Self::Renderer>) -> Result<()> {
        Ok(())
    }

    /// Called every frame to update state and render frames.
    fn on_update(&mut self, cx: &mut Context<Self::UserEvent, Self::Renderer>) -> Result<()>;

    #[cfg(feature = "imgui")]
    fn render_imgui(
        &mut self,
        _cx: &mut Context<Self::UserEvent, Self::Renderer>,
    ) -> Result<&imgui::DrawData>;

    /// Called on engine shutdown to clean up resources.
    fn on_stop(&mut self, _cx: &mut Context<Self::UserEvent, Self::Renderer>) {}

    /// Called on every event.
    fn on_event(
        &mut self,
        _cx: &mut Context<Self::UserEvent, Self::Renderer>,
        _event: Event<Self::UserEvent>,
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

    pub fn title(&mut self, title: impl Into<String>) -> &mut Self {
        self.0.config.window_title = title.into();
        self
    }

    pub fn version(&mut self, version: impl Into<String>) -> &mut Self {
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
    version: String,
    window_create_info: WindowCreateInfo,
    config: Config,
}

impl Default for Engine {
    fn default() -> Self {
        Self {
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
                    .with_title(&self.config.window_title)
                    .with_inner_size(self.window_create_info.size)
                    .with_position(self.window_create_info.positioned.for_monitor(event_loop.primary_monitor(), self.window_create_info.size))
                    .with_resizable(self.window_create_info.resizable)
                    // TODO: Support Exclusive
                    .with_fullscreen(self
                        .config.fullscreen.and_then(|fullscreen| fullscreen.for_monitor(event_loop.primary_monitor()))
                    )
                    .build(event_loop)
                    .context("failed to create window")
                    .map_err(|err| tracing::error!("{err:?}")) else {
                        control_flow.set_exit_with_code(1);
                        return;
                    };

                // TODO: Read from saved configuration
                let settings = RenderSettings::default();
                let Ok(renderer) = Renderer::initialize(&self.config.window_title, &self.version, &window, settings)
                    .map_err(|err| tracing::error!("{err:?}")) else {
                    control_flow.set_exit_with_code(2);
                    return;
                };
                let mut context = Context::<A::UserEvent, A::Renderer>::with_user_events(self.config.clone(), window, event_proxy.clone(), renderer);

                let on_start = app.on_start(&mut context);
                if on_start.is_err() || context.should_quit() {
                    tracing::debug!("quitting after on_start with `Engine::on_stop`");
                    app.on_stop(&mut context);
                    return match on_start {
                        Ok(_) => control_flow.set_exit(),
                        Err(err) => Self::handle_error(control_flow, 1, err),
                    }
                } else {
                    cx = Some(context);
                }
            }

            if let Some(cx) = &mut cx {
                match event {
                    WinitEvent::MainEventsCleared if cx.is_running() => {
                        cx.begin_frame();
                        if let Err(err) = app.on_update(cx) {
                            return Self::handle_error(control_flow, 1, err);
                        }

                        #[cfg(feature = "imgui")]
                        let ui_data = match app.render_imgui(cx) {
                            Ok(ui_data) => ui_data,
                            Err(err) => return Self::handle_error(control_flow, 1, err),
                        };
                        if let Err(err) = cx.draw_frame(#[cfg(feature = "imgui")] ui_data) {
                            return Self::handle_error(control_flow, 2, err);
                        }
                        cx.end_frame();
                    }
                    _ => {
                        tracing::trace!("received event: {event:?}");
                        if let Ok(event) = event.try_into() {
                            cx.on_event(event);
                            app.on_event(cx, event);
                        }
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

    fn handle_error(control_flow: &mut ControlFlow, code: i32, err: Error) {
        tracing::error!("{err:?}");
        control_flow.set_exit_with_code(code);
    }
}
