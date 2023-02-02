//! Core engine logic.

use crate::{
    config::{Config, Fullscreen},
    context::{Context, EngineContext},
    prelude::*,
    render::{RenderSettings, Renderer},
    window::{
        winit::{FullscreenModeExt, PositionedExt},
        Positioned, WindowCreateInfo,
    },
    Result,
};
use anyhow::Context as _;
use std::{fmt::Debug, time::Instant};
use winit::{
    event::{Event as WinitEvent, StartCause},
    event_loop::EventLoopBuilder,
    window::WindowBuilder,
};

pub trait OnUpdate {
    type UserEvent: Copy + Debug + 'static;

    /// Called on engine start for initializing resources and state.
    fn on_start(&mut self, _cx: &mut Context<'_, Self::UserEvent>) -> Result<()> {
        Ok(())
    }

    /// Called every frame to update state and render frames.
    fn on_update(&mut self, cx: &mut Context<'_, Self::UserEvent>) -> Result<()>;

    /// Called on engine shutdown to clean up resources.
    fn on_stop(&mut self, _cx: &mut Context<'_, Self::UserEvent>) {}

    /// Called on every event.
    fn on_event(&mut self, _cx: &mut Context<'_, Self::UserEvent>, _event: Event<Self::UserEvent>) {
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

#[derive(Debug)]
#[must_use]
pub(crate) struct EngineState<T: 'static> {
    engine_cx: EngineContext<T>,
    renderer: Renderer,
    #[cfg(feature = "imgui")]
    imgui: imgui::ImGui,
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
        let mut engine_state = None;

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
                    .map_err(|err| tracing::error!("{err}")) else {
                        control_flow.set_exit_with_code(1);
                        return;
                    };

                let mut engine_cx = EngineContext::<A::UserEvent>::with_user_events(self.config.clone(), window, event_proxy.clone());
                // TODO: Read from saved configuration
                let settings = RenderSettings::default();
                #[cfg(feature = "imgui")]
                let mut imgui = imgui::ImGui::initialize(engine_cx.window());
                let Ok(renderer) = Renderer::initialize(engine_cx.window_title(), &self.version, engine_cx.window(), settings, #[cfg(feature = "imgui")] &mut imgui)
                    .map_err(|err| tracing::error!("{err}")) else {
                    control_flow.set_exit_with_code(2);
                    return;
                };

                let on_start = app.on_start(&mut engine_cx.context());
                if on_start.is_err() || engine_cx.should_quit() {
                    tracing::debug!("quitting after on_start with `Engine::on_stop`");
                    if let Err(ref err) = on_start {
                        tracing::error!("{err}");
                    }
                    app.on_stop(&mut engine_cx.context());
                    control_flow.set_exit_with_code(3);
                } else {
                    engine_state = Some(EngineState { engine_cx, renderer, #[cfg(feature = "imgui")] imgui });
                }
            }

            if let Some(EngineState { engine_cx, renderer, #[cfg(feature = "imgui")] imgui }) = &mut engine_state {
                match event {
                    WinitEvent::MainEventsCleared if engine_cx.is_running() => {
                        let now = Instant::now();
                        let mut cx = engine_cx.begin_frame(now);
                        #[cfg(feature = "imgui")]
                        {
                            imgui.begin_frame(cx.delta_time(), cx.window());
                            cx.ui = Some(imgui.new_frame());
                        }

                        if let Err(err) = app.on_update(&mut cx) {
                            tracing::error!("{err}");
                            control_flow.set_exit_with_code(1);
                            return;
                        }

                        #[cfg(feature = "imgui")]
                        if let Some(ui) = cx.ui.take() {
                            imgui::ImGui::end_frame(ui, cx.window());
                        }

                        if let Err(err) = renderer.draw_frame(&engine_cx.render(#[cfg(feature = "imgui")] imgui)) {
                            tracing::error!("{err}");
                            control_flow.set_exit_with_code(2);
                        };

                        engine_cx.end_frame();
                    }
                    _ => {
                        tracing::trace!("received event: {event:?}");
                        if let Ok(event) = event.try_into() {
                            if let Event::WindowEvent {
                                window_id: _,
                                event: WindowEvent::Resized(size),
                            } = event {
                                renderer.on_resized(size);
                            }
                            engine_cx.on_event(event);
                            #[cfg(feature = "imgui")]
                            imgui.on_event(event);
                            app.on_event(&mut engine_cx.context(), event);
                        }
                    }
                }

                if engine_cx.should_quit() && !engine_cx.is_quitting {
                    engine_cx.is_quitting = true;
                    tracing::debug!("shutting down with `Engine::on_stop`");
                    app.on_stop(&mut engine_cx.context());
                    // on_stop allows aborting the quit request
                    if engine_cx.should_quit() {
                        tracing::debug!("shutting down...");
                        control_flow.set_exit();
                    } else {
                        engine_cx.is_quitting = false;
                    }
                }
            }
        });
    }
}
