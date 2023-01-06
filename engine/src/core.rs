//! Core engine features

use crate::{
    config::Config,
    context::Context,
    event::Event,
    renderer::{Renderer, Shader},
    Error, Result,
};
use anyhow::Context as _;
use derive_builder::Builder;
use std::{
    borrow::Cow,
    fmt::Write,
    thread,
    time::{Duration, Instant},
};
use winit::{
    dpi::{LogicalPosition, LogicalSize, Position},
    event::{ElementState, Event as WinitEvent, VirtualKeyCode, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

pub trait Update {
    /// Called on engine start.
    fn on_start(&mut self, _cx: &mut Context) -> Result<()> {
        Ok(())
    }

    /// Called every frame.
    fn on_update(&mut self, delta_time: f32, cx: &mut Context) -> Result<()>;

    /// Called on engine shutdown.
    fn on_stop(&mut self, _cx: &mut Context) {}

    /// Called on every event.
    fn on_event(&mut self, _delta_time: f32, _event: Event, _cx: &mut Context) {}
}

#[derive(Debug, Builder)]
#[builder(default, build_fn(error = "Error"))]
#[builder_struct_attr(must_use)]
#[must_use]
pub struct Engine {
    #[builder(setter(into))]
    title: Cow<'static, str>,
    #[builder(setter(into))]
    version: Cow<'static, str>,
    width: u32,
    height: u32,
    position: Position,
    config: Config,
    #[builder(setter(each = "shader"))]
    shaders: Vec<Shader>,
}

impl Default for Engine {
    fn default() -> Self {
        Self {
            title: "".into(),
            version: "1.0.0".into(),
            position: Position::from(LogicalPosition::<u32>::default()),
            width: 1024,
            height: 768,
            config: Config::default(),
            shaders: vec![],
        }
    }
}

impl Engine {
    pub fn builder() -> EngineBuilder {
        EngineBuilder::default()
    }

    pub fn run(engine: Engine, mut app: impl Update + 'static) -> Result<()> {
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_title(engine.title.clone())
            .with_inner_size(LogicalSize::new(engine.width, engine.height))
            .with_position(engine.position)
            .build(&event_loop)
            .context("failed to create window")?;

        let renderer =
            Renderer::initialize(&engine.title, &engine.version, &window, &engine.shaders)?;
        let mut cx = Context::new(engine.config, &window, renderer);

        app.on_start(&mut cx)?;

        event_loop.run(move |event, _window_target, control_flow| {
            control_flow.set_poll();

            let current_time = Instant::now();
            let delta_time = current_time - cx.last_frame_time;

            log::trace!("received event: {event:?}");
            match &event {
                WinitEvent::MainEventsCleared if cx.is_running() => {
                    if let Err(err) = app.on_update(delta_time.as_secs_f32(), &mut cx) {
                        log::error!("failed to update application: {err}");
                        control_flow.set_exit_with_code(1);
                    }

                    let end_time = Instant::now();
                    let elapsed = end_time - current_time;
                    cx.fps_timer += delta_time;
                    let remaining = cx
                        .target_frame_rate
                        .checked_sub(elapsed)
                        .unwrap_or_default();
                    if remaining.as_millis() > 0 {
                        if engine.config.limit_frame_rate {
                            thread::sleep(remaining - Duration::from_millis(1));
                        }
                        cx.fps_counter += 1;
                    }

                    let one_second = Duration::from_secs(1);
                    if cx.fps_timer > one_second {
                        cx.title.clear();
                        let _ = write!(cx.title, "{} - FPS: {}", engine.title, cx.fps_counter);
                        window.set_title(&cx.title);
                        cx.fps_timer -= one_second;
                        cx.fps_counter = 0;
                    }

                    cx.last_frame_time = current_time;
                }
                WinitEvent::WindowEvent { event, .. } => match event {
                    WindowEvent::Resized(size) => cx.on_resized(size.width, size.height),
                    #[cfg(debug_assertions)]
                    WindowEvent::KeyboardInput { input, .. } => {
                        if matches!(
                            (input.virtual_keycode, input.state),
                            (Some(VirtualKeyCode::Escape), ElementState::Pressed)
                        ) {
                            control_flow.set_exit();
                        }
                    }
                    WindowEvent::CloseRequested | WindowEvent::Destroyed => {
                        log::debug!("window closed or destroyed");
                        control_flow.set_exit();
                    }
                    _ => (),
                },
                WinitEvent::LoopDestroyed => {
                    log::info!("shutting down...");
                    control_flow.set_exit();
                }
                _ => (),
            }
            if !matches!(
                event,
                WinitEvent::MainEventsCleared
                    | WinitEvent::RedrawEventsCleared
                    | WinitEvent::NewEvents(_)
            ) {
                app.on_event(delta_time.as_secs_f32(), event.into(), &mut cx);
            }

            if cx.should_quit {
                control_flow.set_exit();
            }
        });
    }
}
