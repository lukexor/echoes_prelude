//! Core engine features

use crate::{
    config::Config,
    context::Context,
    event::Event,
    renderer::{Renderer, Shader},
    Result,
};
use anyhow::Context as _;
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
    fn on_start(&mut self, _cx: &mut Context) -> Result<()> {
        Ok(())
    }
    fn on_update(&mut self, delta_time: f32, cx: &mut Context) -> Result<()>;

    fn on_stop(&mut self, _cx: &mut Context) {}

    fn on_event(&mut self, _event: Event, _cx: &mut Context) {}
}

#[derive(Debug)]
#[must_use]
pub struct Engine {
    name: Cow<'static, str>,
    window_title: String,
    width: u32,
    height: u32,
    config: Config,
    shaders: Vec<Shader>,
}

impl Engine {
    pub fn builder() -> EngineBuilder {
        EngineBuilder::new()
    }

    pub fn run(engine: Engine, mut app: impl Update + 'static) -> Result<()> {
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_title(&engine.window_title)
            .with_inner_size(LogicalSize::new(engine.width, engine.height))
            .build(&event_loop)
            .context("failed to create window")?;

        let renderer = Renderer::initialize(engine.name.as_ref(), &window, &engine.shaders)?;
        let mut cx = Context::new(engine.config, renderer);

        app.on_start(&mut cx)?;

        event_loop.run(move |event, _window_target, control_flow| {
            control_flow.set_poll();

            log::trace!("received event: {event:?}");
            match &event {
                WinitEvent::MainEventsCleared if cx.is_running() => {
                    let current_time = Instant::now();
                    let delta_time = current_time - cx.last_frame_time;

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
                        cx.window_title.clear();
                        let _ =
                            write!(cx.window_title, "{} - FPS: {}", engine.name, cx.fps_counter);
                        window.set_title(&cx.window_title);
                        cx.fps_timer -= one_second;
                        cx.fps_counter = 0;
                    }

                    cx.last_frame_time = current_time;
                }
                WinitEvent::WindowEvent { event, .. } => match event {
                    WindowEvent::Resized(size) => cx.renderer.on_resized(size.width, size.height),
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
            app.on_event(event.into(), &mut cx);

            if cx.should_quit {
                control_flow.set_exit();
            }
        });
    }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct EngineBuilder {
    title: Cow<'static, str>,
    position: Position,
    width: u32,
    height: u32,
    shaders: Vec<Shader>,
}

impl Default for EngineBuilder {
    fn default() -> Self {
        Self {
            title: Default::default(),
            position: Position::from(LogicalPosition::<u32>::default()),
            width: 1024,
            height: 768,
            shaders: vec![],
        }
    }
}

impl EngineBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn title(mut self, title: impl Into<Cow<'static, str>>) -> Self {
        self.title = title.into();
        self
    }

    pub fn position(mut self, position: Position) -> Self {
        self.position = position;
        self
    }

    pub fn dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn shader(mut self, shader: Shader) -> Self {
        self.shaders.push(shader);
        self
    }

    pub fn build(self) -> Result<Engine> {
        let config = Config::new();
        let window_title = self.title.to_string();
        Ok(Engine {
            name: self.title,
            window_title,
            width: self.width,
            height: self.height,
            config,
            shaders: self.shaders,
        })
    }
}
