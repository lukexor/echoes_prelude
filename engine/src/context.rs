//! Engine context

use crate::{
    config::Config,
    math::{Mat4, Matrix, Radians},
    renderer::{RenderState, Renderer},
    vector, Result,
};
use std::time::{Duration, Instant};
use winit::{dpi::PhysicalSize, window::Window};

#[derive(Debug)]
#[must_use]
pub struct Context {
    pub(crate) window_title: String,
    pub(crate) last_frame_time: Instant,
    pub(crate) target_frame_rate: Duration,
    pub(crate) fps_counter: usize,
    pub(crate) fps_timer: Duration,
    pub(crate) suspended: bool,
    pub(crate) should_quit: bool,
    pub(crate) config: Config,
    pub(crate) projection: Mat4,
    pub(crate) view: Mat4,
    pub(crate) near_clip: f32,
    pub(crate) far_clip: f32,
    pub(crate) renderer: Renderer,
}

impl Context {
    /// Create a new engine `Context`.
    pub(crate) fn new(config: Config, renderer: Renderer, window: &Window) -> Self {
        let PhysicalSize { width, height } = window.inner_size();
        let near_clip = 0.1;
        let far_clip = 1000.0;
        Self {
            window_title: String::new(),
            last_frame_time: Instant::now(),
            target_frame_rate: Duration::from_secs(1) / config.target_fps,
            fps_counter: 0,
            fps_timer: Duration::default(),
            suspended: false,
            should_quit: false,
            config,
            projection: Matrix::perspective(
                Radians(45.0),
                width as f32 / height as f32,
                near_clip,
                far_clip,
            ),
            view: Matrix::translation(vector!(0.0, 0.0, 10.0)).inverse(),
            near_clip,
            far_clip,
            renderer,
        }
    }

    /// Whether the engine is actively running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        !self.suspended
    }

    /// Begin engine shutdown.
    pub fn quit(&mut self) {
        self.should_quit = true;
    }

    /// Handle window resized event.
    pub fn on_resized(&mut self, width: u32, height: u32) {
        self.projection = Matrix::perspective(
            Radians(45.0),
            width as f32 / height as f32,
            self.near_clip,
            self.far_clip,
        );
        self.renderer.on_resized(width, height);
    }

    /// Draw a frame to the screen.
    pub fn draw_frame(&mut self, mut state: RenderState) -> Result<()> {
        state.projection = self.projection;
        Ok(self.renderer.draw_frame(state)?)
    }

    pub fn set_view(&mut self, view: Mat4) {
        self.view = view;
    }
}
