//! Engine context

use crate::{
    config::Config,
    renderer::{RenderState, Renderer},
    Result,
};
use std::time::{Duration, Instant};
use winit::{dpi::PhysicalSize, window::Window};

#[derive(Debug)]
#[must_use]
pub struct Context {
    pub(crate) title: String,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) start: Instant,
    pub(crate) last_frame_time: Instant,
    pub(crate) target_frame_rate: Duration,
    pub(crate) fps_counter: usize,
    pub(crate) fps_timer: Duration,
    pub(crate) suspended: bool,
    pub(crate) should_quit: bool,
    pub(crate) config: Config,
    pub(crate) renderer: Renderer,
}

impl Context {
    /// Create a new engine `Context`.
    pub(crate) fn new(config: Config, window: &Window, renderer: Renderer) -> Self {
        let PhysicalSize { width, height } = window.inner_size();
        Self {
            title: String::new(),
            width,
            height,
            start: Instant::now(),
            last_frame_time: Instant::now(),
            target_frame_rate: Duration::from_secs(1) / config.target_fps,
            fps_counter: 0,
            fps_timer: Duration::default(),
            suspended: false,
            should_quit: false,
            config,
            renderer,
        }
    }

    /// Whether the engine is actively running.
    #[inline]
    #[must_use]
    pub fn is_running(&self) -> bool {
        !self.suspended
    }

    /// Begin engine shutdown.
    #[inline]
    pub fn quit(&mut self) {
        self.should_quit = true;
    }

    /// Handle window resized event.
    #[inline]
    pub fn on_resized(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.renderer.on_resized(width, height);
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
        self.width
    }

    /// The current window height.
    #[inline]
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }
}
