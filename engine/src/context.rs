//! Engine context

use crate::{
    config::Config,
    renderer::{RenderState, Renderer},
    Result,
};
use std::time::{Duration, Instant};

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
    pub(crate) renderer: Renderer,
}

impl Context {
    pub(crate) fn new(config: Config, renderer: Renderer) -> Self {
        Self {
            window_title: String::new(),
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

    #[must_use]
    pub fn is_running(&self) -> bool {
        !self.suspended
    }

    pub fn quit(&mut self) {
        self.should_quit = true;
    }

    pub fn draw_frame(&mut self, state: RenderState) -> Result<()> {
        Ok(self.renderer.draw_frame(state)?)
    }
}
