//! Engine configuration.

use std::{env, time::Duration};

#[derive(Debug, Clone)]
#[must_use]
pub struct Config {
    pub(crate) window_title: String,
    pub(crate) limit_frame_rate: bool,
    pub(crate) target_fps: u32,
    pub(crate) double_click_speed: Duration,
    pub(crate) fullscreen: Option<Fullscreen>,
    pub(crate) cursor_grab: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            window_title: String::new(),
            limit_frame_rate: env::var("LIMIT_FPS").is_ok(),
            target_fps: env::var("TARGET_FPS")
                .ok()
                .and_then(|target_fps| target_fps.parse::<u32>().ok())
                .unwrap_or(60),
            double_click_speed: Duration::from_millis(400),
            fullscreen: None,
            cursor_grab: false,
        }
    }
}

impl Config {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[must_use]
pub enum Fullscreen {
    Exclusive,
    Borderless,
}
