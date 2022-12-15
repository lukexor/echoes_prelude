//! Engine configuration.

use std::env;

#[derive(Debug, Default, Copy, Clone)]
#[must_use]
pub struct Config {
    pub(crate) limit_frame_rate: bool,
    pub(crate) target_fps: u32,
}

impl Config {
    pub fn new() -> Self {
        Self {
            limit_frame_rate: env::var("LIMIT_FPS").is_ok(),
            target_fps: env::var("TARGET_FPS")
                .ok()
                .and_then(|target_fps| target_fps.parse::<u32>().ok())
                .unwrap_or(60),
        }
    }
}
