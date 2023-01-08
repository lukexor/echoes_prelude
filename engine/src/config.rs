//! Engine configuration.

use crate::prelude::*;
use std::{env, time::Duration};
use winit::monitor::MonitorHandle;

#[derive(Debug, Copy, Clone)]
#[must_use]
pub struct Config {
    pub(crate) limit_frame_rate: bool,
    pub(crate) target_fps: u32,
    pub(crate) double_click_speed: Duration,
    pub(crate) fullscreen_mode: FullscreenMode,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            limit_frame_rate: env::var("LIMIT_FPS").is_ok(),
            target_fps: env::var("TARGET_FPS")
                .ok()
                .and_then(|target_fps| target_fps.parse::<u32>().ok())
                .unwrap_or(60),
            double_click_speed: Duration::from_millis(400),
            fullscreen_mode: FullscreenMode::Windowed,
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
pub enum FullscreenMode {
    Windowed,
    Exclusive,
    Borderless,
}

impl FullscreenMode {
    pub(crate) fn as_monitor(&self, monitor: Option<MonitorHandle>) -> Option<Fullscreen> {
        match self {
            Self::Windowed => None,
            Self::Exclusive => monitor
                .and_then(|monitor| monitor.video_modes().next())
                .map(Fullscreen::Exclusive),
            Self::Borderless => Some(Fullscreen::Borderless(monitor)),
        }
    }
}
