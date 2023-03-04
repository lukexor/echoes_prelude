//! Engine configuration.

use std::{env, path::PathBuf, time::Duration};

#[derive(Debug, Clone)]
#[must_use]
pub struct Config {
    pub(crate) window_title: String,
    pub(crate) limit_frame_rate: bool,
    pub(crate) target_fps: u32,
    pub(crate) double_click_speed: Duration,
    pub(crate) fullscreen: Option<Fullscreen>,
    pub(crate) cursor_grab: bool,
    pub(crate) asset_directory: Option<PathBuf>,
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
            asset_directory: None,
        }
    }
}

impl Config {
    /// Create a new default `Config` instance.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolve the path where an asset conversion is stored, based on whether this path is a
    /// `pix-engine`-provided asset, or an application-provided asset with an optional
    /// `asset_directory` defined.
    #[inline]
    #[must_use]
    pub fn resolve_asset_path(&self, path: PathBuf) -> PathBuf {
        PathBuf::from(env!("OUT_DIR")).join(path)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[must_use]
pub enum Fullscreen {
    Exclusive,
    Borderless,
}
