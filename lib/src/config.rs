//! Game configuration options.

//! Game configuration preferences, modified via the Config menu in-game.
#[derive(Debug, Copy, Clone)]
#[must_use]
pub struct Config {
    pub mouse_sensitivity: f32,
    pub scroll_pixels_per_line: f32,
    pub near_clip: f32,
    pub far_clip: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            mouse_sensitivity: 10.0,
            scroll_pixels_per_line: 12.0,
            near_clip: 0.1,
            far_clip: 1000.0,
        }
    }
}

impl Config {
    pub fn new() -> Self {
        Self::default()
    }
}
