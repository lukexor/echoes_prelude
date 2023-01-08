#[derive(Debug, Copy, Clone)]
#[must_use]
pub struct Config {
    pub mouse_sensitivity: f32,
    pub scroll_pixels_per_line: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            mouse_sensitivity: 10.0,
            scroll_pixels_per_line: 12.0,
        }
    }
}

impl Config {
    pub fn new() -> Self {
        Self::default()
    }
}
