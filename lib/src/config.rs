#[derive(Debug, Copy, Clone)]
#[must_use]
pub struct Config {
    pub mouse_sensitivity: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            mouse_sensitivity: 10.0,
        }
    }
}

impl Config {
    pub fn new() -> Self {
        Self::default()
    }
}
