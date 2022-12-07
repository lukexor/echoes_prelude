//! Application logic.

use anyhow::Result;
use winit::window::Window;

#[derive(Debug, Copy, Clone)]
#[must_use]
pub struct App {
    pub frames: usize,
    pub destroying: bool,
    pub minimized: bool,
    pub resized: bool,
}

impl App {
    pub fn create(_window: &Window) -> Self {
        // TODO finish create
        Self {
            frames: 0,
            destroying: false,
            minimized: false,
            resized: false,
        }
    }

    #[must_use]
    pub fn is_active(&self) -> bool {
        !self.destroying && !self.minimized
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            self.minimized = true;
        } else {
            self.minimized = false;
            self.resized = true;
        }
    }

    pub fn update_and_render(&mut self, _window: &Window) -> Result<()> {
        // TODO finish update_and_render
        self.frames += 1;
        log::debug!("frame: {}", self.frames);
        Ok(())
    }

    pub fn audio_samples(&mut self) -> Result<Vec<f32>> {
        // TODO finish audio_samples
        Ok(vec![])
    }

    pub fn destroy(&mut self) {
        if self.destroying {
            return;
        }
        // TODO finish destroy
        self.destroying = true;
    }
}
