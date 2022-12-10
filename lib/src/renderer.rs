use anyhow::Result;
#[cfg(opengl)]
use opengl::Context;
#[cfg(vulkan)]
use vulkan::Context;
use winit::{dpi::PhysicalSize, window::Window};

#[cfg(opengl)]
mod opengl;
#[cfg(vulkan)]
mod vulkan;

pub trait RendererBackend: Sized {
    fn initialize(application_name: &str, window: &Window) -> Result<Self>;
    fn on_resized(&mut self, width: u32, height: u32);
    fn draw_frame(&mut self) -> Result<()>;
}

#[derive(Debug)]
pub struct Renderer {
    context: Context,
    width: u32,
    height: u32,
}

impl Renderer {
    pub fn initialize(application_name: &str, window: &Window) -> Result<Self> {
        let PhysicalSize { width, height } = window.inner_size();
        Ok(Self {
            context: Context::initialize(application_name, window)?,
            width,
            height,
        })
    }

    pub fn on_resized(&mut self, width: u32, height: u32) {
        if self.width != width || self.height != height {
            self.width = width;
            self.height = height;
            self.context.on_resized(width, height);
        }
    }

    pub fn draw_frame(&mut self) -> Result<()> {
        self.context.draw_frame()
    }
}
