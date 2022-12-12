use anyhow::Result;
use cfg_if::cfg_if;
use winit::{dpi::PhysicalSize, window::Window};

cfg_if! {
    if #[cfg(feature = "vulkan")] {
        mod vulkan;
        use vulkan::Context;
    } else if #[cfg(feature = "opengl")] {
        mod opengl;
        use opengl::Context;
    } else {
        #[derive(Debug)]
        struct Context;
        impl RendererBackend for Context {
            fn initialize(application_name: &str, window: &Window) -> Result<Self> { Ok(Self) }
            fn on_resized(&mut self, width: u32, height: u32) {}
            fn draw_frame(&mut self) -> Result<()> { Ok(()) }
        }
        compile_error!("must select a valid renderer feature: `vulkan` or `opengl`");
    }
}

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