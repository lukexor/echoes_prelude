use anyhow::Result;
use cfg_if::cfg_if;
use std::time::Duration;
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
    // TODO: instead of just shaders, have this be generic platform state?
    fn initialize(application_name: &str, window: &Window, shaders: Shaders) -> Result<Self>;
    fn shutdown(&mut self) -> Result<()>;
    fn on_resized(&mut self, width: u32, height: u32);
    fn begin_frame(&mut self, delta_time: Duration) -> Result<()>;
    fn end_frame(&mut self, delta_time: Duration) -> Result<()>;
}

#[derive(Debug)]
pub struct Renderer {
    context: Context,
    width: u32,
    height: u32,
}

impl Renderer {
    pub fn initialize(application_name: &str, window: &Window, shaders: Shaders) -> Result<Self> {
        let PhysicalSize { width, height } = window.inner_size();
        Ok(Self {
            context: Context::initialize(application_name, window, shaders)?,
            width,
            height,
        })
    }

    pub fn shutdown(&mut self) {
        todo!()
    }

    pub fn on_resized(&mut self, width: u32, height: u32) {
        if self.width != width || self.height != height {
            self.width = width;
            self.height = height;
            self.context.on_resized(width, height);
        }
    }

    pub fn draw_frame(&mut self, render_state: RenderState) -> Result<()> {
        self.context.begin_frame(render_state.delta_time)?;

        self.context.end_frame(render_state.delta_time)?;

        Ok(())
    }
}

#[derive(Debug, Copy, Clone)]
#[must_use]
pub struct RenderState {
    pub(crate) delta_time: Duration,
}

#[derive(Debug, Clone)]
#[must_use]
pub struct Shaders {
    vertex: Vec<u8>,
    fragment: Vec<u8>,
}

impl Shaders {
    pub fn new(vertex: &[u8], fragment: &[u8]) -> Self {
        Self {
            vertex: vertex.to_vec(),
            fragment: fragment.to_vec(),
        }
    }
}
