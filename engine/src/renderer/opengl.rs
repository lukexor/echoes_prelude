use crate::renderer::RendererBackend;
use anyhow::Result;
use winit::window::Window;

#[derive(Debug)]
pub(crate) struct Context;

impl RendererBackend for Context {
    fn initialize(_application_name: &str, _window: &Window) -> Result<Self> {
        compile_error!("opengl renderer is not implemented yet")
    }

    fn on_resized(&mut self, _width: u32, _height: u32) {}

    fn draw_frame(&mut self) -> Result<()> {
        Ok(())
    }
}
