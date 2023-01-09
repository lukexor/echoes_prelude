//! OpenGL renderer backend.

use super::Shader;
use crate::{
    math::Mat4,
    prelude::{dpi::PhysicalSize, window::Window},
    renderer::RendererBackend,
};
use anyhow::Result;

#[derive(Debug)]
pub(crate) struct Context;

impl RendererBackend for Context {
    fn initialize(
        application_name: &str,
        application_version: &str,
        window: &Window,
        shaders: &[Shader],
    ) -> Result<Self> {
        compile_error!("opengl renderer is not implemented yet")
    }

    fn shutdown(&mut self) {}

    fn on_resized(&mut self, size: PhysicalSize<u32>) {}

    fn begin_frame(&mut self, delta_time: f32) -> Result<()> {
        Ok(())
    }

    fn end_frame(&mut self, delta_time: f32) -> Result<()> {
        Ok(())
    }

    fn update_projection_view(&mut self, projection: Mat4, view: Mat4) {}

    fn update_model(&mut self, model: Mat4) {}
}
