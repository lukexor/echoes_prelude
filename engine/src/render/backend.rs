use super::{DrawData, RenderSettings};
#[cfg(feature = "imgui")]
use crate::imgui;
use crate::{
    matrix::Mat4, mesh::MaterialType, prelude::PhysicalSize, vector::Vec4, window::Window, Result,
};
use std::path::Path;

mod vulkan;
pub use vulkan::Context as RenderContext;

#[macro_export]
macro_rules! render_bail {
    ($msg:literal $(,)?) => {
        return ::std::result::Result::Err(
            $crate::Error::Renderer(::anyhow::anyhow!($msg))
        )
    };
    ($err:expr $(,)?) => {
        return ::std::result::Result::Err(
            $crate::Error::Renderer(::anyhow::anyhow!($err))
        )
    };
    ($fmt:expr, $($arg:tt)*) => {
        return ::std::result::Result::Err(
            $crate::Error::Renderer(::anyhow::anyhow!($fmt, $($arg)*))
        )
    };
}

pub trait RenderBackend: Sized {
    /// Initialize the `RendererBackend`.
    fn initialize(
        application_name: &str,
        application_version: &str,
        window: &Window,
        settings: RenderSettings,
    ) -> Result<Self>;

    /// Initialize imgui renderer.
    #[cfg(feature = "imgui")]
    fn initialize_imgui(&mut self, imgui: &mut imgui::ImGui) -> Result<()>;

    /// Handle window resized event.
    fn on_resized(&mut self, size: PhysicalSize<u32>);

    /// Draws a frame.
    fn draw_frame(&mut self, draw_data: &mut DrawData<'_>) -> Result<()>;

    /// Set the clear color for the next frame.
    fn set_clear_color(&mut self, color: Vec4);

    /// Set the clear depth and/or stencil for the next frame.
    fn set_clear_depth_stencil(&mut self, depth: f32, stencil: u32);

    /// Set the projection matrix for the next frame.
    fn set_projection(&mut self, projection: Mat4);

    /// Set the view matrix for the next frame.
    fn set_view(&mut self, view: Mat4);

    /// Set an object transform matrix.
    fn set_object_transform(&mut self, name: &str, transform: Mat4);

    /// Load a mesh into memory.
    fn load_mesh(&mut self, name: &str, filename: &Path) -> Result<()>;

    /// Unload a named mesh from memory.
    fn unload_mesh(&mut self, name: &str) -> Result<()>;

    /// Load a texture asset into memory.
    fn load_texture(&mut self, name: &str, filename: &Path) -> Result<()>;

    /// Unload a named texture from memory.
    fn unload_texture(&mut self, name: &str) -> Result<()>;

    /// Load an object to the current scene.
    fn load_object(
        &mut self,
        name: &str,
        mesh: &str,
        material_type: MaterialType,
        transform: Mat4,
    ) -> Result<()>;

    /// Unload a named object from the current scene.
    fn unload_object(&mut self, name: &str) -> Result<()>;
}
