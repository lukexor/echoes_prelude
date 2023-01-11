use super::RenderSettings;
use crate::{
    math::{Mat4, Vec4},
    mesh::{Mesh, Texture},
    prelude::PhysicalSize,
    window::Window,
    Result,
};
use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(feature = "vulkan")] {
        mod vulkan;
        pub(crate) use vulkan::RenderContext;
    } else {
        compile_error!("must choose a valid renderer feature: `vulkan` is the only option currently");
    }
}

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

pub(crate) trait RenderBackend: Sized {
    /// Initialize the `RendererBackend`.
    fn initialize(
        application_name: &str,
        application_version: &str,
        window: &Window,
        settings: &RenderSettings,
    ) -> Result<Self>;

    /// Handle window resized event.
    fn on_resized(&mut self, size: PhysicalSize<u32>);

    /// Begin rendering a frame to the screen.
    fn draw_frame(&mut self, delta_time: f32) -> Result<()>;

    /// Set the clear color for the next frame.
    fn set_clear_color(&mut self, color: Vec4);

    /// Set the clear depth and/or stencil for the next frame.
    fn set_clear_depth_stencil(&mut self, depth: f32, stencil: u32);

    /// Set the projection matrix for the next frame.
    fn set_projection(&mut self, projection: Mat4);

    /// Set the view matrix for the next frame.
    fn set_view(&mut self, view: Mat4);

    /// Set an object transform matrix.
    fn set_object_transform(&mut self, mesh: &str, transform: Mat4);

    /// Load a mesh into memory.
    fn load_mesh(&mut self, mesh: Mesh) -> Result<()>;

    /// Load a texture into memory.
    fn load_texture(&mut self, texture: Texture, material: &str) -> Result<()>;

    /// Load an object to the current scene.
    fn load_object(&mut self, mesh: String, material: String, transform: Mat4) -> Result<()>;
}
