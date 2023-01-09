//! Traits and types for renderer backends.

use crate::{math::Mat4, prelude::PhysicalSize, window::Window};
use anyhow::{Context as _, Result};
use std::{borrow::Cow, fmt, path::Path};
use tokio::{
    fs::File,
    io::{AsyncReadExt, BufReader},
};

#[cfg(any(feature = "vulkan", not(feature = "opengl")))]
mod vulkan;
#[cfg(any(feature = "vulkan", not(feature = "opengl")))]
use vulkan::Context;

#[cfg(all(feature = "opengl", not(feature = "vulkan")))]
mod opengl;
#[cfg(all(feature = "opengl", not(feature = "vulkan")))]
use opengl::Context;

#[cfg(not(any(feature = "opengl", feature = "vulkan")))]
compile_error!("must choose a valid renderer feature: `vulkan` or `opengl`");

pub trait RendererBackend: Sized {
    /// Initialize the `RendererBackend`.
    fn initialize(
        application_name: &str,
        application_version: &str,
        window: &Window,
        shaders: &[Shader],
    ) -> Result<Self>;

    /// Handle window resized event.
    fn on_resized(&mut self, size: PhysicalSize);

    /// Begin rendering a frame to the screen.
    fn begin_frame(&mut self, delta_time: f32) -> Result<()>;

    /// Finish rendering a frame to the screen.
    fn end_frame(&mut self, delta_time: f32) -> Result<()>;

    /// Update the projection-view matrices for the scene.
    fn update_projection_view(&mut self, projection: Mat4, view: Mat4);

    /// Update the model matrix for the scene.
    fn update_model(&mut self, model: Mat4);
}

#[derive(Debug)]
pub struct Renderer {
    cx: Context,
}

impl Renderer {
    /// Initialize the `Renderer`.
    pub fn initialize(
        application_name: &str,
        applcation_version: &str,
        window: &Window,
        shaders: &[Shader],
    ) -> Result<Self> {
        Ok(Self {
            cx: Context::initialize(application_name, applcation_version, window, shaders)?,
        })
    }

    /// Shutdown the `Renderer`
    pub fn shutdown(&mut self) {
        // TODO
    }

    /// Handle window resized event.
    #[inline]
    pub fn on_resized(&mut self, size: PhysicalSize) {
        self.cx.on_resized(size);
    }

    /// Draw a frame to the screen.
    pub fn draw_frame(&mut self, render_state: RenderState) -> Result<()> {
        self.cx.begin_frame(render_state.delta_time)?;
        self.cx
            .update_projection_view(render_state.projection, render_state.view);
        self.cx.update_model(render_state.model);

        self.cx.end_frame(render_state.delta_time)?;

        Ok(())
    }
}

// TODO: Make fields private
#[derive(Default, Debug, Copy, Clone)]
#[must_use]
pub struct RenderState {
    pub delta_time: f32,
    pub view: Mat4,
    pub projection: Mat4,
    pub model: Mat4, // FIXME: temporary
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[must_use]
pub enum ShaderType {
    Vertex,
    Fragment,
}

#[derive(Clone)]
#[must_use]
pub struct Shader {
    name: Cow<'static, str>,
    ty: ShaderType,
    bytes: Vec<u8>,
}

impl fmt::Debug for Shader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Shader")
            .field("name", &self.name)
            .field("ty", &self.ty)
            .finish_non_exhaustive()
    }
}

impl Shader {
    pub fn from_bytes(name: impl Into<Cow<'static, str>>, ty: ShaderType, bytes: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            ty,
            bytes,
        }
    }

    pub async fn from_path(
        name: impl Into<Cow<'static, str>>,
        ty: ShaderType,
        path: impl AsRef<Path>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let mut file = BufReader::new(
            File::open(path)
                .await
                .with_context(|| format!("failed to open shader: {path:?}"))?,
        );
        let mut bytes = vec![];
        file.read_to_end(&mut bytes)
            .await
            .with_context(|| format!("failed to read shader: {path:?}"))?;
        Ok(Self::from_bytes(name, ty, bytes))
    }

    pub async fn vertex(
        name: impl Into<Cow<'static, str>>,
        path: impl AsRef<Path>,
    ) -> Result<Self> {
        Self::from_path(name, ShaderType::Vertex, path).await
    }

    pub async fn fragment(
        name: impl Into<Cow<'static, str>>,
        path: impl AsRef<Path>,
    ) -> Result<Self> {
        Self::from_path(name, ShaderType::Fragment, path).await
    }
}
