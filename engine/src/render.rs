//! Traits and types for renderer backends.

use self::backend::RenderContext;
use crate::{
    context::Context,
    math::{Mat4, Vec4},
    mesh::{Mesh, Texture},
    window::{PhysicalSize, Window},
    Result,
};
use derive_more::{Deref, DerefMut};

pub(crate) use backend::RenderBackend;
pub use shape::DrawShape;

mod backend;
pub mod shape;

pub trait Render {
    /// Begin rendering a frame to the screen.
    fn draw_frame(&mut self) -> Result<()>;

    /// Set the clear color for the next frame.
    fn set_clear_color(&mut self, color: impl Into<Vec4>);

    /// Set the clear depth and/or stencil for the next frame.
    fn set_clear_depth_stencil(&mut self, depth: f32, stencil: u32);

    /// Set the projection matrix for the next frame.
    fn set_projection(&mut self, projection: impl Into<Mat4>);

    /// Set the view matrix for the next frame.
    fn set_view(&mut self, view: impl Into<Mat4>);

    /// Set an object transform matrix.
    fn set_object_transform(&mut self, mesh: &str, transform: impl Into<Mat4>);

    /// Load a mesh into memory.
    fn load_mesh(&mut self, mesh: impl Into<Mesh>) -> Result<()>;

    /// Load a texture into memory.
    fn load_texture(&mut self, texture: impl Into<Texture>, material: &str) -> Result<()>;

    /// Load an object to the current scene.
    fn load_object(
        &mut self,
        mesh: impl Into<String>,
        material: impl Into<String>,
        transform: impl Into<Mat4>,
    ) -> Result<()>;
}

#[derive(Debug, Deref, DerefMut)]
pub(crate) struct Renderer {
    cx: RenderContext,
}

impl Renderer {
    /// Initialize the `Renderer`.
    pub(crate) fn initialize(
        application_name: &str,
        applcation_version: &str,
        window: &Window,
        settings: &RenderSettings,
    ) -> Result<Self> {
        Ok(Self {
            cx: RenderContext::initialize(application_name, applcation_version, window, settings)?,
        })
    }
}

impl<T> Render for Context<T> {
    /// Draw a frame to the screen.
    #[inline]
    fn draw_frame(&mut self) -> Result<()> {
        self.renderer.draw_frame(self.delta_time.as_secs_f32())
    }

    /// Set the clear color for the next frame.
    #[inline]
    fn set_clear_color(&mut self, color: impl Into<Vec4>) {
        self.renderer.set_clear_color(color.into());
    }

    /// Set the clear depth and/or stencil for the next frame.
    #[inline]
    fn set_clear_depth_stencil(&mut self, depth: f32, stencil: u32) {
        self.renderer.set_clear_depth_stencil(depth, stencil);
    }

    /// Set the projection matrix for the next frame.
    #[inline]
    fn set_projection(&mut self, projection: impl Into<Mat4>) {
        self.renderer.set_projection(projection.into());
    }

    /// Set the view matrix for the next frame.
    #[inline]
    fn set_view(&mut self, view: impl Into<Mat4>) {
        self.renderer.set_view(view.into());
    }

    /// Set an objects transform matrix.
    #[inline]
    fn set_object_transform(&mut self, name: &str, transform: impl Into<Mat4>) {
        self.renderer.set_object_transform(name, transform.into());
    }

    /// Load a mesh into memory.
    #[inline]
    fn load_mesh(&mut self, mesh: impl Into<Mesh>) -> Result<()> {
        self.renderer.load_mesh(mesh.into())
    }

    /// Load a texture into memory.
    #[inline]
    fn load_texture(&mut self, texture: impl Into<Texture>, material: &str) -> Result<()> {
        self.renderer.load_texture(texture.into(), material)
    }

    /// Load a mesh object to render in the current scene.
    #[inline]
    fn load_object(
        &mut self,
        mesh: impl Into<String>,
        material: impl Into<String>,
        transform: impl Into<Mat4>,
    ) -> Result<()> {
        self.renderer
            .load_object(mesh.into(), material.into(), transform.into())
    }
}

impl<T> Context<T> {
    /// Handle window resized event.
    #[inline]
    pub(crate) fn on_resized(&mut self, size: PhysicalSize<u32>) {
        self.renderer.on_resized(size);
    }
}

// TODO: Make fields private
#[derive(Default, Debug, Copy, Clone)]
#[must_use]
pub struct RenderState {
    pub delta_time: f32,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[must_use]
pub enum CullMode {
    None,
    Front,
    Back,
    All,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[must_use]
pub enum FrontFace {
    CounterClockwise,
    Clockwise,
}

#[derive(Debug, Copy, Clone, PartialEq)]
#[must_use]
pub struct RenderSettings {
    sampler_ansiotropy: bool,
    /// Multi-sample Anti-aliasing.
    sample_shading: bool,
    wireframe: bool,
    line_width: f32,
    level_of_detail: f32,
    cull_mode: CullMode,
    front_face: FrontFace,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            sampler_ansiotropy: false,
            // TODO: fix disabling this
            sample_shading: true,
            wireframe: false,
            line_width: 1.0,
            level_of_detail: 1.0,
            cull_mode: CullMode::None, // TODO: Switch to back culling
            front_face: FrontFace::CounterClockwise, // Because Y-axis is inverted in Vulkan
        }
    }
}
