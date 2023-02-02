//! Traits and types for renderer backends.

use crate::{
    mesh::{Mesh, Texture},
    prelude::*,
    window::Window,
    Result,
};

pub(crate) use backend::{RenderBackend, RenderContext};
pub use shape::DrawShape;
use std::fmt;

mod backend;
pub mod shape;

pub trait Render {
    /// Set the clear color for the next frame.
    fn set_clear_color(&mut self, color: impl Into<Vec4>);

    /// Set the clear depth and/or stencil for the next frame.
    fn set_clear_depth_stencil(&mut self, depth: f32, stencil: u32);

    /// Set the projection matrix for the next frame.
    fn set_projection(&mut self, projection: impl Into<Mat4>);

    /// Set the view matrix for the next frame.
    fn set_view(&mut self, view: impl Into<Mat4>);

    /// Set an object transform matrix.
    fn set_object_transform(&mut self, mesh: &'static str, transform: impl Into<Mat4>);

    /// Load a mesh into memory.
    fn load_mesh(&mut self, mesh: impl Into<Mesh>);

    /// Load a texture into memory.
    fn load_texture(&mut self, texture: impl Into<Texture>, material: &'static str);

    /// Load an object to the current scene.
    fn load_object(
        &mut self,
        mesh: impl Into<String>,
        material: impl Into<String>,
        transform: impl Into<Mat4>,
    );
}

#[derive(Debug)]
pub(crate) struct Renderer<C = RenderContext> {
    cx: C,
}

impl<C: RenderBackend> Renderer<C> {
    /// Initialize the `Renderer`.
    pub(crate) fn initialize(
        application_name: &str,
        applcation_version: &str,
        window: &Window,
        settings: RenderSettings,
        #[cfg(feature = "imgui")] imgui: &mut imgui::ImGui,
    ) -> Result<Self> {
        Ok(Self {
            cx: C::initialize(
                application_name,
                applcation_version,
                window,
                settings,
                #[cfg(feature = "imgui")]
                imgui,
            )?,
        })
    }

    /// Handle window resized event.
    pub(crate) fn on_resized(&mut self, size: PhysicalSize<u32>) {
        self.cx.on_resized(size);
    }

    /// Draws a frame.
    pub(crate) fn draw_frame(&mut self, draw_data: &DrawData<'_>) -> Result<()> {
        self.cx.draw_frame(draw_data)
    }
}

impl<T> Render for Context<'_, T> {
    /// Set the clear color for the next frame.
    #[inline]
    fn set_clear_color(&mut self, color: impl Into<Vec4>) {
        self.cx.draw_cmds.push(DrawCmd::ClearColor(color.into()));
    }

    /// Set the clear depth and/or stencil for the next frame.
    #[inline]
    fn set_clear_depth_stencil(&mut self, depth: f32, stencil: u32) {
        self.cx
            .draw_cmds
            .push(DrawCmd::ClearDepthStencil((depth, stencil)));
    }

    /// Set the projection matrix for the next frame.
    #[inline]
    fn set_projection(&mut self, projection: impl Into<Mat4>) {
        self.cx
            .draw_cmds
            .push(DrawCmd::SetProjection(projection.into()));
    }

    /// Set the view matrix for the next frame.
    #[inline]
    fn set_view(&mut self, view: impl Into<Mat4>) {
        self.cx.draw_cmds.push(DrawCmd::SetView(view.into()));
    }

    /// Set an objects transform matrix.
    #[inline]
    fn set_object_transform(&mut self, name: &'static str, transform: impl Into<Mat4>) {
        self.cx.draw_cmds.push(DrawCmd::SetObjectTransform {
            name,
            transform: transform.into(),
        });
    }

    /// Load a mesh into memory.
    #[inline]
    fn load_mesh(&mut self, mesh: impl Into<Mesh>) {
        self.cx.draw_cmds.push(DrawCmd::LoadMesh(mesh.into()));
    }

    /// Load a texture into memory.
    #[inline]
    fn load_texture(&mut self, texture: impl Into<Texture>, material: &'static str) {
        self.cx.draw_cmds.push(DrawCmd::LoadTexture {
            texture: texture.into(),
            material,
        });
    }

    /// Load a mesh object to render in the current scene.
    #[inline]
    fn load_object(
        &mut self,
        mesh: impl Into<String>,
        material: impl Into<String>,
        transform: impl Into<Mat4>,
    ) {
        self.cx.draw_cmds.push(DrawCmd::LoadObject {
            mesh: mesh.into(),
            material: material.into(),
            transform: transform.into(),
        });
    }
}

#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub enum DrawCmd {
    ClearColor(Vec4),
    ClearDepthStencil((f32, u32)),
    SetProjection(Mat4),
    SetView(Mat4),
    SetObjectTransform {
        name: &'static str,
        transform: Mat4,
    },
    LoadMesh(Mesh),
    LoadTexture {
        texture: Texture,
        material: &'static str,
    },
    LoadObject {
        mesh: String,
        material: String,
        transform: Mat4,
    },
}

#[derive(Default)]
#[must_use]
pub struct DrawData<'a> {
    pub(crate) data: &'a [DrawCmd],
    #[cfg(feature = "imgui")]
    pub(crate) imgui: Option<&'a imgui::DrawData>,
}

impl fmt::Debug for DrawData<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DrawData")
            .field("data", &self.data)
            .finish_non_exhaustive()
    }
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
    sample_shading: bool,
    /// Multi-sample Anti-aliasing.
    msaa: bool,
    wireframe: bool,
    line_width: f32,
    level_of_detail: f32,
    cull_mode: CullMode,
    front_face: FrontFace,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            sampler_ansiotropy: true,
            sample_shading: true,
            msaa: true,
            wireframe: false,
            line_width: 1.0,
            level_of_detail: 1.0,
            cull_mode: CullMode::None, // TODO: Switch to back culling
            front_face: FrontFace::CounterClockwise, // Because Y-axis is inverted in Vulkan
        }
    }
}
