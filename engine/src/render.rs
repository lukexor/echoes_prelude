//! Render traits, types and methods.

use crate::{mesh::MaterialType, prelude::*, window::Window, Result};
use std::{fmt, path::PathBuf};

pub use backend::{RenderBackend, RenderContext};

pub mod backend;

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
    fn load_mesh(&mut self, name: impl Into<String>, filename: impl Into<PathBuf>);

    /// Load a texture into memory.
    fn load_texture(&mut self, name: impl Into<String>, filename: impl Into<PathBuf>);

    /// Load an object to the current scene.
    fn load_object(
        &mut self,
        name: impl Into<String>,
        mesh: impl Into<String>,
        material_type: impl Into<MaterialType>,
        transform: impl Into<Mat4>,
    );
}

#[derive(Debug)]
pub struct Renderer<Ctx = RenderContext> {
    cx: Ctx,
}

impl<Ctx: RenderBackend> Renderer<Ctx> {
    /// Initialize the `Renderer`.
    pub fn initialize(
        application_name: &str,
        applcation_version: &str,
        window: &Window,
        settings: RenderSettings,
    ) -> Result<Self> {
        Ok(Self {
            cx: Ctx::initialize(application_name, applcation_version, window, settings)?,
        })
    }

    /// Initialize imgui renderer.
    #[cfg(feature = "imgui")]
    pub fn initialize_imgui(&mut self, imgui: &mut imgui::ImGui) -> Result<()> {
        self.cx.initialize_imgui(imgui)
    }

    /// Handle window resized event.
    pub fn on_resized(&mut self, size: PhysicalSize<u32>) {
        self.cx.on_resized(size);
    }

    /// Draws a frame.
    pub fn draw_frame(&mut self, draw_data: &mut DrawData<'_>) -> Result<()> {
        self.cx.draw_frame(draw_data)
    }
}

impl<T, R: RenderBackend> Render for Context<T, R> {
    /// Set the clear color for the next frame.
    #[inline]
    fn set_clear_color(&mut self, color: impl Into<Vec4>) {
        self.draw_cmds.push(DrawCmd::ClearColor(color.into()));
    }

    /// Set the clear depth and/or stencil for the next frame.
    #[inline]
    fn set_clear_depth_stencil(&mut self, depth: f32, stencil: u32) {
        self.draw_cmds
            .push(DrawCmd::ClearDepthStencil((depth, stencil)));
    }

    /// Set the projection matrix for the next frame.
    #[inline]
    fn set_projection(&mut self, projection: impl Into<Mat4>) {
        self.draw_cmds
            .push(DrawCmd::SetProjection(projection.into()));
    }

    /// Set the view matrix for the next frame.
    #[inline]
    fn set_view(&mut self, view: impl Into<Mat4>) {
        self.draw_cmds.push(DrawCmd::SetView(view.into()));
    }

    /// Set an objects transform matrix.
    #[inline]
    fn set_object_transform(&mut self, name: &'static str, transform: impl Into<Mat4>) {
        self.draw_cmds.push(DrawCmd::SetObjectTransform {
            name,
            transform: transform.into(),
        });
    }

    /// Load a mesh into memory.
    #[inline]
    fn load_mesh(&mut self, name: impl Into<String>, filename: impl Into<PathBuf>) {
        let filename = self.config.resolve_asset_path(filename.into());
        self.draw_cmds.push(DrawCmd::LoadMesh {
            name: name.into(),
            filename,
        });
    }

    /// Load a texture into memory.
    #[inline]
    fn load_texture(&mut self, name: impl Into<String>, filename: impl Into<PathBuf>) {
        let filename = self.config.resolve_asset_path(filename.into());
        self.draw_cmds.push(DrawCmd::LoadTexture {
            name: name.into(),
            filename,
        });
    }

    /// Load a mesh object to render in the current scene.
    #[inline]
    fn load_object(
        &mut self,
        name: impl Into<String>,
        mesh: impl Into<String>,
        material_type: impl Into<MaterialType>,
        transform: impl Into<Mat4>,
    ) {
        self.draw_cmds.push(DrawCmd::LoadObject {
            name: name.into(),
            mesh: mesh.into(),
            material_type: material_type.into(),
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
    LoadMesh {
        name: String,
        filename: PathBuf,
    },
    LoadTexture {
        name: String,
        filename: PathBuf,
    },
    LoadObject {
        name: String,
        mesh: String,
        material_type: MaterialType,
        transform: Mat4,
    },
}

#[must_use]
pub struct DrawData<'a> {
    pub(crate) commands: &'a [DrawCmd],
    #[cfg(feature = "imgui")]
    pub(crate) imgui: &'a imgui::DrawData,
}

impl fmt::Debug for DrawData<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DrawData")
            .field("commands", &self.commands)
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
            sampler_ansiotropy: false,
            sample_shading: false,
            msaa: false,
            wireframe: false,
            line_width: 1.0,
            level_of_detail: 1.0,
            cull_mode: CullMode::None, // TODO: Switch to back culling
            front_face: FrontFace::CounterClockwise, // Because Y-axis is inverted in Vulkan
        }
    }
}
