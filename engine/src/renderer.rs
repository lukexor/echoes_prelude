use anyhow::{Context as _, Result};
use cfg_if::cfg_if;
use std::{
    borrow::Cow,
    fmt, fs,
    io::{BufReader, Read},
    path::Path,
};
use winit::window::Window;

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
    fn initialize(application_name: &str, window: &Window, shaders: &[Shader]) -> Result<Self>;
    fn destroy(&mut self);
    fn on_resized(&mut self, width: u32, height: u32);
    fn begin_frame(&mut self, delta_time: f32) -> Result<()>;
    fn end_frame(&mut self, delta_time: f32) -> Result<()>;
}

#[derive(Debug)]
pub struct Renderer {
    context: Context,
}

impl Renderer {
    pub fn initialize(application_name: &str, window: &Window, shaders: &[Shader]) -> Result<Self> {
        Ok(Self {
            context: Context::initialize(application_name, window, shaders)?,
        })
    }

    pub fn shutdown(&mut self) {
        todo!()
    }

    pub fn on_resized(&mut self, width: u32, height: u32) {
        self.context.on_resized(width, height);
    }

    pub fn draw_frame(&mut self, render_state: RenderState) -> Result<()> {
        self.context.begin_frame(render_state.delta_time)?;

        self.context.end_frame(render_state.delta_time)?;

        Ok(())
    }
}

// TODO: Make fields private
#[derive(Debug, Copy, Clone)]
#[must_use]
pub struct RenderState {
    pub delta_time: f32,
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

    pub fn from_path(
        name: impl Into<Cow<'static, str>>,
        ty: ShaderType,
        path: impl AsRef<Path>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let mut file = BufReader::new(
            fs::File::open(path).with_context(|| format!("failed to open shader: {path:?}"))?,
        );
        let mut bytes = vec![];
        file.read_to_end(&mut bytes)
            .with_context(|| format!("failed to read shader: {path:?}"))?;
        Ok(Self::from_bytes(name, ty, bytes))
    }

    pub fn vertex(name: impl Into<Cow<'static, str>>, path: impl AsRef<Path>) -> Result<Self> {
        Self::from_path(name, ShaderType::Vertex, path)
    }

    pub fn fragment(name: impl Into<Cow<'static, str>>, path: impl AsRef<Path>) -> Result<Self> {
        Self::from_path(name, ShaderType::Fragment, path)
    }
}
