//! Shader management.

use crate::Result;
use anyhow::Context;
use std::{borrow::Cow, fmt, path::Path};
use tokio::{
    fs::File,
    io::{AsyncReadExt, BufReader},
};

pub(crate) const DEFAULT_VERTEX_SHADER: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/default.vert.spv"));
pub(crate) const DEFAULT_FRAGMENT_SHADER: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/texture.frag.spv"));

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[must_use]
pub enum ShaderStage {
    Vertex,
    TessellationControl,
    TessellationEvaluation,
    Geometry,
    Fragment,
    Compute,
}

#[derive(Clone)]
#[must_use]
pub struct Shader {
    pub(crate) name: Cow<'static, str>,
    pub(crate) stage: ShaderStage,
    // NOTE: Required to be owned, so that it can be aligned by renderer
    pub(crate) bytes: Vec<u8>,
}

impl fmt::Debug for Shader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Shader")
            .field("name", &self.name)
            .field("stage", &self.stage)
            .field("size", &self.bytes.len())
            .finish_non_exhaustive()
    }
}

impl Shader {
    pub fn from_bytes(name: impl Into<Cow<'static, str>>, ty: ShaderStage, bytes: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            stage: ty,
            bytes,
        }
    }

    pub async fn from_path(
        name: impl Into<Cow<'static, str>>,
        ty: ShaderStage,
        path: impl AsRef<Path>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let mut file = BufReader::new(
            File::open(path)
                .await
                .with_context(|| format!("failed to open shader file: {path:?}"))?,
        );
        let mut bytes = Vec::with_capacity(2048);
        file.read_to_end(&mut bytes)
            .await
            .with_context(|| format!("failed to read shader: {path:?}"))?;
        bytes.shrink_to_fit();
        Ok(Self::from_bytes(name, ty, bytes))
    }
}
