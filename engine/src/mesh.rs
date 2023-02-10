use crate::{
    matrix::Mat4,
    vector::{Vec2, Vec3, Vec4},
    Result,
};
use anyhow::Context;
use asset_loader::{Asset, MeshAsset, Unpack};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const DEFAULT_MATERIAL: &str = "default_material";
pub(crate) const MAX_OBJECTS: usize = 10_000;

#[derive(Default, Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
#[repr(C)]
#[must_use]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub color: Vec4,
    pub uv: Vec2,
}

impl Vertex {
    /// Create a new `Vertex` instance.
    pub fn new(position: Vec3, normal: Vec3, color: Vec4, uv: Vec2) -> Self {
        Self {
            position,
            normal,
            color,
            uv,
        }
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq)]
#[repr(C)]
#[must_use]
pub(crate) struct ObjectData {
    pub(crate) transform: Mat4,
}

#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub filename: Option<PathBuf>,
}

impl Mesh {
    /// Load a new `Mesh` from an asset file.
    pub async fn from_asset_path(filename: impl AsRef<Path>) -> Result<Self> {
        let filename = filename.as_ref();
        let mut mesh = MeshAsset::load(filename)
            .await
            .with_context(|| format!("failed to load mesh {filename:?}"))?;
        mesh.unpack()
            .await
            .with_context(|| format!("failed to unpack mesh {filename:?})"))?;
        let (vertices, indices) = mesh
            .get_buffers::<Vertex>()
            .with_context(|| format!("failed to deserialize mesh {filename:?}"))?;

        Ok(Self {
            vertices,
            indices,
            filename: Some(filename.to_path_buf()),
        })
    }

    pub fn from_bytes(bytes: Bytes) -> Result<Self> {
        let (vertices, indices) = MeshAsset::buffers_from_bytes::<Vertex>(bytes)
            .context("failed to deserialize mesh from bytes")?;

        Ok(Self {
            vertices,
            indices,
            filename: None,
        })
    }
}
