use anyhow::Context;
use async_compression::{
    tokio::{bufread::DeflateDecoder, write::DeflateEncoder},
    Level,
};
use async_trait::async_trait;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
};
use tokio::{
    io::{self, AsyncReadExt, AsyncWriteExt, BufReader},
    task,
};

pub mod filesystem;

/// Current [Asset] format version.
const ASSET_VERSION: u32 = 1;

/// Results that can be returned from this library.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can be returned from this library.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("this asset is the wrong type")]
    InvalidAssetType,
    #[error("this method requires a destination buffer to be unpacked with")]
    MissingBuffer,
    #[error("this method requires a data buffer to be packed with")]
    MissingData,
    #[error("failed to deserialize asset data")]
    DeserializeError(anyhow::Error),
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Create a profiling timer for a section of code.
#[macro_export]
macro_rules! time {
    ($label:ident) => {
        #[cfg(debug_assertions)]
        let mut $label = Some(::std::time::Instant::now());
    };
    (log: $label:ident) => {
        #[cfg(debug_assertions)]
        match $label {
            Some(label) => {
                ::tracing::debug!("{}: {}", stringify!($label), label.elapsed().as_secs_f32())
            }
            None => ::tracing::warn!("Timer `{}` has been terminated.", stringify!($label)),
        };
    };
    (end: $label:ident) => {{
        #[cfg(debug_assertions)]
        match $label.take() {
            Some(label) => {
                ::tracing::debug!("{}: {}", stringify!($label), label.elapsed().as_secs_f32())
            }
            None => ::tracing::warn!("Timer `{}` has been terminated.", stringify!($label)),
        };
    }};
}

/// Level of compression for `deflate`.
// TODO: Required because async_compression::Level doesn't impl Serialize and is #[non_exhaustive]
// so is difficult to wrap
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[must_use]
#[non_exhaustive]
pub enum CompressionLevel {
    Fastest,
    Best,
    #[default]
    Default,
    Precise(u32),
}

impl From<CompressionLevel> for Level {
    fn from(level: CompressionLevel) -> Self {
        match level {
            CompressionLevel::Fastest => Self::Fastest,
            CompressionLevel::Best => Self::Best,
            CompressionLevel::Default => Self::Default,
            CompressionLevel::Precise(quality) => Self::Precise(quality),
        }
    }
}

/// [Asset] metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[must_use]
pub struct AssetMeta {
    pub version: u32,
    pub compression_level: Option<CompressionLevel>,
    pub original_file: PathBuf,
    pub unpacked_size: usize,
}

impl AssetMeta {
    /// A name identifying the metadata.
    fn name(&self) -> &str {
        self.original_file
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap_or("unknown")
    }

    /// Unpacked size of the associated data.
    fn size(&self) -> usize {
        self.unpacked_size
    }
}

impl Default for AssetMeta {
    fn default() -> Self {
        Self {
            version: ASSET_VERSION,
            compression_level: Some(CompressionLevel::Default),
            original_file: PathBuf::new(),
            unpacked_size: 0,
        }
    }
}

/// Asset types that can be read/written to disk with packed compression.
#[async_trait]
pub trait Asset: Sized + Send + Serialize + DeserializeOwned {
    /// The metadata for the asset.
    fn meta(&self) -> &AssetMeta;

    /// The name of the asset.
    fn name(&self) -> &str {
        self.meta().name()
    }

    /// Unpacked size of the asset.
    fn size(&self) -> usize {
        self.meta().size()
    }

    /// Load an `Asset` from a file path.
    async fn from_file(filename: impl AsRef<Path> + Send) -> Result<Self>;

    /// Save compressed asset data to disk.
    async fn save(&self, path: impl AsRef<Path> + Send) -> Result<()> {
        let mut file = filesystem::create_file(path).await?;
        let data = bincode::serialize(self).context("failed to serialize asset data")?;
        file.write_all(&data).await?;
        file.flush().await?;
        Ok(())
    }

    /// Load compressed asset data from disk.
    async fn load(path: impl AsRef<Path> + Send) -> Result<Self> {
        let mut file = filesystem::open_file(path).await?;
        // Start off with 4MB
        let mut bytes = Vec::with_capacity(1 << 22);
        file.read_to_end(&mut bytes).await?;
        bincode::deserialize(&bytes)
            .context("failed to deserialize asset data")
            .map_err(Error::DeserializeError)
    }

    /// Convert a file from disk to a packed format.
    async fn convert(filename: impl AsRef<Path> + Send) -> Result<PathBuf>
    where
        Self: Pack;
}

/// Asset that can be packed into compressed data format.
#[async_trait]
pub trait Pack {
    /// Pack asset data for saving to disk.
    async fn pack(&mut self) -> Result<()>;
}

/// Asset that can be unpacked from a compressed data format.
#[async_trait]
pub trait Unpack {
    /// Unpack asset data.
    async fn unpack(&mut self) -> Result<()>;
}

/// Asset that can be unpacked from a compressed data format directly into a buffer.
#[async_trait]
pub trait UnpackInto {
    /// Unpack asset data directly into a given byte buffer.
    async fn unpack_into(&self, destination: &mut [u8]) -> Result<()>;
}

/// Generic packing of data for a given compression level.
async fn pack(data: &mut Vec<u8>, compression_level: Option<CompressionLevel>) -> Result<()> {
    if let Some(level) = compression_level {
        let buffer = Vec::with_capacity(data.len());
        let mut encoder = DeflateEncoder::with_quality(buffer, level.into());
        encoder
            .write_all(data)
            .await
            .context("failed to encode asset")?;
        encoder.shutdown().await.context("failed to write asset")?;
        *data = encoder.into_inner();
    }
    Ok(())
}

/// Generic unpacking of data for a given compression level directly into a destination buffer.
async fn unpack_into(
    data: &[u8],
    compression_level: Option<CompressionLevel>,
    destination: &mut [u8],
) -> Result<()> {
    match compression_level {
        None => destination.copy_from_slice(data),
        Some(_) => {
            let mut decoder = DeflateDecoder::new(BufReader::new(data));
            decoder
                .read(destination)
                .await
                .context("failed to decode asset")?;
        }
    }
    Ok(())
}

/// A binary texture asset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[must_use]
pub struct TextureAsset {
    pub meta: AssetMeta,
    pub width: u32,
    pub height: u32,
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

#[async_trait]
impl Asset for TextureAsset {
    /// The metadata for the asset.
    fn meta(&self) -> &AssetMeta {
        &self.meta
    }

    /// Load an `Asset` from a file path.
    async fn from_file(filename: impl AsRef<Path> + Send) -> Result<Self> {
        let filename = filename.as_ref();
        tracing::debug!("loading texture {filename:?}");

        let texture_filename = filename.to_path_buf();
        let (info, pixels) = task::spawn_blocking(move || -> Result<(png::OutputInfo, Vec<u8>)> {
            let decoder = png::Decoder::new(filesystem::open_file_sync(&texture_filename)?);
            tracing::debug!("reading texture {texture_filename:?}");
            let mut reader = decoder.read_info().context("failed to read texture info")?;
            let mut pixels = vec![0; reader.output_buffer_size()];
            let info = reader
                .next_frame(&mut pixels)
                .context("failed to read texture frame {filename:?}")?;
            // let mip_levels = (info.width.max(info.height) as f32).log2().floor() as u32 + 1;

            Ok((info, pixels))
        })
        .await
        .context("failed to join texture thread {filename:?}")?
        .context("failed to read texture {filename:?}")?;

        tracing::debug!(
            "loaded texture {filename:?} successfully, size: {}",
            pixels.len()
        );

        // TODO: Generate mipmaps?
        Ok(Self {
            meta: AssetMeta {
                original_file: filename.to_path_buf(),
                unpacked_size: pixels.len(),
                ..Default::default()
            },
            width: info.width,
            height: info.height,
            data: pixels,
        })
    }

    /// Convert a file from disk to a packed asset format.
    async fn convert(filename: impl AsRef<Path> + Send) -> Result<PathBuf> {
        let filename = filename.as_ref();
        tracing::info!("converting texture asset {filename:?}");

        time!(read_texture);
        let mut texture = Self::from_file(filename).await?;
        time!(end: read_texture);

        time!(pack_texture);
        texture.pack().await?;
        time!(end: pack_texture);

        let new_filename = filename.with_extension("tx");
        texture.save(&new_filename).await?;

        tracing::info!("converted texture asset {filename:?} successsfully");

        Ok(new_filename)
    }
}

#[async_trait]
impl Pack for TextureAsset {
    /// Pack asset data for saving to disk.
    async fn pack(&mut self) -> Result<()> {
        pack(&mut self.data, self.meta.compression_level).await?;
        Ok(())
    }
}

#[async_trait]
impl Unpack for TextureAsset {
    /// Unpack asset data.
    async fn unpack(&mut self) -> Result<()> {
        let mut destination = vec![0; self.meta.unpacked_size];
        unpack_into(&self.data, self.meta.compression_level, &mut destination).await?;
        self.data = destination;
        Ok(())
    }
}

#[async_trait]
impl UnpackInto for TextureAsset {
    /// Unpack asset data directly into a given byte buffer.
    async fn unpack_into(&self, destination: &mut [u8]) -> Result<()> {
        assert!(destination.len() >= self.meta.unpacked_size);
        unpack_into(&self.data, self.meta.compression_level, destination).await?;
        Ok(())
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
#[repr(C)]
#[must_use]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
    pub uv: [f32; 2],
}

/// A binary mesh asset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[must_use]
pub struct MeshAsset {
    pub meta: AssetMeta,
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

impl MeshAsset {
    /// Return the vertex and index buffers from this mesh asset.
    pub fn get_buffers<T: DeserializeOwned>(&self) -> Result<(Vec<T>, Vec<u32>)> {
        let (vertices, indices) =
            bincode::deserialize(&self.data).context("failed to deserialize vertices")?;
        Ok((vertices, indices))
    }

    /// Return the vertex and index buffers from a byte slice.
    pub fn buffers_from_bytes<T: DeserializeOwned>(bytes: &[u8]) -> Result<(Vec<T>, Vec<u32>)> {
        let (vertices, indices) =
            bincode::deserialize(bytes).context("failed to deserialize vertices")?;
        Ok((vertices, indices))
    }
}

#[async_trait]
impl Asset for MeshAsset {
    /// The metadata for the asset.
    fn meta(&self) -> &AssetMeta {
        &self.meta
    }

    /// Load an `Asset` from a file path.
    async fn from_file(filename: impl AsRef<Path> + Send) -> Result<Self> {
        let filename = filename.as_ref();
        tracing::debug!("loading mesh {filename:?}");

        let mut obj_file = filesystem::open_file_sync(filename)?;
        let (models, _) =
            tobj::load_obj_buf_async(&mut obj_file, &tobj::GPU_LOAD_OPTIONS, |_| async {
                Ok((vec![tobj::Material::default()], Default::default()))
            })
            .await
            .with_context(|| format!("failed to load mesh obj: {filename:?}"))?;

        // Vertices / Indices
        let mut vertices = vec![];
        let mut indices: Vec<u32> = vec![];
        for (i, model) in models.into_iter().enumerate() {
            let mesh = &model.mesh;

            tracing::debug!("model[{i}].name = {}", model.name);
            tracing::debug!("model[{i}].mesh.material_id = {:?}", mesh.material_id);
            tracing::debug!("model[{i}].face_count = {}", mesh.face_arities.len());

            tracing::debug!("model[{i}].positions = {}", mesh.positions.len() / 3);
            tracing::debug!("model[{i}].normals = {}", mesh.normals.len());
            tracing::debug!("model[{i}].vertex_colors = {}", mesh.vertex_color.len());
            tracing::debug!("model[{i}].uv = {}", mesh.texcoords.len() / 2);

            let vertices_len = mesh.positions.len() / 3;
            assert!(mesh.positions.len() % 3 == 0);

            vertices.reserve(vertices_len);
            for v in 0..vertices_len {
                let position = [
                    mesh.positions[3 * v],
                    mesh.positions[3 * v + 1],
                    mesh.positions[3 * v + 2],
                ];
                let normal = if mesh.normals.is_empty() {
                    [0.0; 3]
                } else {
                    [
                        mesh.normals[3 * v],
                        mesh.normals[3 * v + 1],
                        mesh.normals[3 * v + 2],
                    ]
                };
                let color = if mesh.vertex_color.is_empty() {
                    if cfg!(debug_assertions) {
                        [normal[0], normal[1], normal[2], 1.0]
                    } else {
                        [1.0; 4]
                    }
                } else {
                    [
                        mesh.vertex_color[3 * v],
                        mesh.vertex_color[3 * v + 1],
                        mesh.vertex_color[3 * v + 2],
                        1.0,
                    ]
                };
                let uv = if mesh.texcoords.is_empty() {
                    [0.0; 2]
                } else {
                    // NOTE: Vulkan y-coordinate is flipped, hence 1.0 - uy
                    [mesh.texcoords[2 * v], 1.0 - mesh.texcoords[2 * v + 1]]
                };

                vertices.push(Vertex {
                    position,
                    normal,
                    color,
                    uv,
                });
            }

            indices.reserve(mesh.indices.len());
            indices.extend(mesh.indices.iter());
        }

        let data =
            bincode::serialize(&(vertices, indices)).context("failed to serialize vertex data")?;

        tracing::debug!(
            "loaded mesh {filename:?} successfully, size: {}",
            data.len()
        );

        Ok(Self {
            meta: AssetMeta {
                original_file: filename.to_path_buf(),
                unpacked_size: data.len(),
                ..Default::default()
            },
            data,
        })
    }

    /// Convert a file from disk to a packed asset format.
    async fn convert(filename: impl AsRef<Path> + Send) -> Result<PathBuf> {
        let filename = filename.as_ref();
        tracing::info!("converting mesh asset {filename:?}");

        time!(read_mesh);
        let mut mesh = MeshAsset::from_file(filename).await?;
        time!(end: read_mesh);

        time!(pack_mesh);
        mesh.pack().await?;
        time!(end: pack_mesh);

        let new_filename = filename.with_extension("mesh");
        mesh.save(&new_filename).await?;

        tracing::info!("converted mesh asset {filename:?} successfully");

        Ok(new_filename)
    }
}

#[async_trait]
impl Pack for MeshAsset {
    /// Pack asset data for saving to disk.
    async fn pack(&mut self) -> Result<()> {
        pack(&mut self.data, self.meta.compression_level).await?;
        Ok(())
    }
}

#[async_trait]
impl Unpack for MeshAsset {
    /// Unpack asset data.
    async fn unpack(&mut self) -> Result<()> {
        let mut destination = vec![0; self.meta.unpacked_size];
        unpack_into(&self.data, self.meta.compression_level, &mut destination).await?;
        self.data = destination;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn texture_convert() {
        let filename = PathBuf::from("tests/checker.png");
        let expected = TextureAsset::from_file(&filename)
            .await
            .expect("valid texture file");
        let converted = TextureAsset::convert(&filename)
            .await
            .expect("valid texture conversion");
        let mut texture = TextureAsset::load(&converted)
            .await
            .expect("valid texture load");
        assert_eq!(texture.meta.version, ASSET_VERSION);
        assert_eq!(texture.meta.original_file, filename);
        assert_eq!(
            texture.meta.compression_level,
            Some(CompressionLevel::Default)
        );
        assert_eq!(texture.width, 920);
        assert_eq!(texture.height, 920);
        assert_eq!(texture.data.len(), 608);

        texture.unpack().await.expect("valid texture unpack");
        assert_eq!(texture, expected);
    }

    #[tokio::test]
    async fn texture_pack_unpack() {
        let mut texture = TextureAsset {
            meta: AssetMeta {
                unpacked_size: 12,
                ..Default::default()
            },
            width: 100,
            height: 200,
            data: vec![
                0xB3, 0xF0, 0x6A, 0xF7, 0xC7, 0xA0, 0xBF, 0xAB, 0x25, 0x3C, 0x57, 0xA8,
            ],
        };
        let expected = texture.clone();
        texture.pack().await.expect("valid texture pack");
        texture.unpack().await.expect("valid texture unpack");
        assert_eq!(texture, expected);
    }

    #[tokio::test]
    async fn mesh_convert() {
        let filename = PathBuf::from("tests/teapot.obj");
        let expected = MeshAsset::from_file(&filename)
            .await
            .expect("valid mesh file");
        let (expected_vertices, expected_indices) = expected
            .get_buffers::<Vertex>()
            .expect("valid mesh deserialization");
        let converted = MeshAsset::convert(&filename)
            .await
            .expect("valid mesh conversion");
        let mut mesh = MeshAsset::load(converted).await.expect("valid mesh load");

        assert_eq!(mesh.meta.version, ASSET_VERSION);
        assert_eq!(mesh.meta.original_file, filename);
        assert_eq!(mesh.meta.compression_level, Some(CompressionLevel::Default));
        assert_eq!(mesh.data.len(), 41140);

        mesh.unpack().await.expect("valid mesh unpack");
        assert_eq!(mesh, expected);

        let (vertices, indices) = mesh
            .get_buffers::<Vertex>()
            .expect("valid mesh deserialization");
        assert_eq!(vertices, expected_vertices);
        assert_eq!(indices, expected_indices);
    }

    #[tokio::test]
    async fn mesh_pack_unpack() {
        let mut mesh = MeshAsset {
            meta: AssetMeta {
                unpacked_size: 12,
                ..Default::default()
            },
            data: vec![
                0xB3, 0xF0, 0x6A, 0xF7, 0xC7, 0xA0, 0xBF, 0xAB, 0x25, 0x3C, 0x57, 0xA8,
            ],
        };
        let expected = mesh.clone();
        mesh.pack().await.expect("valid mesh pack");
        mesh.unpack().await.expect("valid mesh unpack");
        assert_eq!(mesh, expected);
    }
}
