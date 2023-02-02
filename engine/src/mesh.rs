use crate::{
    matrix::Mat4,
    vec2, vec3, vec4,
    vector::{Vec2, Vec3, Vec4},
    Result,
};
use anyhow::Context as _;
use std::{
    fs::{self, File},
    io::BufReader,
    mem,
    path::Path,
};

pub const DEFAULT_MATERIAL: &str = "default_material";
pub(crate) const MAX_OBJECTS: usize = 10_000;

#[derive(Default, Debug, Copy, Clone, PartialEq)]
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
    pub name: String,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Mesh {
    /// Create a new `Mesh` instance.
    #[inline]
    pub fn new(name: impl Into<String>, vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        Self {
            name: name.into(),
            vertices,
            indices,
        }
    }

    /// Load a new `Mesh` from a file with valid OBJ mesh data.
    pub async fn from_file(name: impl Into<String>, filename: impl AsRef<Path>) -> Result<Self> {
        let name = name.into();
        tracing::debug!("loading `{name}` mesh`");

        // Load
        let filename = filename.as_ref();
        let mut obj_file = BufReader::new(
            File::open(
                fs::canonicalize(filename)
                    .with_context(|| format!("failed to find mesh file: {filename:?}"))?,
            )
            .with_context(|| format!("failed to open mesh file: {filename:?}"))?,
        );

        let (models, _) =
            tobj::load_obj_buf_async(&mut obj_file, &tobj::GPU_LOAD_OPTIONS, |_| async {
                Ok((vec![tobj::Material::default()], Default::default()))
            })
            .await
            .context("failed to load OBJ file")?;

        // Vertices / Indices
        let mut vertices = vec![];
        let mut indices = vec![];
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
                let position = vec3!(
                    mesh.positions[3 * v],
                    mesh.positions[3 * v + 1],
                    mesh.positions[3 * v + 2],
                );
                let normal = vec3!(
                    mesh.normals[3 * v],
                    mesh.normals[3 * v + 1],
                    mesh.normals[3 * v + 2],
                );
                let color = if mesh.vertex_color.is_empty() {
                    vec4!(
                        mesh.normals[3 * v],
                        mesh.normals[3 * v + 1],
                        mesh.normals[3 * v + 2],
                        1.0,
                    )
                } else {
                    vec4!(1.0, 1.0, 1.0, 1.0)
                };
                let uv = if mesh.texcoords.is_empty() {
                    vec2!()
                } else {
                    // NOTE: Vulkan y-coordinate is flipped, hence 1.0 - uy
                    vec2!(mesh.texcoords[2 * v], 1.0 - mesh.texcoords[2 * v + 1],)
                };

                let vertex = Vertex::new(position, normal, color, uv);
                vertices.push(vertex);
            }

            indices.reserve(mesh.indices.len());
            indices.extend(mesh.indices.iter());
        }

        tracing::debug!("loaded `{name}` mesh successfully");

        Ok(Self {
            name,
            vertices,
            indices,
        })
    }

    /// Return the size of this mesh in bytes.
    #[inline]
    #[must_use]
    pub fn size(&self) -> usize {
        self.vertices.len() * mem::size_of::<Vertex>()
    }
}

#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct Texture {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub mip_levels: u32,
    pub pixels: Vec<u8>,
    pub size: usize,
}

impl Texture {
    /// Load a new `Texture` from a png file.
    pub async fn from_file(name: impl Into<String>, filename: impl AsRef<Path>) -> Result<Self> {
        let name = name.into();
        tracing::debug!("loading `{name}` texture");

        // Load
        let filename = filename.as_ref();
        // TODO: make async
        let image_file = BufReader::new(
            File::open(
                fs::canonicalize(filename)
                    .with_context(|| format!("failed to find texture file: {filename:?}"))?,
            )
            .with_context(|| format!("failed to open texture file: {filename:?}"))?,
        );

        let decoder = png::Decoder::new(image_file);
        let mut reader = decoder.read_info().context("failed to read texture info")?;
        let mut pixels = vec![0; reader.output_buffer_size()];
        let info = reader
            .next_frame(&mut pixels)
            .context("failed to read texture frame")?;
        let size = info.buffer_size();
        let mip_levels = (info.width.max(info.height) as f32).log2().floor() as u32 + 1;

        tracing::debug!("loaded `{name}` texture successsfully");

        Ok(Self {
            name,
            width: info.width,
            height: info.height,
            mip_levels,
            pixels,
            size,
        })
    }
}
