//! Vulkan meshes and objects.

use super::buffer::AllocatedBuffer;
use crate::{
    matrix::Mat4,
    mesh::{Mesh, Vertex},
    vector::{Vec3, Vec4},
};
use ash::vk;
use derive_more::{Deref, DerefMut};
use std::mem;

#[derive(Deref, DerefMut)]
#[must_use]
pub(crate) struct AllocatedMesh {
    #[deref]
    #[deref_mut]
    pub(crate) handle: Mesh,
    pub(crate) vertex_buffer: AllocatedBuffer,
    pub(crate) index_buffer: AllocatedBuffer,
}

impl AllocatedMesh {
    /// Creates a new `AllocatedMesh` instance.
    pub(crate) fn new(
        mesh: Mesh,
        vertex_buffer: AllocatedBuffer,
        index_buffer: AllocatedBuffer,
    ) -> Self {
        Self {
            handle: mesh,
            vertex_buffer,
            index_buffer,
        }
    }

    /// Destroy a `Mesh` intance.
    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        self.vertex_buffer.destroy(device);
        self.index_buffer.destroy(device);
    }
}

#[derive(Debug, Clone)]
#[repr(C)]
#[must_use]
pub(crate) struct MeshPushConstants {
    pub(crate) data: Vec4,
    pub(crate) transform: Mat4,
}

#[derive(Clone, PartialEq, Eq, Hash)]
#[must_use]
pub(crate) struct Material {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) texture_descriptor_set: Option<vk::DescriptorSet>,
}

impl Material {
    pub(crate) fn new(
        pipeline: vk::Pipeline,
        layout: vk::PipelineLayout,
        texture_descriptor_set: impl Into<Option<vk::DescriptorSet>>,
    ) -> Self {
        Self {
            pipeline,
            pipeline_layout: layout,
            texture_descriptor_set: texture_descriptor_set.into(),
        }
    }
}
#[derive(Clone)]
#[must_use]
pub(crate) struct Object {
    pub(crate) mesh: String,
    pub(crate) material: Material,
    pub(crate) transform: Mat4,
}

#[derive(Clone)]
#[must_use]
pub(crate) struct VertexInputDescription {
    pub(crate) bindings: [vk::VertexInputBindingDescription; 1],
    pub(crate) attributes: [vk::VertexInputAttributeDescription; 4],
}

impl VertexInputDescription {
    pub(crate) fn get() -> Self {
        let main_binding = vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build();

        let mut offset = 0;

        let position_attribute = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0) // location index in vertex shader
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset)
            .build();
        offset += mem::size_of::<Vec3>() as u32;

        let normal_attribute = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1) // location index in vertex shader
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset)
            .build();
        offset += mem::size_of::<Vec3>() as u32;

        let color_attribute = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2) // location index in vertex shader
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset)
            .build();
        offset += mem::size_of::<Vec4>() as u32;

        let uv_attribute = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(3) // location index in vertex shader
            .format(vk::Format::R32G32_SFLOAT)
            .offset(offset)
            .build();

        Self {
            bindings: [main_binding],
            attributes: [
                position_attribute,
                normal_attribute,
                color_attribute,
                uv_attribute,
            ],
        }
    }
}
