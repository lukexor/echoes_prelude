//! Frame

use super::{
    buffer::AllocatedBuffer,
    command_pool,
    debug::Debug,
    device::{Device, QueueFamily},
    swapchain::MAX_FRAMES_IN_FLIGHT,
};
use crate::{
    camera::CameraData,
    mesh::{ObjectData, MAX_OBJECTS},
    scene::SceneData,
};
use anyhow::{Context, Result};
use ash::vk;
use std::{mem, slice};

#[must_use]
pub(crate) struct Frame {
    pub(crate) present_semaphor: vk::Semaphore,
    pub(crate) render_semaphor: vk::Semaphore,
    pub(crate) render_fence: vk::Fence,
    pub(crate) command_pool: vk::CommandPool,
    pub(crate) command_buffers: Vec<vk::CommandBuffer>,
    pub(crate) camera_buffer: AllocatedBuffer,
    pub(crate) object_buffer: AllocatedBuffer,
    pub(crate) descriptor_sets: Vec<vk::DescriptorSet>,
}

impl Frame {
    /// Create a `Frame` instance with [vk::Semaphore]s and [vk::Fence]s.
    pub(crate) fn create(
        device: &Device,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        scene_buffer: &AllocatedBuffer,
        debug: Option<&Debug>,
    ) -> Result<Vec<Self>> {
        tracing::debug!("creating frames");

        let mut frames = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let semaphor_create_info = vk::SemaphoreCreateInfo::builder();
            let present = unsafe { device.create_semaphore(&semaphor_create_info, None) }
                .context("failed to create semaphor")?;
            let render = unsafe { device.create_semaphore(&semaphor_create_info, None) }
                .context("failed to create semaphor")?;

            let fence_create_info =
                vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
            let render_fence = unsafe { device.create_fence(&fence_create_info, None) }
                .context("failed to create semaphor")?;

            let command_pool = command_pool::create(
                device,
                QueueFamily::Graphics,
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            )?;
            let command_buffers = command_pool::create_buffers(device, command_pool, 1)?;

            #[cfg(debug_assertions)]
            if let Some(debug) = debug {
                use ash::vk::Handle;
                debug.set_debug_name(
                    device,
                    "present_semaphor",
                    vk::ObjectType::SEMAPHORE,
                    present.as_raw(),
                );
                debug.set_debug_name(
                    device,
                    "render_semaphor",
                    vk::ObjectType::SEMAPHORE,
                    render.as_raw(),
                );
                debug.set_debug_name(
                    device,
                    "render_fence",
                    vk::ObjectType::FENCE,
                    render_fence.as_raw(),
                );
            }

            let camera_buffer = AllocatedBuffer::create(
                device,
                "camera",
                mem::size_of::<CameraData>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
                debug,
            )?;
            let object_buffer = AllocatedBuffer::create(
                device,
                "object",
                // TODO: Make this growable as objects are added.
                MAX_OBJECTS as u64 * mem::size_of::<ObjectData>() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
                debug,
            )?;

            let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(descriptor_set_layouts);
            let descriptor_sets =
                unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info) }
                    .context("failed to allocate descriptor sets")?;

            let camera_info = vk::DescriptorBufferInfo::builder()
                .buffer(camera_buffer.handle)
                .offset(0)
                .range(mem::size_of::<CameraData>() as u64);
            let scene_info = vk::DescriptorBufferInfo::builder()
                .buffer(scene_buffer.handle)
                .offset(0)
                .range(mem::size_of::<SceneData>() as u64);
            let object_info = vk::DescriptorBufferInfo::builder()
                .buffer(object_buffer.handle)
                .offset(0)
                .range(mem::size_of::<ObjectData>() as u64);

            let camera_write = vk::WriteDescriptorSet::builder()
                .dst_binding(0)
                .dst_set(descriptor_sets[0])
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(slice::from_ref(&camera_info))
                .build();
            let scene_write = vk::WriteDescriptorSet::builder()
                .dst_binding(1)
                .dst_set(descriptor_sets[0])
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .buffer_info(slice::from_ref(&scene_info))
                .build();
            let object_write = vk::WriteDescriptorSet::builder()
                .dst_binding(0)
                .dst_set(descriptor_sets[1])
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(slice::from_ref(&object_info))
                .build();

            let write_sets = [camera_write, scene_write, object_write];
            unsafe {
                device.update_descriptor_sets(&write_sets, &[]);
            };

            let frame = Self {
                present_semaphor: present,
                render_semaphor: render,
                render_fence,
                command_pool,
                command_buffers,
                camera_buffer,
                object_buffer,
                descriptor_sets,
            };
            frames.push(frame);
        }

        tracing::debug!("created frames");

        Ok(frames)
    }

    /// Destroy a `Sync` instance.
    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        self.object_buffer.destroy(device);
        self.camera_buffer.destroy(device);
        device.destroy_command_pool(self.command_pool, None);
        device.destroy_fence(self.render_fence, None);
        device.destroy_semaphore(self.render_semaphor, None);
        device.destroy_semaphore(self.present_semaphor, None);
    }
}
