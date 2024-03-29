//! Vulkan buffer and device memory allocations.

use super::{command_pool, debug::Debug, device::Device};
use anyhow::{Context, Result};
use ash::vk;
use derive_more::{Deref, DerefMut};
use std::{ffi::c_void, mem, ptr, slice};

pub(crate) fn padded_uniform_buffer_size<T>(device: &Device) -> u64 {
    let min_alignment = device
        .info
        .properties
        .limits
        .min_uniform_buffer_offset_alignment;
    let mut aligned_size = mem::size_of::<T>() as u64;
    if min_alignment > 0 {
        aligned_size = (aligned_size + min_alignment - 1) & !(min_alignment - 1);
    }
    aligned_size
}

pub(crate) fn padded_storage_buffer_size<T>(device: &Device) -> u64 {
    let min_alignment = device
        .info
        .properties
        .limits
        .min_storage_buffer_offset_alignment;
    let mut aligned_size = mem::size_of::<T>() as u64;
    if min_alignment > 0 {
        aligned_size = (aligned_size + min_alignment - 1) & !(min_alignment - 1);
    }
    aligned_size
}

#[derive(Default, Deref, DerefMut)]
#[must_use]
pub(crate) struct AllocatedBuffer {
    #[deref]
    #[deref_mut]
    pub(crate) handle: vk::Buffer,
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) mapped_memory: Option<*mut c_void>,
}

impl AllocatedBuffer {
    /// Creates a [vk::Buffer] with associated [vk::DeviceMemory] for the given parameters.
    pub(crate) fn create(
        device: &Device,
        name: &str,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
        #[allow(unused)] debug: Option<&Debug>,
    ) -> Result<Self> {
        tracing::debug!("creating `{name}` buffer, size: {}", size);

        // Buffer
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&buffer_create_info, None)? };

        // Memory
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory_allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(device.memory_type_index(properties, requirements)?);

        // Allocate
        let memory = unsafe { device.allocate_memory(&memory_allocate_info, None) }
            .context("failed to allocate buffer memory")?;
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }
            .context("failed to bind to buffer memory")?;

        #[cfg(debug_assertions)]
        if let Some(debug) = debug {
            use ash::vk::Handle;
            debug.set_debug_name(device, name, vk::ObjectType::BUFFER, buffer.as_raw());
            debug.set_debug_name(device, name, vk::ObjectType::DEVICE_MEMORY, memory.as_raw());
        }

        tracing::debug!("created `{name}` buffer successfully");

        Ok(AllocatedBuffer {
            handle: buffer,
            memory,
            mapped_memory: None,
        })
    }

    /// Create a generic memory-mapped `Buffer` instance of arbitrary length.
    pub(crate) fn create_mapped(
        device: &Device,
        name: &str,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
        #[allow(unused)] debug: Option<&Debug>,
    ) -> Result<Self> {
        tracing::debug!("creating `{name}` mapped buffer");

        let buffer = Self::create(device, name, size, usage, properties, debug)?;
        let mapped_memory = unsafe {
            device
                .map_memory(buffer.memory, 0, size, vk::MemoryMapFlags::empty())
                .context("failed to map buffer memory")?
        };

        tracing::debug!("created `{name}` mapped buffer successfully");

        Ok(Self {
            mapped_memory: Some(mapped_memory),
            ..buffer
        })
    }

    /// Create a generic array `Buffer` instance like a Vertex or Index buffer.
    pub(crate) fn create_array<T>(
        device: &Device,
        name: &str,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        usage: vk::BufferUsageFlags,
        array: &[T],
        #[allow(unused)] debug: Option<&Debug>,
    ) -> Result<Self> {
        let size = std::mem::size_of_val(array) as u64;

        tracing::debug!("creating `{name}` array buffer");

        // Staging Buffer
        let staging = Self::create(
            device,
            name,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
            debug,
        )?;

        // Copy Staging
        unsafe {
            let memory = device
                .map_memory(staging.memory, 0, size, vk::MemoryMapFlags::empty())
                .context("failed to map array buffer memory")?;
            ptr::copy_nonoverlapping(array.as_ptr(), memory.cast(), array.len());
            device.unmap_memory(staging.memory);
        }

        // Buffer
        let buffer = Self::create(
            device,
            name,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            debug,
        )?;

        // Copy
        staging.copy(device, command_pool, queue, &buffer, size)?;

        // Cleanup
        unsafe {
            device.destroy_buffer(staging.handle, None);
            device.free_memory(staging.memory, None);
        }

        tracing::debug!("created `{name}` array buffer successfully");

        Ok(buffer)
    }

    /// Copy one [Buffer] another.
    pub(crate) fn copy(
        &self,
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        destination: &AllocatedBuffer,
        size: vk::DeviceSize,
    ) -> Result<()> {
        // TODO: need to wait for device idle?
        command_pool::immediate_command(device, command_pool, queue, |command_buffer| {
            let region = vk::BufferCopy::builder().size(size);
            unsafe {
                device.cmd_copy_buffer(
                    command_buffer,
                    self.handle,
                    destination.handle,
                    slice::from_ref(&region),
                );
            };
        })?;
        Ok(())
    }

    /// Copy a [vk::Buffer] to a [vk::Image].
    pub(crate) fn copy_to_image(
        &self,
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        image: vk::Image,
        width: u32,
        height: u32,
    ) -> Result<()> {
        command_pool::immediate_command(device, command_pool, queue, |command_buffer| {
            let region = vk::BufferImageCopy::builder()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                )
                .image_offset(vk::Offset3D::default())
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                });

            unsafe {
                device.cmd_copy_buffer_to_image(
                    command_buffer,
                    self.handle,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    slice::from_ref(&region),
                );
            }
        })?;
        Ok(())
    }

    /// Destroy a `Buffer` instance.
    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) {
        device.free_memory(self.memory, None);
        device.destroy_buffer(self.handle, None);
    }
}
