//! Vulkan Command Pool and Command Buffers.

use super::device::{Device, QueueFamily};
use anyhow::{Context, Result};
use ash::vk;
use std::slice;

/// Create a [`vk::CommandPool`] instance.
pub(crate) fn create(
    device: &Device,
    queue_family: QueueFamily,
    flags: vk::CommandPoolCreateFlags,
) -> Result<vk::CommandPool> {
    tracing::debug!("creating command pool on queue family: {queue_family:?}");

    let queue_index = device.info.queue_family_indices.get(queue_family);

    let pool_create_info = vk::CommandPoolCreateInfo::builder()
        .flags(flags)
        .queue_family_index(queue_index);
    let pool = unsafe { device.create_command_pool(&pool_create_info, None) }
        .context("failed to create command pool")?;

    tracing::debug!("created command pool successfully");

    Ok(pool)
}

/// Create a [`vk::CommandBuffer`] instance.
pub(crate) fn create_buffers(
    device: &ash::Device,
    pool: vk::CommandPool,
    buffer_count: u32,
) -> Result<Vec<vk::CommandBuffer>> {
    let buffer_alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(buffer_count);
    let buffers = unsafe { device.allocate_command_buffers(&buffer_alloc_info) }
        .context("failed to allocate command buffers")?;
    Ok(buffers)
}

/// Submits an immediate command to a [`vk::CommandBuffer`].
pub(crate) fn immediate_command<F: FnOnce(vk::CommandBuffer)>(
    device: &ash::Device,
    pool: vk::CommandPool,
    queue: vk::Queue,
    f: F,
) -> Result<()> {
    tracing::debug!("submitting immediate command.");

    // Allocate
    let command_allocate_info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(pool)
        .command_buffer_count(1);

    let command_buffer = unsafe { device.allocate_command_buffers(&command_allocate_info) }
        .context("failed to allocate command buffer")?[0];

    // Commands
    let command_begin_info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { device.begin_command_buffer(command_buffer, &command_begin_info) }
        .context("failed to begin command buffer")?;

    f(command_buffer);

    unsafe { device.end_command_buffer(command_buffer) }.context("failed to end command buffer")?;

    // Submit
    let command_buffers = [command_buffer];
    let submit_info = vk::SubmitInfo::builder().command_buffers(&command_buffers);

    unsafe {
        device.queue_submit(queue, slice::from_ref(&submit_info), vk::Fence::null())?;
        device.queue_wait_idle(queue)?;
        // Cleanup
        device.free_command_buffers(pool, &command_buffers);
    }

    tracing::debug!("submitted immediate command successfully");

    Ok(())
}
