//! Vulkan images and allocated memory.

use super::{
    buffer::AllocatedBuffer,
    command_pool::{begin_one_time_command, end_one_time_command},
    debug::Debug,
    device::Device,
};
use crate::{render::RenderSettings, render_bail, Result};
use anyhow::Context;
use ash::vk;
use asset_loader::{Asset, TextureAsset, UnpackInto};
use derive_more::{Deref, DerefMut};
use std::slice;

#[derive(Deref, DerefMut)]
#[must_use]
pub(crate) struct AllocatedImage {
    #[deref]
    #[deref_mut]
    pub(crate) handle: vk::Image,
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) view: vk::ImageView,
}

impl AllocatedImage {
    /// Create an `Image` instance with a [vk::Image] handle and associated [`vk::DeviceMemory`] with the given parameters.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create(
        device: &Device,
        name: &str,
        width: u32,
        height: u32,
        mip_levels: u32,
        samples: vk::SampleCountFlags,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        properties: vk::MemoryPropertyFlags,
        #[allow(unused)] debug: Option<&Debug>,
    ) -> Result<(vk::Image, vk::DeviceMemory)> {
        tracing::debug!("creating `{name}` image");

        // Image
        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(format)
            .tiling(tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .samples(samples)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image =
            unsafe { device.create_image(&image_info, None) }.context("failed to create image")?;

        // Memory
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let memory_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(device.memory_type_index(properties, requirements)?);
        let memory = unsafe { device.allocate_memory(&memory_info, None) }
            .context("failed to allocate image memory")?;
        unsafe { device.bind_image_memory(image, memory, 0) }
            .context("failed to bind image memory")?;

        #[cfg(debug_assertions)]
        if let Some(debug) = debug {
            use ash::vk::Handle;
            debug.set_debug_name(device, name, vk::ObjectType::IMAGE, image.as_raw());
            debug.set_debug_name(device, name, vk::ObjectType::DEVICE_MEMORY, memory.as_raw());
        }

        tracing::debug!("created `{name}` image successfully");

        Ok((image, memory))
    }

    /// Destroy an `Image` intance.
    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        device.destroy_image_view(self.view, None);
        device.free_memory(self.memory, None);
        device.destroy_image(self.handle, None);
    }

    /// Create a [vk::Image] instance to be used as a [`vk::ImageUsageFlags::COLOR_ATTACHMENT`].
    pub(crate) fn create_color(
        device: &Device,
        format: vk::Format,
        extent: vk::Extent2D,
        samples: vk::SampleCountFlags,
        #[allow(unused)] debug: Option<&Debug>,
    ) -> Result<Self> {
        // Image
        let name = "color_attachment";
        let mip_levels = 1;
        let (image, memory) = Self::create(
            device,
            name,
            extent.width,
            extent.height,
            mip_levels,
            samples,
            format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            debug,
        )?;

        // Image View
        let view = Self::create_view(
            device,
            name,
            image,
            format,
            vk::ImageAspectFlags::COLOR,
            mip_levels,
            debug,
        )?;

        tracing::debug!("created color image successfully");

        Ok(Self {
            handle: image,
            memory,
            view,
        })
    }

    /// Create a [vk::Image] instance to be used as a [`vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT`].
    pub(crate) fn create_depth(
        instance: &ash::Instance,
        device: &Device,
        extent: vk::Extent2D,
        samples: vk::SampleCountFlags,
        #[allow(unused)] debug: Option<&Debug>,
    ) -> Result<Self> {
        // Image
        let format = device.get_depth_format(instance, vk::ImageTiling::OPTIMAL)?;
        let name = "depth_attachment";
        let mip_levels = 1;
        let (image, memory) = Self::create(
            device,
            name,
            extent.width,
            extent.height,
            mip_levels,
            samples,
            format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            debug,
        )?;

        // Image View
        let view = Self::create_view(
            device,
            name,
            image,
            format,
            vk::ImageAspectFlags::DEPTH,
            mip_levels,
            debug,
        )?;

        tracing::debug!("created depth image successfully");

        Ok(Self {
            handle: image,
            memory,
            view,
        })
    }

    /// Create a [vk::Image] instance to be used as a texture.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_texture(
        instance: &ash::Instance,
        device: &Device,
        command_pool: vk::CommandPool,
        graphics_queue: vk::Queue,
        texture: &TextureAsset,
        #[allow(unused)] debug: Option<&Debug>,
    ) -> Result<Self> {
        let texture_name = texture.name();

        tracing::debug!(
            "creating texture named `{texture_name}`, size: {}",
            texture.size()
        );

        // Staging Buffer
        let staging_buffer = AllocatedBuffer::create(
            device,
            texture_name,
            texture.size() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
            debug,
        )?;

        // Copy Staging
        unsafe {
            let memory = device
                .map_memory(
                    staging_buffer.memory,
                    0,
                    texture.size() as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .context("failed to map texture buffer memory")?;
            let memory = slice::from_raw_parts_mut(memory.cast(), texture.size());
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            rt.block_on(texture.unpack_into(memory))
                .context("failed to unpack texture data")?;
            device.handle.unmap_memory(staging_buffer.memory);
        };

        // Texture Image
        // TODO: MIP
        let mip_levels = 1;
        let (image, memory) = AllocatedImage::create(
            device,
            texture_name,
            texture.width,
            texture.height,
            mip_levels,
            vk::SampleCountFlags::TYPE_1,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            debug,
        )?;

        // Transition
        {
            let command_buffer = begin_one_time_command(device, command_pool)?;
            let image_barrier = vk::ImageMemoryBarrier::builder()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(mip_levels)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                )
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);

            #[allow(trivial_casts)]
            unsafe {
                device.handle.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[] as &[vk::MemoryBarrier; 0],
                    &[] as &[vk::BufferMemoryBarrier; 0],
                    slice::from_ref(&image_barrier),
                );
            };
            end_one_time_command(device, command_pool, command_buffer, graphics_queue)?;
        }

        // Copy
        staging_buffer.copy_to_image(
            device,
            command_pool,
            graphics_queue,
            image,
            texture.width,
            texture.height,
        )?;

        // Mipmap
        // TODO: move to asset loader
        let format = vk::Format::R8G8B8A8_SRGB;
        Self::generate_mipmaps(
            instance,
            device,
            command_pool,
            graphics_queue,
            image,
            format,
            texture.width,
            texture.height,
            mip_levels,
        )?;

        let view = Self::create_view(
            device,
            texture_name,
            image,
            format,
            vk::ImageAspectFlags::COLOR,
            mip_levels,
            debug,
        )?;

        // Cleanup
        unsafe {
            staging_buffer.destroy(device);
        }

        tracing::debug!("created texture named `{texture_name}` successfully");

        Ok(Self {
            handle: image,
            memory,
            view,
        })
    }

    /// Create a [`vk::ImageView`] instance.
    pub(crate) fn create_view(
        device: &ash::Device,
        name: &str,
        image: vk::Image,
        format: vk::Format,
        aspects: vk::ImageAspectFlags,
        mip_levels: u32,
        #[allow(unused)] debug: Option<&Debug>,
    ) -> Result<vk::ImageView> {
        tracing::debug!("creating `{name}` image view");

        let view_create_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(aspects)
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            );
        let view = unsafe { device.create_image_view(&view_create_info, None) }
            .context("failed to create image view")?;

        #[cfg(debug_assertions)]
        if let Some(debug) = debug {
            use ash::vk::Handle;
            debug.set_debug_name(device, name, vk::ObjectType::IMAGE_VIEW, view.as_raw());
        }

        tracing::debug!("created `{name}` image view successfully");

        Ok(view)
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_mipmaps(
        instance: &ash::Instance,
        device: &Device,
        command_pool: vk::CommandPool,
        graphics_queue: vk::Queue,
        image: vk::Image,
        format: vk::Format,
        width: u32,
        height: u32,
        mip_levels: u32,
    ) -> Result<()> {
        tracing::debug!("generating mipmaps");

        // Check Support
        let properties =
            unsafe { instance.get_physical_device_format_properties(device.physical, format) };
        if !properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
        {
            render_bail!("device does not support linear blitting!");
        }

        // Mipmap
        let command_buffer = begin_one_time_command(device, command_pool)?;

        let mut barrier = vk::ImageMemoryBarrier::builder()
            .image(image)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .layer_count(1)
                    .level_count(1)
                    .build(),
            );

        let mut mip_width = width as i32;
        let mut mip_height = height as i32;

        for i in 1..mip_levels {
            barrier.subresource_range.base_mip_level = i - 1;
            barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

            #[allow(trivial_casts)]
            unsafe {
                device.handle.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[] as &[vk::MemoryBarrier; 0],
                    &[] as &[vk::BufferMemoryBarrier; 0],
                    slice::from_ref(&barrier),
                );
            }

            let blit = vk::ImageBlit::builder()
                .src_offsets([
                    vk::Offset3D::default(),
                    vk::Offset3D {
                        x: mip_width,
                        y: mip_height,
                        z: 1,
                    },
                ])
                .src_subresource(
                    vk::ImageSubresourceLayers::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(i - 1)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                )
                .dst_offsets([
                    vk::Offset3D::default(),
                    vk::Offset3D {
                        x: (mip_width / 2).max(1),
                        y: (mip_height / 2).max(1),
                        z: 1,
                    },
                ])
                .dst_subresource(
                    vk::ImageSubresourceLayers::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(i)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                );

            unsafe {
                device.handle.cmd_blit_image(
                    command_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    slice::from_ref(&blit),
                    vk::Filter::LINEAR,
                );
            }

            barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
            barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            #[allow(trivial_casts)]
            unsafe {
                device.handle.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[] as &[vk::MemoryBarrier; 0],
                    &[] as &[vk::BufferMemoryBarrier; 0],
                    slice::from_ref(&barrier),
                );
            }

            mip_width = (mip_width / 2).max(1);
            mip_height = (mip_height / 2).max(1);
        }

        barrier.subresource_range.base_mip_level = mip_levels - 1;
        barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        #[allow(trivial_casts)]
        unsafe {
            device.handle.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[] as &[vk::MemoryBarrier; 0],
                &[] as &[vk::BufferMemoryBarrier; 0],
                slice::from_ref(&barrier),
            );
        }

        end_one_time_command(device, command_pool, command_buffer, graphics_queue)?;

        tracing::debug!("generated mipmaps successfully");

        Ok(())
    }

    pub(crate) fn create_sampler(
        device: &ash::Device,
        settings: &RenderSettings,
    ) -> Result<vk::Sampler> {
        tracing::debug!("creating sampler");

        let sampler_create_info = vk::SamplerCreateInfo::builder()
            .min_filter(vk::Filter::NEAREST)
            .mag_filter(vk::Filter::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(settings.sampler_ansiotropy)
            .max_anisotropy(16.0)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .min_lod(0.0) // Optional
            .max_lod(settings.level_of_detail)
            .mip_lod_bias(0.0);

        let sampler = unsafe { device.create_sampler(&sampler_create_info, None) }
            .context("failed to create sampler")?;

        tracing::debug!("created sampler successfully");

        Ok(sampler)
    }
}
