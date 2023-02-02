//! Vulkan swapchain.

use super::{device::Device, surface::Surface};
use crate::{render_bail, window::PhysicalSize, Result};
use anyhow::Context;
use ash::{extensions::khr, vk};
use derive_more::{Deref, DerefMut};

pub(crate) const MAX_FRAMES_IN_FLIGHT: usize = 2;

#[derive(Clone, Deref, DerefMut)]
#[must_use]
pub(crate) struct Swapchain {
    pub(crate) loader: khr::Swapchain,
    #[deref]
    #[deref_mut]
    pub(crate) handle: vk::SwapchainKHR,
    pub(crate) format: vk::Format,
    pub(crate) extent: vk::Extent2D,
    pub(crate) _images: Vec<vk::Image>,
    pub(crate) image_views: Vec<vk::ImageView>,
}

impl Swapchain {
    /// Create a `Swapchain` instance containing a handle to [`vk::SwapchainKHR`],
    /// [vk::Format], [`vk::Extent2D`] and the associated swapchain [vk::Image]s and
    /// [`vk::ImageView`]s.
    pub(crate) fn create(
        instance: &ash::Instance,
        device: &Device,
        surface: &Surface,
        size: PhysicalSize<u32>,
        old_swapchain: Option<vk::SwapchainKHR>,
    ) -> Result<Self> {
        tracing::debug!("creating swapchain");

        let Some(swapchain_support) = device.info.swapchain_support.as_ref() else {
            render_bail!("{} does not support swapchains", device.info.name);
        };

        // Select swapchain format
        let surface_format = swapchain_support
            .formats
            .iter()
            .find(|surface_format| {
                surface_format.format == vk::Format::B8G8R8A8_SRGB
                    && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| swapchain_support.formats.first())
            .copied()
            .context("failed to find a valid swapchain format")?;

        // Select present mode
        let present_mode = swapchain_support
            .present_modes
            .iter()
            .find(|&&present_mode| present_mode == vk::PresentModeKHR::MAILBOX)
            .copied()
            .unwrap_or(vk::PresentModeKHR::FIFO);

        // Select extent
        // NOTE: current_extent.width equal to u32::MAX means that the swapchain image
        // resolution can differ from the window resolution
        let capabilities = unsafe {
            surface
                .loader
                .get_physical_device_surface_capabilities(device.physical, surface.handle)
        }
        .context("Failed to query for surface capabilities.")?;
        let image_extent = if capabilities.current_extent.width == u32::MAX {
            let min = capabilities.min_image_extent;
            let max = capabilities.max_image_extent;
            vk::Extent2D {
                width: size.width.clamp(min.width, max.width),
                height: size.height.clamp(min.height, max.height),
            }
        } else {
            capabilities.current_extent
        };

        // Determine image_count
        let min_image_count = capabilities.min_image_count + 1;
        let min_image_count =
            if capabilities.max_image_count > 0 && min_image_count > capabilities.max_image_count {
                min_image_count.min(capabilities.max_image_count)
            } else {
                min_image_count
            };

        // Select image sharing mode, concurrent vs exclusive
        let indices = device.info.queue_family_indices;
        let queues = [indices.graphics, indices.present];
        let (image_sharing_mode, queue_family_indices) = if indices.graphics != indices.present {
            (vk::SharingMode::CONCURRENT, queues.as_slice())
        } else {
            (vk::SharingMode::EXCLUSIVE, [].as_slice())
        };

        // Create info
        let mut swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface.handle)
            .min_image_count(min_image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
            .image_extent(image_extent)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(image_sharing_mode)
            .queue_family_indices(queue_family_indices)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .image_array_layers(1);
        if let Some(old_swapchain) = old_swapchain {
            swapchain_create_info = swapchain_create_info.old_swapchain(old_swapchain);
        }

        // Create swapchain
        let swapchain_loader = khr::Swapchain::new(instance, device);
        // SAFETY: All create_info values are set correctly above with valid lifetimes.
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None) }
            .context("failed to create swapchain")?;

        // Get images
        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }
            .context("failed to get swapchain images")?;

        // Create image views
        let image_views = images
            .iter()
            .map(|&image| {
                let image_view_create_info = vk::ImageViewCreateInfo::builder()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .components(vk::ComponentMapping::default())
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1)
                            .build(),
                    )
                    .image(image);
                let image_view = unsafe { device.create_image_view(&image_view_create_info, None) }
                    .context("failed to create image view")?;
                Ok(image_view)
            })
            .collect::<Result<Vec<_>>>()?;

        tracing::debug!("created swapchain successfully");

        Ok(Self {
            loader: swapchain_loader,
            handle: swapchain,
            format: surface_format.format,
            extent: image_extent,
            _images: images,
            image_views,
        })
    }

    /// Create a list of [vk::Framebuffer`]s for this swapchain.
    pub(crate) fn create_framebuffers(
        &self,
        device: &ash::Device,
        render_pass: vk::RenderPass,
        color_view: Option<vk::ImageView>,
        depth_view: Option<vk::ImageView>,
    ) -> Result<Vec<vk::Framebuffer>> {
        tracing::debug!("creating framebuffers");

        let framebuffers = self
            .image_views
            .iter()
            .map(|&image_view| {
                let attachments = if let Some((color_view, depth_view)) = color_view.zip(depth_view)
                {
                    vec![color_view, depth_view, image_view]
                } else if let Some(depth_view) = depth_view {
                    vec![image_view, depth_view]
                } else {
                    vec![image_view]
                };
                let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(self.extent.width)
                    .height(self.extent.height)
                    .layers(1);
                let framebuffer =
                    unsafe { device.create_framebuffer(&framebuffer_create_info, None) }
                        .context("failed to create framebuffer")?;
                Ok(framebuffer)
            })
            .collect::<Result<Vec<_>>>()
            .context("failed to create image buffers")?;

        tracing::debug!("created framebuffers successfully");

        Ok(framebuffers)
    }

    /// Destroy a `Swapchain` instance.
    pub(crate) unsafe fn destroy(
        &self,
        device: &ash::Device,
        allocator: Option<&vk::AllocationCallbacks>,
    ) {
        for image_view in self.image_views.iter().copied() {
            device.destroy_image_view(image_view, allocator);
        }
        self.loader.destroy_swapchain(self.handle, allocator);
    }
}
