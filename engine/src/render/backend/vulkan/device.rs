//! Vulkan Devices.

use super::{surface::Surface, ENABLED_LAYER_NAMES};
use crate::{platform, render::RenderSettings, render_bail, Result};
use anyhow::Context;
use ash::vk;
use derive_more::{Deref, DerefMut};
use std::{collections::HashSet, ffi::CStr};

#[derive(Deref, DerefMut)]
pub(crate) struct Device {
    pub(crate) physical: vk::PhysicalDevice,
    #[deref]
    #[deref_mut]
    pub(crate) handle: ash::Device,
    pub(crate) info: DeviceInfo,
    pub(crate) graphics_queue: vk::Queue,
    pub(crate) present_queue: vk::Queue,
    pub(crate) _transfer_queue: vk::Queue,
}

impl Device {
    /// Select a preferred [`vk::PhysicalDevice`] and create a logical [ash::Device].
    pub(crate) fn create(
        instance: &ash::Instance,
        surface: &Surface,
        settings: &RenderSettings,
    ) -> Result<Self> {
        tracing::debug!("selecting physical device and creating logical device");

        let (physical_device, info) = Self::select_physical_device(instance, surface, settings)?;
        tracing::debug!("selected physical device successfully");

        let device = Self::create_logical_device(instance, physical_device, &info, settings)?;
        tracing::debug!("created logical device successfully");

        let [graphics_queue, present_queue, transfer_queue] = [
            QueueFamily::Graphics,
            QueueFamily::Present,
            QueueFamily::Transfer,
        ]
        .map(|family| unsafe { device.get_device_queue(info.queue_family_indices.get(family), 0) });

        Ok(Self {
            physical: physical_device,
            handle: device,
            info,
            graphics_queue,
            present_queue,
            _transfer_queue: transfer_queue,
        })
    }

    /// Get a preferred depth [vk::Format] with the given [`vk::ImageTiling`].
    pub(crate) fn get_depth_format(
        &self,
        instance: &ash::Instance,
        tiling: vk::ImageTiling,
    ) -> Result<vk::Format> {
        let candidates = [
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ];
        let features = vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT;
        let format = candidates
            .iter()
            .find(|&&format| {
                let properties = unsafe {
                    instance.get_physical_device_format_properties(self.physical, format)
                };
                match tiling {
                    vk::ImageTiling::LINEAR => properties.linear_tiling_features.contains(features),
                    vk::ImageTiling::OPTIMAL => {
                        properties.optimal_tiling_features.contains(features)
                    }
                    _ => false,
                }
            })
            .copied()
            .context("failed to find supported depth format!")?;
        Ok(format)
    }

    /// Get the [`vk::MemoryType`] index with the desired [`vk::MemoryPropertyFlags`] and
    /// [`vk::MemoryRequirements`].
    pub(crate) fn memory_type_index(
        &self,
        properties: vk::MemoryPropertyFlags,
        requirements: vk::MemoryRequirements,
    ) -> Result<u32> {
        let memory = self.info.memory_properties;
        let index = (0..memory.memory_type_count)
            .find(|&index| {
                let suitable = (requirements.memory_type_bits & (1 << index)) != 0;
                let memory_type = memory.memory_types[index as usize];
                suitable && memory_type.property_flags.contains(properties)
            })
            .context("failed to find suitable memory type.")?;
        Ok(index)
    }

    /// Select a preferred [`vk::PhysicalDevice`].
    fn select_physical_device(
        instance: &ash::Instance,
        surface: &Surface,
        settings: &RenderSettings,
    ) -> Result<(vk::PhysicalDevice, DeviceInfo)> {
        tracing::debug!("selecting physical device");

        // Select physical device
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .context("failed to enumerate physical devices")?;
        if physical_devices.is_empty() {
            render_bail!("failed to find any devices with vulkan support");
        }

        let mut devices = physical_devices
            .into_iter()
            .filter_map(|physical_device| {
                DeviceInfo::query(instance, physical_device, surface, settings)
                    .map_or(None, |info| {
                        (info.rating > 0).then_some((physical_device, info))
                    })
            })
            .collect::<Vec<(vk::PhysicalDevice, DeviceInfo)>>();
        devices.sort_by_key(|(_, info)| info.rating);
        let selected_device = devices
            .pop()
            .context("failed to find a suitable physical device")?;

        tracing::debug!("selected physical device successfully");

        Ok(selected_device)
    }

    /// Create a logical [vk::Device] instance.
    fn create_logical_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        info: &DeviceInfo,
        settings: &RenderSettings,
    ) -> Result<ash::Device> {
        tracing::debug!("creating logical device");

        let queue_priorities = [1.0];
        let queue_create_infos = info
            .queue_family_indices
            .unique()
            .iter()
            .map(|&index| {
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(index)
                    .queue_priorities(&queue_priorities)
                    .build()
            })
            .collect::<Vec<vk::DeviceQueueCreateInfo>>();
        let enabled_features = vk::PhysicalDeviceFeatures::builder()
            .sampler_anisotropy(settings.sampler_ansiotropy && info.sampler_anisotropy_support)
            .sample_rate_shading(settings.sample_shading);
        let enabled_layer_names = ENABLED_LAYER_NAMES.map(CStr::as_ptr);
        let enabled_extensions = info
            .required_extensions
            .iter()
            .copied()
            .map(CStr::as_ptr)
            .collect::<Vec<_>>();
        let mut shader_draw_features =
            vk::PhysicalDeviceShaderDrawParametersFeatures::builder().shader_draw_parameters(true);
        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&enabled_features)
            .enabled_layer_names(&enabled_layer_names)
            .enabled_extension_names(&enabled_extensions)
            .push_next(&mut shader_draw_features);

        // SAFETY: All create_info values are set correctly above with valid lifetimes.
        let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }
            .context("failed to create logical device")?;

        tracing::debug!("creating logical device");

        Ok(device)
    }
}

#[derive(Debug, Clone)]
#[must_use]
#[allow(unused)]
pub(crate) struct DeviceInfo {
    pub(crate) name: String,
    pub(crate) properties: vk::PhysicalDeviceProperties,
    pub(crate) memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub(crate) features: vk::PhysicalDeviceFeatures,
    pub(crate) queue_family_indices: QueueFamilyIndices,
    pub(crate) rating: u32,
    pub(crate) msaa_samples: vk::SampleCountFlags,
    pub(crate) swapchain_support: Option<SwapchainSupport>,
    pub(crate) required_extensions: Vec<&'static CStr>,
    pub(crate) sampler_anisotropy_support: bool,
}

impl DeviceInfo {
    /// Query [`vk::PhysicalDevice`] properties and generate a rating score.
    pub(crate) fn query(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        surface: &Surface,
        settings: &RenderSettings,
    ) -> Result<Self> {
        let mut rating = 10;

        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let features = unsafe { instance.get_physical_device_features(physical_device) };
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let msaa_samples = if settings.msaa {
            Self::get_max_sample_count(properties, &mut rating)
        } else {
            vk::SampleCountFlags::TYPE_1
        };
        let queue_family_indices = QueueFamily::query(instance, physical_device, surface)?;
        let swapchain_support = SwapchainSupport::query(physical_device, surface)
            .map_err(|err| tracing::error!("{err:?}"))
            .ok();
        let sampler_anisotropy_support = features.sampler_anisotropy == 1;

        // Bonus if graphics + present are the same queue
        if queue_family_indices.graphics == queue_family_indices.present {
            rating += 100;
        }
        // Bonus if has a dedicated transfer queue
        if queue_family_indices.graphics != queue_family_indices.transfer {
            rating += 500;
        }
        // Large bonus for being a discrete GPU
        if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            rating += 1000;
        }
        // Higher texture resolution bonus
        rating += properties.limits.max_image_dimension2_d;
        if sampler_anisotropy_support {
            rating += 100;
        }

        // SAFETY: device_name is provided by Vulkan and is a valid CStr.
        let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_string_lossy()
            .to_string();
        let required_extensions = Self::required_extensions(instance, physical_device);
        if required_extensions.len() != platform::REQUIRED_DEVICE_EXTENSIONS.len()
            || swapchain_support.is_none()
        {
            tracing::warn!("Device `{name}` does not meet minimum device requirements");
            rating = 0;
        }

        tracing::debug!("Device Information:\nName: {}\nRating: {}\nQueue Family Indices: {:?}\nSwapchain Support: {}\nAnsiotropy Support: {}\nExtensions Supported: {:?}",
            name,
            rating,
            queue_family_indices,
            swapchain_support.is_some(),
            sampler_anisotropy_support,
            required_extensions,
        );

        Ok(Self {
            name,
            properties,
            memory_properties,
            features,
            queue_family_indices,
            rating,
            msaa_samples,
            swapchain_support,
            required_extensions,
            sampler_anisotropy_support,
        })
    }

    /// Return the maximum usable sample count for multisampling for a [`vk::PhysicalDevice`].
    pub(crate) fn get_max_sample_count(
        properties: vk::PhysicalDeviceProperties,
        rating: &mut u32,
    ) -> vk::SampleCountFlags {
        let count = properties
            .limits
            .framebuffer_color_sample_counts
            .min(properties.limits.framebuffer_depth_sample_counts);
        if count.contains(vk::SampleCountFlags::TYPE_64) {
            *rating += 64;
            vk::SampleCountFlags::TYPE_64
        } else if count.contains(vk::SampleCountFlags::TYPE_32) {
            *rating += 32;
            vk::SampleCountFlags::TYPE_32
        } else if count.contains(vk::SampleCountFlags::TYPE_16) {
            *rating += 16;
            vk::SampleCountFlags::TYPE_16
        } else if count.contains(vk::SampleCountFlags::TYPE_8) {
            *rating += 8;
            vk::SampleCountFlags::TYPE_8
        } else if count.contains(vk::SampleCountFlags::TYPE_4) {
            *rating += 4;
            vk::SampleCountFlags::TYPE_4
        } else if count.contains(vk::SampleCountFlags::TYPE_2) {
            *rating += 2;
            vk::SampleCountFlags::TYPE_2
        } else {
            *rating += 1;
            vk::SampleCountFlags::TYPE_1
        }
    }

    /// Whether a [`vk::PhysicalDevice`] supports the required device extensions for this platform.
    pub(crate) fn required_extensions(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Vec<&'static CStr> {
        let device_extensions =
            unsafe { instance.enumerate_device_extension_properties(physical_device) };
        let Ok(device_extensions) = device_extensions else {
                tracing::error!("failed to enumerate device extensions");
                return vec![];
            };
        let extension_names = device_extensions
            .iter()
            // SAFETY: extension_name is provided by Vulkan and is a valid CStr.
            .map(|extension_properties| unsafe {
                CStr::from_ptr(extension_properties.extension_name.as_ptr())
            })
            .collect::<HashSet<_>>();
        platform::REQUIRED_DEVICE_EXTENSIONS
            .iter()
            .filter(|&extension_name| extension_names.contains(extension_name))
            .copied()
            .collect::<Vec<_>>()
    }
}

#[derive(Debug, Clone)]
#[must_use]
pub(crate) struct SwapchainSupport {
    pub(crate) formats: Vec<vk::SurfaceFormatKHR>,
    pub(crate) present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    /// Queries a [`vk::PhysicalDevice`] for [`vk::SwapchainKHR`] extension support.
    pub(crate) fn query(physical_device: vk::PhysicalDevice, surface: &Surface) -> Result<Self> {
        let formats = unsafe {
            surface
                .loader
                .get_physical_device_surface_formats(physical_device, **surface)
        }
        .context("Failed to query for surface formats.")?;

        let present_modes = unsafe {
            surface
                .loader
                .get_physical_device_surface_present_modes(physical_device, **surface)
        }
        .context("Failed to query for surface present mode.")?;

        if formats.is_empty() || present_modes.is_empty() {
            render_bail!("no available swapchain formats or present_modes");
        }

        Ok(Self {
            formats,
            present_modes,
        })
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[must_use]
#[allow(unused)]
pub(crate) enum QueueFamily {
    Graphics,
    Present,
    Compute,
    Transfer,
}

impl QueueFamily {
    /// Queries for desired [vk::Queue] families from a [`vk::PhysicalDevice`].
    pub(crate) fn query(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        surface: &Surface,
    ) -> Result<QueueFamilyIndices> {
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let mut queue_family_indices = QueueFamilyIndicesBuilder::default();
        let mut min_transfer_score = 255;
        for (index, queue_family) in queue_families.iter().enumerate() {
            let queue_index = index as u32;
            let mut current_transfer_score = 0;

            let has_surface_support = unsafe {
                surface.loader.get_physical_device_surface_support(
                    physical_device,
                    queue_index,
                    surface.handle,
                )
            }
            .unwrap_or_default();

            if queue_family_indices.graphics.is_none() {
                current_transfer_score += 1;
                queue_family_indices.graphics = Some(queue_index);
            }
            // Prioritize queues supporting both graphics and present
            if queue_family_indices.graphics.is_some() && has_surface_support {
                queue_family_indices.present = Some(queue_index);
                current_transfer_score += 1;
            }

            if queue_family.queue_flags.contains(vk::QueueFlags::COMPUTE)
                && queue_family_indices.compute.is_none()
            {
                queue_family_indices.compute = Some(queue_index);
                current_transfer_score += 1;
            }

            if queue_family.queue_flags.contains(vk::QueueFlags::TRANSFER)
                && current_transfer_score <= min_transfer_score
            {
                min_transfer_score = current_transfer_score;
                if queue_family_indices.transfer.is_none() {
                    queue_family_indices.transfer = Some(queue_index);
                }
            }

            if queue_family_indices.is_complete() {
                break;
            }
        }

        // If we didn't find a Graphics family supporting Present, just find use first one
        // found
        if queue_family_indices.present.is_none() {
            if let Some(queue_index) = queue_families
                .iter()
                .enumerate()
                .map(|(queue_index, _)| queue_index as u32)
                .find(|&queue_index| {
                    unsafe {
                        surface.loader.get_physical_device_surface_support(
                            physical_device,
                            queue_index,
                            surface.handle,
                        )
                    }
                    .unwrap_or_default()
                })
            {
                queue_family_indices.present = Some(queue_index);
            }
        }

        queue_family_indices.build()
    }
}

#[derive(Debug, Copy, Clone)]
#[must_use]
pub(crate) struct QueueFamilyIndices {
    pub(crate) graphics: u32,
    pub(crate) present: u32,
    pub(crate) compute: u32,
    pub(crate) transfer: u32,
}

impl QueueFamilyIndices {
    /// Get the index for this [`QueueFamily`].
    #[inline]
    pub(crate) fn get(&self, family: QueueFamily) -> u32 {
        match family {
            QueueFamily::Graphics => self.graphics,
            QueueFamily::Present => self.present,
            QueueFamily::Compute => self.compute,
            QueueFamily::Transfer => self.transfer,
        }
    }

    /// Return a unique list of [`QueueFamily`] indices.
    #[inline]
    pub(crate) fn unique(&self) -> HashSet<u32> {
        HashSet::from_iter([self.graphics, self.present, self.compute, self.transfer])
    }
}

#[derive(Default, Debug, Copy, Clone)]
#[must_use]
struct QueueFamilyIndicesBuilder {
    graphics: Option<u32>,
    present: Option<u32>,
    compute: Option<u32>,
    transfer: Option<u32>,
}

impl QueueFamilyIndicesBuilder {
    /// Build [`QueueFamilyIndices`]/
    ///
    /// # Errors
    ///
    /// If any one of the [`QueueFamily`] indices are missing, then an error is returned.
    #[inline]
    pub(crate) fn build(&self) -> Result<QueueFamilyIndices> {
        match (self.graphics, self.present, self.compute, self.transfer) {
            (Some(graphics), Some(present), Some(compute), Some(transfer)) => {
                Ok(QueueFamilyIndices {
                    graphics,
                    present,
                    compute,
                    transfer,
                })
            }
            _ => render_bail!("missing required queue families"),
        }
    }

    /// Whether the [`QueueFamilyIndices`] contains all of the desired queues.
    #[inline]
    #[must_use]
    pub(crate) fn is_complete(&self) -> bool {
        self.graphics.is_some()
            && self.present.is_some()
            && self.compute.is_some()
            && self.transfer.is_some()
    }
}
