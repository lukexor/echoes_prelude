use super::RendererBackend;
use anyhow::{bail, Context as _, Result};
use ash::vk;
use std::{
    ffi::{c_void, CStr, CString},
    fmt,
};
use winit::{dpi::PhysicalSize, window::Window};

use crate::renderer::vulkan::image::Image;
use debug::{Debug, VALIDATION_LAYER_NAME};
use device::{Device, QueueFamily};
use pipeline::Pipeline;
use surface::Surface;
use swapchain::Swapchain;

pub(crate) struct Context {
    _entry: ash::Entry,
    instance: ash::Instance,
    surface: Surface,
    device: Device,
    swapchain: Swapchain,
    render_pass: vk::RenderPass,
    color_image: Image,
    depth_image: Image,
    pipeline: Pipeline,
    framebuffers: Vec<vk::Framebuffer>,
    #[cfg(debug_assertions)]
    debug: Debug,
    resized: Option<(u32, u32)>,
}

impl Drop for Context {
    fn drop(&mut self) {
        // NOTE: Drop-order is important!
        //
        // SAFETY:
        //
        // 1. Elements are destroyed in the correct order
        // 2. Only elements that have been allocated are destroyed, ensured by `initalize` being
        //    the only way to construct a Context with all values initialized.
        #[allow(clippy::expect_used)]
        unsafe {
            self.device
                .logical_device
                .device_wait_idle()
                .expect("failed to wait for device idle");

            self.destroy_swapchain();

            let device = &self.device.logical_device;
            device.destroy_device(None);
            self.surface
                .loader
                .destroy_surface(self.surface.handle, None);
            self.debug
                .utils
                .destroy_debug_utils_messenger(self.debug.messenger, None);
            self.instance.destroy_instance(None);

            log::info!("destroyed vulkan context");
        };
    }
}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Context")
            .field("surface", &self.surface)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl RendererBackend for Context {
    /// Initialize `Context`.
    fn initialize(application_name: &str, window: &Window) -> Result<Self> {
        log::info!("initializing vulkan renderer backend");

        let entry = ash::Entry::linked();
        let instance = Self::create_instance(application_name, &entry)?;
        #[cfg(debug_assertions)]
        let debug = Debug::create(&entry, &instance)?;
        let surface = Surface::create(&entry, &instance, window)?;
        let device = Device::create(&instance, &surface)?;
        let graphics_queue = device.queue_family(QueueFamily::Graphics)?;
        let present_queue = device.queue_family(QueueFamily::Present)?;
        let PhysicalSize { width, height } = window.inner_size();
        let swapchain = Swapchain::create(&instance, &device, &surface, width, height)?;
        let render_pass = Self::create_render_pass(&instance, &device, swapchain.format)?;
        // ubo_layout
        let pipeline = Pipeline::create(&device, swapchain.extent, render_pass)?;
        let color_image = Image::create_color(
            &instance,
            &device,
            &swapchain,
            #[cfg(debug_assertions)]
            &debug,
        )?;
        let depth_image = Image::create_depth(
            &instance,
            &device,
            &swapchain,
            #[cfg(debug_assertions)]
            &debug,
        )?;
        let framebuffers = Self::create_framebuffers(
            &device,
            &swapchain,
            render_pass,
            color_image.view,
            depth_image.view,
        )?;
        // command_pools
        // texture_image
        // texture_image_view
        // texture_sampler
        // model
        // vertex_buffer
        // index_buffer
        // uniform_buffers
        // descriptor_pool
        // descriptor_sets
        // command_buffers
        // sync

        log::info!("initialized vulkan renderer backend successfully");

        Ok(Context {
            _entry: entry,
            instance,
            surface,
            device,
            swapchain,
            render_pass,
            color_image,
            depth_image,
            pipeline,
            framebuffers,
            #[cfg(debug_assertions)]
            debug,
            resized: None,
        })
    }

    /// Handle window resized event.
    fn on_resized(&mut self, width: u32, height: u32) {
        self.resized = Some((width, height));
    }

    /// Draw a frame to the [Window] surface.
    fn draw_frame(&mut self) -> Result<()> {
        if let Some((width, height)) = self.resized.take() {
            self.recreate_swapchain(width, height)?;
        }
        Ok(())
    }
}

impl Context {
    /// Create a Vulkan [Instance].
    fn create_instance(application_name: &str, entry: &ash::Entry) -> Result<ash::Instance> {
        // Application Info
        let application_info = vk::ApplicationInfo::builder()
            .application_name(&CString::new(application_name)?)
            .application_version(vk::API_VERSION_1_0)
            .engine_name(&CString::new("PixEngine")?)
            .engine_version(vk::API_VERSION_1_0)
            .api_version(vk::API_VERSION_1_3)
            .build();

        // Validation Layer check
        #[cfg(debug_assertions)]
        if !entry
            .enumerate_instance_layer_properties()?
            .iter()
            .any(|layer| {
                // SAFETY: This layer name is provided by Vulkan and is a valid Cstr.
                let layer_name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
                layer_name == VALIDATION_LAYER_NAME
            })
        {
            bail!("validation layer requested but is unsupported");
        }

        // Instance Creation
        let enabled_layer_names = [
            #[cfg(debug_assertions)]
            VALIDATION_LAYER_NAME.as_ptr(),
        ];
        let enabled_extension_names = platform::enabled_extension_names();
        let instance_create_flags = platform::instance_create_flags();
        let mut create_info = vk::InstanceCreateInfo::builder()
            .application_info(&application_info)
            .enabled_layer_names(&enabled_layer_names)
            .enabled_extension_names(&enabled_extension_names)
            .flags(instance_create_flags)
            .build();

        // Debug Creation
        #[cfg(debug_assertions)]
        let debug_create_info = Debug::build_debug_create_info();
        #[cfg(debug_assertions)]
        {
            let p_next: *const vk::DebugUtilsMessengerCreateInfoEXT = &debug_create_info;
            create_info.p_next = p_next as *const c_void;
        }

        // SAFETY: All create_info values are set correctly above with valid lifetimes.
        unsafe { entry.create_instance(&create_info, None) }
            .context("failed to create vulkan instance")
    }

    /// Create a [`vk::RenderPass`] instance.
    fn create_render_pass(
        instance: &ash::Instance,
        device: &Device,
        format: vk::Format,
    ) -> Result<vk::RenderPass> {
        log::debug!("creating vulkan render pass");

        // Attachments
        let color_attachment = vk::AttachmentDescription::builder()
            .format(format)
            .samples(device.info.msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();
        let depth_stencil_attachment = vk::AttachmentDescription::builder()
            .format(device.get_depth_format(instance, vk::ImageTiling::OPTIMAL)?)
            .samples(device.info.msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .build();

        let color_resolve_attachment = vk::AttachmentDescription::builder()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .build();

        // Subpasses
        let color_attachment_refs = [vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build()];
        let depth_stencil_attachment_ref = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .build();
        let color_resolve_attachment_refs = [vk::AttachmentReference::builder()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build()];

        let subpasses = [vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_refs)
            .depth_stencil_attachment(&depth_stencil_attachment_ref)
            .resolve_attachments(&color_resolve_attachment_refs)
            .build()];

        // Dependencies
        let dependencies = [vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .build()];

        // Create
        let attachments = &[
            color_attachment,
            depth_stencil_attachment,
            color_resolve_attachment,
        ];
        let render_pass_create_info = vk::RenderPassCreateInfo::builder()
            .attachments(attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        let render_pass = unsafe {
            device
                .logical_device
                .create_render_pass(&render_pass_create_info, None)
        }
        .context("failed to create vulkan render pass")?;

        log::debug!("created vulkan render pass successfully");

        Ok(render_pass)
    }

    /// Create a list of [vk::Framebuffer`]s.
    fn create_framebuffers(
        device: &Device,
        swapchain: &Swapchain,
        render_pass: vk::RenderPass,
        color_attachment: vk::ImageView,
        depth_attachment: vk::ImageView,
    ) -> Result<Vec<vk::Framebuffer>> {
        swapchain
            .image_views
            .iter()
            .map(|&image_view| {
                let attachments = [color_attachment, depth_attachment, image_view];
                let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(swapchain.extent.width)
                    .height(swapchain.extent.height)
                    .layers(1)
                    .build();
                // SAFETY: TODO
                unsafe {
                    device
                        .logical_device
                        .create_framebuffer(&framebuffer_create_info, None)
                }
                .context("failed to create vulkan framebuffer")
            })
            .collect::<Result<Vec<_>>>()
            .context("failed to create vulkan image buffers")
    }

    /// Destroys the current swapchain and re-creates it with a new width/height.
    fn recreate_swapchain(&mut self, width: u32, height: u32) -> Result<()> {
        // SAFETY: TODO
        unsafe { self.device.logical_device.device_wait_idle() }
            .context("failed to wait for device idle")?;
        // SAFETY: TODO
        unsafe { self.destroy_swapchain() };

        self.swapchain =
            Swapchain::create(&self.instance, &self.device, &self.surface, width, height)?;
        self.render_pass =
            Self::create_render_pass(&self.instance, &self.device, self.swapchain.format)?;
        self.pipeline = Pipeline::create(&self.device, self.swapchain.extent, self.render_pass)?;
        self.color_image = Image::create_color(
            &self.instance,
            &self.device,
            &self.swapchain,
            #[cfg(debug_assertions)]
            &self.debug,
        )?;
        self.depth_image = Image::create_depth(
            &self.instance,
            &self.device,
            &self.swapchain,
            #[cfg(debug_assertions)]
            &self.debug,
        )?;
        self.framebuffers = Self::create_framebuffers(
            &self.device,
            &self.swapchain,
            self.render_pass,
            self.color_image.view,
            self.depth_image.view,
        )?;
        // uniform buffers
        // descriptor pool
        // descriptor sets
        // command buffers
        // resize images_in_flight

        Ok(())
    }

    /// Destroys the current [`vk::SwapchainKHR`] and associated [`vk::ImageView`]s. Used to clean
    /// up Vulkan when dropped and to re-create swapchain on window resize.
    unsafe fn destroy_swapchain(&mut self) {
        let device = &self.device.logical_device;

        for image in [&self.color_image, &self.depth_image] {
            device.destroy_image_view(image.view, None);
            device.free_memory(image.memory, None);
            device.destroy_image(image.handle, None);
        }
        self.framebuffers
            .iter()
            .for_each(|&framebuffer| device.destroy_framebuffer(framebuffer, None));
        device.destroy_pipeline(self.pipeline.handle, None);
        device.destroy_pipeline_layout(self.pipeline.layout, None);
        device.destroy_render_pass(self.render_pass, None);
        self.swapchain.image_views.iter().for_each(|&image_view| {
            device.destroy_image_view(image_view, None);
        });
        self.swapchain
            .loader
            .destroy_swapchain(self.swapchain.handle, None);
    }
}

mod surface {
    use super::platform;
    use anyhow::Result;
    use ash::{extensions::khr, vk};
    use std::fmt;
    use winit::{dpi::PhysicalSize, window::Window};

    #[derive(Clone)]
    #[must_use]
    pub(crate) struct Surface {
        pub(crate) handle: vk::SurfaceKHR,
        pub(crate) loader: khr::Surface,
        pub(crate) width: u32,
        pub(crate) height: u32,
    }

    impl fmt::Debug for Surface {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("Surface")
                .field("width", &self.width)
                .field("height", &self.height)
                .finish_non_exhaustive()
        }
    }

    impl Surface {
        /// Create a platform-agnostic surface instance.
        pub(crate) fn create(
            entry: &ash::Entry,
            instance: &ash::Instance,
            window: &Window,
        ) -> Result<Self> {
            log::debug!("creating vulkan surface");
            let handle = platform::create_surface(entry, instance, window)?;
            let loader = khr::Surface::new(entry, instance);
            let PhysicalSize { width, height } = window.inner_size();
            log::debug!("vulkan surface created successfully");
            Ok(Self {
                handle,
                loader,
                width,
                height,
            })
        }
    }
}

mod device {
    use super::{platform, Surface, VALIDATION_LAYER_NAME};
    use anyhow::{bail, Context as _, Result};
    use ash::vk;
    use std::{
        collections::{hash_map::Entry, HashMap, HashSet},
        ffi::CStr,
        fmt,
    };

    #[derive(Clone)]
    #[must_use]
    pub(crate) struct Device {
        pub(crate) physical_device: vk::PhysicalDevice,
        pub(crate) logical_device: ash::Device,
        pub(crate) queue_family_indices: QueueFamilyIndices,
        pub(crate) info: DeviceInfo,
    }

    impl fmt::Debug for Device {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("Device")
                .field("queue_family_indices", &self.queue_family_indices)
                .field("info", &self.info)
                .finish_non_exhaustive()
        }
    }

    impl Device {
        /// Select a preferred [`vk::PhysicalDevice`] and create a logical [ash::Device].
        pub(crate) fn create(instance: &ash::Instance, surface: &Surface) -> Result<Self> {
            log::debug!("creating vulkan device");

            let (physical_device, queue_family_indices, info) =
                Self::select_physical_device(instance, surface)?;
            let logical_device =
                Self::create_logical_device(instance, physical_device, &queue_family_indices)?;

            log::debug!("created vulkan device successfully");

            Ok(Self {
                physical_device,
                logical_device,
                queue_family_indices,
                info,
            })
        }

        pub(crate) fn queue_family(&self, family: QueueFamily) -> Result<vk::Queue> {
            Ok(unsafe {
                self.logical_device.get_device_queue(
                    *self
                        .queue_family_indices
                        .get(&family)
                        .context("failed to get queue family: {family:?}")?,
                    0,
                )
            })
        }

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
            candidates
                .iter()
                .find(|&&format| {
                    let properties = unsafe {
                        instance.get_physical_device_format_properties(self.physical_device, format)
                    };
                    match tiling {
                        vk::ImageTiling::LINEAR => {
                            properties.linear_tiling_features.contains(features)
                        }
                        vk::ImageTiling::OPTIMAL => {
                            properties.optimal_tiling_features.contains(features)
                        }
                        _ => false,
                    }
                })
                .copied()
                .context("failed to find supported depth format!")
        }

        pub(crate) fn memory_type_index(
            &self,
            instance: &ash::Instance,
            properties: vk::MemoryPropertyFlags,
            requirements: vk::MemoryRequirements,
        ) -> Result<u32> {
            // SAFETY: TODO
            let memory =
                unsafe { instance.get_physical_device_memory_properties(self.physical_device) };
            (0..memory.memory_type_count)
                .find(|&i| {
                    let suitable = (requirements.memory_type_bits & (1 << i)) != 0;
                    let memory_type = memory.memory_types[i as usize];
                    suitable && memory_type.property_flags.contains(properties)
                })
                .context("failed to find suitable vulkan memory type.")
        }

        /// Select a preferred [`vk::PhysicalDevice`].
        fn select_physical_device(
            instance: &ash::Instance,
            surface: &Surface,
        ) -> Result<(vk::PhysicalDevice, QueueFamilyIndices, DeviceInfo)> {
            // Select physical device
            // SAFETY: TODO
            let physical_devices = unsafe { instance.enumerate_physical_devices() }
                .context("failed to enumerate physical devices")?;
            if physical_devices.is_empty() {
                bail!("failed to find any devices with vulkan support");
            }

            let mut devices = physical_devices
            .into_iter()
            .filter_map(|physical_device| {
                let (queue_family_indices, info) = DeviceInfo::query(instance, physical_device, surface);
                log::debug!("Device Information:\nName: {}\nRating: {}\nQueue Family Indices: {:?}\nSwapchain Support: {}\nAnsiotropy Support: {}\nExtension Support: {}",
                    &info.name,
                    info.rating,
                    &queue_family_indices,
                    info.swapchain_support.is_some(),
                    info.sampler_anisotropy_support,
                    info.extension_support,
                );
                (info.rating > 0).then_some((physical_device, queue_family_indices, info))
            })
            .collect::<Vec<(vk::PhysicalDevice, QueueFamilyIndices, DeviceInfo)>>();
            devices.sort_by_key(|(_, _, info)| info.rating);
            let selected_device = devices
                .pop()
                .context("failed to find a suitable physical device")?;

            Ok(selected_device)
        }

        /// Create a logical [vk::Device].
        fn create_logical_device(
            instance: &ash::Instance,
            physical_device: vk::PhysicalDevice,
            queue_family_indices: &QueueFamilyIndices,
        ) -> Result<ash::Device> {
            let queue_priorities = [1.0];
            let unique_families: HashSet<u32> =
                HashSet::from_iter(queue_family_indices.values().copied());
            let queue_create_infos = unique_families
                .iter()
                .map(|&index| {
                    vk::DeviceQueueCreateInfo::builder()
                        .queue_family_index(index)
                        .queue_priorities(&queue_priorities)
                        .build()
                })
                .collect::<Vec<vk::DeviceQueueCreateInfo>>();
            let enabled_features = vk::PhysicalDeviceFeatures::builder()
                .sampler_anisotropy(true)
                .sample_rate_shading(true)
                .build();
            let enabled_layer_names = [
                #[cfg(debug_assertions)]
                VALIDATION_LAYER_NAME.as_ptr(),
            ];
            let enabled_extension_names = platform::required_device_extensions();
            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_create_infos)
                .enabled_features(&enabled_features)
                .enabled_layer_names(&enabled_layer_names)
                .enabled_extension_names(&enabled_extension_names)
                .build();
            // SAFETY: All create_info values are set correctly above with valid lifetimes.
            unsafe { instance.create_device(physical_device, &device_create_info, None) }
                .context("failed to create vulkan logical device")
        }
    }

    #[derive(Debug, Clone)]
    #[must_use]
    pub(crate) struct DeviceInfo {
        pub(crate) name: String,
        pub(crate) properties: vk::PhysicalDeviceProperties,
        pub(crate) memory_properties: vk::PhysicalDeviceMemoryProperties,
        pub(crate) features: vk::PhysicalDeviceFeatures,
        pub(crate) rating: u32,
        pub(crate) msaa_samples: vk::SampleCountFlags,
        pub(crate) swapchain_support: Option<SwapchainSupport>,
        pub(crate) extension_support: bool,
        pub(crate) sampler_anisotropy_support: bool,
    }

    impl DeviceInfo {
        /// Query [`vk::PhysicalDevice`] properties and generate a rating score.
        fn query(
            instance: &ash::Instance,
            physical_device: vk::PhysicalDevice,
            surface: &Surface,
        ) -> (QueueFamilyIndices, Self) {
            let mut rating = 10;

            // SAFETY: TODO
            let properties = unsafe { instance.get_physical_device_properties(physical_device) };
            // SAFETY: TODO
            let features = unsafe { instance.get_physical_device_features(physical_device) };
            let memory_properties =
                unsafe { instance.get_physical_device_memory_properties(physical_device) };
            let msaa_samples = Self::get_max_sample_count(properties, &mut rating);
            let queue_family_indices = QueueFamily::query(instance, physical_device, surface);
            let swapchain_support = SwapchainSupport::query(physical_device, surface)
                .map_err(|err| log::error!("{err}"))
                .ok();

            // Bonus if graphics + present are the same queue
            if queue_family_indices.get(&QueueFamily::Graphics)
                == queue_family_indices.get(&QueueFamily::Present)
            {
                rating += 100;
            }
            // Bonus if has a dedicated transfer queue
            if queue_family_indices.get(&QueueFamily::Graphics)
                != queue_family_indices.get(&QueueFamily::Transfer)
            {
                rating += 500;
            }
            // Large bonus for being a discrete GPU
            if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                rating += 1000;
            }
            // Higher texture resolution bonus
            rating += properties.limits.max_image_dimension2_d;

            // SAFETY: device_name is provided by Vulkan and is a valid CStr.
            let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
                .to_string_lossy()
                .to_string();
            let extension_support = Self::supports_required_extensions(instance, physical_device);
            if !extension_support
                || swapchain_support.is_none()
                || !QueueFamily::is_complete(&queue_family_indices)
            {
                log::warn!("Device `{name}` does not meet minimum device requirements");
                rating = 0;
            }

            (
                queue_family_indices,
                Self {
                    name,
                    properties,
                    memory_properties,
                    features,
                    rating,
                    msaa_samples,
                    swapchain_support,
                    extension_support,
                    sampler_anisotropy_support: features.sampler_anisotropy == 1,
                },
            )
        }

        /// Return the maximum usable sample count for this device.
        fn get_max_sample_count(
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

        /// Whether a physical device supports the required extensions for this platform.
        fn supports_required_extensions(
            instance: &ash::Instance,
            physical_device: vk::PhysicalDevice,
        ) -> bool {
            // SAFETY: TODO
            let device_extensions =
                unsafe { instance.enumerate_device_extension_properties(physical_device) };
            let Ok(device_extensions) = device_extensions else {
                log::error!("failed to enumerate vulkan device extensions");
                return false;
            };
            let extension_names = device_extensions
                .iter()
                // SAFETY: extension_name is provided by Vulkan and is a valid CStr.
                .map(|extension_properties| unsafe {
                    CStr::from_ptr(extension_properties.extension_name.as_ptr())
                })
                .collect::<HashSet<_>>();
            platform::required_device_extensions()
                .iter()
                // SAFETY: required_device_extensions are static strings provided by Ash and are valid
                // CStrs.
                .map(|&extension| unsafe { CStr::from_ptr(extension) })
                .all(|extension| extension_names.contains(extension))
        }
    }

    #[derive(Debug, Clone)]
    #[must_use]
    pub(crate) struct SwapchainSupport {
        pub(crate) capabilities: vk::SurfaceCapabilitiesKHR,
        pub(crate) formats: Vec<vk::SurfaceFormatKHR>,
        pub(crate) present_modes: Vec<vk::PresentModeKHR>,
    }

    impl SwapchainSupport {
        fn query(physical_device: vk::PhysicalDevice, surface: &Surface) -> Result<Self> {
            // SAFETY: TODO
            let capabilities = unsafe {
                surface
                    .loader
                    .get_physical_device_surface_capabilities(physical_device, surface.handle)
            }
            .context("Failed to query for surface capabilities.")?;

            // SAFETY: TODO
            let formats = unsafe {
                surface
                    .loader
                    .get_physical_device_surface_formats(physical_device, surface.handle)
            }
            .context("Failed to query for surface formats.")?;

            // SAFETY: TODO
            let present_modes = unsafe {
                surface
                    .loader
                    .get_physical_device_surface_present_modes(physical_device, surface.handle)
            }
            .context("Failed to query for surface present mode.")?;

            if formats.is_empty() || present_modes.is_empty() {
                bail!("no available swapchain formats or present_modes");
            }

            Ok(Self {
                capabilities,
                formats,
                present_modes,
            })
        }
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    #[must_use]
    pub(crate) enum QueueFamily {
        Graphics,
        Present,
        Compute,
        Transfer,
    }

    type QueueFamilyIndices = HashMap<QueueFamily, u32>;

    impl QueueFamily {
        /// Find desired [vk::Queue] families from a [`vk::PhysicalDevice`].
        fn query(
            instance: &ash::Instance,
            physical_device: vk::PhysicalDevice,
            surface: &Surface,
        ) -> QueueFamilyIndices {
            // SAFETY: TODO
            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

            let mut queue_family_indices = HashMap::default();
            let mut min_transfer_score = 255;
            for (index, queue_family) in queue_families.iter().enumerate() {
                let queue_index = index as u32;
                let mut current_transfer_score = 0;

                // SAFETY: TODO
                let has_surface_support = unsafe {
                    surface.loader.get_physical_device_surface_support(
                        physical_device,
                        queue_index,
                        surface.handle,
                    )
                }
                .unwrap_or_default();

                queue_family_indices
                    .entry(QueueFamily::Graphics)
                    .or_insert_with(|| {
                        current_transfer_score += 1;
                        queue_index
                    });
                // Prioritize queues supporting both graphics and present
                if queue_family_indices.contains_key(&QueueFamily::Graphics) && has_surface_support
                {
                    queue_family_indices.insert(QueueFamily::Present, queue_index);
                    current_transfer_score += 1;
                }

                if queue_family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    queue_family_indices
                        .entry(QueueFamily::Compute)
                        .or_insert(queue_index);
                    current_transfer_score += 1;
                }

                if queue_family.queue_flags.contains(vk::QueueFlags::TRANSFER)
                    && current_transfer_score <= min_transfer_score
                {
                    min_transfer_score = current_transfer_score;
                    queue_family_indices
                        .entry(QueueFamily::Transfer)
                        .or_insert(queue_index);
                }

                if QueueFamily::is_complete(&queue_family_indices) {
                    break;
                }
            }

            // If we didn't find a Graphics family supporting Present, just find use first one
            // found
            if let Entry::Vacant(entry) = queue_family_indices.entry(QueueFamily::Present) {
                if let Some(queue_index) = queue_families
                    .iter()
                    .enumerate()
                    .map(|(queue_index, _)| queue_index as u32)
                    .find(|&queue_index| {
                        // SAFETY: TODO
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
                    entry.insert(queue_index);
                }
            }

            queue_family_indices
        }

        fn is_complete(indices: &QueueFamilyIndices) -> bool {
            indices.contains_key(&Self::Graphics)
                && indices.contains_key(&Self::Present)
                && indices.contains_key(&Self::Compute)
                && indices.contains_key(&Self::Transfer)
        }
    }
}

mod swapchain {
    use super::{
        device::{Device, QueueFamily},
        surface::Surface,
    };
    use anyhow::{bail, Context, Result};
    use ash::{extensions::khr, vk};

    #[derive(Clone)]
    #[must_use]
    pub(crate) struct Swapchain {
        pub(crate) loader: khr::Swapchain,
        pub(crate) handle: vk::SwapchainKHR,
        pub(crate) format: vk::Format,
        pub(crate) extent: vk::Extent2D,
        pub(crate) images: Vec<vk::Image>,
        pub(crate) image_views: Vec<vk::ImageView>,
        pub(crate) max_frames_in_flight: u32,
    }

    impl Swapchain {
        pub(crate) fn create(
            instance: &ash::Instance,
            device: &Device,
            surface: &Surface,
            width: u32,
            height: u32,
        ) -> Result<Self> {
            log::debug!("creating vulkan swapchain");

            let Some(swapchain_support) = device.info.swapchain_support.as_ref() else {
                bail!("{} does not support swapchains", device.info.name);
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
            let capabilities = &swapchain_support.capabilities;
            let image_extent = if capabilities.current_extent.width == u32::MAX {
                let vk::Extent2D {
                    width: min_width,
                    height: min_height,
                } = capabilities.min_image_extent;
                let vk::Extent2D {
                    width: max_width,
                    height: max_height,
                } = capabilities.max_image_extent;
                vk::Extent2D::builder()
                    .width(width.clamp(min_width, max_width))
                    .height(height.clamp(min_height, max_height))
                    .build()
            } else {
                capabilities.current_extent
            };

            // Determine image_count
            let min_image_count = capabilities.min_image_count + 1;
            let min_image_count = if capabilities.max_image_count > 0
                && min_image_count > capabilities.max_image_count
            {
                min_image_count.min(capabilities.max_image_count)
            } else {
                min_image_count
            };

            // Select image sharing mode, concurrent vs exclusive
            let (image_sharing_mode, queue_family_indices) = match (
                device.queue_family_indices.get(&QueueFamily::Graphics),
                device.queue_family_indices.get(&QueueFamily::Present),
            ) {
                (Some(&graphics), Some(&present)) if graphics != present => {
                    (vk::SharingMode::CONCURRENT, vec![graphics, present])
                }
                _ => (vk::SharingMode::EXCLUSIVE, vec![]),
            };

            // Create info
            let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
                .surface(surface.handle)
                .min_image_count(min_image_count)
                .image_color_space(surface_format.color_space)
                .image_format(surface_format.format)
                .image_extent(image_extent)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(image_sharing_mode)
                .queue_family_indices(&queue_family_indices)
                .pre_transform(capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .image_array_layers(1)
                .build();

            // Create swapchain
            let swapchain_loader = khr::Swapchain::new(instance, &device.logical_device);
            // SAFETY: All create_info values are set correctly above with valid lifetimes.
            let swapchain =
                unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None) }
                    .context("failed to create vulkan swapchain")?;

            // Get images
            // SAFETY: TODO
            let images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }
                .context("failed to get vulkan swapchain images")?;

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
                        .image(image)
                        .build();
                    unsafe {
                        device
                            .logical_device
                            .create_image_view(&image_view_create_info, None)
                    }
                    .context("failed to create vulkan image view")
                })
                .collect::<Result<Vec<_>>>()?;

            // Get depth images

            log::debug!("created vulkan swapchain successfully");

            Ok(Self {
                loader: swapchain_loader,
                handle: swapchain,
                format: surface_format.format,
                extent: image_extent,
                images,
                image_views,
                max_frames_in_flight: min_image_count - 1,
            })
        }
    }
}

mod pipeline {
    use super::device::Device;
    use crate::math::Vertex;
    use anyhow::{bail, Context, Result};
    use ash::vk;
    use std::ffi::CString;

    #[derive(Clone)]
    #[must_use]
    pub(crate) struct Pipeline {
        pub(crate) handle: vk::Pipeline,
        pub(crate) layout: vk::PipelineLayout,
    }

    impl Pipeline {
        const VERTEX_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/primary.vert.spv"));
        const FRAGMENT_SHADER: &[u8] =
            include_bytes!(concat!(env!("OUT_DIR"), "/primary.frag.spv"));

        pub(crate) fn create(
            device: &Device,
            extent: vk::Extent2D,
            render_pass: vk::RenderPass,
        ) -> Result<Self> {
            log::debug!("creating vulkan graphics pipeline");

            // Shader Stages
            let vertex_shader_module = Self::create_shader_module(device, Self::VERTEX_SHADER)?;
            let fragment_shader_module = Self::create_shader_module(device, Self::FRAGMENT_SHADER)?;

            let shader_entry_name = CString::new("main")?;
            let shader_stages = [
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vertex_shader_module)
                    .name(&shader_entry_name)
                    .build(),
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(fragment_shader_module)
                    .name(&shader_entry_name)
                    .build(),
            ];

            // Vertex Input State
            let binding_descriptions = [Vertex::binding_description()];
            let attribute_descriptions = Vertex::attribute_descriptions();
            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_binding_descriptions(&binding_descriptions)
                .vertex_attribute_descriptions(&attribute_descriptions)
                .build();

            // Input Assembly State
            let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                .primitive_restart_enable(false)
                .build();

            // Viewport State
            let viewports = [vk::Viewport::builder()
                .x(0.0)
                .y(0.0)
                .width(extent.width as f32)
                .height(extent.height as f32)
                .min_depth(0.0)
                .max_depth(1.0)
                .build()];
            let scissors = [vk::Rect2D::builder()
                .offset(vk::Offset2D::default())
                .extent(extent)
                .build()];
            let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
                .viewports(&viewports)
                .scissors(&scissors)
                .build();

            // Rasterization State
            let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE) // Because Y-axis is inverted in Vulkan
                .depth_bias_enable(false)
                .build();

            // Multisample State
            let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
                // TODO: May impact performance
                .sample_shading_enable(true)
                // Closer to 1 is smoother
                .min_sample_shading(0.2)
                .rasterization_samples(device.info.msaa_samples)
                .build();

            // Depth Stencil State
            let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS)
                .depth_bounds_test_enable(false)
                .stencil_test_enable(false)
                .build();

            // Color Blend State
            let attachments = [vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                .alpha_blend_op(vk::BlendOp::ADD)
                .build()];
            let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
                .logic_op_enable(false)
                .logic_op(vk::LogicOp::COPY)
                .attachments(&attachments)
                .blend_constants([0.0; 4]);

            // Push Constants
            let vert_push_constant_ranges = vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::VERTEX)
                .offset(0)
                .size(64) // 16 * 4 byte floats
                .build();
            let frag_push_constant_ranges = vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .offset(64)
                .size(4) // 1 * 4 byte float
                .build();

            // Layout
            // let set_layouts = [descriptor_set_layout];
            let push_constant_ranges = [vert_push_constant_ranges, frag_push_constant_ranges];
            let layout_info = vk::PipelineLayoutCreateInfo::builder()
                // .set_layouts(&set_layouts)
                .push_constant_ranges(&push_constant_ranges)
                .build();
            // SAFETY: TODO
            let layout = unsafe {
                device
                    .logical_device
                    .create_pipeline_layout(&layout_info, None)
            }
            .context("failed to create vulkan graphics pipeline layout")?;

            // Create
            let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
                .stages(&shader_stages)
                .vertex_input_state(&vertex_input_state)
                .input_assembly_state(&input_assembly_state)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterization_state)
                .multisample_state(&multisample_state)
                .depth_stencil_state(&depth_stencil_state)
                .color_blend_state(&color_blend_state)
                .layout(layout)
                .render_pass(render_pass)
                .subpass(0)
                .build();

            // SAFETY: TODO
            let pipeline = unsafe {
                device.logical_device.create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[pipeline_create_info],
                    None,
                )
            }
            .map_err(|(_, err)| err)
            .context("failed to create vulkan graphics pipeline")?
            .into_iter()
            .next()
            .context("no graphics pipelines were created")?;

            // Cleanup
            // SAFETY: TODO
            unsafe {
                device
                    .logical_device
                    .destroy_shader_module(vertex_shader_module, None);
                device
                    .logical_device
                    .destroy_shader_module(fragment_shader_module, None);
            }

            log::debug!("created vulkan graphics pipeline successfully");

            Ok(Self {
                handle: pipeline,
                layout,
            })
        }

        fn create_shader_module(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule> {
            let bytecode = bytecode.to_vec();
            // SAFETY: We check for prefix/suffix below and bail if not aligned correctly.
            let (prefix, code, suffix) = unsafe { bytecode.align_to::<u32>() };
            if !prefix.is_empty() || !suffix.is_empty() {
                bail!("shader bytecode is not properly aligned.");
            }
            let shader_module_info = vk::ShaderModuleCreateInfo::builder().code(code);
            unsafe {
                device
                    .logical_device
                    .create_shader_module(&shader_module_info, None)
            }
            .context("failed to create vulkan shader module")
        }
    }
}

mod image {
    #[cfg(debug_assertions)]
    use super::debug::Debug;
    use super::{device::Device, swapchain::Swapchain};
    use anyhow::{Context, Result};
    use ash::vk::{self, Handle};
    use std::ffi::CString;

    #[derive(Clone)]
    #[must_use]
    pub(crate) struct Image {
        pub(crate) handle: vk::Image,
        pub(crate) memory: vk::DeviceMemory,
        pub(crate) view: vk::ImageView,
    }

    impl Image {
        pub(crate) fn create_color(
            instance: &ash::Instance,
            device: &Device,
            swapchain: &Swapchain,
            #[cfg(debug_assertions)] debug: &Debug,
        ) -> Result<Self> {
            log::debug!("creating vulkan color image");

            // Image
            let (image, memory) = Self::create_image(
                "color",
                instance,
                device,
                swapchain,
                swapchain.format,
                1,
                vk::ImageTiling::OPTIMAL,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                #[cfg(debug_assertions)]
                debug,
            )?;

            // Image View
            let view = Self::create_view(
                device,
                image,
                swapchain.format,
                vk::ImageAspectFlags::COLOR,
                1,
            )?;

            log::debug!("created vulkan color image successfully");

            Ok(Self {
                handle: image,
                memory,
                view,
            })
        }

        pub(crate) fn create_depth(
            instance: &ash::Instance,
            device: &Device,
            swapchain: &Swapchain,
            #[cfg(debug_assertions)] debug: &Debug,
        ) -> Result<Self> {
            log::debug!("creating vulkan depth image");

            let format = device.get_depth_format(instance, vk::ImageTiling::OPTIMAL)?;

            // Image
            let (image, memory) = Self::create_image(
                "depth",
                instance,
                device,
                swapchain,
                format,
                1,
                vk::ImageTiling::OPTIMAL,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                #[cfg(debug_assertions)]
                debug,
            )?;

            // Image View
            let view = Self::create_view(device, image, format, vk::ImageAspectFlags::DEPTH, 1)?;

            log::debug!("created vulkan depth image successfully");

            Ok(Self {
                handle: image,
                memory,
                view,
            })
        }

        #[allow(clippy::too_many_arguments)]
        fn create_image(
            name: &str,
            instance: &ash::Instance,
            device: &Device,
            swapchain: &Swapchain,
            format: vk::Format,
            mip_levels: u32,
            tiling: vk::ImageTiling,
            usage: vk::ImageUsageFlags,
            properties: vk::MemoryPropertyFlags,
            #[cfg(debug_assertions)] debug: &Debug,
        ) -> Result<(vk::Image, vk::DeviceMemory)> {
            // Image
            let image_info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .extent(vk::Extent3D {
                    width: swapchain.extent.width,
                    height: swapchain.extent.height,
                    depth: 1,
                })
                .mip_levels(mip_levels)
                .array_layers(1)
                .format(format)
                .tiling(tiling)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .usage(usage)
                .samples(device.info.msaa_samples)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .build();

            // SAFETY: TODO
            let image = unsafe { device.logical_device.create_image(&image_info, None) }
                .context("failed to create vulkan image")?;

            // Debug Name

            #[cfg(debug_assertions)]
            let name = CString::new(name)?;
            #[cfg(debug_assertions)]
            let debug_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(vk::ObjectType::IMAGE)
                .object_name(&name)
                .object_handle(image.as_raw())
                .build();
            #[cfg(debug_assertions)]
            unsafe {
                debug
                    .utils
                    .set_debug_utils_object_name(device.logical_device.handle(), &debug_info)
            }
            .context("failed to set debug utils object name")?;

            // Memory

            let requirements =
                unsafe { device.logical_device.get_image_memory_requirements(image) };
            let memory_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(requirements.size)
                .memory_type_index(device.memory_type_index(instance, properties, requirements)?)
                .build();
            let memory = unsafe { device.logical_device.allocate_memory(&memory_info, None) }
                .context("failed to allocate vulkan image memory")?;
            // SAFETY: TODO
            unsafe { device.logical_device.bind_image_memory(image, memory, 0) }
                .context("failed to bind vulkan image memory")?;

            Ok((image, memory))
        }

        fn create_view(
            device: &Device,
            image: vk::Image,
            format: vk::Format,
            aspects: vk::ImageAspectFlags,
            mip_levels: u32,
        ) -> Result<vk::ImageView> {
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
                )
                .build();
            // SAFETY: TODO
            let view = unsafe {
                device
                    .logical_device
                    .create_image_view(&view_create_info, None)
            }
            .context("failed to create vulkan image view")?;

            Ok(view)
        }
    }
}

mod platform {
    use anyhow::{Context, Result};
    use ash::{
        extensions::{ext, khr, mvk},
        vk, Entry, Instance,
    };
    use winit::window::Window;

    /// Return a list of enabled Vulkan [ash::Instance] extensions for macOS.
    #[cfg(target_os = "macos")]
    pub(crate) fn enabled_extension_names() -> Vec<*const i8> {
        vec![
            khr::Surface::name().as_ptr(),
            #[cfg(debug_assertions)]
            ext::DebugUtils::name().as_ptr(),
            mvk::MacOSSurface::name().as_ptr(),
            vk::KhrPortabilityEnumerationFn::name().as_ptr(),
            vk::KhrGetPhysicalDeviceProperties2Fn::name().as_ptr(),
        ]
    }

    /// Return a list of enabled Vulkan [ash::Instance] extensions for Linux.
    #[cfg(all(unix, not(target_os = "macos"),))]
    pub(crate) fn enabled_extension_names() -> Vec<*const i8> {
        vec![
            khr::Surface::name().as_ptr(),
            #[cfg(debug_assertions)]
            ext::DebugUtils::name().as_ptr(),
            khr::XlibSurface::name().as_ptr(),
        ]
    }

    /// Return a list of enabled Vulkan [ash::Instance] extensions for Windows.
    #[cfg(windows)]
    pub(crate) fn enabled_extension_names() -> Vec<*const i8> {
        vec![
            khr::Surface::name().as_ptr(),
            #[cfg(debug_assertions)]
            ext::DebugUtils::name().as_ptr(),
            khr::Win32Surface::name().as_ptr(),
        ]
    }

    /// Return a set of [`vk::InstanceCreateFlags`] for macOS.
    #[cfg(target_os = "macos")]
    pub(crate) fn instance_create_flags() -> vk::InstanceCreateFlags {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    }

    /// Return a set of [`vk::InstanceCreateFlags`] for all platforms other than macOS.
    #[cfg(not(target_os = "macos"))]
    pub(crate) fn instance_create_flags() -> vk::InstanceCreateFlags {
        vk::InstanceCreateFlags::default()
    }

    /// Create a [`vk::SurfaceKHR`] instance for the current [Window] for macOS.
    #[cfg(target_os = "macos")]
    pub(crate) fn create_surface(
        entry: &Entry,
        instance: &Instance,
        window: &Window,
    ) -> Result<vk::SurfaceKHR> {
        use cocoa::{
            appkit::{NSView, NSWindow},
            base::{id as cocoa_id, YES},
        };
        use metal::{MetalLayer, MetalLayerRef};
        use objc::runtime::Object;
        use std::mem;
        use winit::platform::macos::WindowExtMacOS;

        log::debug!("creating macOS metal surface");

        let layer = MetalLayer::new();
        layer.set_edge_antialiasing_mask(0);
        layer.set_presents_with_transaction(false);
        layer.remove_all_animations();

        // SAFETY: TODO
        let wnd: cocoa_id = unsafe { mem::transmute(window.ns_window()) };
        let view = unsafe { wnd.contentView() };
        unsafe { layer.set_contents_scale(view.backingScaleFactor()) };
        let metal_layer_ref: *const MetalLayerRef = layer.as_ref();
        unsafe { view.setLayer(metal_layer_ref as *mut Object) };
        unsafe { view.setWantsLayer(YES) };

        let surface_create_info = vk::MacOSSurfaceCreateInfoMVK::builder()
            .view(window.ns_view())
            .build();

        let macos_surface = mvk::MacOSSurface::new(entry, instance);
        // SAFETY: All create_info values are set correctly above with valid lifetimes.
        unsafe { macos_surface.create_mac_os_surface(&surface_create_info, None) }
            .context("failed to create vulkan surface")
    }

    /// Create a [`vk::SurfaceKHR`] instance for the current [Window] for Linux.
    #[cfg(all(unix, not(target_os = "macos"),))]
    pub(crate) fn create_surface(
        entry: &Entry,
        instance: &Instance,
        window: &Window,
    ) -> Result<vk::SurfaceKHR> {
        use std::ptr;
        use winit::platform::unix::WindowExtUnix;

        log::debug!("creating Linux XLIB surface");

        let x11_display = window
            .xlib_display()
            .context("failed to get XLIB display")?;
        let x11_window = winadow.xlib_window().context("failed to get XLIB window")?;
        let surface_create_info = vk::XlibSurfaceCreateInfoKHR::builder()
            .window(x11_window as vk::Window)
            .dpy(x11_display as *mut vk::Display)
            .build();
        let xlib_surface = khr::XlibSurface::new(entry, instance);
        // SAFETY: All create_info values are set correctly above with valid lifetimes.
        unsafe { xlib_surface.create_xlib_surface(&surface_create_info, None) }
            .context("failed to create vulkan surface")
    }

    /// Create a [`vk::SurfaceKHR`] instance for the current [Window] for Windows.
    #[cfg(windows)]
    pub(crate) fn create_surface(
        entry: &Entry,
        instance: &Instance,
        window: &Window,
    ) -> Result<vk::SurfaceKHR> {
        use std::os::raw::c_void;
        use std::ptr;
        use winapi::um::libloaderapi::GetModuleHandleW;
        use winit::platform::windows::WindowExtWindows;

        log::debug!("creating win32 surface");

        let hinstance = GetModuleHandleW(ptr::null()) as *const c_void;
        let surface_create_info = vk::Win32SurfaceCreateInfoKHR::builder()
            .hinstance(hinstance)
            .hwnd(window.hwnd())
            .build();
        let win32_surface = khr::Win32Surface::new(entry, instance);
        // SAFETY: All create_info values are set correctly above with valid lifetimes.
        unsafe { win32_surface.create_win32_surface(&surface_create_info, None) }
            .context("failed to create vulkan surface")
    }

    /// Return a list of required [`vk::PhysicalDevice`] extensions for macOS.
    #[cfg(target_os = "macos")]
    pub(crate) fn required_device_extensions() -> Vec<*const i8> {
        vec![
            khr::Swapchain::name().as_ptr(),
            vk::KhrPortabilitySubsetFn::name().as_ptr(),
        ]
    }

    /// Return a list of required [`vk::PhysicalDevice`] extensions for all platforms other than
    /// macOS.
    #[cfg(not(target_os = "macos"))]
    pub(crate) fn required_device_extensions() -> Vec<*const i8> {
        vec![khr::Swapchain::name().as_ptr()]
    }
}

mod vertex {
    use crate::math::{Vec3, Vertex};
    use ash::vk;
    use std::mem::size_of;

    impl Vertex {
        pub(crate) fn binding_description() -> vk::VertexInputBindingDescription {
            vk::VertexInputBindingDescription::builder()
                .binding(0)
                .stride(size_of::<Vertex>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX)
                .build()
        }

        pub(crate) fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
            let pos = vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0) // points to shader.vert location
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0)
                .build();
            let color = vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1) // points to shader.vert location
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(size_of::<Vec3>() as u32)
                .build();
            let texcoord = vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(2) // points to shader.vert location
                .format(vk::Format::R32G32_SFLOAT)
                .offset((size_of::<Vec3>() + size_of::<Vec3>()) as u32)
                .build();
            [pos, color, texcoord]
        }
    }
}

mod debug {
    use anyhow::Result;
    use ash::{extensions::ext, vk};
    use std::ffi::{c_void, CStr};

    // SAFETY: This static string has been verified as a valid CStr.
    pub(crate) const VALIDATION_LAYER_NAME: &CStr =
        unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };

    #[derive(Clone)]
    #[cfg(debug_assertions)]
    pub(crate) struct Debug {
        pub(crate) utils: ext::DebugUtils,
        pub(crate) messenger: vk::DebugUtilsMessengerEXT,
    }

    #[cfg(debug_assertions)]
    impl Debug {
        /// Create a `Debug` instance with [ext::DebugUtils] and [vk::DebugUtilsMessengerEXT].
        pub(crate) fn create(entry: &ash::Entry, instance: &ash::Instance) -> Result<Self> {
            log::debug!("creating vulkan debug utils");
            let utils = ext::DebugUtils::new(entry, instance);
            let debug_create_info = Self::build_debug_create_info();
            // SAFETY: All create_info values are set correctly above with valid lifetimes.
            let messenger =
                unsafe { utils.create_debug_utils_messenger(&debug_create_info, None)? };
            log::debug!("vulkan debug utils created successfully");
            Ok(Self { utils, messenger })
        }

        /// Build [vk::DebugUtilsMessengerCreateInfoEXT].
        pub(crate) fn build_debug_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT {
            vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                )
                .pfn_user_callback(Some(debug_callback))
                .build()
        }
    }

    /// Callback function used in Debug Utils.
    extern "system" fn debug_callback(
        severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
        data: *const vk::DebugUtilsMessengerCallbackDataEXT,
        _: *mut c_void,
    ) -> vk::Bool32 {
        let msg_type = match msg_type {
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
            vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
            vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
            _ => "[Unknown]",
        };
        // SAFETY: This message is provided by Vulkan and is a valid CStr.
        let message = unsafe { CStr::from_ptr((*data).p_message) }.to_string_lossy();
        match severity {
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::error!("{msg_type} {message}"),
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::warn!("{msg_type} {message}"),
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::info!("{msg_type} {message}"),
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => log::trace!("{msg_type} {message}"),
            _ => log::debug!("{msg_type} {message}"),
        };
        vk::FALSE
    }
}
