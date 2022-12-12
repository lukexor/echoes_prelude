use crate::renderer::RendererBackend;
use anyhow::{bail, Context as _, Result};
use ash::{extensions::ext, vk};
use std::{
    collections::{hash_map::Entry, HashMap, HashSet},
    ffi::{c_void, CStr, CString},
    fmt,
};
use surface::Surface;
use winit::window::Window;

// SAFETY: This static string has been verified as a valid CStr.
const VALIDATION_LAYER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };

pub(crate) struct Context {
    _entry: ash::Entry,
    instance: ash::Instance,
    surface: Surface,
    device: Device,
    #[cfg(debug_assertions)]
    debug: Debug,
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
        unsafe {
            self.device.logical_device.destroy_device(None);
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
        let debug = Debug::new(&entry, &instance)?;
        let surface = Surface::create(&entry, &instance, window)?;
        let device = Device::create(&instance, &surface)?;
        let graphics_queue = device.queue_family(QueueFamily::Graphics)?;
        let present_queue = device.queue_family(QueueFamily::Present)?;
        // let swapchain =
        //     Self::create_swapchain(&instance, &device, &physical_device, window, &surface)?;
        // let image_views = Self::create_image_views(&device, &swapchain)?;
        // let msaa_samples = Self::get_max_sample_count(&instance, &physical_device)?;
        // let render_pass = Self::create_render_pass(
        //     &instance,
        //     &device,
        //     &physical_device,
        //     &swapchain,
        //     &msaa_samples,
        // )?;
        // let ubo_layout = Self::create_descriptor_set_layout(&device)?;
        // let graphics_pipeline = Self::create_graphics_pipeline(
        //     &device,
        //     &render_pass,
        //     &swapchain,
        //     &ubo_layout,
        //     &msaa_samples,
        // )?;
        // let command_pool = Self::create_command_pool(&device)?;
        // let color_image =
        //     Self::create_color_image(&device, &swapchain, &physical_device, &msaa_samples)?;
        // let depth_image = Self::create_depth_image(
        //     &instance,
        //     &device,
        //     &physical_device,
        //     &command_pool,
        //     &swapchain,
        //     &physical_device,
        //     &msaa_samples,
        // )?;
        // let framebuffers = Self::create_framebuffers(
        //     &device,
        //     &render_pass,
        //     &image_views,
        //     &depth_image,
        //     &color_image,
        //     &swapchain,
        // )?;
        // let texture_image = Self::create_texture_image(&instance, &device, &command_pool)?;
        // let texture_image_view = Self::create_texture_image_view(&device, &texture_image)?;
        // let texture_sampler = Self::create_texture_sampler(&device, &texture_image)?;
        // let model = Self::load_model()?;
        // let vertex_buffer = Self::create_vertex_buffer(&instance, &device, &command_pool, &model)?;
        // let index_buffer = Self::create_index_buffer(&instance, &device, &command_pool, &model)?;
        // let uniform_buffers = Self::create_uniform_buffers(&device, &swapchain)?;
        // let descriptor_pool = Self::create_descriptor_pool(&device, &swapchain)?;
        // let descriptor_sets = Self::create_descriptor_sets(
        //     &device,
        //     &descriptor_pool,
        //     &ubo_layout,
        //     &uniform_buffers,
        //     &texture_image_view,
        //     &texture_sampler,
        //     &swapchain,
        // )?;
        // let command_buffers = Self::create_command_buffers(
        //     &device,
        //     &command_pool,
        //     &graphics_pipeline,
        //     &framebuffers,
        //     &render_pass,
        //     &swapchain,
        //     &vertex_buffer,
        //     &index_buffer,
        //     &descriptor_sets,
        //     &model,
        // )?;
        // let sync = Self::create_sync_objects(&device)?;

        log::info!("initialized vulkan renderer successfully");

        Ok(Context {
            _entry: entry,
            instance,
            surface,
            device,
            #[cfg(debug_assertions)]
            debug,
        })
    }

    /// Handle window resized event.
    fn on_resized(&mut self, _width: u32, _height: u32) {}

    /// Draw a frame to the windows surface.
    fn draw_frame(&mut self) -> Result<()> {
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
}

#[derive(Clone)]
#[must_use]
struct Device {
    physical_device: vk::PhysicalDevice,
    logical_device: ash::Device,
    queue_family_indices: QueueFamilyIndices,
    info: DeviceInfo,
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
    /// Select a preferred [vk::PhysicalDevice] and create a logical [ash::Device].
    fn create(instance: &ash::Instance, surface: &Surface) -> Result<Self> {
        let (physical_device, queue_family_indices, info) =
            Self::select_physical_device(instance, surface)?;
        let logical_device =
            Self::create_logical_device(instance, physical_device, &queue_family_indices)?;
        Ok(Self {
            physical_device,
            logical_device,
            queue_family_indices,
            info,
        })
    }

    /// Select a preferred [vk::PhysicalDevice].
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
            .build();
        let enabled_layer_names = [
            #[cfg(debug_assertions)]
            VALIDATION_LAYER_NAME.as_ptr(),
        ];
        let enabled_extension_names = platform::required_device_extensions();
        let create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&enabled_features)
            .enabled_layer_names(&enabled_layer_names)
            .enabled_extension_names(&enabled_extension_names)
            .build();
        // SAFETY: All create_info values are set correctly above with valid lifetimes.
        unsafe { instance.create_device(physical_device, &create_info, None) }
            .context("failed to create vulkan logical device")
    }

    fn queue_family(&self, family: QueueFamily) -> Result<vk::Queue> {
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
}

#[derive(Debug, Clone)]
#[must_use]
struct DeviceInfo {
    name: String,
    properties: vk::PhysicalDeviceProperties,
    features: vk::PhysicalDeviceFeatures,
    rating: u32,
    swapchain_support: Option<SwapchainSupport>,
    extension_support: bool,
    sampler_anisotropy_support: bool,
}

impl DeviceInfo {
    /// Query [vk::PhysicalDevice] properties and generate a rating score.
    fn query(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        surface: &Surface,
    ) -> (QueueFamilyIndices, Self) {
        // SAFETY: TODO
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        // SAFETY: TODO
        let features = unsafe { instance.get_physical_device_features(physical_device) };
        let queue_family_indices = QueueFamily::query(instance, physical_device, surface);
        let swapchain_support = SwapchainSupport::query(physical_device, surface)
            .map_err(|err| {
                log::error!("{err}");
            })
            .ok();

        let mut rating = 10;
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
                features,
                rating,
                swapchain_support,
                extension_support,
                sampler_anisotropy_support: features.sampler_anisotropy == 1,
            },
        )
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
struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
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

        Ok(Self {
            capabilities,
            formats,
            present_modes,
        })
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[must_use]
enum QueueFamily {
    Graphics,
    Present,
    Compute,
    Transfer,
}

type QueueFamilyIndices = HashMap<QueueFamily, u32>;

impl QueueFamily {
    /// Find desired [vk::Queue] families from a [vk::PhysicalDevice].
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
            if queue_family_indices.contains_key(&QueueFamily::Graphics) && has_surface_support {
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
            let handle = platform::create_surface(entry, instance, window)?;
            let loader = khr::Surface::new(entry, instance);
            let PhysicalSize { width, height } = window.inner_size();
            Ok(Self {
                handle,
                loader,
                width,
                height,
            })
        }
    }
}

#[derive(Clone)]
#[cfg(debug_assertions)]
struct Debug {
    utils: ext::DebugUtils,
    messenger: vk::DebugUtilsMessengerEXT,
}

#[cfg(debug_assertions)]
impl Debug {
    /// Create a `Debug` instance with [ext::DebugUtils] and [vk::DebugUtilsMessengerEXT].
    fn new(entry: &ash::Entry, instance: &ash::Instance) -> Result<Self> {
        let utils = ext::DebugUtils::new(entry, instance);
        let create_info = Self::build_debug_create_info();
        // SAFETY: All create_info values are set correctly above with valid lifetimes.
        let messenger = unsafe { utils.create_debug_utils_messenger(&create_info, None)? };
        Ok(Self { utils, messenger })
    }

    /// Build [vk::DebugUtilsMessengerCreateInfoEXT].
    fn build_debug_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT {
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

mod platform {
    use anyhow::{Context, Result};
    use ash::{
        extensions::{ext, khr, mvk},
        vk, Entry, Instance,
    };
    use winit::window::Window;

    /// Return a list of enabled Vulkan [Instance] extensions.
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

    /// Return a list of enabled Vulkan [Instance] extensions.
    #[cfg(all(unix, not(target_os = "macos"),))]
    pub(crate) fn enabled_extension_names() -> Vec<*const i8> {
        vec![
            khr::Surface::name().as_ptr(),
            #[cfg(debug_assertions)]
            ext::DebugUtils::name().as_ptr(),
            khr::XlibSurface::name().as_ptr(),
        ]
    }

    /// Return a list of enabled Vulkan [Instance] extensions.
    #[cfg(windows)]
    pub(crate) fn enabled_extension_names() -> Vec<*const i8> {
        vec![
            khr::Surface::name().as_ptr(),
            #[cfg(debug_assertions)]
            ext::DebugUtils::name().as_ptr(),
            khr::Win32Surface::name().as_ptr(),
        ]
    }

    /// Return a set of [vk::InstanceCreateFlags].
    #[cfg(target_os = "macos")]
    pub(crate) fn instance_create_flags() -> vk::InstanceCreateFlags {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    }

    /// Return a set of [vk::InstanceCreateFlags].
    #[cfg(not(target_os = "macos"))]
    pub(crate) fn instance_create_flags() -> vk::InstanceCreateFlags {
        vk::InstanceCreateFlags::default()
    }

    /// Create a [vk::SurfaceKHR] instance for the current [Window].
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

        let create_info = vk::MacOSSurfaceCreateInfoMVK::builder()
            .view(window.ns_view())
            .build();

        let macos_surface = mvk::MacOSSurface::new(entry, instance);
        // SAFETY: All create_info values are set correctly above with valid lifetimes.
        unsafe { macos_surface.create_mac_os_surface(&create_info, None) }
            .context("failed to create vulkan surface")
    }

    /// Create a [vk::SurfaceKHR] instance for the current [Window].
    #[cfg(all(unix, not(target_os = "macos"),))]
    pub(crate) fn create_surface(
        entry: &Entry,
        instance: &Instance,
        window: &Window,
    ) -> Result<vk::SurfaceKHR> {
        use std::ptr;
        use winit::platform::unix::WindowExtUnix;

        let x11_display = window
            .xlib_display()
            .context("failed to get XLIB display")?;
        let x11_window = winadow.xlib_window().context("failed to get XLIB window")?;
        let create_info = vk::XlibSurfaceCreateInfoKHR::builder()
            .window(x11_window as vk::Window)
            .dpy(x11_display as *mut vk::Display)
            .build();
        let xlib_surface = khr::XlibSurface::new(entry, instance);
        // SAFETY: All create_info values are set correctly above with valid lifetimes.
        unsafe { xlib_surface.create_xlib_surface(&create_info, None) }
            .context("failed to create vulkan surface")
    }

    /// Create a [vk::SurfaceKHR] instance for the current [Window].
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

        let hinstance = GetModuleHandleW(ptr::null()) as *const c_void;
        let create_info = vk::Win32SurfaceCreateInfoKHR::builder()
            .hinstance(hinstance)
            .hwnd(window.hwnd())
            .build();
        let win32_surface = khr::Win32Surface::new(entry, instance);
        // SAFETY: All create_info values are set correctly above with valid lifetimes.
        unsafe { win32_surface.create_win32_surface(&create_info, None) }
            .context("failed to create vulkan surface")
    }

    /// Return a list of required device extensions.
    #[cfg(target_os = "macos")]
    pub(crate) fn required_device_extensions() -> Vec<*const i8> {
        vec![
            khr::Swapchain::name().as_ptr(),
            vk::KhrPortabilitySubsetFn::name().as_ptr(),
        ]
    }

    /// Return a list of required device extensions.
    #[cfg(not(target_os = "macos"))]
    pub(crate) fn required_device_extensions() -> Vec<*const i8> {
        vec![khr::Swapchain::name().as_ptr()]
    }
}
