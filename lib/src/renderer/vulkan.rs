use crate::renderer::RendererBackend;
use anyhow::{bail, Context as _, Result};
use ash::{extensions::ext, vk};
use device::Device;
use std::{
    ffi::{c_void, CStr, CString},
    fmt,
};
use surface::Surface;
use winit::window::Window;

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
        unsafe {
            // NOTE: Drop-order is important here
            // self.device.device.destroy_device(None);
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
        write!(f, "Context {{ }}")
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
        let enabled_instance_extensions = platform::enabled_instance_extensions();
        let instance_create_flags = platform::instance_create_flags();
        let mut create_info = vk::InstanceCreateInfo::builder()
            .application_info(&application_info)
            .enabled_layer_names(&enabled_layer_names)
            .enabled_extension_names(&enabled_instance_extensions)
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

        unsafe { entry.create_instance(&create_info, None) }
            .context("failed to create vulkan instance")
    }
}

mod surface {
    use super::platform;
    use anyhow::Result;
    use ash::{extensions::khr, vk};
    use winit::{dpi::PhysicalSize, window::Window};

    #[must_use]
    pub(crate) struct Surface {
        pub(crate) handle: vk::SurfaceKHR,
        pub(crate) loader: khr::Surface,
        pub(crate) width: u32,
        pub(crate) height: u32,
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

mod device {
    use super::{platform, surface::Surface};
    use anyhow::{bail, Context, Result};
    use ash::vk;
    use std::{collections::HashSet, ffi::CStr};

    #[derive(Default, Debug, Copy, Clone)]
    #[must_use]
    pub(crate) struct Device {
        pub(crate) physical_device: vk::PhysicalDevice,
        // device: vk::Device,
    }

    impl Device {
        /// Picks a [vk::PhysicalDevice] and creates a [vk::Device].
        pub(crate) fn create(instance: &ash::Instance, surface: &Surface) -> Result<Device> {
            let physical_devices = unsafe { instance.enumerate_physical_devices() }
                .context("failed to enumerate physical devices")?;

            if physical_devices.is_empty() {
                bail!("failed to find any devices with vulkan support");
            }

            let mut device_infos = physical_devices
                .into_iter()
                .map(|physical_device| DeviceInfo::create(instance, physical_device, surface))
                .collect::<Vec<DeviceInfo>>();
            device_infos.sort_by_key(|device_info| device_info.rating);
            let selected_device = device_infos
                .pop()
                .context("failed to find a suitable physical device")?;
            // let device = Self::create_logical_device(&instance, &physical_device, &surface)?;

            Ok(Device {
                physical_device: selected_device.physical_device,
                // device: todo!(),
            })
        }
    }

    #[derive(Default, Debug)]
    #[must_use]
    struct DeviceInfo {
        physical_device: vk::PhysicalDevice,
        rating: u32,
        queue_family_indices: QueueFamilyIndices,
        swapchain_support: Option<SwapchainSupport>,
        sampler_anisotropy_support: bool,
    }

    impl DeviceInfo {
        /// Create device information for a [vk::PhysicalDevice] for rating suitability.
        fn create(
            instance: &ash::Instance,
            physical_device: vk::PhysicalDevice,
            surface: &Surface,
        ) -> Self {
            let device_properties =
                unsafe { instance.get_physical_device_properties(physical_device) };
            let device_features = unsafe { instance.get_physical_device_features(physical_device) };
            let queue_family_indices = QueueFamilyIndices::get(instance, physical_device, surface);
            let swapchain_support = SwapchainSupport::query(physical_device, surface)
                .map_err(|err| {
                    log::error!("{err}");
                })
                .ok();

            let mut rating = 0;
            if device_properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                rating += 1000;
            }
            rating += device_properties.limits.max_image_dimension2_d;

            if !Self::supports_required_extensions(instance, physical_device)
                || !queue_family_indices.is_valid()
            {
                rating = 0;
            }

            Self {
                physical_device,
                rating,
                queue_family_indices,
                swapchain_support,
                sampler_anisotropy_support: device_features.sampler_anisotropy == 1,
            }
        }

        /// Whether a physical device supports the required extensions for this platform.
        fn supports_required_extensions(
            instance: &ash::Instance,
            physical_device: vk::PhysicalDevice,
        ) -> bool {
            let device_extensions =
                unsafe { instance.enumerate_device_extension_properties(physical_device) };
            let Ok(device_extensions) = device_extensions else {
                log::error!("failed to enumerate vulkan device extensions");
                return false;
            };
            let extension_names = device_extensions
                .iter()
                .map(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) })
                .collect::<HashSet<_>>();
            platform::required_device_extensions()
                .iter()
                .all(|extension| extension_names.contains(extension))
        }
    }

    #[derive(Default, Debug)]
    #[must_use]
    struct SwapchainSupport {
        capabilities: vk::SurfaceCapabilitiesKHR,
        formats: Vec<vk::SurfaceFormatKHR>,
        present_modes: Vec<vk::PresentModeKHR>,
    }

    impl SwapchainSupport {
        fn query(physical_device: vk::PhysicalDevice, surface: &Surface) -> Result<Self> {
            let capabilities = unsafe {
                surface
                    .loader
                    .get_physical_device_surface_capabilities(physical_device, surface.handle)
            }
            .context("Failed to query for surface capabilities.")?;

            let formats = unsafe {
                surface
                    .loader
                    .get_physical_device_surface_formats(physical_device, surface.handle)
            }
            .context("Failed to query for surface formats.")?;

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

    #[derive(Default, Debug, Copy, Clone)]
    #[must_use]
    struct QueueFamilyIndices {
        graphics: Option<u32>,
        present: Option<u32>,
        compute: Option<u32>,
        transfer: Option<u32>,
    }

    impl QueueFamilyIndices {
        /// Find [vk::Queue] families.
        fn get(
            instance: &ash::Instance,
            physical_device: vk::PhysicalDevice,
            surface: &Surface,
        ) -> Self {
            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

            let mut queue_family_indices = Self::default();
            let mut fallback_present_index = None;
            let mut min_transfer_score = 255;
            for (i, queue_family) in queue_families.iter().enumerate() {
                let mut current_transfer_score = 0;

                let has_surface_support = unsafe {
                    surface.loader.get_physical_device_surface_support(
                        physical_device,
                        i as u32,
                        surface.handle,
                    )
                }
                .unwrap_or_default();

                if queue_family_indices.graphics.is_none()
                    && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                {
                    queue_family_indices.graphics = Some(i as u32);
                    current_transfer_score += 1;

                    // Prioritize queues supporting both graphics and present
                    if has_surface_support {
                        queue_family_indices.present = Some(i as u32);
                        current_transfer_score += 1;
                    }
                }

                if fallback_present_index.is_none() && has_surface_support {
                    fallback_present_index = Some(i as u32);
                }

                if queue_family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    queue_family_indices.compute = Some(i as u32);
                    current_transfer_score += 1;
                }

                if queue_family.queue_flags.contains(vk::QueueFlags::TRANSFER)
                    && current_transfer_score <= min_transfer_score
                {
                    min_transfer_score = current_transfer_score;
                    queue_family_indices.transfer = Some(i as u32);
                }

                if queue_family_indices.is_valid() {
                    break;
                }
            }

            if queue_family_indices.present.is_none() {
                queue_family_indices.present = fallback_present_index;
            }

            queue_family_indices
        }

        fn is_valid(&self) -> bool {
            self.graphics.is_some()
                && self.present.is_some()
                && self.compute.is_some()
                && self.transfer.is_some()
        }
    }
}

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
    use std::ffi::CStr;
    use winit::window::Window;

    /// Return a list of enabled Vulkan [Instance] extensions.
    #[cfg(target_os = "macos")]
    pub(crate) fn enabled_instance_extensions() -> Vec<*const i8> {
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
    pub(crate) fn enabled_instance_extensions() -> Vec<*const i8> {
        vec![
            khr::Surface::name().as_ptr(),
            #[cfg(debug_assertions)]
            ext::DebugUtils::name().as_ptr(),
            khr::XlibSurface::name().as_ptr(),
        ]
    }

    /// Return a list of enabled Vulkan [Instance] extensions.
    #[cfg(windows)]
    pub(crate) fn enabled_instance_extensions() -> Vec<*const i8> {
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
        use winit::platform::unix::WindowExtUnix;

        let create_info = vk::XlibSurfaceCreateInfoKHR::builder()
            .window(window.xlib_window()?)
            .build();
        let xlib_surface = khr::XlibSurface::new(entry, instance);
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
        use winit::platform::windows::WindowExtWindows;

        let create_info = vk::Win32SurfaceCreateInfoKHR::builder()
            .hwnd(window.hwnd())
            .build();
        let win32_surface = khr::Win32Surface::new(entry, instance);
        unsafe { win32_surface.create_win32_surface(&create_info, None) }
            .context("failed to create vulkan surface")
    }

    /// Return a list of required device extensions.
    #[cfg(target_os = "macos")]
    pub(crate) fn required_device_extensions() -> Vec<&'static CStr> {
        vec![
            khr::Swapchain::name(),
            vk::KhrPortabilityEnumerationFn::name(),
        ]
    }

    /// Return a list of required device extensions.
    #[cfg(not(target_os = "macos"))]
    pub(crate) fn required_device_extensions() -> Vec<&'static Cstr> {
        vec![khr::Swapchain::name()]
    }
}
