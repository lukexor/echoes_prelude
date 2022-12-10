use crate::renderer::RendererBackend;
use anyhow::{bail, Context as _, Result};
#[cfg(target_os = "macos")]
use ash::extensions::mvk;
use ash::{
    extensions::{ext, khr},
    vk, Entry, Instance,
};
use std::{
    ffi::{c_void, CStr, CString},
    fmt,
};
use winit::window::Window;

const VALIDATION_LAYER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };

pub(crate) struct Context {
    _entry: Entry,
    instance: Instance,
    #[cfg(debug_assertions)]
    debug_utils: ext::DebugUtils,
    #[cfg(debug_assertions)]
    debug_messenger: vk::DebugUtilsMessengerEXT,
}

impl Drop for Context {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl fmt::Debug for Context {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
    }
}

impl RendererBackend for Context {
    fn initialize(application_name: &str, _window: &Window) -> Result<Self> {
        log::debug!("initializing vulkan renderer backend");

        let entry = Entry::linked();
        let instance = Self::create_instance(application_name, &entry)?;
        #[cfg(debug_assertions)]
        let (debug_utils, debug_messenger) = Self::create_debug_utils(&entry, &instance)?;

        log::debug!("initalized vulkan successfully");

        Ok(Context {
            _entry: entry,
            instance,
            #[cfg(debug_assertions)]
            debug_utils,
            #[cfg(debug_assertions)]
            debug_messenger,
        })
    }

    fn on_resized(&mut self, _width: u32, _height: u32) {}

    fn draw_frame(&mut self) -> Result<()> {
        Ok(())
    }
}

impl Context {
    fn create_instance(application_name: &str, entry: &Entry) -> Result<Instance> {
        let application_info = vk::ApplicationInfo::builder()
            .application_name(&CString::new(application_name)?)
            .application_version(vk::API_VERSION_1_0)
            .engine_name(&CString::new("PixEngine")?)
            .engine_version(vk::API_VERSION_1_0)
            .api_version(vk::API_VERSION_1_3)
            .build();

        // Validation Layer support
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
        let enabled_extension_names = [
            khr::Surface::name().as_ptr(),
            #[cfg(debug_assertions)]
            ext::DebugUtils::name().as_ptr(),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            mvk::MacOSSurface::name().as_ptr(),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            vk::KhrPortabilityEnumerationFn::name().as_ptr(),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            vk::KhrGetPhysicalDeviceProperties2Fn::name().as_ptr(),
            #[cfg(windows)]
            khr::Win32Surface::name().as_ptr(),
            #[cfg(all(
                unix,
                not(target_os = "android"),
                not(target_os = "macos"),
                not(target_os = "ios")
            ))]
            khr::XlibSurface::name().as_ptr(),
        ];
        let mut create_info = vk::InstanceCreateInfo::builder()
            .application_info(&application_info)
            .enabled_layer_names(&enabled_layer_names)
            .enabled_extension_names(&enabled_extension_names)
            .flags(if cfg!(any(target_os = "macos", target_os = "ios")) {
                vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
            } else {
                vk::InstanceCreateFlags::default()
            })
            .build();

        // Debug Utils Creation
        #[cfg(debug_assertions)]
        let debug_create_info = build_debug_create_info();
        #[cfg(debug_assertions)]
        {
            let p_next: *const vk::DebugUtilsMessengerCreateInfoEXT = &debug_create_info;
            create_info.p_next = p_next as *const c_void;
        }

        unsafe { entry.create_instance(&create_info, None) }
            .context("failed to create vulkan instance")
    }

    #[cfg(debug_assertions)]
    fn create_debug_utils(
        entry: &Entry,
        instance: &Instance,
    ) -> Result<(ext::DebugUtils, vk::DebugUtilsMessengerEXT)> {
        let debug_utils = ext::DebugUtils::new(entry, instance);
        let create_info = build_debug_create_info();
        let debug_messenger =
            unsafe { debug_utils.create_debug_utils_messenger(&create_info, None)? };
        Ok((debug_utils, debug_messenger))
    }

    fn shutdown(&mut self) {
        unsafe {
            #[cfg(debug_assertions)]
            self.debug_utils
                .destroy_debug_utils_messenger(self.debug_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}

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
