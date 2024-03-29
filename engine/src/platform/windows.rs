//! Windows platform-specific implementation.

#[cfg(any(feature = "vulkan", not(feature = "opengl")))]
pub(crate) use vulkan::*;

#[cfg(any(feature = "vulkan", not(feature = "opengl")))]
mod vulkan {
    use crate::window::Window;
    use anyhow::{Context, Result};
    use ash::{
        extensions::{ext, khr},
        vk, Entry, Instance,
    };
    use std::ffi::CStr;

    /// Set of [`vk::InstanceCreateFlags`] for Windows.
    pub(crate) const INSTANCE_CREATE_FLAGS: vk::InstanceCreateFlags =
        vk::InstanceCreateFlags::empty();

    /// Count of required Vulkan [ash::Instance] extensions for Windows.
    #[cfg(debug_assertions)]
    const REQUIRED_EXTENSIONS_COUNT: usize = 3;
    #[cfg(not(debug_assertions))]
    const REQUIRED_EXTENSIONS_COUNT: usize = 2;

    /// List of required Vulkan [ash::Instance] extensions for Windows.
    pub(crate) const REQUIRED_EXTENSIONS: [&CStr; REQUIRED_EXTENSIONS_COUNT] = [
        khr::Surface::name(),
        #[cfg(debug_assertions)]
        ext::DebugUtils::name(),
        khr::Win32Surface::name(),
    ];

    /// Return a list of required [`vk::PhysicalDevice`] extensions for Windows.
    pub(crate) const REQUIRED_DEVICE_EXTENSIONS: [&CStr; 1] = [khr::Swapchain::name()];

    /// Create a [`vk::SurfaceKHR`] instance for the current [Window] for Windows.
    pub(crate) fn create_surface(
        entry: &Entry,
        instance: &Instance,
        window: &Window,
    ) -> Result<vk::SurfaceKHR> {
        use std::os::raw::c_void;
        use std::ptr;
        use winapi::um::libloaderapi::GetModuleHandleW;
        use winit::platform::windows::WindowExtWindows;

        tracing::debug!("creating win32 surface");

        let hinstance = unsafe { GetModuleHandleW(ptr::null()) };
        let surface_create_info = vk::Win32SurfaceCreateInfoKHR::builder()
            .hinstance(hinstance as *const c_void)
            .hwnd(window.hwnd() as *const c_void);
        let win32_surface = khr::Win32Surface::new(entry, instance);
        // SAFETY: All create_info values are set correctly above with valid lifetimes.
        unsafe { win32_surface.create_win32_surface(&surface_create_info, None) }
            .context("failed to create surface")
    }
}
