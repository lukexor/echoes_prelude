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
    use std::{collections::HashSet, ffi::CStr};

    /// Return a set of [`vk::InstanceCreateFlags`] for Linux.
    pub(crate) fn instance_create_flags() -> vk::InstanceCreateFlags {
        vk::InstanceCreateFlags::default()
    }

    /// Return a list of required Vulkan [ash::Instance] extensions for Windows.
    pub(crate) fn required_extensions() -> Vec<*const i8> {
        vec![
            khr::Surface::name().as_ptr(),
            #[cfg(debug_assertions)]
            ext::DebugUtils::name().as_ptr(),
            khr::Win32Surface::name().as_ptr(),
        ]
    }

    /// Return a list of required [`vk::PhysicalDevice`] extensions for windows.
    pub(crate) fn required_device_extensions(
        _supported_extensions: &HashSet<&CStr>,
    ) -> Vec<*const i8> {
        vec![khr::Swapchain::name().as_ptr()]
    }

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

        log::debug!("creating win32 surface");

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
