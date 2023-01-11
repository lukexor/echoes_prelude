//! Linux platform-specific implementation.

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

    /// Set of [`vk::InstanceCreateFlags`] for Linux.
    pub(crate) const INSTANCE_CREATE_FLAGS: vk::InstanceCreateFlags =
        vk::InstanceCreateFlags::default();

    /// Count of required Vulkan [ash::Instance] extensions for Linux.
    #[cfg(debug_assertions)]
    const REQUIRED_EXTENSIONS_COUNT: usize = 3;
    #[cfg(not(debug_assertions))]
    const REQUIRED_EXTENSIONS_COUNT: usize = 2;

    /// List of required Vulkan [ash::Instance] extensions for Linux.
    pub(crate) const REQUIRED_EXTENSIONS: [*const i8; REQUIRED_EXTENSIONS_COUNT] = [
        khr::Surface::name().as_ptr(),
        #[cfg(debug_assertions)]
        ext::DebugUtils::name().as_ptr(),
        khr::XlibSurface::name().as_ptr(),
    ];

    /// Return a list of required [`vk::PhysicalDevice`] extensions for Linux.
    pub(crate) const REQUIRED_DEVICE_EXTENSIONS: [*const i8; 1] = [khr::Swapchain::name().as_ptr()];

    /// Return a list of optional [`vk::PhysicalDevice`] extensions for Linux.
    #[inline]
    pub(crate) fn optional_device_extensions(
        supported_extensions: &HashSet<&CStr>,
    ) -> Vec<*const i8> {
        vec![]
    }

    /// Create a [`vk::SurfaceKHR`] instance for the current [Window] for Linux.
    pub(crate) fn create_surface(
        entry: &Entry,
        instance: &Instance,
        window: &Window,
    ) -> Result<vk::SurfaceKHR> {
        use winit::platform::unix::WindowExtUnix;

        tracing::debug!("creating Linux XLIB surface");

        let x11_display = window
            .xlib_display()
            .context("failed to get XLIB display")?;
        let x11_window = window.xlib_window().context("failed to get XLIB window")?;
        let surface_create_info = vk::XlibSurfaceCreateInfoKHR::builder()
            .window(x11_window)
            .dpy(x11_display as *mut vk::Display);
        let xlib_surface = khr::XlibSurface::new(entry, instance);
        // SAFETY: All create_info values are set correctly above with valid lifetimes.
        unsafe { xlib_surface.create_xlib_surface(&surface_create_info, None) }
            .context("failed to create surface")
    }
}
