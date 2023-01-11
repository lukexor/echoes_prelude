//! macOS platform-specific implementation.

#[cfg(any(feature = "vulkan", not(feature = "opengl")))]
pub(crate) use vulkan::*;

#[cfg(any(feature = "vulkan", not(feature = "opengl")))]
mod vulkan {
    use crate::window::Window;
    use anyhow::{Context, Result};
    use ash::{
        extensions::{ext, khr, mvk},
        vk, Entry, Instance,
    };
    use std::ffi::CStr;

    /// Set of [`vk::InstanceCreateFlags`] for macOS.
    pub(crate) const INSTANCE_CREATE_FLAGS: vk::InstanceCreateFlags =
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;

    /// Count of required Vulkan [`ash::Instance`] extensions for macOS.
    #[cfg(debug_assertions)]
    const REQUIRED_EXTENSIONS_COUNT: usize = 6;
    #[cfg(not(debug_assertions))]
    const REQUIRED_EXTENSIONS_COUNT: usize = 5;

    /// List of required Vulkan [ash::Instance] extensions for macOS.
    pub(crate) const REQUIRED_EXTENSIONS: [&CStr; REQUIRED_EXTENSIONS_COUNT] = [
        khr::Surface::name(),
        mvk::MacOSSurface::name(),
        ext::MetalSurface::name(),
        vk::KhrPortabilityEnumerationFn::name(),
        vk::KhrGetPhysicalDeviceProperties2Fn::name(),
        #[cfg(debug_assertions)]
        ext::DebugUtils::name(),
    ];

    /// Count of required Vulkan [`vk::PhysicalDevice`] extensions for macOS.
    const REQUIRED_DEVICE_EXTENSIONS_COUNT: usize = 2;

    /// Return a list of required [`vk::PhysicalDevice`] extensions for macOS.
    pub(crate) const REQUIRED_DEVICE_EXTENSIONS: [&CStr; REQUIRED_DEVICE_EXTENSIONS_COUNT] =
        [khr::Swapchain::name(), vk::KhrPortabilitySubsetFn::name()];

    /// Create a [`vk::SurfaceKHR`] instance for the current [Window] for macOS.
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

        tracing::debug!("creating macOS surface");

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

        let surface_create_info = vk::MacOSSurfaceCreateInfoMVK::builder().view(window.ns_view());

        let macos_surface = mvk::MacOSSurface::new(entry, instance);
        // SAFETY: All create_info values are set correctly above with valid lifetimes.
        let surface = unsafe { macos_surface.create_mac_os_surface(&surface_create_info, None) }
            .context("failed to create surface")?;

        tracing::debug!("created macOS surface successfully");

        Ok(surface)
    }
}
