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
    use std::{collections::HashSet, ffi::CStr};

    /// Return a set of [`vk::InstanceCreateFlags`] for macOS.
    pub(crate) fn instance_create_flags() -> vk::InstanceCreateFlags {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    }

    /// Return a list of required Vulkan [ash::Instance] extensions for macOS.
    pub(crate) fn required_extensions() -> Vec<*const i8> {
        vec![
            khr::Surface::name().as_ptr(),
            #[cfg(debug_assertions)]
            ext::DebugUtils::name().as_ptr(),
            mvk::MacOSSurface::name().as_ptr(),
            vk::KhrPortabilityEnumerationFn::name().as_ptr(),
            vk::KhrGetPhysicalDeviceProperties2Fn::name().as_ptr(),
        ]
    }

    /// Return a list of required [`vk::PhysicalDevice`] extensions for macOS.
    pub(crate) fn required_device_extensions(
        supported_extensions: &HashSet<&CStr>,
    ) -> Vec<*const i8> {
        let mut required_extensions = vec![
            khr::Swapchain::name().as_ptr(),
            vk::KhrPortabilitySubsetFn::name().as_ptr(),
        ];
        if supported_extensions.contains(ext::MetalSurface::name()) {
            required_extensions.push(ext::MetalSurface::name().as_ptr());
        }
        required_extensions
    }

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

        let surface_create_info = vk::MacOSSurfaceCreateInfoMVK::builder().view(window.ns_view());

        let macos_surface = mvk::MacOSSurface::new(entry, instance);
        // SAFETY: All create_info values are set correctly above with valid lifetimes.
        unsafe { macos_surface.create_mac_os_surface(&surface_create_info, None) }
            .context("failed to create surface")
    }
}
