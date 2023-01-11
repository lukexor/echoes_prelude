//! Vulkan surface.

use crate::platform;
use crate::window::Window;
use anyhow::Result;
use ash::{extensions::khr, vk};
use derive_more::{Deref, DerefMut};

#[derive(Deref, DerefMut)]
#[must_use]
pub(crate) struct Surface {
    pub(crate) loader: khr::Surface,
    #[deref]
    #[deref_mut]
    pub(crate) handle: vk::SurfaceKHR,
}

impl Surface {
    /// Create a platform-agnostic `Surface` instance.
    pub(crate) fn create(
        entry: &ash::Entry,
        instance: &ash::Instance,
        window: &Window,
    ) -> Result<Self> {
        tracing::debug!("creating surface");

        let loader = khr::Surface::new(entry, instance);
        let handle = platform::create_surface(entry, instance, window)?;

        tracing::debug!("surface created successfully");

        Ok(Self { loader, handle })
    }

    pub(crate) unsafe fn destroy(&self, allocator: Option<&vk::AllocationCallbacks>) {
        self.loader.destroy_surface(self.handle, allocator);
    }
}
