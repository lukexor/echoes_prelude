#![allow(unused)]
use crate::{render_bail, Result};
use anyhow::Context;
use ash::vk;
use fnv::FnvHashMap;
use std::{
    collections::hash_map::Entry,
    hash::{Hash, Hasher},
    slice,
};

/// Multiplier constants for a range of descriptor types. This can be tuned as needed based on
/// usage patterns.
const POOL_SIZES: [(vk::DescriptorType, f32); 11] = [
    (vk::DescriptorType::SAMPLER, 0.5),
    (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 4.0),
    (vk::DescriptorType::SAMPLED_IMAGE, 4.0),
    (vk::DescriptorType::STORAGE_IMAGE, 1.0),
    (vk::DescriptorType::UNIFORM_TEXEL_BUFFER, 1.0),
    (vk::DescriptorType::STORAGE_TEXEL_BUFFER, 1.0),
    (vk::DescriptorType::UNIFORM_BUFFER, 2.0),
    (vk::DescriptorType::STORAGE_BUFFER, 2.0),
    (vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC, 1.0),
    (vk::DescriptorType::STORAGE_BUFFER_DYNAMIC, 1.0),
    (vk::DescriptorType::INPUT_ATTACHMENT, 0.5),
];

/// A [`vk::DescriptorSet`] allocator re-uses [`vk::DescriptorPool`]s and creates new pools as
/// necessary when existing pools run out of memory.
#[must_use]
pub(crate) struct DescriptorAllocator {
    device: ash::Device,
    current_pool: Option<vk::DescriptorPool>,
    used_pools: Vec<vk::DescriptorPool>,
    free_pools: Vec<vk::DescriptorPool>,
}

impl DescriptorAllocator {
    /// Initialize the allocator with a given device.
    pub(crate) fn initialize(device: ash::Device) -> Self {
        Self {
            device,
            current_pool: None,
            used_pools: vec![],
            free_pools: vec![],
        }
    }

    /// Destroy the allocator and all [`vk::DescriptorPool`]s.
    pub(crate) unsafe fn destroy(&mut self) {
        for pool in self.free_pools.drain(..).chain(self.used_pools.drain(..)) {
            self.device.destroy_descriptor_pool(pool, None);
        }
    }

    /// Allocate a new [`vk::DescriptorSet`] from an existing [`vk::DescriptorPool`]. Creates a new
    /// pool if one is not created yet or the current pool ran out of memory.
    pub(crate) fn allocate(
        &mut self,
        layouts: &[vk::DescriptorSetLayout],
    ) -> Result<vk::DescriptorSet> {
        let current_pool = match self.current_pool {
            Some(pool) => pool,
            None => {
                let pool = self.get_or_create_pool()?;
                self.current_pool = Some(pool);
                pool
            }
        };
        let descriptor_set_alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(current_pool)
            .set_layouts(layouts);
        let descriptor_sets_alloc_result = unsafe {
            self.device
                .allocate_descriptor_sets(&descriptor_set_alloc_info)
        };
        match descriptor_sets_alloc_result {
            Ok(descriptor_sets) => Ok(descriptor_sets[0]),
            Err(err) => match err {
                vk::Result::ERROR_FRAGMENTED_POOL | vk::Result::ERROR_OUT_OF_POOL_MEMORY => {
                    let pool = self.get_or_create_pool()?;
                    self.current_pool = Some(pool);
                    let descriptor_sets = unsafe {
                        self.device
                            .allocate_descriptor_sets(&descriptor_set_alloc_info)
                    }?;
                    Ok(descriptor_sets[0])
                }
                _ => render_bail!("failed to allocate descriptor sets"),
            },
        }
    }

    /// Reset used [`vk::DescriptorPool`]s, returning them to a list of free pools.
    pub(crate) fn reset_pools(&mut self) -> Result<()> {
        while let Some(pool) = self.used_pools.pop() {
            unsafe {
                self.device
                    .reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty())
            }?;
            self.free_pools.push(pool);
        }
        self.current_pool = None;
        Ok(())
    }

    /// Gets a new [`vk::DescriptorPool`] from the list of free pools, if any are left. Otherwise,
    /// a new pool is created.
    fn get_or_create_pool(&mut self) -> Result<vk::DescriptorPool> {
        let pool = if let Some(pool) = self.free_pools.pop() {
            pool
        } else {
            let count = 1000;
            let flags = vk::DescriptorPoolCreateFlags::empty();
            let pool_sizes = POOL_SIZES.map(|(ty, size)| {
                vk::DescriptorPoolSize::builder()
                    .ty(ty)
                    .descriptor_count((size * count as f32) as u32)
                    .build()
            });
            let pool_create_info = vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&pool_sizes)
                .max_sets(count)
                .flags(flags);
            unsafe { self.device.create_descriptor_pool(&pool_create_info, None) }
                .context("failed to create descriptor pool")?
        };
        self.used_pools.push(pool);
        Ok(pool)
    }
}

/// A list of [`vk::DescriptorSetLayoutBinding`]s used to cache [`vk::DescriptorSetLayout`]s.
#[must_use]
pub(crate) struct DescriptorLayoutInfo {
    bindings: Vec<vk::DescriptorSetLayoutBinding>,
}

impl PartialEq for DescriptorLayoutInfo {
    fn eq(&self, rhs: &Self) -> bool {
        if self.bindings.len() != rhs.bindings.len() {
            false
        } else {
            self.bindings
                .iter()
                .zip(rhs.bindings.iter())
                .any(|(lhs, rhs)| {
                    lhs.binding != rhs.binding
                        || lhs.descriptor_type != rhs.descriptor_type
                        || lhs.descriptor_count != rhs.descriptor_count
                        || lhs.stage_flags != rhs.stage_flags
                })
        }
    }
}

impl Eq for DescriptorLayoutInfo {}

impl Hash for DescriptorLayoutInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for binding in &self.bindings {
            binding.binding.hash(state);
            binding.descriptor_type.hash(state);
            binding.descriptor_count.hash(state);
            binding.stage_flags.hash(state);
        }
    }
}

/// A [`vk::DescriptorSetLayout`] cache to avoid creating duplicate descriptor set layouts.
#[must_use]
pub(crate) struct DescriptorLayoutCache {
    device: ash::Device,
    cache: FnvHashMap<DescriptorLayoutInfo, vk::DescriptorSetLayout>,
}

impl DescriptorLayoutCache {
    /// Initialize the cache with a given device.
    pub(crate) fn initialize(device: ash::Device) -> Self {
        Self {
            device,
            cache: FnvHashMap::default(),
        }
    }

    /// Destroy the cache and all [`vk::DescriptorSetLayout`]s.
    pub(crate) unsafe fn destroy(&mut self) {
        for (_, layout) in self.cache.drain() {
            self.device.destroy_descriptor_set_layout(layout, None);
        }
    }

    /// Return a [`vk::DescriptorSetLayout`] from the cache, or create a new one and cache it.
    pub(crate) fn create_descriptor_set_layout(
        &mut self,
        layout_create_info: &vk::DescriptorSetLayoutCreateInfo,
    ) -> Result<vk::DescriptorSetLayout> {
        let mut layout_info = DescriptorLayoutInfo {
            bindings: unsafe {
                slice::from_raw_parts(
                    layout_create_info.p_bindings,
                    layout_create_info.binding_count as usize,
                )
            }
            .to_vec(),
        };
        layout_info.bindings.sort_by_key(|binding| binding.binding);
        let layout = match self.cache.entry(layout_info) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let layout = unsafe {
                    self.device
                        .create_descriptor_set_layout(layout_create_info, None)
                }
                .context("failed to create descriptor set layout")?;
                *entry.insert(layout)
            }
        };
        Ok(layout)
    }
}

/// A [`vk::DescriptorSetLayout`] builder with an internal layout cache and growable descriptor
/// pool allocator.
#[must_use]
pub(crate) struct DescriptorBuilder<'a> {
    write_sets: Vec<vk::WriteDescriptorSet>,
    bindings: Vec<vk::DescriptorSetLayoutBinding>,
    cache: &'a mut DescriptorLayoutCache,
    alloc: &'a mut DescriptorAllocator,
}

impl<'a> DescriptorBuilder<'a> {
    /// Create a new `DescriptorBuilder` with an existing cache and allocator.
    pub(crate) fn new(
        cache: &'a mut DescriptorLayoutCache,
        alloc: &'a mut DescriptorAllocator,
    ) -> Self {
        Self {
            write_sets: vec![],
            bindings: vec![],
            cache,
            alloc,
        }
    }

    /// Bind a [`vk::DescriptorBufferInfo`] to the [`vk::DescriptorSetLayout`].
    pub(crate) fn bind_buffer(
        &mut self,
        binding: u32,
        buffer_info: &'a [vk::DescriptorBufferInfo],
        ty: vk::DescriptorType,
        flags: vk::ShaderStageFlags,
    ) -> &mut Self {
        let layout_binding = vk::DescriptorSetLayoutBinding::builder()
            .descriptor_type(ty)
            .stage_flags(flags)
            .binding(binding)
            .build();
        self.bindings.push(layout_binding);
        let write_set = vk::WriteDescriptorSet::builder()
            .descriptor_type(ty)
            .buffer_info(buffer_info)
            .dst_binding(binding)
            .build();
        self.write_sets.push(write_set);
        self
    }

    /// Bind a [`vk::DescriptorImageInfo`] to the [`vk::DescriptorSetLayout`].
    pub(crate) fn bind_image(
        &mut self,
        binding: u32,
        image_info: &'a [vk::DescriptorImageInfo],
        ty: vk::DescriptorType,
        flags: vk::ShaderStageFlags,
    ) -> &mut Self {
        let layout_binding = vk::DescriptorSetLayoutBinding::builder()
            .descriptor_type(ty)
            .stage_flags(flags)
            .binding(binding)
            .build();
        self.bindings.push(layout_binding);
        let write_set = vk::WriteDescriptorSet::builder()
            .descriptor_type(ty)
            .image_info(image_info)
            .dst_binding(binding)
            .build();
        self.write_sets.push(write_set);
        self
    }

    /// Build the [`vk::DescriptorSetLayout`].
    pub(crate) fn build(mut self) -> Result<vk::DescriptorSetLayout> {
        let layout_create_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&self.bindings)
            .build();
        let layout = self
            .cache
            .create_descriptor_set_layout(&layout_create_info)?;
        let descriptor_set = self.alloc.allocate(slice::from_ref(&layout))?;

        for write_set in &mut self.write_sets {
            write_set.dst_set = descriptor_set;
        }

        unsafe {
            self.alloc
                .device
                .update_descriptor_sets(&self.write_sets, &[]);
        }

        Ok(layout)
    }
}
