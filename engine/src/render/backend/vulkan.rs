//! Vulkan renderer backendvulkanrs

#[cfg(debug_assertions)]
use self::debug::VALIDATION_LAYER_NAME;
use self::{
    buffer::{padded_uniform_buffer_size, AllocatedBuffer},
    debug::Debug,
    descriptors::{DescriptorAllocator, DescriptorLayoutCache},
    device::{Device, QueueFamily},
    frame::Frame,
    image::AllocatedImage,
    mesh::{AllocatedMesh, Material, Object},
    pipeline::set_layouts,
    surface::Surface,
    swapchain::{Swapchain, MAX_FRAMES_IN_FLIGHT},
};
use super::{RenderBackend, RenderSettings};
use crate::{
    camera::CameraData,
    hash_map,
    mesh::{MaterialType, Mesh, ObjectData},
    platform,
    prelude::*,
    render::{DrawCmd, DrawData},
    render_bail,
    scene::SceneData,
    window::Window,
    Error, Result,
};
use anyhow::Context as _;
use ash::vk;
use asset_loader::{Asset, TextureAsset};
use fnv::FnvHashMap;
use semver::Version;
use std::{
    ffi::{CStr, CString},
    fmt, mem,
    path::Path,
    ptr, slice,
};
use tokio::runtime::{self, Runtime};

mod buffer;
mod command_pool;
mod descriptors;
mod device;
mod frame;
mod image;
#[cfg(feature = "imgui")]
mod imgui_renderer;
mod mesh;
mod pipeline;
mod surface;
mod swapchain;

#[cfg(debug_assertions)]
pub(crate) const ENABLED_LAYER_NAMES: [&CStr; 1] = [VALIDATION_LAYER_NAME];
#[cfg(not(debug_assertions))]
pub(crate) const ENABLED_LAYER_NAMES: [&CStr; 0] = [];

pub struct Context {
    size: PhysicalSize<u32>,
    resized: bool,
    frame_number: usize,
    current_frame: usize,
    settings: RenderSettings,
    runtime: Runtime,

    _entry: ash::Entry,
    instance: ash::Instance,
    debug: Option<Debug>,

    surface: Surface,
    device: Device,
    swapchain: Swapchain,
    frames: Vec<Frame>,
    render_pass: vk::RenderPass,

    descriptor_alloc: DescriptorAllocator,
    descriptor_layout_cache: DescriptorLayoutCache,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    global_descriptor_pool: vk::DescriptorPool,
    pipeline_layout: vk::PipelineLayout,
    pipelines: Vec<vk::Pipeline>,
    graphics_command_pool: vk::CommandPool,

    clear_color: vk::ClearValue,
    clear_depth: vk::ClearValue,
    viewport: vk::Viewport,
    scissor: vk::Rect2D,

    camera_view: CameraData,
    scene_attributes: SceneData,
    scene_buffer: AllocatedBuffer,
    objects: FnvHashMap<String, Object>,
    materials: FnvHashMap<MaterialType, Material>,
    meshes: FnvHashMap<String, AllocatedMesh>,
    textures: FnvHashMap<String, AllocatedImage>,

    sampler: vk::Sampler,
    color_image: Option<AllocatedImage>,
    depth_image: AllocatedImage,
    primary_framebuffers: Vec<vk::Framebuffer>,

    #[cfg(feature = "imgui")]
    imgui_renderer: Option<imgui_renderer::Renderer>,
}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Context").finish_non_exhaustive()
    }
}

impl RenderBackend for Context {
    /// Initialize Vulkan `Context`.
    fn initialize(
        application_name: &str,
        application_version: &str,
        window: &Window,
        settings: RenderSettings,
    ) -> Result<Self> {
        tracing::debug!("initializing vulkan renderer backend");

        let entry = ash::Entry::linked();
        let instance = create_instance(&entry, application_name, application_version)?;
        let debug = if cfg!(debug_assertions) {
            Some(Debug::create(&entry, &instance)?)
        } else {
            None
        };

        let surface = Surface::create(&entry, &instance, window)?;
        let device = Device::create(&instance, &surface, &settings)?;

        let size = window.inner_size().into();
        let swapchain = Swapchain::create(&instance, &device, &surface, size, None)?;
        let depth_format = device.get_depth_format(&instance, vk::ImageTiling::OPTIMAL)?;
        let render_pass = create_render_pass(&device, swapchain.format, depth_format, &settings)?;

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: swapchain.extent.width as f32,
            height: swapchain.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain.extent,
        };

        let descriptor_alloc = DescriptorAllocator::initialize(device.handle.clone());
        let descriptor_layout_cache = DescriptorLayoutCache::initialize(device.handle.clone());
        let (descriptor_set_layouts, global_descriptor_pool) =
            pipeline::create_global_descriptor_pool(&device)?;

        let scene_attributes = SceneData::default();
        let scene_buffer = AllocatedBuffer::create(
            &device,
            "scene",
            MAX_FRAMES_IN_FLIGHT as u64 * padded_uniform_buffer_size::<SceneData>(&device),
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
            debug.as_ref(),
        )?;
        let frames = Frame::create(
            &device,
            global_descriptor_pool,
            &[
                descriptor_set_layouts[set_layouts::GLOBAL],
                descriptor_set_layouts[set_layouts::OBJECT],
            ],
            &scene_buffer,
            debug.as_ref(),
        )?;

        let (pipeline_layout, pipelines) = pipeline::create_default(
            &device,
            viewport,
            scissor,
            render_pass,
            &descriptor_set_layouts,
            &settings,
        )?;

        let graphics_command_pool = command_pool::create(
            &device,
            QueueFamily::Graphics,
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        )?;

        let sampler = AllocatedImage::create_sampler(&device, &settings)?;

        let materials = hash_map! {
            MaterialType::Default => Material::new(pipelines[pipeline::DEFAULT], pipeline_layout, None),
        };
        let color_image = if settings.msaa {
            Some(AllocatedImage::create_color(
                &device,
                swapchain.format,
                swapchain.extent,
                device.info.msaa_samples,
                debug.as_ref(),
            )?)
        } else {
            None
        };
        let depth_image = AllocatedImage::create_depth(
            &instance,
            &device,
            swapchain.extent,
            device.info.msaa_samples,
            debug.as_ref(),
        )?;
        let primary_framebuffers = swapchain.create_framebuffers(
            &device,
            render_pass,
            color_image.as_ref().map(|image| image.view),
            Some(depth_image.view),
        )?;

        tracing::debug!("initialized vulkan renderer backend successfully");

        Ok(Self {
            size,
            resized: false,
            frame_number: 0,
            current_frame: 0,
            settings,
            runtime: runtime::Builder::new_multi_thread().enable_all().build()?,

            _entry: entry,
            instance,
            debug,

            surface,
            device,
            swapchain,
            frames,
            render_pass,

            descriptor_alloc,
            descriptor_layout_cache,
            descriptor_set_layouts,
            global_descriptor_pool,
            pipeline_layout,
            pipelines,
            graphics_command_pool,

            clear_color: vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            clear_depth: vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
            viewport,
            scissor,

            scene_attributes,
            scene_buffer,
            camera_view: CameraData::default(),
            objects: FnvHashMap::default(),
            materials,
            meshes: FnvHashMap::default(),
            textures: FnvHashMap::default(),

            sampler,
            color_image,
            depth_image,
            primary_framebuffers,

            #[cfg(feature = "imgui")]
            imgui_renderer: None,
        })
    }

    /// Initialize imgui renderer.
    #[cfg(feature = "imgui")]
    fn initialize_imgui(&mut self, imgui: &mut imgui::ImGui) -> Result<()> {
        self.imgui_renderer = Some(imgui_renderer::Renderer::initialize(
            &self.instance,
            &self.device,
            &self.swapchain,
            self.graphics_command_pool,
            imgui,
        )?);
        Ok(())
    }

    /// Handle window resized event.
    #[inline]
    fn on_resized(&mut self, size: PhysicalSize<u32>) {
        if self.size != size {
            tracing::debug!("received resized event {size:?}. pending swapchain recreation.");
            self.size = size;
            self.resized = true;
        }
    }

    /// Called at the end of a frame to submit updates to the renderer.
    #[inline]
    fn draw_frame(&mut self, draw_data: &mut DrawData<'_>) -> Result<()> {
        for cmd in draw_data.commands {
            match cmd {
                DrawCmd::ClearColor(color) => self.set_clear_color(*color),
                DrawCmd::ClearDepthStencil((depth, stencil)) => {
                    self.set_clear_depth_stencil(*depth, *stencil);
                }
                DrawCmd::SetProjection(projection) => self.set_projection(*projection),
                DrawCmd::SetView(view) => self.set_view(*view),
                DrawCmd::SetObjectTransform { name, transform } => {
                    self.set_object_transform(name, *transform);
                }
                DrawCmd::LoadMesh { name, filename } => self.load_mesh(name, filename)?,
                DrawCmd::LoadTexture { name, filename } => self.load_texture(name, filename)?,
                DrawCmd::LoadObject {
                    name,
                    mesh,
                    material_type,
                    transform,
                } => self.load_object(name, mesh, material_type.clone(), *transform)?,
            }
        }

        assert!(self.current_frame < self.frames.len());
        let render_fence = self.frames[self.current_frame].render_fence;
        unsafe {
            self.device
                .wait_for_fences(slice::from_ref(&render_fence), true, u64::MAX)?;
        }

        let present_semaphor = self.frames[self.current_frame].present_semaphor;
        let image_index = {
            let result = unsafe {
                self.swapchain.loader.acquire_next_image(
                    self.swapchain.handle,
                    u64::MAX,
                    present_semaphor,
                    vk::Fence::null(),
                )
            };
            match result {
                Ok((index, _is_sub_optimal)) => index as usize,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    tracing::debug!("swapchain image out of date");
                    self.recreate_swapchain()?;
                    return Ok(());
                }
                Err(err) => return Err(err.into()),
            }
        };

        self.draw_objects(
            self.frames[self.current_frame].command_buffers[0],
            image_index,
        )?;
        #[cfg(feature = "imgui")]
        if let Some(imgui_renderer) = &mut self.imgui_renderer {
            imgui_renderer.draw(
                &self.device,
                &self.swapchain,
                self.frames[self.current_frame].command_buffers[1],
                image_index,
                draw_data.imgui,
            )?;
        }

        #[cfg(feature = "imgui")]
        let command_buffers = if self.imgui_renderer.is_some() {
            &self.frames[self.current_frame].command_buffers
        } else {
            slice::from_ref(&self.frames[self.current_frame].command_buffers[0])
        };
        #[cfg(not(feature = "imgui"))]
        let command_buffers = slice::from_ref(&self.frames[self.current_frame].command_buffers[0]);
        let wait_stages = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        let render_semaphor = self.frames[self.current_frame].render_semaphor;
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(slice::from_ref(&present_semaphor))
            .wait_dst_stage_mask(slice::from_ref(&wait_stages))
            .command_buffers(command_buffers)
            .signal_semaphores(slice::from_ref(&render_semaphor))
            .build();

        unsafe {
            self.device.reset_fences(slice::from_ref(&render_fence))?;
            self.device.queue_submit(
                self.device.graphics_queue,
                slice::from_ref(&submit_info),
                render_fence,
            )?;
        }

        let image_index = image_index as u32;
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(slice::from_ref(&render_semaphor))
            .swapchains(slice::from_ref(&self.swapchain.handle))
            .image_indices(slice::from_ref(&image_index));

        let result = unsafe {
            self.swapchain
                .loader
                .queue_present(self.device.present_queue, &present_info)
        };

        let is_resized = match result {
            Ok(_) => self.resized,
            Err(vk_result) => match vk_result {
                vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => true,
                err => return Err(err.into()),
            },
        };

        if is_resized {
            tracing::debug!("swapchain is suboptimal for surface");
            self.recreate_swapchain()?;
            self.resized = false;
        }

        self.frame_number += 1;
        self.current_frame += 1;
        if self.current_frame == MAX_FRAMES_IN_FLIGHT {
            self.current_frame = 0;
        }

        Ok(())
    }

    /// Set the clear color for the next frame.
    #[inline]
    fn set_clear_color(&mut self, color: Vec4) {
        self.clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: color.to_array(),
            },
        };
    }

    /// Set the clear depth and/or stencil for the next frame.
    #[inline]
    fn set_clear_depth_stencil(&mut self, depth: f32, stencil: u32) {
        self.clear_depth = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
        };
    }

    /// Set the projection matrix for the next frame.
    #[inline]
    fn set_projection(&mut self, projection: Mat4) {
        self.camera_view.projection = projection;
    }

    /// Set the view matrix for the next frame.
    #[inline]
    fn set_view(&mut self, view: Mat4) {
        self.camera_view.view = view;
    }

    /// Set an objects transform matrix.
    #[inline]
    fn set_object_transform(&mut self, name: &str, transform: Mat4) {
        if let Some(object) = self.objects.get_mut(name) {
            object.transform = transform;
        }
    }

    /// Load a mesh into memory.
    fn load_mesh(&mut self, name: &str, filename: &Path) -> Result<()> {
        let mesh = self.runtime.block_on(Mesh::from_asset_path(filename))?;

        // TODO: allocate from larger buffer pool
        let vertex_buffer = AllocatedBuffer::create_array(
            &self.device,
            "vertex",
            self.graphics_command_pool,
            self.device.graphics_queue,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &mesh.vertices,
            self.debug.as_ref(),
        )?;
        let index_buffer = AllocatedBuffer::create_array(
            &self.device,
            "index",
            self.graphics_command_pool,
            self.device.graphics_queue,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &mesh.indices,
            self.debug.as_ref(),
        )?;

        self.meshes.insert(
            name.to_string(),
            AllocatedMesh::new(mesh, vertex_buffer, index_buffer),
        );

        Ok(())
    }

    /// Unload a named mesh from memory.
    fn unload_mesh(&mut self, name: &str) -> Result<()> {
        if let Some(mut mesh) = self.meshes.remove(name) {
            unsafe {
                self.device.device_wait_idle()?;
                mesh.destroy(&self.device);
            };
            Ok(())
        } else {
            render_bail!("mesh {name} is not loaded");
        }
    }

    /// Load a texture into memory.
    fn load_texture(&mut self, name: &str, filename: &Path) -> Result<()> {
        let texture = self
            .runtime
            .block_on(TextureAsset::load(filename))
            .context("failed to load texture {filename:?}")?;
        let texture_image = self.runtime.block_on(AllocatedImage::create_texture(
            &self.instance,
            &self.device,
            self.graphics_command_pool,
            self.device.graphics_queue,
            &texture,
            self.debug.as_ref(),
        ))?;

        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.global_descriptor_pool)
            .set_layouts(slice::from_ref(
                &self.descriptor_set_layouts[set_layouts::TEXTURE],
            ));
        let descriptor_sets = unsafe {
            self.device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
        }
        .context("failed to allocate descriptor sets")?;

        let material = Material::new(
            self.pipelines[pipeline::TEXTURE],
            self.pipeline_layout,
            descriptor_sets[0],
        );
        self.materials
            .insert(MaterialType::Texture(name.to_string()), material);

        let texture_info = vk::DescriptorImageInfo::builder()
            .sampler(self.sampler)
            .image_view(texture_image.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let texture_write = vk::WriteDescriptorSet::builder()
            .dst_binding(0)
            .dst_set(descriptor_sets[0])
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(slice::from_ref(&texture_info))
            .build();

        let write_sets = [texture_write];
        unsafe {
            self.device.update_descriptor_sets(&write_sets, &[]);
        };

        self.textures.insert(name.to_string(), texture_image);
        Ok(())
    }

    /// Unload a named texture from memory.
    fn unload_texture(&mut self, name: &str) -> Result<()> {
        if let Some(mut texture) = self.textures.remove(name) {
            unsafe {
                self.device.device_wait_idle()?;
                texture.destroy(&self.device);
            };
            Ok(())
        } else {
            render_bail!("mesh {name} is not loaded");
        }
    }

    /// Load a mesh object to render in the current scene.
    fn load_object(
        &mut self,
        name: &str,
        mesh_name: &str,
        material_type: MaterialType,
        transform: Mat4,
    ) -> Result<()> {
        if !self.meshes.contains_key(mesh_name) {
            render_bail!("mesh {} does not exist", mesh_name);
        }
        let material = self
            .materials
            .get(&material_type)
            .cloned()
            .with_context(|| format!("material `{material_type}` does not exist"))?;
        self.objects.insert(
            name.to_string(),
            Object {
                mesh: mesh_name.to_string(),
                material,
                transform,
            },
        );
        Ok(())
    }

    /// Unload a named object from the current scene.
    fn unload_object(&mut self, name: &str) -> Result<()> {
        self.objects
            .remove(name)
            .map_or_else(|| render_bail!("object {name} is not loaded"), |_| Ok(()))
    }
}

impl Context {
    /// Re-creates the swapchain, scheduling the old one for destruction.
    fn recreate_swapchain(&mut self) -> Result<()> {
        tracing::debug!("re-creating swapchain: {:?}", self.size);

        unsafe {
            // TODO: Clean up old swapchain without blocking
            self.device.device_wait_idle()?;

            for &framebuffer in &self.primary_framebuffers {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            self.swapchain.destroy(&self.device, None);
            if let Some(image) = &mut self.color_image {
                image.destroy(&self.device);
            }
            self.depth_image.destroy(&self.device);
        }

        self.swapchain =
            Swapchain::create(&self.instance, &self.device, &self.surface, self.size, None)?;
        self.viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: self.swapchain.extent.width as f32,
            height: self.swapchain.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        self.scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.swapchain.extent,
        };

        // TODO: re-create pipelines if swapchain extent changes
        // TODO: May need to recreate render_passes to account for HDR changes across monitors
        self.color_image = if self.settings.msaa {
            Some(AllocatedImage::create_color(
                &self.device,
                self.swapchain.format,
                self.swapchain.extent,
                self.device.info.msaa_samples,
                self.debug.as_ref(),
            )?)
        } else {
            None
        };
        self.depth_image = AllocatedImage::create_depth(
            &self.instance,
            &self.device,
            self.swapchain.extent,
            self.device.info.msaa_samples,
            self.debug.as_ref(),
        )?;
        self.primary_framebuffers = self.swapchain.create_framebuffers(
            &self.device,
            self.render_pass,
            self.color_image.as_ref().map(|image| image.view),
            Some(self.depth_image.view),
        )?;
        #[cfg(feature = "imgui")]
        if let Some(imgui_renderer) = &mut self.imgui_renderer {
            imgui_renderer.recreate_framebuffers(&self.device, &self.swapchain)?;
        }

        tracing::debug!("re-created swapchain successfully");

        Ok(())
    }

    /// Updates the camera buffer for this frame
    fn update_camera_buffer(&mut self) -> Result<()> {
        self.camera_view.projection_view = self.camera_view.projection * self.camera_view.view;
        let camera_buffer = &self.frames[self.current_frame].camera_buffer;
        match camera_buffer.mapped_memory {
            Some(memory) => unsafe {
                ptr::copy_nonoverlapping(&self.camera_view, memory.cast(), 1);
            },
            None => render_bail!("camera buffer is not mapped correctly"),
        }
        Ok(())
    }

    // TODO: Refine scene lighting functionality
    /// Updates the scene buffer for this frame.
    fn update_scene_buffer(&mut self) -> Result<()> {
        // let (frame_sin, frame_cos) = (self.frame_number as f32 / 120.0).sin_cos();
        // self.scene_attributes.ambient_color = vec4!(frame_sin, 0.0, frame_cos, 1.0);
        self.scene_attributes.ambient_color = vec4!(0.0, 0.0, 0.0, 1.0);

        unsafe {
            let memory = self
                .device
                .map_memory(
                    self.scene_buffer.memory,
                    self.current_frame as u64
                        * padded_uniform_buffer_size::<SceneData>(&self.device),
                    mem::size_of::<SceneData>() as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .context("failed to map scene buffer memory")?;
            ptr::copy_nonoverlapping(&self.scene_attributes, memory.cast(), 1);
            self.device.unmap_memory(self.scene_buffer.memory);
        }

        Ok(())
    }

    /// Draw objects for this frame.
    fn draw_objects(
        &mut self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
    ) -> Result<()> {
        self.update_camera_buffer()?;
        self.update_scene_buffer()?;

        let object_buffer = &self.frames[self.current_frame].object_buffer;
        let Some(object_data) = object_buffer.mapped_memory.map(|memory| {
            unsafe { slice::from_raw_parts_mut::<ObjectData>(memory.cast(), self.objects.len()) }
        }) else {
            render_bail!("object memory is not mapped correctly");
        };

        // Commands
        unsafe {
            self.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
        }
        let command_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &command_begin_info)?;
        }

        let clear_values = [self.clear_color, self.clear_depth];
        let render_pass_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(self.primary_framebuffers[image_index])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain.extent,
            })
            .clear_values(&clear_values);
        unsafe {
            self.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );
        }

        let mut last_mesh = None;
        let mut last_material = None;
        for (i, (_, object)) in self.objects.iter().enumerate() {
            object_data[i].transform = object.transform;

            let mesh = self
                .meshes
                .get(&object.mesh)
                .with_context(|| format!("mesh {} does not exist", object.mesh))?;
            let material = &object.material;

            // Bind material pipeline
            if Some(material) != last_material {
                unsafe {
                    self.device.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        material.pipeline,
                    );

                    self.device.cmd_set_viewport(
                        command_buffer,
                        0,
                        slice::from_ref(&self.viewport),
                    );
                    self.device
                        .cmd_set_scissor(command_buffer, 0, slice::from_ref(&self.scissor));

                    let scene_size = padded_uniform_buffer_size::<SceneData>(&self.device) as u32;
                    let scene_offsets = [self.current_frame as u32 * scene_size];
                    self.device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        material.pipeline_layout,
                        0,
                        slice::from_ref(&self.frames[self.current_frame].descriptor_sets[0]),
                        &scene_offsets,
                    );
                    self.device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        material.pipeline_layout,
                        1,
                        slice::from_ref(&self.frames[self.current_frame].descriptor_sets[1]),
                        &[],
                    );

                    if let Some(descriptor_set) = &material.texture_descriptor_set {
                        self.device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            material.pipeline_layout,
                            2,
                            slice::from_ref(descriptor_set),
                            &[],
                        );
                    }
                }
                last_material = Some(material);
            }

            if Some(&object.mesh) != last_mesh {
                unsafe {
                    self.device.cmd_bind_vertex_buffers(
                        command_buffer,
                        0,
                        slice::from_ref(&mesh.vertex_buffer.handle),
                        slice::from_ref(&0),
                    );
                    self.device.cmd_bind_index_buffer(
                        command_buffer,
                        *mesh.index_buffer,
                        0,
                        vk::IndexType::UINT32, // NOTE: Must match indices datatype
                    );
                }
                last_mesh = Some(&object.mesh);
            }

            unsafe {
                self.device.cmd_draw_indexed(
                    command_buffer,
                    mesh.indices.len() as u32,
                    1,
                    0,
                    0,
                    i as u32,
                );
            }
        }

        unsafe {
            self.device.cmd_end_render_pass(command_buffer);
            self.device.end_command_buffer(command_buffer)?;
        }

        Ok(())
    }
}

impl Drop for Context {
    /// Cleans up all Vulkan resources.
    fn drop(&mut self) {
        tracing::debug!("destroying vulkan context");

        // NOTE: Drop-order is important!
        //
        // SAFETY:
        //
        // 1. Elements are destroyed in the correct order
        // 2. Only elements that have been allocated are destroyed, ensured by `initialize` being
        //    the only way to construct a Context with all values initialized.
        unsafe {
            if let Err(err) = self.device.device_wait_idle() {
                tracing::error!("failed to wait for device idle: {err:?}");
                return;
            }

            #[cfg(feature = "imgui")]
            if let Some(imgui_renderer) = &mut self.imgui_renderer {
                imgui_renderer.destroy(&self.device);
            }

            for &framebuffer in self.primary_framebuffers.iter() {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            self.swapchain.destroy(&self.device, None);

            if let Some(image) = &mut self.color_image {
                image.destroy(&self.device);
            }
            self.depth_image.destroy(&self.device);
            for mesh in self.meshes.values_mut() {
                mesh.destroy(&self.device);
            }
            for texture in self.textures.values_mut() {
                texture.destroy(&self.device);
            }
            for pipeline in self.pipelines.iter().copied() {
                self.device.destroy_pipeline(pipeline, None);
            }
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_pool(self.global_descriptor_pool, None);
            for layout in self.descriptor_set_layouts.iter().copied() {
                self.device.destroy_descriptor_set_layout(layout, None);
            }
            self.device.destroy_render_pass(self.render_pass, None);
            for frame in &mut self.frames {
                frame.destroy(&self.device);
            }
            self.scene_buffer.destroy(&self.device);
            self.device.destroy_sampler(self.sampler, None);
            self.device
                .destroy_command_pool(self.graphics_command_pool, None);
            self.device.destroy_device(None);
            self.surface.destroy(None);
            if let Some(debug) = &mut self.debug {
                debug.destroy();
            }
            self.instance.destroy_instance(None);
        }

        tracing::debug!("destroyed vulkan context");
    }
}

/// Create a Vulkan [Instance].
pub(crate) fn create_instance(
    entry: &ash::Entry,
    application_name: &str,
    application_version: &str,
) -> Result<ash::Instance> {
    tracing::debug!("creating vulkan instance");

    // Application Info
    let application_name = CString::new(application_name)
        .with_context(|| format!("failed to convert `{application_name}` to CString"))?;
    let application_version = Version::parse(application_version)
        .with_context(|| format!("failed to parse version: {application_version}"))?;
    let engine_name = CString::new(env!("CARGO_PKG_NAME"))
        .with_context(|| format!("failed to convert `{}` to CString", env!("CARGO_PKG_NAME")))?;
    let application_info = vk::ApplicationInfo::builder()
        .application_name(&application_name)
        .application_version(vk::make_api_version(
            0,
            application_version.major as u32,
            application_version.minor as u32,
            application_version.patch as u32,
        ))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(
            0,
            env!("CARGO_PKG_VERSION_MAJOR")
                .parse::<u32>()
                .context("failed to parse MAJOR version")?,
            env!("CARGO_PKG_VERSION_MINOR")
                .parse::<u32>()
                .context("failed to parse MINOR version")?,
            env!("CARGO_PKG_VERSION_PATCH")
                .parse::<u32>()
                .context("failed to parse PATCH version")?,
        ))
        .api_version(vk::API_VERSION_1_2);

    // Validate debug layer is supported
    #[cfg(debug_assertions)]
    if !entry
        .enumerate_instance_layer_properties()?
        .iter()
        .any(|layer| unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) } == VALIDATION_LAYER_NAME)
    {
        render_bail!("validation layer is not supported");
    }

    // Validate required instance extensions
    let extension_properties = entry
        .enumerate_instance_extension_properties(None)
        .context("failed to enumerate instance extension properties")?;
    if !platform::REQUIRED_EXTENSIONS.iter().all(|&extension_name| {
        extension_properties
            .iter()
            .any(|property| unsafe { CStr::from_ptr(property.extension_name.as_ptr()) } == extension_name)
    }) {
        render_bail!("required vulkan extensions are not supported");
    }

    // Instance Creation
    let enabled_extension_names = platform::REQUIRED_EXTENSIONS.map(CStr::as_ptr);
    let enabled_layer_names = ENABLED_LAYER_NAMES.map(CStr::as_ptr);
    let create_info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_extension_names(&enabled_extension_names)
        .enabled_layer_names(&enabled_layer_names)
        .flags(platform::INSTANCE_CREATE_FLAGS);

    // Debug Creation
    #[cfg(debug_assertions)]
    let mut debug_create_info = Debug::build_debug_create_info();
    #[cfg(debug_assertions)]
    let create_info = create_info.push_next(&mut debug_create_info);

    // SAFETY: All create_info values are set correctly above with valid lifetimes.
    let instance = unsafe { entry.create_instance(&create_info, None) }
        .context("failed to create vulkan instance")?;

    tracing::debug!("created vulkan instance successfully");

    Ok(instance)
}

/// Create the primary [`vk::RenderPass`] instance.
fn create_render_pass(
    device: &Device,
    color_format: vk::Format,
    depth_format: vk::Format,
    settings: &RenderSettings,
) -> Result<vk::RenderPass> {
    tracing::debug!("creating primary render pass");

    // Attachments
    let color_attachment = vk::AttachmentDescription::builder()
        .format(color_format)
        .samples(device.info.msaa_samples)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(if settings.msaa {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        } else {
            vk::ImageLayout::PRESENT_SRC_KHR
        })
        .build();
    let depth_stencil_attachment = vk::AttachmentDescription::builder()
        .format(depth_format)
        .samples(device.info.msaa_samples)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .build();
    // Only used if settings.msaa is enabled
    let color_resolve_attachment = vk::AttachmentDescription::builder()
        .format(color_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .build();

    // Subpasses
    let color_attachment_ref = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build();
    let depth_stencil_attachment_ref = vk::AttachmentReference::builder()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .build();
    // Only used if msaa is enabled
    let color_resolve_attachment_ref = vk::AttachmentReference::builder()
        .attachment(2)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build();

    let mut subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(slice::from_ref(&color_attachment_ref))
        .depth_stencil_attachment(&depth_stencil_attachment_ref);
    if settings.msaa {
        subpass = subpass.resolve_attachments(slice::from_ref(&color_resolve_attachment_ref));
    }

    // Dependencies
    let dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        )
        .build();

    // Create
    let mut attachments = vec![color_attachment, depth_stencil_attachment];
    if settings.msaa {
        attachments.push(color_resolve_attachment);
    }
    let render_pass_create_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .subpasses(slice::from_ref(&subpass))
        .dependencies(slice::from_ref(&dependency));

    let render_pass = unsafe { device.create_render_pass(&render_pass_create_info, None) }
        .context("failed to create render pass")?;

    tracing::debug!("created primary render pass successfully");

    Ok(render_pass)
}

// /// Create a new material pipeline.
// fn create_material(
//     &mut self,
//     pipeline: vk::Pipeline,
//     layout: vk::PipelineLayout,
//     texture_descriptor_set: Option<vk::DescriptorSet>,
// ) {
// }

mod shader {
    use crate::{
        prelude::{Shader, ShaderStage},
        render_bail, Result,
    };
    use anyhow::Context;
    use ash::vk;
    use std::{borrow::Cow, ffi::CStr};

    /// Build a [`vk::PipelineShaderStageCreateInfo`].
    pub(crate) fn build_stage_info(
        stage: vk::ShaderStageFlags,
        module: vk::ShaderModule,
    ) -> Result<vk::PipelineShaderStageCreateInfo> {
        // SAFETY: This is static string with a nul character.
        const SHADER_ENTRY_NAME: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };
        Ok(vk::PipelineShaderStageCreateInfo::builder()
            .stage(stage)
            .module(module)
            .name(SHADER_ENTRY_NAME)
            .build())
    }

    /// Create a [`vk::ShaderModule`] instance from bytecode.
    pub(crate) fn create_module(
        device: &ash::Device,
        name: impl Into<Cow<'static, str>>,
        stage: ShaderStage,
        bytes: Vec<u8>,
    ) -> Result<vk::ShaderModule> {
        tracing::debug!("creating shader module");

        let shader = Shader::from_bytes(name, stage, bytes);
        // SAFETY: We check for prefix/suffix below and bail if not aligned correctly.
        let (prefix, code, suffix) = unsafe { shader.bytes.align_to::<u32>() };
        if !prefix.is_empty() || !suffix.is_empty() {
            render_bail!("shader bytecode is not properly aligned.");
        }
        let shader_module_info = vk::ShaderModuleCreateInfo::builder().code(code);

        let shader_module = unsafe { device.create_shader_module(&shader_module_info, None) }
            .context("failed to create shader module")?;

        tracing::debug!("created shader module successfully");

        Ok(shader_module)
    }

    impl From<ShaderStage> for vk::ShaderStageFlags {
        fn from(stage: ShaderStage) -> Self {
            match stage {
                ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
                ShaderStage::TessellationControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
                ShaderStage::TessellationEvaluation => {
                    vk::ShaderStageFlags::TESSELLATION_EVALUATION
                }
                ShaderStage::Geometry => vk::ShaderStageFlags::GEOMETRY,
                ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
                ShaderStage::Compute => vk::ShaderStageFlags::COMPUTE,
            }
        }
    }
}

impl From<vk::Result> for Error {
    fn from(err: vk::Result) -> Self {
        Self::Renderer(anyhow::anyhow!("{err:?}"))
    }
}

mod debug {
    use anyhow::{Context, Result};
    use ash::{extensions::ext, vk};
    use std::ffi::{c_void, CStr};

    // SAFETY: This static string has been verified as a valid CStr.
    #[cfg(debug_assertions)]
    pub(crate) const VALIDATION_LAYER_NAME: &CStr =
        unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };

    #[derive(Clone)]
    pub(crate) struct Debug {
        pub(crate) utils: ext::DebugUtils,
        pub(crate) messenger: vk::DebugUtilsMessengerEXT,
    }

    impl Debug {
        /// Create a `Debug` instance with [`ext::DebugUtils`] and [`vk::DebugUtilsMessengerEXT`].
        pub(crate) fn create(entry: &ash::Entry, instance: &ash::Instance) -> Result<Self> {
            tracing::debug!("creating debug utils");

            let utils = ext::DebugUtils::new(entry, instance);
            let debug_create_info = Self::build_debug_create_info();
            // SAFETY: All create_info values are set correctly above with valid lifetimes.
            let messenger = unsafe { utils.create_debug_utils_messenger(&debug_create_info, None) }
                .context("failed to create debug utils messenger")?;

            tracing::debug!("debug utils created successfully");

            Ok(Self { utils, messenger })
        }

        /// Destroy a `Debug` instance.
        pub(crate) unsafe fn destroy(&mut self) {
            self.utils
                .destroy_debug_utils_messenger(self.messenger, None);
        }

        /// Build [`vk::DebugUtilsMessengerCreateInfoEXT`] with desired message severity and
        /// message types..
        pub(crate) fn build_debug_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT {
            vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
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

        /// Sets a debug name on a Vulkan handle.
        #[cfg(debug_assertions)]
        pub(crate) fn set_debug_name(
            &self,
            device: &ash::Device,
            name: &str,
            object_type: vk::ObjectType,
            handle: u64,
        ) {
            let Ok(name) = std::ffi::CString::new(name) else {
                tracing::warn!("failed to create CString from `{name}`");
                return;
            };
            let debug_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(object_type)
                .object_name(&name)
                .object_handle(handle);
            if let Err(err) = unsafe {
                self.utils
                    .set_debug_utils_object_name(device.handle(), &debug_info)
            } {
                tracing::warn!("failed to set debug utils object name for `{name:?}`: {err:?}");
            }
        }
    }

    /// Callback function used by [`ext::DebugUtils`] when validation layers are enabled.
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
        if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
            tracing::error!("{msg_type} {message}");
        } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
            tracing::warn!("{msg_type} {message}");
        } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
            tracing::debug!("{msg_type} {message}");
        } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE {
            tracing::trace!("{msg_type} {message}");
        };
        vk::FALSE
    }
}
