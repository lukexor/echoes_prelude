use super::{RendererBackend, Shader};
use anyhow::{bail, Context as _, Result};
use ash::vk;
use std::{
    ffi::{c_void, CStr, CString},
    fmt,
    mem::size_of,
    ptr, slice,
    time::Instant,
};
use winit::{dpi::PhysicalSize, window::Window};

use crate::{
    math::{Matrix, Radians, UniformBufferObject},
    vector,
};
use command_pool::CommandPool;
use debug::{Debug, VALIDATION_LAYER_NAME};
use descriptor::Descriptor;
use device::Buffer;
use device::{Device, QueueFamily};
use image::Image;
use model::Model;
use pipeline::Pipeline;
use surface::Surface;
use swapchain::Swapchain;
use sync::Syncs;

pub(crate) struct Context {
    start: Instant,
    frame: usize,
    width: u32,
    height: u32,
    resized: bool,
    shaders: Vec<Shader>,

    _entry: ash::Entry,      // Needs to out-live `instance`
    instance: ash::Instance, // Needs to out-live `device`
    // TODO: custom allocator
    // allocator: Option<vk::AllocationCallbacks>
    #[cfg(debug_assertions)]
    debug: Debug,

    surface: Surface,
    device: Device,
    swapchain: Swapchain,
    render_pass: vk::RenderPass,
    command_pools: Vec<CommandPool>,

    textures: Vec<Image>,
    sampler: vk::Sampler,

    uniform_buffers: Vec<Buffer>,
    descriptor: Descriptor,
    pipeline: Pipeline,

    color_image: Image,
    depth_image: Image,
    framebuffers: Vec<vk::Framebuffer>,

    models: Vec<Model>,
    vertex_buffer: Buffer,
    index_buffer: Buffer,

    syncs: Syncs,
}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TODO: debug
        f.debug_struct("Context")
            .field("surface", &self.surface)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl RendererBackend for Context {
    /// Initialize Vulkan `Context`.
    fn initialize(application_name: &str, window: &Window, shaders: &[Shader]) -> Result<Self> {
        log::info!("initializing vulkan renderer backend");

        let entry = ash::Entry::linked();
        let instance = Self::create_instance(application_name, &entry)?;
        #[cfg(debug_assertions)]
        let debug = Debug::create(&entry, &instance)?;

        let surface = Surface::create(&entry, &instance, window)?;
        let device = Device::create(&instance, &surface)?;
        let PhysicalSize { width, height } = window.inner_size();
        let swapchain = Swapchain::create(&instance, &device, &surface, width, height)?;
        let render_pass = Self::create_render_pass(&instance, &device, swapchain.format)?;
        let command_pools = Self::create_command_pools(&device, &swapchain)?;

        let graphics_queue = device.queue_family(QueueFamily::Graphics)?;
        let texture = Image::create_texture(
            "viking_room",
            &instance,
            &device,
            &command_pools[0],
            graphics_queue,
            "assets/viking_room.png",
            #[cfg(debug_assertions)]
            &debug,
        )?;
        let sampler = Image::create_sampler(&device.logical_device, texture.mip_levels)?;

        let uniform_buffers = device.create_uniform_buffers(&swapchain)?;
        let descriptor = Descriptor::create(
            &device.logical_device,
            &swapchain,
            &uniform_buffers,
            texture.view,
            sampler,
        )?;
        let pipeline = Pipeline::create(
            &device,
            swapchain.extent,
            render_pass,
            descriptor.set_layout,
            shaders,
        )?;

        let color_image = Image::create_color(
            &device,
            &swapchain,
            #[cfg(debug_assertions)]
            &debug,
        )?;
        let depth_image = Image::create_depth(
            &instance,
            &device,
            &swapchain,
            #[cfg(debug_assertions)]
            &debug,
        )?;
        let framebuffers = Self::create_framebuffers(
            &device,
            &swapchain,
            render_pass,
            color_image.view,
            depth_image.view,
        )?;

        let model = Model::load("viking_room", "assets/viking_room.obj")?;
        let vertex_buffer =
            device.create_vertex_buffer(&command_pools[0], graphics_queue, &model.vertices)?;
        let index_buffer =
            device.create_index_buffer(&command_pools[0], graphics_queue, &model.indices)?;

        let syncs = Syncs::create(&device.logical_device, &swapchain)?;

        log::info!("initialized vulkan renderer backend successfully");

        Ok(Context {
            start: Instant::now(),
            frame: 0,
            width,
            height,
            resized: false,
            shaders: shaders.to_vec(),

            _entry: entry,
            instance,
            #[cfg(debug_assertions)]
            debug,

            surface,
            device,
            swapchain,
            render_pass,
            command_pools,

            textures: vec![texture],
            sampler,

            uniform_buffers,
            descriptor,
            pipeline,

            color_image,
            depth_image,
            framebuffers,

            models: vec![model],
            vertex_buffer,
            index_buffer,

            syncs,
        })
    }

    /// Shutdown the Vulkan `Context`, freeing any resources.
    fn shutdown(&mut self) -> Result<()> {
        log::info!("destroying vulkan context");

        // NOTE: Drop-order is important!
        //
        // SAFETY:
        //
        // 1. Elements are destroyed in the correct order
        // 2. Only elements that have been allocated are destroyed, ensured by `initalize` being
        //    the only way to construct a Context with all values initialized.
        unsafe {
            self.destroy_swapchain()?;

            let device = &self.device.logical_device;
            self.syncs.destroy(device);
            self.index_buffer.destroy(device);
            self.vertex_buffer.destroy(device);
            device.destroy_sampler(self.sampler, None);
            self.textures
                .iter()
                .for_each(|texture| texture.destroy(device));
            self.command_pools
                .iter()
                .for_each(|command_pool| command_pool.destroy(device));
            device.destroy_device(None);
            self.surface.destroy();
            self.debug.destroy();
            self.instance.destroy_instance(None);
        }

        log::info!("destroyed vulkan context");

        Ok(())
    }

    /// Handle window resized event.
    fn on_resized(&mut self, width: u32, height: u32) {
        if self.width != width || self.height != height {
            log::debug!("received resized event {width}x{height}. pending swapchain recreation.");
            self.width = width;
            self.height = height;
            self.resized = true;
        }
    }

    /// Begin drawing a frame to the [Window] surface.
    fn begin_frame(&mut self, _delta_time: f32) -> Result<()> {
        let in_flight_fence = self.syncs.in_flight_fences[self.frame];

        self.device
            .wait_for_fences(slice::from_ref(&in_flight_fence), true, u64::MAX)?;

        // SAFETY: TODO
        let result = self.swapchain.acquire_next_image(
            u64::MAX,
            self.syncs.images_available[self.frame],
            vk::Fence::null(),
        );
        let image_index = match result {
            Ok((index, _)) => index as usize,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                log::warn!("swapchain image out of date");
                return self.recreate_swapchain(self.width, self.height);
            }
            Err(err) => bail!(err),
        };

        let image_in_flight = self.syncs.images_in_flight[image_index];
        if image_in_flight != vk::Fence::null() {
            self.device
                .wait_for_fences(slice::from_ref(&image_in_flight), true, u64::MAX)?;
        }

        self.syncs.images_in_flight[image_index] = in_flight_fence;
        unsafe {
            self.update_command_buffer(image_index)?;
            self.update_uniform_buffer(image_index)?;
        }

        let wait_semaphores = [self.syncs.images_available[self.frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [self.command_pools[image_index].buffers[0]];
        let signal_semaphores = [self.syncs.renders_finished[self.frame]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);

        let graphics_queue = self.device.queue_family(QueueFamily::Graphics)?;
        self.device
            .reset_fences(slice::from_ref(&in_flight_fence))?;
        self.device.queue_submit(
            graphics_queue,
            slice::from_ref(&submit_info),
            in_flight_fence,
        )?;

        let swapchains = [self.swapchain.handle];
        let image_indices = [image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let present_queue = self.device.queue_family(QueueFamily::Present)?;
        let result = self.swapchain.queue_present(present_queue, &present_info);
        let requires_recreate = matches!(result, Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR));
        if self.resized || requires_recreate {
            log::warn!("swapchain is suboptimal for surface");
            self.recreate_swapchain(self.width, self.height)?;
            self.resized = false;
        } else if let Err(err) = result {
            bail!(err);
        }

        self.frame = (self.frame + 1) % self.swapchain.max_frames_in_flight;

        Ok(())
    }

    /// End drawing a frame.
    fn end_frame(&mut self, _delta_time: f32) -> Result<()> {
        Ok(())
    }
}

impl Drop for Context {
    /// Cleans up all Vulkan resources.
    fn drop(&mut self) {
        if let Err(err) = self.shutdown() {
            log::error!("failed to shutdown vulkan context: {err}");
        }
    }
}

impl Context {
    // TODO: refactor
    unsafe fn update_command_buffer(&mut self, image_index: usize) -> Result<()> {
        // Reset
        let pool = self.command_pools[image_index].handle;
        self.device
            .logical_device
            .reset_command_pool(pool, vk::CommandPoolResetFlags::empty())?;

        let buffer = self.command_pools[image_index].buffers[0];

        // Commands
        let command_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        self.device
            .logical_device
            .begin_command_buffer(buffer, &command_begin_info)?;

        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };
        let depth_clear_value = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };
        let clear_values = [color_clear_value, depth_clear_value];
        let render_pass_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffers[image_index])
            .render_area(
                vk::Rect2D::builder()
                    .offset(vk::Offset2D::default())
                    .extent(self.swapchain.extent)
                    .build(),
            )
            .clear_values(&clear_values);

        self.device.logical_device.cmd_begin_render_pass(
            buffer,
            &render_pass_info,
            vk::SubpassContents::SECONDARY_COMMAND_BUFFERS,
        );

        let secondary_command_buffer = self.update_secondary_command_buffer(image_index, 0)?;
        self.device
            .logical_device
            .cmd_execute_commands(buffer, slice::from_ref(&secondary_command_buffer));

        self.device.logical_device.cmd_end_render_pass(buffer);

        self.device.logical_device.end_command_buffer(buffer)?;

        Ok(())
    }

    // TODO: refactor
    unsafe fn update_secondary_command_buffer(
        &mut self,
        image_index: usize,
        model_index: usize,
    ) -> Result<vk::CommandBuffer> {
        let device = &self.device.logical_device;

        let pool = self.command_pools[image_index].handle;
        let buffers = &mut self.command_pools[image_index].secondary_buffers;
        buffers.clear();
        while model_index >= buffers.len() {
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::SECONDARY)
                .command_buffer_count(1);
            let buffer = device.allocate_command_buffers(&allocate_info)?[0];
            buffers.push(buffer);
        }

        let buffer = buffers[model_index];

        let inheritance_info = vk::CommandBufferInheritanceInfo::builder()
            .render_pass(self.render_pass)
            .subpass(0)
            .framebuffer(self.framebuffers[image_index]);
        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
            .inheritance_info(&inheritance_info);

        device.begin_command_buffer(buffer, &info)?;

        // Model
        let time = self.start.elapsed().as_secs_f32();
        let model = Matrix::rotation_z(Radians(time * 20.0f32.to_radians()));
        let (_, model_bytes, _) = model.as_slice().align_to::<u8>();

        let opacity = (model_index + 1) as f32 * 1.0;
        let opacity_bytes = &opacity.to_ne_bytes()[..];

        device.cmd_bind_pipeline(
            buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline.handle,
        );
        device.cmd_bind_vertex_buffers(
            buffer,
            0,
            slice::from_ref(&self.vertex_buffer.handle),
            slice::from_ref(&0),
        );
        // Must match data.indices datatype
        device.cmd_bind_index_buffer(buffer, self.index_buffer.handle, 0, vk::IndexType::UINT32);
        device.cmd_bind_descriptor_sets(
            buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline.layout,
            0,
            slice::from_ref(&self.descriptor.sets[image_index]),
            &[],
        );
        device.cmd_push_constants(
            buffer,
            self.pipeline.layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            model_bytes,
        );
        device.cmd_push_constants(
            buffer,
            self.pipeline.layout,
            vk::ShaderStageFlags::FRAGMENT,
            64,
            opacity_bytes,
        );
        device.cmd_draw_indexed(buffer, self.models[0].indices.len() as u32, 1, 0, 0, 0);

        device.end_command_buffer(buffer)?;

        Ok(buffer)
    }

    // TODO: refactor
    unsafe fn update_uniform_buffer(&self, image_index: usize) -> Result<()> {
        let device = &self.device.logical_device;

        // View / Projection
        let view = Matrix::look_at(
            vector!(2.0, 2.0, 2.0),
            vector!(0.0, 0.0, 0.0),
            vector!(0.0, 0.0, 1.0),
        );
        let vk::Extent2D { width, height } = self.swapchain.extent;
        let mut projection = Matrix::perspective(
            Radians(45.0f32.to_radians()),
            width as f32 / height as f32,
            0.1,
            10.0,
        );
        projection[5] *= -1.0; // Y-axis is inverted in Vulkan

        let ubo = UniformBufferObject { view, projection };

        // Copy
        let uniform_buffer_memory = self.uniform_buffers[image_index].memory;
        let memory = device.map_memory(
            uniform_buffer_memory,
            0,
            size_of::<UniformBufferObject>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        ptr::copy_nonoverlapping(&ubo, memory.cast(), 1);
        device.unmap_memory(uniform_buffer_memory);

        Ok(())
    }

    /// Create a Vulkan [Instance].
    fn create_instance(application_name: &str, entry: &ash::Entry) -> Result<ash::Instance> {
        log::debug!("creating vulkan instance");

        // Application Info
        let application_name = CString::new(application_name)?;
        let engine_name = CString::new("Echoes Engine")?;
        let application_info = vk::ApplicationInfo::builder()
            .application_name(&application_name)
            .application_version(vk::API_VERSION_1_0)
            .engine_name(&engine_name)
            .engine_version(vk::API_VERSION_1_0)
            .api_version(vk::API_VERSION_1_3);

        // Validation Layer check
        #[cfg(debug_assertions)]
        if !entry
            .enumerate_instance_layer_properties()?
            .iter()
            .any(|layer| {
                // SAFETY: This layer name is provided by Vulkan and is a valid Cstr.
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
        let enabled_extension_names = platform::enabled_extension_names();
        let instance_create_flags = platform::instance_create_flags();
        let mut create_info = vk::InstanceCreateInfo::builder()
            .application_info(&application_info)
            .enabled_layer_names(&enabled_layer_names)
            .enabled_extension_names(&enabled_extension_names)
            .flags(instance_create_flags);

        // Debug Creation
        #[cfg(debug_assertions)]
        let debug_create_info = Debug::build_debug_create_info();
        #[cfg(debug_assertions)]
        {
            let p_next: *const vk::DebugUtilsMessengerCreateInfoEXT = &debug_create_info;
            create_info.p_next = p_next as *const c_void;
        }

        // SAFETY: All create_info values are set correctly above with valid lifetimes.
        let instance = unsafe { entry.create_instance(&create_info, None) }
            .context("failed to create vulkan instance")?;

        log::debug!("created vulkan instance successfully");

        Ok(instance)
    }

    /// Create a [`vk::RenderPass`] instance.
    fn create_render_pass(
        instance: &ash::Instance,
        device: &Device,
        format: vk::Format,
    ) -> Result<vk::RenderPass> {
        log::debug!("creating render pass");

        // Attachments
        let color_attachment = vk::AttachmentDescription::builder()
            .format(format)
            .samples(device.info.msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let depth_stencil_attachment = vk::AttachmentDescription::builder()
            .format(device.get_depth_format(instance, vk::ImageTiling::OPTIMAL)?)
            .samples(device.info.msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_resolve_attachment = vk::AttachmentDescription::builder()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        // Subpasses
        let color_attachment_ref = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let depth_stencil_attachment_ref = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let color_resolve_attachment_ref = vk::AttachmentReference::builder()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(slice::from_ref(&color_attachment_ref))
            .depth_stencil_attachment(&depth_stencil_attachment_ref)
            .resolve_attachments(slice::from_ref(&color_resolve_attachment_ref));

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
            );

        // Create
        let attachments = [
            color_attachment.build(),
            depth_stencil_attachment.build(),
            color_resolve_attachment.build(),
        ];
        let render_pass_create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(slice::from_ref(&subpass))
            .dependencies(slice::from_ref(&dependency));

        let render_pass = unsafe {
            device
                .logical_device
                .create_render_pass(&render_pass_create_info, None)
        }
        .context("failed to create render pass")?;

        log::debug!("created render pass successfully");

        Ok(render_pass)
    }

    /// Create a list of [vk::Framebuffer`]s.
    fn create_framebuffers(
        device: &Device,
        swapchain: &Swapchain,
        render_pass: vk::RenderPass,
        color_attachment: vk::ImageView,
        depth_attachment: vk::ImageView,
    ) -> Result<Vec<vk::Framebuffer>> {
        log::debug!("creating framebuffers");

        let framebuffers = swapchain
            .image_views
            .iter()
            .map(|&image_view| {
                let attachments = [color_attachment, depth_attachment, image_view];
                let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(swapchain.extent.width)
                    .height(swapchain.extent.height)
                    .layers(1);
                // SAFETY: TODO
                unsafe {
                    device
                        .logical_device
                        .create_framebuffer(&framebuffer_create_info, None)
                }
                .context("failed to create framebuffer")
            })
            .collect::<Result<Vec<_>>>()
            .context("failed to create image buffers")?;

        log::debug!("created framebuffers successfully");

        Ok(framebuffers)
    }

    /// Create a list of [`CommandPool`]s.
    fn create_command_pools(device: &Device, swapchain: &Swapchain) -> Result<Vec<CommandPool>> {
        log::debug!("creating command pools");

        // One global pool + one pool for each swapchain image
        let mut command_pools = Vec::with_capacity(swapchain.images.len() + 1);
        let buffer_count = 1;

        // Global Pool
        command_pools.push(CommandPool::create(
            device,
            QueueFamily::Graphics,
            buffer_count,
        )?);

        // Per-framebuffer Pool
        for _ in 0..swapchain.images.len() {
            command_pools.push(CommandPool::create(
                device,
                QueueFamily::Graphics,
                buffer_count,
            )?);
        }

        log::debug!("created command pools successfully");

        Ok(command_pools)
    }

    /// Destroys the current swapchain and re-creates it with a new width/height.
    fn recreate_swapchain(&mut self, width: u32, height: u32) -> Result<()> {
        log::debug!("re-creating swapchain for {width}x{height}");

        self.destroy_swapchain()?;

        self.device.update(&self.surface)?;
        self.swapchain =
            Swapchain::create(&self.instance, &self.device, &self.surface, width, height)?;
        self.render_pass =
            Self::create_render_pass(&self.instance, &self.device, self.swapchain.format)?;
        self.uniform_buffers = self.device.create_uniform_buffers(&self.swapchain)?;
        self.descriptor = Descriptor::create(
            &self.device.logical_device,
            &self.swapchain,
            &self.uniform_buffers,
            self.textures[0].view,
            self.sampler,
        )?;
        self.pipeline = Pipeline::create(
            &self.device,
            self.swapchain.extent,
            self.render_pass,
            self.descriptor.set_layout,
            &self.shaders,
        )?;
        self.color_image = Image::create_color(
            &self.device,
            &self.swapchain,
            #[cfg(debug_assertions)]
            &self.debug,
        )?;
        self.depth_image = Image::create_depth(
            &self.instance,
            &self.device,
            &self.swapchain,
            #[cfg(debug_assertions)]
            &self.debug,
        )?;
        self.framebuffers = Self::create_framebuffers(
            &self.device,
            &self.swapchain,
            self.render_pass,
            self.color_image.view,
            self.depth_image.view,
        )?;
        self.syncs
            .images_in_flight
            .resize(self.swapchain.images.len(), vk::Fence::null());

        log::debug!("re-created swapchain successfully");

        Ok(())
    }

    /// Destroys the current [`vk::SwapchainKHR`] and associated [`vk::ImageView`]s. Used to clean
    /// up Vulkan when dropped and to re-create swapchain on window resize.
    fn destroy_swapchain(&mut self) -> Result<()> {
        // SAFETY: TODO
        log::debug!("destroying swapchain");

        unsafe {
            // TODO: Instead of pausing, and destroying and re-creating entirely, try to use the
            // previous swapchain to recreate and then destroy it when it's finished being used.
            self.device.wait_idle()?;

            let device = &self.device.logical_device;

            self.descriptor.destroy(device);
            self.uniform_buffers
                .iter()
                .for_each(|uniform_buffer| uniform_buffer.destroy(device));
            [&self.color_image, &self.depth_image]
                .iter()
                .for_each(|image| image.destroy(device));
            self.framebuffers
                .iter()
                .for_each(|&framebuffer| device.destroy_framebuffer(framebuffer, None));
            self.pipeline.destroy(device);
            device.destroy_render_pass(self.render_pass, None);
            self.swapchain.destroy(device);
        }

        log::debug!("destroyed swapchain succesfully");

        Ok(())
    }
}

mod surface {
    use super::platform;
    use anyhow::Result;
    use ash::{extensions::khr, vk};
    use std::fmt;
    use winit::{dpi::PhysicalSize, window::Window};

    #[derive(Clone)]
    #[must_use]
    pub(crate) struct Surface {
        pub(crate) handle: vk::SurfaceKHR,
        pub(crate) loader: khr::Surface,
        pub(crate) width: u32,
        pub(crate) height: u32,
    }

    impl fmt::Debug for Surface {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("Surface")
                .field("width", &self.width)
                .field("height", &self.height)
                .finish_non_exhaustive()
        }
    }

    impl Surface {
        /// Create a platform-agnostic `Surface` instance.
        pub(crate) fn create(
            entry: &ash::Entry,
            instance: &ash::Instance,
            window: &Window,
        ) -> Result<Self> {
            log::debug!("creating surface");

            let handle = platform::create_surface(entry, instance, window)?;
            let loader = khr::Surface::new(entry, instance);
            let PhysicalSize { width, height } = window.inner_size();

            log::debug!("vulkan surface created successfully");

            Ok(Self {
                handle,
                loader,
                width,
                height,
            })
        }

        /// Destroy a `Surface` instance.
        // SAFETY: TODO
        pub(crate) unsafe fn destroy(&self) {
            self.loader.destroy_surface(self.handle, None);
        }
    }
}

mod device {
    use super::{
        command_pool::CommandPool, platform, swapchain::Swapchain, Surface, VALIDATION_LAYER_NAME,
    };
    use crate::math::{UniformBufferObject, Vertex};
    use anyhow::{bail, Context as _, Result};
    use ash::vk;
    use std::{
        collections::{hash_map::Entry, HashMap, HashSet},
        ffi::CStr,
        fmt,
        mem::size_of,
        ptr, slice,
    };

    #[derive(Clone)]
    #[must_use]
    pub(crate) struct Device {
        pub(crate) physical_device: vk::PhysicalDevice,
        pub(crate) logical_device: ash::Device,
        pub(crate) queue_family_indices: QueueFamilyIndices,
        pub(crate) info: DeviceInfo,
    }

    impl fmt::Debug for Device {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("Device")
                .field("queue_family_indices", &self.queue_family_indices)
                .field("info", &self.info)
                .finish_non_exhaustive()
        }
    }

    impl Device {
        /// Select a preferred [`vk::PhysicalDevice`] and create a logical [ash::Device].
        pub(crate) fn create(instance: &ash::Instance, surface: &Surface) -> Result<Self> {
            log::debug!("creating device");

            let (physical_device, queue_family_indices, info) =
                Self::select_physical_device(instance, surface)?;
            let logical_device =
                Self::create_logical_device(instance, physical_device, &queue_family_indices)?;

            log::debug!("created device successfully");

            Ok(Self {
                physical_device,
                logical_device,
                queue_family_indices,
                info,
            })
        }

        /// Updates device capabilities if the window surface changes.
        pub(crate) fn update(&mut self, surface: &Surface) -> Result<()> {
            self.info.swapchain_support = SwapchainSupport::query(self.physical_device, surface)
                .map_err(|err| log::error!("{err}"))
                .ok();
            Ok(())
        }

        /// Waits for the [ash::Device] to be idle.
        pub(crate) fn wait_idle(&self) -> Result<()> {
            log::trace!("waiting for device idle");
            // SAFETY: TODO
            unsafe { self.logical_device.device_wait_idle() }
                .context("failed to wait for device idle")
        }

        /// Waits for the given [vk::Fence]s to be signaled.
        pub(crate) fn wait_for_fences(
            &self,
            fences: &[vk::Fence],
            wait_all: bool,
            timeout: u64,
        ) -> Result<()> {
            log::trace!("waiting for device fences");
            // SAFETY: TODO
            unsafe {
                self.logical_device
                    .wait_for_fences(fences, wait_all, timeout)
            }
            .context("failed to wait for device fence")
        }

        /// Resets the given [vk::Fence]s.
        pub(crate) fn reset_fences(&self, fences: &[vk::Fence]) -> Result<()> {
            log::trace!("resetting device fences");
            // SAFETY: TODO
            unsafe { self.logical_device.reset_fences(fences) }
                .context("failed to reset device fences")
        }

        /// Submit commands to a [vk::Queue].
        pub(crate) fn queue_submit(
            &self,
            queue: vk::Queue,
            submits: &[vk::SubmitInfo],
            fence: vk::Fence,
        ) -> Result<()> {
            log::trace!("submitting to device queue {queue:?}");
            // SAFETY: TODO
            unsafe { self.logical_device.queue_submit(queue, submits, fence) }
                .context("failed to submit to device queue {queue:?}")
        }

        /// Get a [vk::Queue] handle to a given [`QueueFamily`].
        pub(crate) fn queue_family(&self, family: QueueFamily) -> Result<vk::Queue> {
            Ok(unsafe {
                self.logical_device.get_device_queue(
                    *self
                        .queue_family_indices
                        .get(&family)
                        .context("failed to get queue family: {family:?}")?,
                    0,
                )
            })
        }

        /// Get a preferred depth [vk::Format] with the given [`vk::ImageTiling`].
        pub(crate) fn get_depth_format(
            &self,
            instance: &ash::Instance,
            tiling: vk::ImageTiling,
        ) -> Result<vk::Format> {
            let candidates = [
                vk::Format::D32_SFLOAT,
                vk::Format::D32_SFLOAT_S8_UINT,
                vk::Format::D24_UNORM_S8_UINT,
            ];
            let features = vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT;
            candidates
                .iter()
                .find(|&&format| {
                    let properties = self.format_properties(instance, format);
                    match tiling {
                        vk::ImageTiling::LINEAR => {
                            properties.linear_tiling_features.contains(features)
                        }
                        vk::ImageTiling::OPTIMAL => {
                            properties.optimal_tiling_features.contains(features)
                        }
                        _ => false,
                    }
                })
                .copied()
                .context("failed to find supported depth format!")
        }

        /// Get the [`vk::MemoryType`] index with the desired [`vk::MemoryPropertyFlags`] and
        /// [`vk::MemoryRequirements`].
        pub(crate) fn memory_type_index(
            &self,
            properties: vk::MemoryPropertyFlags,
            requirements: vk::MemoryRequirements,
        ) -> Result<u32> {
            let memory = self.info.memory_properties;
            (0..memory.memory_type_count)
                .find(|&index| {
                    let suitable = (requirements.memory_type_bits & (1 << index)) != 0;
                    let memory_type = memory.memory_types[index as usize];
                    suitable && memory_type.property_flags.contains(properties)
                })
                .context("failed to find suitable memory type.")
        }

        /// Get supported [`vk::FormatProperties`] for the [`vk::PhysicalDevice`].
        pub(crate) fn format_properties(
            &self,
            instance: &ash::Instance,
            format: vk::Format,
        ) -> vk::FormatProperties {
            // SAFETY: TODO
            unsafe { instance.get_physical_device_format_properties(self.physical_device, format) }
        }

        /// Creates a [vk::Buffer] with associated [vk::DeviceMemory] for the given parameters.
        pub(crate) fn create_buffer(
            &self,
            size: vk::DeviceSize,
            usage: vk::BufferUsageFlags,
            properties: vk::MemoryPropertyFlags,
        ) -> Result<Buffer> {
            log::debug!("creating buffer");

            // Buffer
            let buffer_create_info = vk::BufferCreateInfo::builder()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            // SAFETY: TODO
            let buffer = unsafe {
                self.logical_device
                    .create_buffer(&buffer_create_info, None)?
            };

            // Memory
            // SAFETY: TODO
            let requirements =
                unsafe { self.logical_device.get_buffer_memory_requirements(buffer) };
            let memory_allocate_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(requirements.size)
                .memory_type_index(self.memory_type_index(properties, requirements)?);

            // Allocate
            // SAFETY: TODO
            let memory = unsafe {
                self.logical_device
                    .allocate_memory(&memory_allocate_info, None)
            }
            .context("failed to allocate buffer memory")?;
            unsafe { self.logical_device.bind_buffer_memory(buffer, memory, 0) }
                .context("failed to bind to buffer memory")?;

            log::debug!("created buffer successfully");

            Ok(Buffer {
                handle: buffer,
                memory,
            })
        }

        /// Create a Uniform `Buffer` instance.
        pub(crate) fn create_uniform_buffers(&self, swapchain: &Swapchain) -> Result<Vec<Buffer>> {
            log::debug!("creating uniform buffers");

            let size = size_of::<UniformBufferObject>() as u64;
            let buffers = (0..swapchain.images.len())
                .map(|_| {
                    self.create_buffer(
                        size,
                        vk::BufferUsageFlags::UNIFORM_BUFFER,
                        vk::MemoryPropertyFlags::HOST_COHERENT
                            | vk::MemoryPropertyFlags::HOST_VISIBLE,
                    )
                })
                .collect::<Result<Vec<_>>>()
                .context("failed to create uniform buffers")?;

            log::debug!("created uniform buffers successfully");

            Ok(buffers)
        }

        /// Create a [Vertex] `Buffer` instance.
        pub(crate) fn create_vertex_buffer(
            &self,
            command_pool: &CommandPool,
            queue: vk::Queue,
            vertices: &[Vertex],
        ) -> Result<Buffer> {
            log::debug!("creating vertex buffer");

            // NOTE: size_of::<T> must match typeof `vertices`
            let size = (size_of::<Vertex>() * vertices.len()) as u64;

            // Staging Buffer
            let staging_buffer = self.create_buffer(
                size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
            )?;

            // Copy Staging
            // SAFETY: TODO
            unsafe {
                let memory = self
                    .logical_device
                    .map_memory(staging_buffer.memory, 0, size, vk::MemoryMapFlags::empty())
                    .context("failed to map vertex buffer memory")?;
                ptr::copy_nonoverlapping(vertices.as_ptr(), memory.cast(), vertices.len());
                self.logical_device.unmap_memory(staging_buffer.memory);
            }

            // Buffer
            let buffer = self.create_buffer(
                size,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;

            // Copy
            self.copy_buffer(
                command_pool,
                queue,
                staging_buffer.handle,
                buffer.handle,
                size,
            )?;

            // Cleanup
            // SAFETY: TODO
            unsafe {
                self.logical_device
                    .destroy_buffer(staging_buffer.handle, None);
                self.logical_device.free_memory(staging_buffer.memory, None);
            }

            log::debug!("created vertex buffer successfully");

            Ok(buffer)
        }

        /// Create an Index `Buffer` instance.
        pub(crate) fn create_index_buffer(
            &self,
            command_pool: &CommandPool,
            queue: vk::Queue,
            indices: &[u32],
        ) -> Result<Buffer> {
            // TODO: This is mostly duplicate code from create_vertex_buffer, extract out
            log::debug!("creating index buffer");

            // NOTE: size_of::<T> must match typeof `indices`
            let size = (size_of::<u32>() * indices.len()) as u64;

            // Staging Buffer
            let staging_buffer = self.create_buffer(
                size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
            )?;

            // Copy Staging
            // SAFETY: TODO
            unsafe {
                let memory = self
                    .logical_device
                    .map_memory(staging_buffer.memory, 0, size, vk::MemoryMapFlags::empty())
                    .context("failed to map index buffer memory")?;
                ptr::copy_nonoverlapping(indices.as_ptr(), memory.cast(), indices.len());
                self.logical_device.unmap_memory(staging_buffer.memory);
            }

            // Buffer
            let buffer = self.create_buffer(
                size,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;

            // Copy
            self.copy_buffer(
                command_pool,
                queue,
                staging_buffer.handle,
                buffer.handle,
                size,
            )?;

            // Cleanup
            // SAFETY: TODO
            unsafe {
                self.logical_device
                    .destroy_buffer(staging_buffer.handle, None);
                self.logical_device.free_memory(staging_buffer.memory, None);
            }

            log::debug!("created index buffer successfully");

            Ok(buffer)
        }

        /// Copy one [Buffer] another.
        pub(crate) fn copy_buffer(
            &self,
            command_pool: &CommandPool,
            queue: vk::Queue,
            source: vk::Buffer,
            destination: vk::Buffer,
            size: vk::DeviceSize,
        ) -> Result<()> {
            let command_buffer = command_pool.begin_one_time_command(&self.logical_device)?;

            let region = vk::BufferCopy::builder().size(size);
            // SAFETY: TODO
            unsafe {
                self.logical_device.cmd_copy_buffer(
                    command_buffer,
                    source,
                    destination,
                    slice::from_ref(&region),
                );
            };

            command_pool.end_one_time_command(&self.logical_device, command_buffer, queue)?;
            Ok(())
        }

        /// Copy a [vk::Buffer] to a [vk::Image].
        pub(crate) fn copy_buffer_to_image(
            &self,
            command_pool: &CommandPool,
            queue: vk::Queue,
            buffer: vk::Buffer,
            image: vk::Image,
            width: u32,
            height: u32,
        ) -> Result<()> {
            let command_buffer = command_pool.begin_one_time_command(&self.logical_device)?;

            let region = vk::BufferImageCopy::builder()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                )
                .image_offset(vk::Offset3D::default())
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                });

            // SAFETY: TODO
            unsafe {
                self.logical_device.cmd_copy_buffer_to_image(
                    command_buffer,
                    buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    slice::from_ref(&region),
                );
            }

            command_pool.end_one_time_command(&self.logical_device, command_buffer, queue)?;

            Ok(())
        }

        /// Select a preferred [`vk::PhysicalDevice`].
        fn select_physical_device(
            instance: &ash::Instance,
            surface: &Surface,
        ) -> Result<(vk::PhysicalDevice, QueueFamilyIndices, DeviceInfo)> {
            log::debug!("selecting physical device");

            // Select physical device
            // SAFETY: TODO
            let physical_devices = unsafe { instance.enumerate_physical_devices() }
                .context("failed to enumerate physical devices")?;
            if physical_devices.is_empty() {
                bail!("failed to find any devices with vulkan support");
            }

            let mut devices = physical_devices
            .into_iter()
            .filter_map(|physical_device| {
                let (queue_family_indices, info) = DeviceInfo::query(instance, physical_device, surface);
                log::debug!("Device Information:\nName: {}\nRating: {}\nQueue Family Indices: {:?}\nSwapchain Support: {}\nAnsiotropy Support: {}\nExtension Support: {}",
                    &info.name,
                    info.rating,
                    &queue_family_indices,
                    info.swapchain_support.is_some(),
                    info.sampler_anisotropy_support,
                    info.extension_support,
                );
                (info.rating > 0).then_some((physical_device, queue_family_indices, info))
            })
            .collect::<Vec<(vk::PhysicalDevice, QueueFamilyIndices, DeviceInfo)>>();
            devices.sort_by_key(|(_, _, info)| info.rating);
            let selected_device = devices
                .pop()
                .context("failed to find a suitable physical device")?;

            log::debug!("selected physical device successfully");

            Ok(selected_device)
        }

        /// Create a logical [vk::Device] instance.
        fn create_logical_device(
            instance: &ash::Instance,
            physical_device: vk::PhysicalDevice,
            queue_family_indices: &QueueFamilyIndices,
        ) -> Result<ash::Device> {
            log::debug!("creating logical device");

            let queue_priorities = [1.0];
            let unique_families: HashSet<u32> =
                HashSet::from_iter(queue_family_indices.values().copied());
            let queue_create_infos = unique_families
                .iter()
                .map(|&index| {
                    vk::DeviceQueueCreateInfo::builder()
                        .queue_family_index(index)
                        .queue_priorities(&queue_priorities)
                        .build()
                })
                .collect::<Vec<vk::DeviceQueueCreateInfo>>();
            let enabled_features = vk::PhysicalDeviceFeatures::builder()
                .sampler_anisotropy(true)
                .sample_rate_shading(true);
            let enabled_layer_names = [
                #[cfg(debug_assertions)]
                VALIDATION_LAYER_NAME.as_ptr(),
            ];
            let enabled_extension_names = platform::required_device_extensions();
            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_create_infos)
                .enabled_features(&enabled_features)
                .enabled_layer_names(&enabled_layer_names)
                .enabled_extension_names(&enabled_extension_names);
            // SAFETY: All create_info values are set correctly above with valid lifetimes.
            let device =
                unsafe { instance.create_device(physical_device, &device_create_info, None) }
                    .context("failed to create logical device")?;

            log::debug!("creating logical device");

            Ok(device)
        }
    }

    #[derive(Clone)]
    #[must_use]
    pub(crate) struct Buffer {
        pub(crate) handle: vk::Buffer,
        pub(crate) memory: vk::DeviceMemory,
    }

    impl Buffer {
        /// Destroy a `Buffer` instance.
        pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
            device.free_memory(self.memory, None);
            device.destroy_buffer(self.handle, None);
        }
    }

    #[derive(Debug, Clone)]
    #[must_use]
    pub(crate) struct DeviceInfo {
        pub(crate) name: String,
        pub(crate) properties: vk::PhysicalDeviceProperties,
        pub(crate) memory_properties: vk::PhysicalDeviceMemoryProperties,
        pub(crate) features: vk::PhysicalDeviceFeatures,
        pub(crate) rating: u32,
        pub(crate) msaa_samples: vk::SampleCountFlags,
        pub(crate) swapchain_support: Option<SwapchainSupport>,
        pub(crate) extension_support: bool,
        pub(crate) sampler_anisotropy_support: bool,
    }

    impl DeviceInfo {
        /// Query [`vk::PhysicalDevice`] properties and generate a rating score.
        fn query(
            instance: &ash::Instance,
            physical_device: vk::PhysicalDevice,
            surface: &Surface,
        ) -> (QueueFamilyIndices, Self) {
            let mut rating = 10;

            // SAFETY: TODO
            let properties = unsafe { instance.get_physical_device_properties(physical_device) };
            let features = unsafe { instance.get_physical_device_features(physical_device) };
            let memory_properties =
                unsafe { instance.get_physical_device_memory_properties(physical_device) };

            let msaa_samples = Self::get_max_sample_count(properties, &mut rating);
            let queue_family_indices = QueueFamily::query(instance, physical_device, surface);
            let swapchain_support = SwapchainSupport::query(physical_device, surface)
                .map_err(|err| log::error!("{err}"))
                .ok();

            // Bonus if graphics + present are the same queue
            if queue_family_indices.get(&QueueFamily::Graphics)
                == queue_family_indices.get(&QueueFamily::Present)
            {
                rating += 100;
            }
            // Bonus if has a dedicated transfer queue
            if queue_family_indices.get(&QueueFamily::Graphics)
                != queue_family_indices.get(&QueueFamily::Transfer)
            {
                rating += 500;
            }
            // Large bonus for being a discrete GPU
            if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                rating += 1000;
            }
            // Higher texture resolution bonus
            rating += properties.limits.max_image_dimension2_d;

            // SAFETY: device_name is provided by Vulkan and is a valid CStr.
            let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
                .to_string_lossy()
                .to_string();
            let extension_support = Self::supports_required_extensions(instance, physical_device);
            if !extension_support
                || swapchain_support.is_none()
                || !QueueFamily::is_complete(&queue_family_indices)
            {
                log::warn!("Device `{name}` does not meet minimum device requirements");
                rating = 0;
            }

            (
                queue_family_indices,
                Self {
                    name,
                    properties,
                    memory_properties,
                    features,
                    rating,
                    msaa_samples,
                    swapchain_support,
                    extension_support,
                    sampler_anisotropy_support: features.sampler_anisotropy == 1,
                },
            )
        }

        /// Return the maximum usable sample count for multisampling for a [`vk::PhysicalDevice`].
        fn get_max_sample_count(
            properties: vk::PhysicalDeviceProperties,
            rating: &mut u32,
        ) -> vk::SampleCountFlags {
            let count = properties
                .limits
                .framebuffer_color_sample_counts
                .min(properties.limits.framebuffer_depth_sample_counts);
            if count.contains(vk::SampleCountFlags::TYPE_64) {
                *rating += 64;
                vk::SampleCountFlags::TYPE_64
            } else if count.contains(vk::SampleCountFlags::TYPE_32) {
                *rating += 32;
                vk::SampleCountFlags::TYPE_32
            } else if count.contains(vk::SampleCountFlags::TYPE_16) {
                *rating += 16;
                vk::SampleCountFlags::TYPE_16
            } else if count.contains(vk::SampleCountFlags::TYPE_8) {
                *rating += 8;
                vk::SampleCountFlags::TYPE_8
            } else if count.contains(vk::SampleCountFlags::TYPE_4) {
                *rating += 4;
                vk::SampleCountFlags::TYPE_4
            } else if count.contains(vk::SampleCountFlags::TYPE_2) {
                *rating += 2;
                vk::SampleCountFlags::TYPE_2
            } else {
                *rating += 1;
                vk::SampleCountFlags::TYPE_1
            }
        }

        /// Whether a [`vk::PhysicalDevice`] supports the required device extensions for this platform.
        fn supports_required_extensions(
            instance: &ash::Instance,
            physical_device: vk::PhysicalDevice,
        ) -> bool {
            // SAFETY: TODO
            let device_extensions =
                unsafe { instance.enumerate_device_extension_properties(physical_device) };
            let Ok(device_extensions) = device_extensions else {
                log::error!("failed to enumerate device extensions");
                return false;
            };
            let extension_names = device_extensions
                .iter()
                // SAFETY: extension_name is provided by Vulkan and is a valid CStr.
                .map(|extension_properties| unsafe {
                    CStr::from_ptr(extension_properties.extension_name.as_ptr())
                })
                .collect::<HashSet<_>>();
            platform::required_device_extensions()
                .iter()
                // SAFETY: required_device_extensions are static strings provided by Ash and are valid
                // CStrs.
                .map(|&extension| unsafe { CStr::from_ptr(extension) })
                .all(|extension| extension_names.contains(extension))
        }
    }

    #[derive(Debug, Clone)]
    #[must_use]
    pub(crate) struct SwapchainSupport {
        pub(crate) capabilities: vk::SurfaceCapabilitiesKHR,
        pub(crate) formats: Vec<vk::SurfaceFormatKHR>,
        pub(crate) present_modes: Vec<vk::PresentModeKHR>,
    }

    impl SwapchainSupport {
        /// Queries a [`vk::PhysicalDevice`] for [`vk::SwapchainKHR`] extension support.
        fn query(physical_device: vk::PhysicalDevice, surface: &Surface) -> Result<Self> {
            // SAFETY: TODO
            let capabilities = unsafe {
                surface
                    .loader
                    .get_physical_device_surface_capabilities(physical_device, surface.handle)
            }
            .context("Failed to query for surface capabilities.")?;

            // SAFETY: TODO
            let formats = unsafe {
                surface
                    .loader
                    .get_physical_device_surface_formats(physical_device, surface.handle)
            }
            .context("Failed to query for surface formats.")?;

            // SAFETY: TODO
            let present_modes = unsafe {
                surface
                    .loader
                    .get_physical_device_surface_present_modes(physical_device, surface.handle)
            }
            .context("Failed to query for surface present mode.")?;

            if formats.is_empty() || present_modes.is_empty() {
                bail!("no available swapchain formats or present_modes");
            }

            Ok(Self {
                capabilities,
                formats,
                present_modes,
            })
        }
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    #[must_use]
    pub(crate) enum QueueFamily {
        Graphics,
        Present,
        Compute,
        Transfer,
    }

    type QueueFamilyIndices = HashMap<QueueFamily, u32>;

    impl QueueFamily {
        /// Queries for desired [vk::Queue] families from a [`vk::PhysicalDevice`].
        fn query(
            instance: &ash::Instance,
            physical_device: vk::PhysicalDevice,
            surface: &Surface,
        ) -> QueueFamilyIndices {
            // SAFETY: TODO
            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

            let mut queue_family_indices = HashMap::default();
            let mut min_transfer_score = 255;
            for (index, queue_family) in queue_families.iter().enumerate() {
                let queue_index = index as u32;
                let mut current_transfer_score = 0;

                // SAFETY: TODO
                let has_surface_support = unsafe {
                    surface.loader.get_physical_device_surface_support(
                        physical_device,
                        queue_index,
                        surface.handle,
                    )
                }
                .unwrap_or_default();

                queue_family_indices
                    .entry(QueueFamily::Graphics)
                    .or_insert_with(|| {
                        current_transfer_score += 1;
                        queue_index
                    });
                // Prioritize queues supporting both graphics and present
                if queue_family_indices.contains_key(&QueueFamily::Graphics) && has_surface_support
                {
                    queue_family_indices.insert(QueueFamily::Present, queue_index);
                    current_transfer_score += 1;
                }

                if queue_family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    queue_family_indices
                        .entry(QueueFamily::Compute)
                        .or_insert(queue_index);
                    current_transfer_score += 1;
                }

                if queue_family.queue_flags.contains(vk::QueueFlags::TRANSFER)
                    && current_transfer_score <= min_transfer_score
                {
                    min_transfer_score = current_transfer_score;
                    queue_family_indices
                        .entry(QueueFamily::Transfer)
                        .or_insert(queue_index);
                }

                if QueueFamily::is_complete(&queue_family_indices) {
                    break;
                }
            }

            // If we didn't find a Graphics family supporting Present, just find use first one
            // found
            if let Entry::Vacant(entry) = queue_family_indices.entry(QueueFamily::Present) {
                if let Some(queue_index) = queue_families
                    .iter()
                    .enumerate()
                    .map(|(queue_index, _)| queue_index as u32)
                    .find(|&queue_index| {
                        // SAFETY: TODO
                        unsafe {
                            surface.loader.get_physical_device_surface_support(
                                physical_device,
                                queue_index,
                                surface.handle,
                            )
                        }
                        .unwrap_or_default()
                    })
                {
                    entry.insert(queue_index);
                }
            }

            queue_family_indices
        }

        /// Whether the `QueueFamilyIndices` contains all of the desired queues.
        fn is_complete(indices: &QueueFamilyIndices) -> bool {
            indices.contains_key(&Self::Graphics)
                && indices.contains_key(&Self::Present)
                && indices.contains_key(&Self::Compute)
                && indices.contains_key(&Self::Transfer)
        }
    }
}

mod swapchain {
    use super::{
        device::{Device, QueueFamily},
        surface::Surface,
    };
    use anyhow::{bail, Context, Result};
    use ash::{extensions::khr, prelude::VkResult, vk};

    #[derive(Clone)]
    #[must_use]
    pub(crate) struct Swapchain {
        pub(crate) loader: khr::Swapchain,
        pub(crate) handle: vk::SwapchainKHR,
        pub(crate) format: vk::Format,
        pub(crate) extent: vk::Extent2D,
        pub(crate) images: Vec<vk::Image>,
        pub(crate) image_views: Vec<vk::ImageView>,
        pub(crate) max_frames_in_flight: usize,
    }

    impl Swapchain {
        /// Create a `Swapchain` instance containing a handle to [`vk::SwapchainKHR`],
        /// [vk::Format], [`vk::Extent2D`] and the associated swapchain [vk::Image]s and
        /// [`vk::ImageView`]s.
        pub(crate) fn create(
            instance: &ash::Instance,
            device: &Device,
            surface: &Surface,
            width: u32,
            height: u32,
        ) -> Result<Self> {
            log::debug!("creating swapchain");

            let Some(swapchain_support) = device.info.swapchain_support.as_ref() else {
                bail!("{} does not support swapchains", device.info.name);
            };

            // Select swapchain format
            let surface_format = swapchain_support
                .formats
                .iter()
                .find(|surface_format| {
                    surface_format.format == vk::Format::B8G8R8A8_SRGB
                        && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
                .or_else(|| swapchain_support.formats.first())
                .copied()
                .context("failed to find a valid swapchain format")?;

            // Select present mode
            let present_mode = swapchain_support
                .present_modes
                .iter()
                .find(|&&present_mode| present_mode == vk::PresentModeKHR::MAILBOX)
                .copied()
                .unwrap_or(vk::PresentModeKHR::FIFO);

            // Select extent
            // NOTE: current_extent.width equal to u32::MAX means that the swapchain image
            // resolution can differ from the window resolution
            let capabilities = &swapchain_support.capabilities;
            let image_extent = if capabilities.current_extent.width == u32::MAX {
                let vk::Extent2D {
                    width: min_width,
                    height: min_height,
                } = capabilities.min_image_extent;
                let vk::Extent2D {
                    width: max_width,
                    height: max_height,
                } = capabilities.max_image_extent;
                vk::Extent2D::builder()
                    .width(width.clamp(min_width, max_width))
                    .height(height.clamp(min_height, max_height))
                    .build()
            } else {
                capabilities.current_extent
            };

            // Determine image_count
            let min_image_count = capabilities.min_image_count + 1;
            let min_image_count = if capabilities.max_image_count > 0
                && min_image_count > capabilities.max_image_count
            {
                min_image_count.min(capabilities.max_image_count)
            } else {
                min_image_count
            };

            // Select image sharing mode, concurrent vs exclusive
            let (image_sharing_mode, queue_family_indices) = match (
                device.queue_family_indices.get(&QueueFamily::Graphics),
                device.queue_family_indices.get(&QueueFamily::Present),
            ) {
                (Some(&graphics), Some(&present)) if graphics != present => {
                    (vk::SharingMode::CONCURRENT, vec![graphics, present])
                }
                _ => (vk::SharingMode::EXCLUSIVE, vec![]),
            };

            // Create info
            let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
                .surface(surface.handle)
                .min_image_count(min_image_count)
                .image_color_space(surface_format.color_space)
                .image_format(surface_format.format)
                .image_extent(image_extent)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(image_sharing_mode)
                .queue_family_indices(&queue_family_indices)
                .pre_transform(capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .image_array_layers(1);

            // Create swapchain
            let swapchain_loader = khr::Swapchain::new(instance, &device.logical_device);
            // SAFETY: All create_info values are set correctly above with valid lifetimes.
            let swapchain =
                unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None) }
                    .context("failed to create swapchain")?;

            // Get images
            // SAFETY: TODO
            let images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }
                .context("failed to get swapchain images")?;

            // Create image views
            let image_views = images
                .iter()
                .map(|&image| {
                    let image_view_create_info = vk::ImageViewCreateInfo::builder()
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(surface_format.format)
                        .components(vk::ComponentMapping::default())
                        .subresource_range(
                            vk::ImageSubresourceRange::builder()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1)
                                .build(),
                        )
                        .image(image);
                    unsafe {
                        device
                            .logical_device
                            .create_image_view(&image_view_create_info, None)
                    }
                    .context("failed to create image view")
                })
                .collect::<Result<Vec<_>>>()?;

            log::debug!("created swapchain successfully");

            Ok(Self {
                loader: swapchain_loader,
                handle: swapchain,
                format: surface_format.format,
                extent: image_extent,
                images,
                image_views,
                max_frames_in_flight: min_image_count as usize - 1,
            })
        }

        /// Destroy a `Swapchain` instance.
        // SAFETY: TODO
        pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
            self.image_views.iter().for_each(|&image_view| {
                device.destroy_image_view(image_view, None);
            });
            self.loader.destroy_swapchain(self.handle, None);
        }

        /// Acquire next [vk::SwapchainKHR] images index.
        pub(crate) fn acquire_next_image(
            &self,
            timeout: u64,
            semaphore: vk::Semaphore,
            fence: vk::Fence,
        ) -> VkResult<(u32, bool)> {
            // SAFETY: TODO
            unsafe {
                self.loader
                    .acquire_next_image(self.handle, timeout, semaphore, fence)
            }
        }

        /// Present to a given [vk::Queue]. On success, returns whether swapchain is suboptimal for
        /// the surface.
        pub(crate) fn queue_present(
            &self,
            queue: vk::Queue,
            present_info: &vk::PresentInfoKHR,
        ) -> VkResult<bool> {
            // SAFETY: TODO
            unsafe { self.loader.queue_present(queue, present_info) }
        }
    }
}

mod pipeline {
    use super::device::Device;
    use crate::{
        math::Vertex,
        renderer::{Shader, ShaderType},
    };
    use anyhow::{Context, Result};
    use ash::vk;
    use std::{ffi::CString, slice};

    #[derive(Clone)]
    #[must_use]
    pub(crate) struct Pipeline {
        pub(crate) handle: vk::Pipeline,
        pub(crate) layout: vk::PipelineLayout,
    }

    impl Pipeline {
        /// Create a `Pipeline` instancing containing a handle to [vk::Pipeline].
        pub(crate) fn create(
            device: &Device,
            extent: vk::Extent2D,
            render_pass: vk::RenderPass,
            descriptor_set_layout: vk::DescriptorSetLayout,
            shaders: &[Shader],
        ) -> Result<Self> {
            log::debug!("creating graphics pipeline");

            let device_info = &device.info;
            let device = &device.logical_device;

            // Shader Stages
            // TODO: Make shaders configuration more flexible
            let vertex_shader_module = shaders
                .iter()
                .find(|shader| shader.ty == ShaderType::Vertex)
                .context("failed to find a valid vertex shader")?
                .create_shader_module(device)?;
            let fragment_shader_module = shaders
                .iter()
                .find(|shader| shader.ty == ShaderType::Fragment)
                .context("failed to find a valid fragment shader")?
                .create_shader_module(device)?;

            let shader_entry_name = CString::new("main")?;
            let shader_stages = [
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vertex_shader_module)
                    .name(&shader_entry_name)
                    .build(),
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(fragment_shader_module)
                    .name(&shader_entry_name)
                    .build(),
            ];

            // Vertex Input State
            let binding_descriptions = [Vertex::binding_description()];
            let attribute_descriptions = Vertex::attribute_descriptions();
            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_binding_descriptions(&binding_descriptions)
                .vertex_attribute_descriptions(&attribute_descriptions);

            // Input Assembly State
            let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                .primitive_restart_enable(false);

            // Viewport State
            let viewport = vk::Viewport::builder()
                .x(0.0)
                .y(0.0)
                .width(extent.width as f32)
                .height(extent.height as f32)
                .min_depth(0.0)
                .max_depth(1.0);
            let scissor = vk::Rect2D::builder()
                .offset(vk::Offset2D::default())
                .extent(extent);
            let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
                .viewports(slice::from_ref(&viewport))
                .scissors(slice::from_ref(&scissor));

            // Rasterization State
            let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                // TODO: Disabled for now while rotating
                // .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE) // Because Y-axis is inverted in Vulkan
                .depth_clamp_enable(false)
                .depth_bias_enable(false);

            // Multisample State
            let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
                // TODO: May impact performance
                .sample_shading_enable(true)
                // Closer to 1 is smoother
                .min_sample_shading(0.2)
                .rasterization_samples(device_info.msaa_samples);

            // Depth Stencil State
            let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS)
                .depth_bounds_test_enable(false)
                .stencil_test_enable(false);

            // Color Blend State
            let attachment = vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                .alpha_blend_op(vk::BlendOp::ADD);
            let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
                .logic_op_enable(false)
                .logic_op(vk::LogicOp::COPY)
                .attachments(slice::from_ref(&attachment))
                .blend_constants([0.0; 4]);

            // Push Constants
            let vert_push_constant_ranges = vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::VERTEX)
                .offset(0)
                .size(64); // 16 * 4 byte floats
            let frag_push_constant_ranges = vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .offset(64)
                .size(4); // 1 * 4 byte float

            // Layout
            let set_layouts = [descriptor_set_layout];
            let push_constant_ranges = [
                vert_push_constant_ranges.build(),
                frag_push_constant_ranges.build(),
            ];
            let layout_info = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&push_constant_ranges);
            // SAFETY: TODO
            let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
                .context("failed to create graphics pipeline layout")?;

            // Create
            let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
                .stages(&shader_stages)
                .vertex_input_state(&vertex_input_state)
                .input_assembly_state(&input_assembly_state)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterization_state)
                .multisample_state(&multisample_state)
                .depth_stencil_state(&depth_stencil_state)
                .color_blend_state(&color_blend_state)
                .layout(layout)
                .render_pass(render_pass)
                .subpass(0);

            // SAFETY: TODO
            let pipeline = unsafe {
                device.create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    slice::from_ref(&pipeline_create_info),
                    None,
                )
            }
            .map_err(|(_, err)| err)
            .context("failed to create graphics pipeline")?
            .into_iter()
            .next()
            .context("no graphics pipelines were created")?;

            // Cleanup
            // SAFETY: TODO
            unsafe {
                device.destroy_shader_module(vertex_shader_module, None);
                device.destroy_shader_module(fragment_shader_module, None);
            }

            log::debug!("created graphics pipeline successfully");

            Ok(Self {
                handle: pipeline,
                layout,
            })
        }

        /// Destroy a `Pipeline` instance.
        // SAFETY: TODO
        pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
            device.destroy_pipeline(self.handle, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

mod shader {
    use crate::renderer::Shader;
    use anyhow::{bail, Context, Result};
    use ash::vk;

    impl Shader {
        /// Create a [`vk::ShaderModule`] instance from bytecode.
        pub(crate) fn create_shader_module(
            &self,
            device: &ash::Device,
        ) -> Result<vk::ShaderModule> {
            log::debug!("creating shader module");

            // SAFETY: We check for prefix/suffix below and bail if not aligned correctly.
            let (prefix, code, suffix) = unsafe { self.bytes.align_to::<u32>() };
            if !prefix.is_empty() || !suffix.is_empty() {
                bail!("shader bytecode is not properly aligned.");
            }
            let shader_module_info = vk::ShaderModuleCreateInfo::builder().code(code);

            let shader_module = unsafe { device.create_shader_module(&shader_module_info, None) }
                .context("failed to create shader module")?;

            log::debug!("created shader module successfully");

            Ok(shader_module)
        }
    }
}

mod image {
    #[cfg(debug_assertions)]
    use super::debug::Debug;
    use super::{command_pool::CommandPool, device::Device, swapchain::Swapchain};
    use anyhow::{bail, Context, Result};
    use ash::vk::{self, Handle};
    use std::{ffi::CString, fs, io::BufReader, path::Path, ptr, slice};

    #[derive(Clone)]
    #[must_use]
    pub(crate) struct Image {
        pub(crate) handle: vk::Image,
        pub(crate) memory: vk::DeviceMemory,
        pub(crate) view: vk::ImageView,
        pub(crate) mip_levels: u32,
    }

    impl Image {
        /// Create an `Image` instance with a [vk::Image] handle and associated [`vk::DeviceMemory`] with the given parameters.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn create(
            name: &str,
            device: &Device,
            width: u32,
            height: u32,
            mip_levels: u32,
            samples: vk::SampleCountFlags,
            format: vk::Format,
            tiling: vk::ImageTiling,
            usage: vk::ImageUsageFlags,
            properties: vk::MemoryPropertyFlags,
            #[cfg(debug_assertions)] debug: &Debug,
        ) -> Result<(vk::Image, vk::DeviceMemory)> {
            log::debug!("creating image");

            // Image
            let image_info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(mip_levels)
                .array_layers(1)
                .format(format)
                .tiling(tiling)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .usage(usage)
                .samples(samples)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            // SAFETY: TODO
            let image = unsafe { device.logical_device.create_image(&image_info, None) }
                .context("failed to create image")?;

            // Debug Name

            #[cfg(debug_assertions)]
            let name = CString::new(name)?;
            #[cfg(debug_assertions)]
            let debug_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(vk::ObjectType::IMAGE)
                .object_name(&name)
                .object_handle(image.as_raw());
            #[cfg(debug_assertions)]
            unsafe {
                debug
                    .utils
                    .set_debug_utils_object_name(device.logical_device.handle(), &debug_info)
            }
            .context("failed to set debug utils object name")?;

            // Memory

            let requirements =
                unsafe { device.logical_device.get_image_memory_requirements(image) };
            let memory_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(requirements.size)
                .memory_type_index(device.memory_type_index(properties, requirements)?);
            let memory = unsafe { device.logical_device.allocate_memory(&memory_info, None) }
                .context("failed to allocate image memory")?;
            // SAFETY: TODO
            unsafe { device.logical_device.bind_image_memory(image, memory, 0) }
                .context("failed to bind image memory")?;

            log::debug!("created image successfully");

            Ok((image, memory))
        }

        /// Destroy an `Image` intance.
        // SAFETY: TODO
        pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
            device.destroy_image_view(self.view, None);
            device.free_memory(self.memory, None);
            device.destroy_image(self.handle, None);
        }

        /// Create a [vk::Image] instance to be used as a [`vk::ImageUsageFlags::COLOR_ATTACHMENT`].
        pub(crate) fn create_color(
            device: &Device,
            swapchain: &Swapchain,
            #[cfg(debug_assertions)] debug: &Debug,
        ) -> Result<Self> {
            log::debug!("creating color image");

            // Image
            let mip_levels = 1;
            let (image, memory) = Self::create(
                "color",
                device,
                swapchain.extent.width,
                swapchain.extent.height,
                mip_levels,
                device.info.msaa_samples,
                swapchain.format,
                vk::ImageTiling::OPTIMAL,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                #[cfg(debug_assertions)]
                debug,
            )?;

            // Image View
            let view = Self::create_view(
                device,
                image,
                swapchain.format,
                vk::ImageAspectFlags::COLOR,
                mip_levels,
            )?;

            log::debug!("created color image successfully");

            Ok(Self {
                handle: image,
                memory,
                view,
                mip_levels,
            })
        }

        /// Create a [vk::Image] instance to be used as a [`vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT`].
        pub(crate) fn create_depth(
            instance: &ash::Instance,
            device: &Device,
            swapchain: &Swapchain,
            #[cfg(debug_assertions)] debug: &Debug,
        ) -> Result<Self> {
            log::debug!("creating depth image");

            let format = device.get_depth_format(instance, vk::ImageTiling::OPTIMAL)?;

            // Image
            let mip_levels = 1;
            let (image, memory) = Self::create(
                "depth",
                device,
                swapchain.extent.width,
                swapchain.extent.height,
                mip_levels,
                device.info.msaa_samples,
                format,
                vk::ImageTiling::OPTIMAL,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                #[cfg(debug_assertions)]
                debug,
            )?;

            // Image View
            let view = Self::create_view(
                device,
                image,
                format,
                vk::ImageAspectFlags::DEPTH,
                mip_levels,
            )?;

            log::debug!("created depth image successfully");

            Ok(Self {
                handle: image,
                memory,
                view,
                mip_levels,
            })
        }

        /// Create a [vk::Image] instance to be used as a texture.
        pub(crate) fn create_texture(
            name: &str,
            instance: &ash::Instance,
            device: &Device,
            command_pool: &CommandPool,
            graphics_queue: vk::Queue,
            filename: impl AsRef<Path>,
            #[cfg(debug_assertions)] debug: &Debug,
        ) -> Result<Self> {
            log::debug!("creating texture named `{name}`");

            // Load Texture
            let filename = filename.as_ref();
            let image = BufReader::new(
                fs::File::open(
                    fs::canonicalize(filename)
                        .with_context(|| format!("failed to find texture file: {filename:?}"))?,
                )
                .with_context(|| format!("failed to open texture file: {filename:?}"))?,
            );

            let decoder = png::Decoder::new(image);
            let mut reader = decoder.read_info()?;
            let mut pixels = vec![0; reader.output_buffer_size()];
            let info = reader.next_frame(&mut pixels)?;
            let size = info.buffer_size() as u64;

            let mip_levels = (info.width.max(info.height) as f32).log2().floor() as u32 + 1;

            // Staging Buffer
            let staging_buffer = device.create_buffer(
                size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
            )?;

            // Copy Staging
            // SAFETY: TODO
            unsafe {
                let memory = device
                    .logical_device
                    .map_memory(staging_buffer.memory, 0, size, vk::MemoryMapFlags::empty())
                    .context("failed to map staging buffer memory")?;
                // SAFETY: TODO
                ptr::copy_nonoverlapping(pixels.as_ptr(), memory.cast(), pixels.len());
                device.logical_device.unmap_memory(staging_buffer.memory);
            };

            // Texture Image
            let (image, memory) = Image::create(
                name,
                device,
                info.width,
                info.height,
                mip_levels,
                vk::SampleCountFlags::TYPE_1,
                vk::Format::R8G8B8A8_SRGB,
                vk::ImageTiling::OPTIMAL,
                vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                #[cfg(debug_assertions)]
                debug,
            )?;

            // Transition
            {
                let command_buffer = command_pool.begin_one_time_command(&device.logical_device)?;
                let image_barrier = vk::ImageMemoryBarrier::builder()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(mip_levels)
                            .base_array_layer(0)
                            .layer_count(1)
                            .build(),
                    )
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);

                // SAFETY: TODO
                #[allow(trivial_casts)]
                unsafe {
                    device.logical_device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[] as &[vk::MemoryBarrier; 0],
                        &[] as &[vk::BufferMemoryBarrier; 0],
                        slice::from_ref(&image_barrier),
                    );
                };
                command_pool.end_one_time_command(
                    &device.logical_device,
                    command_buffer,
                    graphics_queue,
                )?;
            }

            // Copy
            device.copy_buffer_to_image(
                command_pool,
                graphics_queue,
                staging_buffer.handle,
                image,
                info.width,
                info.height,
            )?;

            // Mipmap
            let format = vk::Format::R8G8B8A8_SRGB;
            Self::generate_mipmaps(
                instance,
                device,
                command_pool,
                graphics_queue,
                image,
                format,
                info.width,
                info.height,
                mip_levels,
            )?;

            let view = Self::create_view(
                device,
                image,
                format,
                vk::ImageAspectFlags::COLOR,
                mip_levels,
            )?;

            // Cleanup
            // SAFETY: TODO
            unsafe {
                staging_buffer.destroy(&device.logical_device);
            }

            log::debug!("created texture named `{name}` successfully");

            Ok(Self {
                handle: image,
                memory,
                view,
                mip_levels,
            })
        }

        /// Create a [`vk::ImageView`] instance.
        pub(crate) fn create_view(
            device: &Device,
            image: vk::Image,
            format: vk::Format,
            aspects: vk::ImageAspectFlags,
            mip_levels: u32,
        ) -> Result<vk::ImageView> {
            log::debug!("creating image view");

            let view_create_info = vk::ImageViewCreateInfo::builder()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(aspects)
                        .base_mip_level(0)
                        .level_count(mip_levels)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                );
            // SAFETY: TODO
            let view = unsafe {
                device
                    .logical_device
                    .create_image_view(&view_create_info, None)
            }
            .context("failed to create image view")?;

            log::debug!("created image view successfully");

            Ok(view)
        }

        #[allow(clippy::too_many_arguments)]
        fn generate_mipmaps(
            instance: &ash::Instance,
            device: &Device,
            command_pool: &CommandPool,
            graphics_queue: vk::Queue,
            image: vk::Image,
            format: vk::Format,
            width: u32,
            height: u32,
            mip_levels: u32,
        ) -> Result<()> {
            log::debug!("generating mipmaps");

            // Check Support
            let properties = device.format_properties(instance, format);
            if !properties
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
            {
                bail!("vulkan device does not support linear blitting!");
            }

            // Mipmap
            let command_buffer = command_pool.begin_one_time_command(&device.logical_device)?;

            let mut barrier = vk::ImageMemoryBarrier::builder()
                .image(image)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_array_layer(0)
                        .layer_count(1)
                        .level_count(1)
                        .build(),
                );

            let mut mip_width = width as i32;
            let mut mip_height = height as i32;

            for i in 1..mip_levels {
                barrier.subresource_range.base_mip_level = i - 1;
                barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
                barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

                // SAFETY: TODO
                #[allow(trivial_casts)]
                unsafe {
                    device.logical_device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[] as &[vk::MemoryBarrier; 0],
                        &[] as &[vk::BufferMemoryBarrier; 0],
                        slice::from_ref(&barrier),
                    );
                }

                let blit = vk::ImageBlit::builder()
                    .src_offsets([
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: mip_width,
                            y: mip_height,
                            z: 1,
                        },
                    ])
                    .src_subresource(
                        vk::ImageSubresourceLayers::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(i - 1)
                            .base_array_layer(0)
                            .layer_count(1)
                            .build(),
                    )
                    .dst_offsets([
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: (mip_width / 2).max(1),
                            y: (mip_height / 2).max(1),
                            z: 1,
                        },
                    ])
                    .dst_subresource(
                        vk::ImageSubresourceLayers::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(i)
                            .base_array_layer(0)
                            .layer_count(1)
                            .build(),
                    );

                // SAFETY: TODO
                unsafe {
                    device.logical_device.cmd_blit_image(
                        command_buffer,
                        image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        slice::from_ref(&blit),
                        vk::Filter::LINEAR,
                    );
                }

                barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
                barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

                // SAFETY: TODO
                #[allow(trivial_casts)]
                unsafe {
                    device.logical_device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        &[] as &[vk::MemoryBarrier; 0],
                        &[] as &[vk::BufferMemoryBarrier; 0],
                        slice::from_ref(&barrier),
                    );
                }

                mip_width = (mip_width / 2).max(1);
                mip_height = (mip_height / 2).max(1);
            }

            barrier.subresource_range.base_mip_level = mip_levels - 1;
            barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            // SAFETY: TODO
            #[allow(trivial_casts)]
            unsafe {
                device.logical_device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[] as &[vk::MemoryBarrier; 0],
                    &[] as &[vk::BufferMemoryBarrier; 0],
                    slice::from_ref(&barrier),
                );
            }

            command_pool.end_one_time_command(
                &device.logical_device,
                command_buffer,
                graphics_queue,
            )?;

            log::debug!("generated mipmaps successfully");

            Ok(())
        }

        pub(crate) fn create_sampler(device: &ash::Device, mip_levels: u32) -> Result<vk::Sampler> {
            log::debug!("creating sampler");

            let sampler_create_info = vk::SamplerCreateInfo::builder()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                .anisotropy_enable(true)
                .max_anisotropy(16.0)
                .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
                .unnormalized_coordinates(false)
                .compare_enable(false)
                .compare_op(vk::CompareOp::ALWAYS)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .min_lod(0.0) // Optional
                .max_lod(mip_levels as f32)
                .mip_lod_bias(0.0);

            let sampler = unsafe { device.create_sampler(&sampler_create_info, None) }
                .context("failed to create sampler")?;

            log::debug!("created sampler successfully");

            Ok(sampler)
        }
    }
}

mod command_pool {
    use super::device::{Device, QueueFamily};
    use anyhow::{Context, Result};
    use ash::vk;
    use std::slice;

    #[derive(Clone)]
    #[must_use]
    pub(crate) struct CommandPool {
        pub(crate) handle: vk::CommandPool,
        pub(crate) buffers: Vec<vk::CommandBuffer>,
        pub(crate) secondary_buffers: Vec<vk::CommandBuffer>,
    }

    impl CommandPool {
        /// Create a `CommandPool` instance containing a handle to a [`vk::CommandPool`] and a list
        /// of [`vk::CommandBuffer`]s.
        pub(crate) fn create(
            device: &Device,
            queue_family: QueueFamily,
            buffer_count: u32,
        ) -> Result<Self> {
            log::debug!("creating command pool on queue family: {queue_family:?}");

            let queue_index = device
                .queue_family_indices
                .get(&queue_family)
                .context("{queue_family:?} queue family not found")?;

            let pool_create_info = vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(*queue_index);
            let pool = unsafe {
                device
                    .logical_device
                    .create_command_pool(&pool_create_info, None)
            }
            .context("failed to create command pool")?;

            let buffer_alloc_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(buffer_count);
            let buffers = unsafe {
                device
                    .logical_device
                    .allocate_command_buffers(&buffer_alloc_info)
            }
            .context("failed to allocate command buffers")?;

            log::debug!("created command pool successfully");

            Ok(Self {
                handle: pool,
                buffers,
                secondary_buffers: vec![],
            })
        }

        /// Destroy a `CommandPool` instance.
        // SAFETY: TODO
        pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
            device.destroy_command_pool(self.handle, None);
        }

        /// Creates a one-time [`vk::CommandBuffer`] instance to write commands to.
        pub(crate) fn begin_one_time_command(
            &self,
            device: &ash::Device,
        ) -> Result<vk::CommandBuffer> {
            log::debug!("beginning single time command. creating command buffer");

            // Allocate
            let command_allocate_info = vk::CommandBufferAllocateInfo::builder()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(self.handle)
                .command_buffer_count(1);

            let command_buffer = unsafe { device.allocate_command_buffers(&command_allocate_info) }
                .context("failed to allocate command buffer")?[0];

            // Commands
            let command_begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe { device.begin_command_buffer(command_buffer, &command_begin_info) }
                .context("failed to begin command buffer")?;

            log::debug!("created command buffer successfully");

            Ok(command_buffer)
        }

        /// Finishes a one-time [`vk::CommandBuffer`] and submits it to the queue.
        pub(crate) fn end_one_time_command(
            &self,
            device: &ash::Device,
            command_buffer: vk::CommandBuffer,
            queue: vk::Queue,
        ) -> Result<()> {
            log::debug!("finishing single time command");

            unsafe { device.end_command_buffer(command_buffer) }
                .context("failed to end command buffer")?;

            // Submit
            let command_buffers = [command_buffer];
            let submit_info = vk::SubmitInfo::builder().command_buffers(&command_buffers);

            // SAFETY: TODO
            unsafe {
                device.queue_submit(queue, slice::from_ref(&submit_info), vk::Fence::null())?;
                device.queue_wait_idle(queue)?;
            }

            // Cleanup
            unsafe {
                device.free_command_buffers(self.handle, &command_buffers);
            }

            log::debug!("finished single time command successfully");

            Ok(())
        }
    }
}

mod descriptor {
    use super::{device::Buffer, swapchain::Swapchain};
    use crate::math::UniformBufferObject;
    use anyhow::{Context, Result};
    use ash::vk;
    use std::{mem::size_of, slice};

    #[derive(Clone)]
    #[must_use]
    pub(crate) struct Descriptor {
        pub(crate) pool: vk::DescriptorPool,
        pub(crate) set_layout: vk::DescriptorSetLayout,
        pub(crate) sets: Vec<vk::DescriptorSet>,
    }

    impl Descriptor {
        /// Create a `Descriptor` instance with a [`vk::DescriptorPool`], [`vk::DescriptorSetLayout`] and list of [`vk::DescriptorSet`]s.
        pub(crate) fn create(
            device: &ash::Device,
            swapchain: &Swapchain,
            uniform_buffers: &[Buffer],
            image_view: vk::ImageView,
            sampler: vk::Sampler,
        ) -> Result<Self> {
            // Pool
            log::debug!("creating descriptor pool");

            let count = swapchain.images.len() as u32;

            let ubo_size = vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(count);
            let sampler_size = vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(count);

            let pool_sizes = [ubo_size.build(), sampler_size.build()];
            let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&pool_sizes)
                .max_sets(count);

            let pool = unsafe { device.create_descriptor_pool(&descriptor_pool_info, None) }
                .context("failed to create descriptor pool")?;

            log::debug!("created descriptor pool successfully");

            log::debug!("creating descriptor set layout");

            // Set Layout
            let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX);

            let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT);

            let bindings = [ubo_binding.build(), sampler_binding.build()];
            let descriptor_set_create_info =
                vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);

            let set_layout =
                unsafe { device.create_descriptor_set_layout(&descriptor_set_create_info, None) }
                    .context("failed to create descriptor set layout")?;

            log::debug!("created descriptor set layout successfully");

            log::debug!("creating descriptor sets");

            // Allocate
            let count = swapchain.images.len();
            let set_layouts = vec![set_layout; count];
            let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(pool)
                .set_layouts(&set_layouts);

            // SAFETY: TODO
            let sets = unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info) }
                .context("failed to allocate descriptor sets")?;

            // Update
            for i in 0..count {
                let buffer_info = vk::DescriptorBufferInfo::builder()
                    .buffer(uniform_buffers[i].handle)
                    .offset(0)
                    .range(size_of::<UniformBufferObject>() as u64);

                let ubo_write = vk::WriteDescriptorSet::builder()
                    .dst_set(sets[i])
                    .dst_binding(0) // points to shader.vert binding
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(slice::from_ref(&buffer_info));

                let image_info = vk::DescriptorImageInfo::builder()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(image_view)
                    .sampler(sampler);

                let sampler_write = vk::WriteDescriptorSet::builder()
                    .dst_set(sets[i])
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(slice::from_ref(&image_info));

                // SAFETY: TODO
                let descriptor_writes = [ubo_write.build(), sampler_write.build()];
                #[allow(trivial_casts)]
                unsafe {
                    device.update_descriptor_sets(
                        &descriptor_writes,
                        &[] as &[vk::CopyDescriptorSet; 0],
                    );
                };
            }

            log::debug!("created descriptor sets successfully");

            Ok(Self {
                pool,
                set_layout,
                sets,
            })
        }

        /// Destroy a `Descriptor` instance.
        // SAFETY: TODO
        pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_descriptor_pool(self.pool, None);
        }
    }
}

mod sync {
    use super::swapchain::Swapchain;
    use anyhow::{Context, Result};
    use ash::vk;

    #[derive(Clone)]
    #[must_use]
    pub(crate) struct Syncs {
        pub(crate) images_available: Vec<vk::Semaphore>,
        pub(crate) renders_finished: Vec<vk::Semaphore>,
        pub(crate) in_flight_fences: Vec<vk::Fence>,
        pub(crate) images_in_flight: Vec<vk::Fence>,
    }

    impl Syncs {
        /// Create a `Sync` instance with [vk::Semaphore]s and [vk::Fence]s.
        pub(crate) fn create(device: &ash::Device, swapchain: &Swapchain) -> Result<Self> {
            log::debug!("creating semaphor and fences");

            let semaphor_create_info = vk::SemaphoreCreateInfo::builder();
            let images_available = (0..swapchain.max_frames_in_flight)
                .map(|_| {
                    // SAFETY: TODO
                    unsafe { device.create_semaphore(&semaphor_create_info, None) }
                        .context("failed to create semaphor")
                })
                .collect::<Result<Vec<_>>>()?;
            let renders_finished = (0..swapchain.max_frames_in_flight)
                .map(|_| {
                    // SAFETY: TODO
                    unsafe { device.create_semaphore(&semaphor_create_info, None) }
                        .context("failed to create semaphor")
                })
                .collect::<Result<Vec<_>>>()?;

            let fence_create_info =
                vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
            let in_flight_fences = (0..swapchain.max_frames_in_flight)
                .map(|_| {
                    // SAFETY: TODO
                    unsafe { device.create_fence(&fence_create_info, None) }
                        .context("failed to create semaphor")
                })
                .collect::<Result<Vec<_>>>()?;

            let images_in_flight = swapchain
                .images
                .iter()
                .map(|_| vk::Fence::null())
                .collect::<Vec<_>>();

            log::debug!("created semaphor and fences successfully");

            Ok(Self {
                images_available,
                renders_finished,
                in_flight_fences,
                images_in_flight,
            })
        }

        /// Destroy a `Sync` instance.
        // SAFETY: TODO
        pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
            // images_in_flight are swapped with in_flight_fences during rendering,
            // so they
            self.in_flight_fences
                .iter()
                .for_each(|&fence| device.destroy_fence(fence, None));
            self.renders_finished
                .iter()
                .for_each(|&fence| device.destroy_semaphore(fence, None));
            self.images_available
                .iter()
                .for_each(|&fence| device.destroy_semaphore(fence, None));
        }
    }
}

mod model {
    use crate::{math::Vertex, vector};
    use anyhow::{Context, Result};
    use std::{fs, io::BufReader, path::Path};

    #[derive(Clone, Debug)]
    #[must_use]
    pub(crate) struct Model {
        pub(crate) name: String,
        pub(crate) vertices: Vec<Vertex>,
        pub(crate) indices: Vec<u32>,
    }

    impl Model {
        pub(crate) fn load(name: impl Into<String>, filename: impl AsRef<Path>) -> Result<Self> {
            let name = name.into();
            log::debug!("loading model named `{name}`");

            // Load Model
            let filename = filename.as_ref();
            let mut obj_reader = BufReader::new(
                fs::File::open(
                    fs::canonicalize(filename)
                        .with_context(|| format!("failed to find model file: {filename:?}"))?,
                )
                .with_context(|| format!("failed to open model file: {filename:?}"))?,
            );

            let (models, _) = tobj::load_obj_buf(
                &mut obj_reader,
                &tobj::LoadOptions {
                    triangulate: true,
                    ..Default::default()
                },
                |_| Ok((vec![tobj::Material::default()], Default::default())),
            )?;

            // Vertices / Indices
            let mut vertices = vec![];
            let mut indices = vec![];
            for mut model in models.into_iter() {
                let vertices_len = model.mesh.positions.len() / 3;
                vertices.reserve(vertices_len);
                indices.reserve(model.mesh.indices.len());
                for index in 0..vertices_len {
                    let position_offset = index * 3;
                    let texcoord_offset = index * 2;

                    let position = &model.mesh.positions;
                    let texcoords = &model.mesh.texcoords;

                    let vertex = Vertex::new(
                        vector!(
                            position[position_offset],
                            position[position_offset + 1],
                            position[position_offset + 2],
                        ),
                        vector!(1.0, 1.0, 1.0),
                        vector!(
                            texcoords[texcoord_offset],
                            1.0 - texcoords[texcoord_offset + 1],
                        ),
                    );
                    vertices.push(vertex);
                }
                indices.append(&mut model.mesh.indices);
            }

            log::debug!("loaded model named `{name}` succesfully");

            Ok(Self {
                name,
                vertices,
                indices,
            })
        }
    }
}

mod vertex {
    use crate::math::{Vec3, Vertex};
    use ash::vk;
    use std::mem::size_of;

    impl Vertex {
        /// Return the [`vk::VertexInputBindingDescription`] for the `Vertex` struct.
        pub(crate) fn binding_description() -> vk::VertexInputBindingDescription {
            vk::VertexInputBindingDescription::builder()
                .binding(0)
                .stride(size_of::<Vertex>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX)
                .build()
        }

        /// Return the list of [`vk::VertexInputAttributeDescription`]s for the `Vertex` struct.
        pub(crate) fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
            let pos = vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0) // points to vertex shader location
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0);
            let color = vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1) // points to vertex shader location
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(size_of::<Vec3>() as u32);
            let texcoord = vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(2) // points to vertex shader location
                .format(vk::Format::R32G32_SFLOAT)
                .offset((size_of::<Vec3>() + size_of::<Vec3>()) as u32);
            [pos.build(), color.build(), texcoord.build()]
        }
    }
}

mod debug {
    use anyhow::Result;
    use ash::{extensions::ext, vk};
    use std::ffi::{c_void, CStr};

    // SAFETY: This static string has been verified as a valid CStr.
    pub(crate) const VALIDATION_LAYER_NAME: &CStr =
        unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };

    #[derive(Clone)]
    #[cfg(debug_assertions)]
    pub(crate) struct Debug {
        pub(crate) utils: ext::DebugUtils,
        pub(crate) messenger: vk::DebugUtilsMessengerEXT,
    }

    #[cfg(debug_assertions)]
    impl Debug {
        /// Create a `Debug` instance with [`ext::DebugUtils`] and [`vk::DebugUtilsMessengerEXT`].
        pub(crate) fn create(entry: &ash::Entry, instance: &ash::Instance) -> Result<Self> {
            log::debug!("creating debug utils");

            let utils = ext::DebugUtils::new(entry, instance);
            let debug_create_info = Self::build_debug_create_info();
            // SAFETY: All create_info values are set correctly above with valid lifetimes.
            let messenger =
                unsafe { utils.create_debug_utils_messenger(&debug_create_info, None)? };

            log::debug!("vulkan debug utils created successfully");

            Ok(Self { utils, messenger })
        }

        /// Destroy a `Debug` instance.
        // SAFETY: TODO
        pub(crate) unsafe fn destroy(&self) {
            self.utils
                .destroy_debug_utils_messenger(self.messenger, None);
        }

        /// Build [`vk::DebugUtilsMessengerCreateInfoEXT`] with desired message severity and
        /// message types..
        pub(crate) fn build_debug_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT {
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
        match severity {
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::error!("{msg_type} {message}"),
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::warn!("{msg_type} {message}"),
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::info!("{msg_type} {message}"),
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => log::trace!("{msg_type} {message}"),
            _ => log::debug!("{msg_type} {message}"),
        };
        vk::FALSE
    }
}

mod platform {
    use anyhow::{Context, Result};
    use ash::{
        extensions::{ext, khr, mvk},
        vk, Entry, Instance,
    };
    use winit::window::Window;

    /// Return a list of enabled Vulkan [ash::Instance] extensions for macOS.
    #[cfg(target_os = "macos")]
    pub(crate) fn enabled_extension_names() -> Vec<*const i8> {
        vec![
            khr::Surface::name().as_ptr(),
            #[cfg(debug_assertions)]
            ext::DebugUtils::name().as_ptr(),
            mvk::MacOSSurface::name().as_ptr(),
            vk::KhrPortabilityEnumerationFn::name().as_ptr(),
            vk::KhrGetPhysicalDeviceProperties2Fn::name().as_ptr(),
        ]
    }

    /// Return a list of enabled Vulkan [ash::Instance] extensions for Linux.
    #[cfg(all(unix, not(target_os = "macos"),))]
    pub(crate) fn enabled_extension_names() -> Vec<*const i8> {
        vec![
            khr::Surface::name().as_ptr(),
            #[cfg(debug_assertions)]
            ext::DebugUtils::name().as_ptr(),
            khr::XlibSurface::name().as_ptr(),
        ]
    }

    /// Return a list of enabled Vulkan [ash::Instance] extensions for Windows.
    #[cfg(windows)]
    pub(crate) fn enabled_extension_names() -> Vec<*const i8> {
        vec![
            khr::Surface::name().as_ptr(),
            #[cfg(debug_assertions)]
            ext::DebugUtils::name().as_ptr(),
            khr::Win32Surface::name().as_ptr(),
        ]
    }

    /// Return a set of [`vk::InstanceCreateFlags`] for macOS.
    #[cfg(target_os = "macos")]
    pub(crate) fn instance_create_flags() -> vk::InstanceCreateFlags {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    }

    /// Return a set of [`vk::InstanceCreateFlags`] for all platforms other than macOS.
    #[cfg(not(target_os = "macos"))]
    pub(crate) fn instance_create_flags() -> vk::InstanceCreateFlags {
        vk::InstanceCreateFlags::default()
    }

    /// Create a [`vk::SurfaceKHR`] instance for the current [Window] for macOS.
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

    /// Create a [`vk::SurfaceKHR`] instance for the current [Window] for Linux.
    #[cfg(all(unix, not(target_os = "macos"),))]
    pub(crate) fn create_surface(
        entry: &Entry,
        instance: &Instance,
        window: &Window,
    ) -> Result<vk::SurfaceKHR> {
        use std::ptr;
        use winit::platform::unix::WindowExtUnix;

        log::debug!("creating Linux XLIB surface");

        let x11_display = window
            .xlib_display()
            .context("failed to get XLIB display")?;
        let x11_window = winadow.xlib_window().context("failed to get XLIB window")?;
        let surface_create_info = vk::XlibSurfaceCreateInfoKHR::builder()
            .window(x11_window as vk::Window)
            .dpy(x11_display as *mut vk::Display);
        let xlib_surface = khr::XlibSurface::new(entry, instance);
        // SAFETY: All create_info values are set correctly above with valid lifetimes.
        unsafe { xlib_surface.create_xlib_surface(&surface_create_info, None) }
            .context("failed to create surface")
    }

    /// Create a [`vk::SurfaceKHR`] instance for the current [Window] for Windows.
    #[cfg(windows)]
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

        let hinstance = GetModuleHandleW(ptr::null()) as *const c_void;
        let surface_create_info = vk::Win32SurfaceCreateInfoKHR::builder()
            .hinstance(hinstance)
            .hwnd(window.hwnd());
        let win32_surface = khr::Win32Surface::new(entry, instance);
        // SAFETY: All create_info values are set correctly above with valid lifetimes.
        unsafe { win32_surface.create_win32_surface(&surface_create_info, None) }
            .context("failed to create surface")
    }

    /// Return a list of required [`vk::PhysicalDevice`] extensions for macOS.
    #[cfg(target_os = "macos")]
    pub(crate) fn required_device_extensions() -> Vec<*const i8> {
        vec![
            khr::Swapchain::name().as_ptr(),
            vk::KhrPortabilitySubsetFn::name().as_ptr(),
        ]
    }

    /// Return a list of required [`vk::PhysicalDevice`] extensions for all platforms other than
    /// macOS.
    #[cfg(not(target_os = "macos"))]
    pub(crate) fn required_device_extensions() -> Vec<*const i8> {
        vec![khr::Swapchain::name().as_ptr()]
    }
}
