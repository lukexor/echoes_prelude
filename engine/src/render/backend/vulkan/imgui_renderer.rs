use super::{
    device::Device,
    swapchain::{Swapchain, MAX_FRAMES_IN_FLIGHT},
};
use crate::prelude::imgui;
use anyhow::{Context, Result};
use ash::vk;
use std::{mem::ManuallyDrop, slice};

pub(crate) struct Renderer {
    render_pass: vk::RenderPass,
    renderer: ManuallyDrop<imgui_rs_vulkan_renderer::Renderer>,
    framebuffers: Vec<vk::Framebuffer>,
}

impl Renderer {
    /// Initialize the `imgui` `Renderer`.
    pub(crate) fn initialize(
        instance: &ash::Instance,
        device: &Device,
        swapchain: &Swapchain,
        command_pool: vk::CommandPool,
        imgui: &mut imgui::ImGui,
    ) -> Result<Self> {
        use imgui_rs_vulkan_renderer::{Options, Renderer};

        tracing::debug!("initializing imgui renderer ");

        // TODO fonts
        let render_pass = create_render_pass(device, swapchain.format)?;
        let renderer = Renderer::with_default_allocator(
            instance,
            device.physical,
            device.handle.clone(),
            device.graphics_queue,
            command_pool,
            render_pass,
            imgui,
            Some(Options {
                in_flight_frames: MAX_FRAMES_IN_FLIGHT,
                ..Default::default()
            }),
        )
        .context("failed to create imgui renderer")?;

        let framebuffers = swapchain.create_framebuffers(device, render_pass, None, None)?;

        tracing::debug!("initialized imgui renderer successfully");

        Ok(Self {
            render_pass,
            renderer: ManuallyDrop::new(renderer),
            framebuffers,
        })
    }

    /// Recreate `imgui` framebuffers with a new swapchain.
    pub(crate) fn recreate_framebuffers(
        &mut self,
        device: &Device,
        swapchain: &Swapchain,
    ) -> Result<()> {
        for &framebuffer in &self.framebuffers {
            unsafe { device.destroy_framebuffer(framebuffer, None) };
        }
        self.framebuffers = swapchain.create_framebuffers(device, self.render_pass, None, None)?;
        Ok(())
    }

    /// Draw imgui for this frame.
    pub(crate) fn draw(
        &mut self,
        device: &Device,
        swapchain: &Swapchain,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        draw_data: &imgui::DrawData,
    ) -> Result<()> {
        // Commands
        unsafe {
            device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
        }
        let command_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device.begin_command_buffer(command_buffer, &command_begin_info)?;
        }

        let render_pass_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffers[image_index])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: swapchain.extent,
            });
        unsafe {
            device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );
        }

        self.renderer
            .cmd_draw(command_buffer, draw_data)
            .context("failed to draw imgui")?;

        unsafe {
            device.cmd_end_render_pass(command_buffer);
            device.end_command_buffer(command_buffer)?;
        }

        Ok(())
    }

    /// Destroy the `imgui` `Renderer` instance.
    pub(crate) unsafe fn destroy(&mut self, device: &Device) {
        ManuallyDrop::drop(&mut self.renderer);
        for &framebuffer in &self.framebuffers {
            device.destroy_framebuffer(framebuffer, None);
        }
        device.destroy_render_pass(self.render_pass, None);
    }
}

/// Create the gui [`vk::RenderPass`] instance.
fn create_render_pass(device: &Device, color_format: vk::Format) -> Result<vk::RenderPass> {
    tracing::debug!("creating gui render pass");

    // Attachments
    let color_attachment = vk::AttachmentDescription::builder()
        .format(color_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .build();

    // Subpasses
    let color_attachment_ref = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build();
    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(slice::from_ref(&color_attachment_ref));

    // Dependencies
    let dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .build();

    // Create
    let attachments = vec![color_attachment];
    let render_pass_create_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .subpasses(slice::from_ref(&subpass))
        .dependencies(slice::from_ref(&dependency));

    let render_pass = unsafe { device.create_render_pass(&render_pass_create_info, None) }
        .context("failed to create render pass")?;

    tracing::debug!("created gui render pass successfully");

    Ok(render_pass)
}
