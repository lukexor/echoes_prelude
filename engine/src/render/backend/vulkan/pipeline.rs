//! Vulkan pipelines.

use super::{
    device::Device,
    mesh::{MeshPushConstants, VertexInputDescription},
    shader,
};
use crate::{
    prelude::ShaderStage,
    render::{CullMode, FrontFace, RenderSettings},
    shader::{DEFAULT_FRAGMENT_SHADER, DEFAULT_VERTEX_SHADER},
    Result,
};
use anyhow::Context;
use ash::vk;
use std::{mem, slice};

pub(crate) mod set_layouts {
    pub(crate) const GLOBAL: usize = 0;
    pub(crate) const OBJECT: usize = 1;
    pub(crate) const TEXTURE: usize = 2;
}

/// Create default [`vk::Pipeline`]s.
pub(crate) fn create_default(
    device: &Device,
    viewport: vk::Viewport,
    scissor: vk::Rect2D,
    render_pass: vk::RenderPass,
    descriptor_set_layouts: &[vk::DescriptorSetLayout],
    settings: &RenderSettings,
) -> Result<(vk::PipelineLayout, Vec<vk::Pipeline>)> {
    // Shader Stages
    let vertex_shader_module = shader::create_module(
        device,
        "default_vertex",
        ShaderStage::Vertex,
        DEFAULT_VERTEX_SHADER.to_vec(),
    )?;
    let fragment_shader_module = shader::create_module(
        device,
        "default_fragment",
        ShaderStage::Fragment,
        DEFAULT_FRAGMENT_SHADER.to_vec(),
    )?;
    let vertex_stage_info =
        shader::build_stage_info(vk::ShaderStageFlags::VERTEX, vertex_shader_module)?;
    let fragment_stage_info =
        shader::build_stage_info(vk::ShaderStageFlags::FRAGMENT, fragment_shader_module)?;

    let vertex_description = VertexInputDescription::get();
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(&vertex_description.bindings)
        .vertex_attribute_descriptions(&vertex_description.attributes)
        .build();
    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false)
        .build();
    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
        .polygon_mode(if settings.wireframe {
            vk::PolygonMode::LINE
        } else {
            vk::PolygonMode::FILL
        })
        .line_width(settings.line_width)
        .cull_mode(settings.cull_mode.into())
        .front_face(settings.front_face.into())
        .rasterizer_discard_enable(false)
        .depth_clamp_enable(false)
        .depth_bias_enable(false)
        .depth_bias_constant_factor(0.0)
        .depth_bias_clamp(0.0)
        .depth_bias_slope_factor(0.0)
        .build();
    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(settings.sample_shading)
        .min_sample_shading(1.0) // Closer to 1 is smoother
        .rasterization_samples(if settings.sample_shading {
            device.info.msaa_samples
        } else {
            vk::SampleCountFlags::TYPE_1
        })
        .build();

    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
        .blend_enable(false) // TODO: maybe enable?
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        // NOTE: vk::BlendFactor::ONE instead of vk::BlendFactor::SRC_ALPHA is for pre-multiplied alpha
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD)
        .build();
    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false)
        .min_depth_bounds(0.0)
        .max_depth_bounds(1.0)
        .build();

    let push_constant_ranges = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(mem::size_of::<MeshPushConstants>() as u32)
        .build();
    let layout_info = vk::PipelineLayoutCreateInfo::builder()
        .push_constant_ranges(slice::from_ref(&push_constant_ranges))
        .set_layouts(descriptor_set_layouts);
    let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
        .context("failed to create graphics pipeline layout")?;

    let shader_stages = [vertex_stage_info, fragment_stage_info];
    let mut pipeline_builder = PipelineBuilder::new();
    pipeline_builder
        .shader_stages(&shader_stages)
        .vertex_input_state(vertex_input_state)
        .input_assembly_state(input_assembly_state)
        .viewport(viewport)
        .scissor(scissor)
        .rasterization_state(rasterization_state)
        .color_blend_attachment(color_blend_attachment)
        .depth_stencil_state(depth_stencil_state)
        .multisample_state(multisample_state)
        .layout(layout);

    let pipeline = pipeline_builder.build(device, render_pass)?;

    // Cleanup
    unsafe {
        device.destroy_shader_module(vertex_shader_module, None);
        device.destroy_shader_module(fragment_shader_module, None);
    }

    Ok((layout, vec![pipeline]))
}

#[derive(Default, Clone)]
#[must_use]
pub(crate) struct PipelineBuilder<'a> {
    pub(crate) shader_stages: &'a [vk::PipelineShaderStageCreateInfo],
    pub(crate) vertex_input_state: vk::PipelineVertexInputStateCreateInfo,
    pub(crate) input_assembly_state: vk::PipelineInputAssemblyStateCreateInfo,
    pub(crate) depth_stencil_state: vk::PipelineDepthStencilStateCreateInfo,
    pub(crate) viewport: vk::Viewport,
    pub(crate) scissor: vk::Rect2D,
    pub(crate) rasterization_state: vk::PipelineRasterizationStateCreateInfo,
    pub(crate) color_blend_attachment: vk::PipelineColorBlendAttachmentState,
    pub(crate) multisample_state: vk::PipelineMultisampleStateCreateInfo,
    pub(crate) layout: vk::PipelineLayout,
}

#[allow(unused)]
impl<'a> PipelineBuilder<'a> {
    /// Create a new `PipelineBuilder`.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Set [`vk::PipelineShaderStageCreateInfo`]s.
    #[inline]
    pub(crate) fn shader_stages(
        &mut self,
        shader_stages: &'a [vk::PipelineShaderStageCreateInfo],
    ) -> &mut Self {
        self.shader_stages = shader_stages;
        self
    }

    /// Set [`vk::PipelineVertexInputStateCreateInfo`].
    #[inline]
    pub(crate) fn vertex_input_state(
        &mut self,
        vertex_input_state: vk::PipelineVertexInputStateCreateInfo,
    ) -> &mut Self {
        self.vertex_input_state = vertex_input_state;
        self
    }

    /// Set [`vk::PipelineInputAssemblyStateCreateInfo`].
    #[inline]
    pub(crate) fn input_assembly_state(
        &mut self,
        input_assembly_state: vk::PipelineInputAssemblyStateCreateInfo,
    ) -> &mut Self {
        self.input_assembly_state = input_assembly_state;
        self
    }

    /// Set [`vk::PipelineDepthStencilStateCreateInfo`].
    #[inline]
    pub(crate) fn depth_stencil_state(
        &mut self,
        depth_stencil_state: vk::PipelineDepthStencilStateCreateInfo,
    ) -> &mut Self {
        self.depth_stencil_state = depth_stencil_state;
        self
    }

    /// Set [`vk::Viewport`].
    #[inline]
    pub(crate) fn viewport(&mut self, viewport: vk::Viewport) -> &mut Self {
        self.viewport = viewport;
        self
    }

    /// Set [`vk::Rect2D`].
    #[inline]
    pub(crate) fn scissor(&mut self, scissor: vk::Rect2D) -> &mut Self {
        self.scissor = scissor;
        self
    }

    /// Set [`vk::PipelineRasterizationStateCreateInfo`].
    #[inline]
    pub(crate) fn rasterization_state(
        &mut self,
        rasterization_state: vk::PipelineRasterizationStateCreateInfo,
    ) -> &mut Self {
        self.rasterization_state = rasterization_state;
        self
    }

    /// Set [`vk::PipelineColorBlendAttachmentState`].
    #[inline]
    pub(crate) fn color_blend_attachment(
        &mut self,
        color_blend_attachment: vk::PipelineColorBlendAttachmentState,
    ) -> &mut Self {
        self.color_blend_attachment = color_blend_attachment;
        self
    }

    /// Set [`vk::PipelineMultisampleStateCreateInfo`].
    #[inline]
    pub(crate) fn multisample_state(
        &mut self,
        multisample_state: vk::PipelineMultisampleStateCreateInfo,
    ) -> &mut Self {
        self.multisample_state = multisample_state;
        self
    }

    /// Set [`vk::PipelineLayout`].
    #[inline]
    pub(crate) fn layout(&mut self, layout: vk::PipelineLayout) -> &mut Self {
        self.layout = layout;
        self
    }

    /// Build [Pipeline].
    #[inline]
    pub(crate) fn build(
        &self,
        device: &ash::Device,
        render_pass: vk::RenderPass,
    ) -> Result<vk::Pipeline> {
        tracing::debug!("creating graphics pipeline");

        // Viewport State
        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(slice::from_ref(&self.viewport))
            .scissors(slice::from_ref(&self.scissor));

        let color_blend_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(slice::from_ref(&self.color_blend_attachment))
            .blend_constants([0.0; 4]);

        // Dynamic State
        let dynamic_states = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::LINE_WIDTH,
        ];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

        // Create
        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(self.shader_stages)
            .vertex_input_state(&self.vertex_input_state)
            .input_assembly_state(&self.input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&self.rasterization_state)
            .multisample_state(&self.multisample_state)
            .depth_stencil_state(&self.depth_stencil_state)
            .color_blend_state(&color_blend_info)
            .dynamic_state(&dynamic_state)
            // TODO: tessellation
            // .tessellation_state(&tessellation_state)
            .layout(self.layout)
            .render_pass(render_pass)
            .subpass(0);

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

        tracing::debug!("created graphics pipeline successfully");

        Ok(pipeline)
    }
}

/// Create global [`vk::DescriptorSetLayout`] and [`vk::DescriptorPool`] instances.
pub(crate) fn create_global_descriptor_pool(
    device: &ash::Device,
) -> Result<(Vec<vk::DescriptorSetLayout>, vk::DescriptorPool)> {
    // Pool
    tracing::debug!("creating descriptor pool");

    let camera_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .build();
    let scene_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(1)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
        .build();
    let object_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .build();
    let texture_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .build();

    let create_set_layout = |bindings| {
        let descriptor_set_create_info =
            vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);
        unsafe { device.create_descriptor_set_layout(&descriptor_set_create_info, None) }
            .context("failed to create descriptor set layout")
    };
    let global_bindings = [camera_binding, scene_binding];
    let object_bindings = [object_binding];
    let texture_bindings = [texture_binding];
    let mut set_layouts = vec![Default::default(); 3];
    set_layouts[set_layouts::GLOBAL] = create_set_layout(&global_bindings)?;
    set_layouts[set_layouts::OBJECT] = create_set_layout(&object_bindings)?;
    set_layouts[set_layouts::TEXTURE] = create_set_layout(&texture_bindings)?;

    let count = 10;
    let pool_sizes = [
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(count)
            .build(),
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .descriptor_count(count)
            .build(),
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(count)
            .build(),
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(count)
            .build(),
    ];

    let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&pool_sizes)
        .max_sets(count);

    let pool = unsafe { device.create_descriptor_pool(&descriptor_pool_info, None) }
        .context("failed to create descriptor pool")?;

    tracing::debug!("created descriptor pool successfully");

    Ok((set_layouts, pool))
}

impl From<CullMode> for vk::CullModeFlags {
    fn from(value: CullMode) -> Self {
        match value {
            CullMode::None => Self::NONE,
            CullMode::Front => Self::FRONT,
            CullMode::Back => Self::BACK,
            CullMode::All => Self::FRONT_AND_BACK,
        }
    }
}

impl From<FrontFace> for vk::FrontFace {
    fn from(value: FrontFace) -> Self {
        match value {
            FrontFace::Clockwise => Self::CLOCKWISE,
            FrontFace::CounterClockwise => Self::COUNTER_CLOCKWISE,
        }
    }
}
