#include "PipelineBuilder.hpp"

void PipelineBuilder::clear(){
    shader_stages.clear();

    input_assembly = {};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;

    rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;

    color_blending = {};

    multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;

    depth_stencil = {};
    depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

    rendering_info = {};
    rendering_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;

    color_attachment_format = VK_FORMAT_UNDEFINED;

    pipeline_layout = {};
}
void PipelineBuilder::setShaders(VkShaderModule vertex_shader, VkShaderModule fragment_shader) {
    shader_stages.clear();
    VkPipelineShaderStageCreateInfo vertex_info = {};
    vertex_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertex_info.pNext = nullptr;
    vertex_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertex_info.module = vertex_shader;
    vertex_info.pName = "main";

    VkPipelineShaderStageCreateInfo fragment_info = {};
    fragment_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragment_info.pNext = nullptr;
    fragment_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragment_info.module = fragment_shader;
    fragment_info.pName = "main";

    shader_stages = {vertex_info, fragment_info};
}
void PipelineBuilder::setTopology(VkPrimitiveTopology topology) {
    input_assembly.topology = topology;
    input_assembly.primitiveRestartEnable = VK_FALSE;
}
void PipelineBuilder::setPolygonMode(VkPolygonMode mode) {
    rasterizer.polygonMode = mode;
    rasterizer.lineWidth = 1.0f;
}
void PipelineBuilder::setCullingMode(VkCullModeFlags mode, VkFrontFace face) {
    rasterizer.cullMode = mode;
    rasterizer.frontFace = face;
}
void PipelineBuilder::setMultisamplingNone() {
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f;
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;
}
void PipelineBuilder::disableBlending() {
    color_blending.blendEnable = VK_FALSE;
    color_blending.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
}
/*
VK_BLEND_FACTOR_ONE = additive
VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA = alphablend
*/
void PipelineBuilder::enableBlending(VkBlendFactor dst_blend_factor) {
    color_blending.blendEnable = VK_TRUE;
    color_blending.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    color_blending.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    color_blending.dstColorBlendFactor = dst_blend_factor;
    color_blending.colorBlendOp = VK_BLEND_OP_ADD;

    color_blending.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    color_blending.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    color_blending.alphaBlendOp = VK_BLEND_OP_ADD;

}
void PipelineBuilder::disableDepthtest() {
    depth_stencil.depthTestEnable = VK_FALSE;
    depth_stencil.depthWriteEnable = VK_FALSE;
    depth_stencil.depthCompareOp = VK_COMPARE_OP_NEVER;
    depth_stencil.depthBoundsTestEnable = VK_FALSE;
    depth_stencil.stencilTestEnable = VK_FALSE;
    depth_stencil.front = {};
    depth_stencil.back = {};
    depth_stencil.minDepthBounds = 0.0f;
    depth_stencil.maxDepthBounds = 1.0f;
}
void PipelineBuilder::enableDepthtest(VkBool32 depth_write_enable, VkCompareOp op) {
    depth_stencil.depthTestEnable = VK_TRUE;
    depth_stencil.depthWriteEnable = depth_write_enable;
    depth_stencil.depthCompareOp = op;
    depth_stencil.depthBoundsTestEnable = VK_FALSE;
    depth_stencil.stencilTestEnable = VK_FALSE;
    depth_stencil.front = {};
    depth_stencil.back = {};
    depth_stencil.minDepthBounds = 0.0f;
    depth_stencil.maxDepthBounds = 1.0f;
}
void PipelineBuilder::setColorAttachmentFormat(VkFormat format) {
    color_attachment_format = format;
    rendering_info.colorAttachmentCount = 1;
    rendering_info.pColorAttachmentFormats = &color_attachment_format;
}
void PipelineBuilder::setDepthAttachmentFormat(VkFormat format) {
    rendering_info.depthAttachmentFormat = format;
}
VkPipeline PipelineBuilder::buildPipeline(VkDevice device) {
    VkPipelineViewportStateCreateInfo viewport_state = {};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.pNext = nullptr;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    //Only rendering to one color attachment, no blending
    VkPipelineColorBlendStateCreateInfo colorblending_info = {};
    colorblending_info.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorblending_info.pNext = nullptr;
    colorblending_info.logicOpEnable = VK_FALSE;
    colorblending_info.logicOp = VK_LOGIC_OP_COPY;
    colorblending_info.attachmentCount = 1;
    colorblending_info.pAttachments = &color_blending;

    VkPipelineVertexInputStateCreateInfo vertex_input_info = {};
    vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    //Define dynamic state
    VkDynamicState state[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamic_state_info = {};
    dynamic_state_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state_info.pDynamicStates = &state[0];
    dynamic_state_info.dynamicStateCount = 2;

    VkGraphicsPipelineCreateInfo graphics_pipeline_info = {};
    graphics_pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    //set pnext to render info for dynamic rendering
    graphics_pipeline_info.pNext = &rendering_info;

    graphics_pipeline_info.stageCount = (uint32_t)shader_stages.size();
    graphics_pipeline_info.pStages = shader_stages.data();
    graphics_pipeline_info.pVertexInputState = &vertex_input_info;
    graphics_pipeline_info.pInputAssemblyState = &input_assembly;
    graphics_pipeline_info.pViewportState = &viewport_state;
    graphics_pipeline_info.pRasterizationState = &rasterizer;
    graphics_pipeline_info.pMultisampleState = &multisampling;
    graphics_pipeline_info.pColorBlendState = &colorblending_info;
    graphics_pipeline_info.pDepthStencilState = &depth_stencil;
    graphics_pipeline_info.layout = pipeline_layout;
    graphics_pipeline_info.pDynamicState = &dynamic_state_info;

    VkPipeline pipeline;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &graphics_pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
        return VK_NULL_HANDLE; // failed to create graphics pipeline
    }
    else {
        return pipeline;
    }
};