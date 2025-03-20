#ifndef PIPELINEBUILDER_HPP
#define PIPELINEBUILDER_HPP

#include <vulkan/vulkan.h>
#include <vector>

class PipelineBuilder {
    public:
        std::vector<VkPipelineShaderStageCreateInfo> shader_stages;
        VkPipelineInputAssemblyStateCreateInfo input_assembly;
        VkPipelineRasterizationStateCreateInfo rasterizer;
        VkPipelineColorBlendAttachmentState color_blending;
        VkPipelineMultisampleStateCreateInfo multisampling;
        VkPipelineDepthStencilStateCreateInfo depth_stencil;
        VkPipelineRenderingCreateInfo rendering_info;
        VkFormat color_attachment_format;
        VkPipelineLayout pipeline_layout;
        
        PipelineBuilder() { clear(); };

        void clear();
        void setShaders(VkShaderModule vertex_shader, VkShaderModule fragment_shader);

        void setTopology(VkPrimitiveTopology topology);
        void setPolygonMode(VkPolygonMode mode);
        void setCullingMode(VkCullModeFlags mode, VkFrontFace face);
        void setMultisamplingNone();

        void disableBlending();
        void enableBlending(VkBlendFactor dst_blend_factor);

        void disableDepthtest();
        void enableDepthtest(VkBool32 depth_write_enable, VkCompareOp op);

        void setColorAttachmentFormat(VkFormat format);
        void setDepthAttachmentFormat(VkFormat format);

        VkPipeline buildPipeline(VkDevice device);
};

#endif