#include "Engine.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <VkBootstrap.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <cmath>
#include <iostream>
#include <fstream>

#include "Swapchain.hpp"
#include "Image.hpp"
#include "PipelineBuilder.hpp"

Engine::Engine()
{
}

Engine::~Engine()
{
    if(!cleanedUp){
        cleanup();
    }
}

void Engine::cleanup(){
    vkDeviceWaitIdle(m_device);

    vkDestroyPipelineLayout(m_device, meshPipelineLayout, nullptr);
    vkDestroyPipeline(m_device, meshPipeline, nullptr);

    vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
	vkDestroyDescriptorSetLayout(m_device, m_descriptorLayout, nullptr);

    for (int i = 0; i < 2; i++) {
        vkDestroyCommandPool(m_device, frameData[i].commandPool, nullptr);

        vkDestroyFence(m_device, frameData[i].renderFence, nullptr);
        vkDestroySemaphore(m_device, frameData[i].renderSemaphore, nullptr);
        vkDestroySemaphore(m_device, frameData[i].swapchainSemaphore, nullptr);
    }

    drawImage->destroy();
    depthImage->destroy();

    vmaDestroyAllocator(m_allocator);

    m_swapchain->destroySwapchain();

	vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
	vkDestroyDevice(m_device, nullptr);
		
	vkb::destroy_debug_utils_messenger(m_instance, m_debugMessenger);
	vkDestroyInstance(m_instance, nullptr);
	SDL_DestroyWindow(m_pWindow);

    cleanedUp = true;
}

void Engine::init(){
    cleanedUp = false;
    initSDL3();
    
    initVulkan();
    
    initSwapchain();

    initDrawResources();
    
    initCommands();
    
    initSynchronization();    

    initDescriptors();

    initPipelines();
}
//Initialization
void Engine::initSDL3(){
    SDL_InitFlags SdlInitFlags = 0;
    SdlInitFlags |= SDL_INIT_VIDEO;
    
    SDL_WindowFlags SdlWindowFlags = 0;
    SdlWindowFlags |= SDL_WINDOW_VULKAN;
    
    SDL_Init(SdlInitFlags);
    m_pWindow = SDL_CreateWindow("Engine", 1600, 900, SdlWindowFlags);
}

void Engine::initVulkan(){
    vkb::InstanceBuilder instanceBuilder;

    vkb::SystemInfo sysInfo = vkb::SystemInfo::get_system_info().value();

    if (sysInfo.validation_layers_available && m_bUseValidation) {
        instanceBuilder.request_validation_layers();
    }
    if (sysInfo.debug_utils_available && m_bUseDebugMessenger) {
        instanceBuilder.use_default_debug_messenger();
    }
    instanceBuilder.require_api_version(1, 3, 0);

    vkb::Result<vkb::Instance> instanceBuilderRet = instanceBuilder.build();
    if (!instanceBuilderRet) {
        throw std::runtime_error(instanceBuilderRet.error().message() + "\n");
    }
    vkb::Instance vkbInstance = instanceBuilderRet.value();
    m_instance = vkbInstance.instance;

    if(vkbInstance.debug_messenger && m_bUseDebugMessenger){
        m_debugMessenger = vkbInstance.debug_messenger;
    }       

    
    if(!SDL_Vulkan_CreateSurface(m_pWindow, m_instance, nullptr, &m_surface)){
       const char* err = SDL_GetError();
       std::cout << err << std::endl;
    }

	VkPhysicalDeviceVulkan13Features features13{ 
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .synchronization2 = VK_TRUE,
        .dynamicRendering = VK_TRUE
    };

	VkPhysicalDeviceVulkan12Features features12{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .descriptorIndexing = VK_TRUE,
        .shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
        .shaderStorageBufferArrayNonUniformIndexing = VK_TRUE,
        .shaderStorageImageArrayNonUniformIndexing = VK_TRUE,

        .descriptorBindingSampledImageUpdateAfterBind = VK_TRUE,
        .descriptorBindingStorageImageUpdateAfterBind = VK_TRUE,
        .descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE,
        .descriptorBindingUpdateUnusedWhilePending = VK_TRUE,

        .descriptorBindingPartiallyBound = VK_TRUE,
        .runtimeDescriptorArray = VK_TRUE,
        .bufferDeviceAddress = VK_TRUE
    };

    vkb::PhysicalDeviceSelector vkbSelector{ vkbInstance };
	vkb::PhysicalDevice physicalDevice = vkbSelector
		.set_minimum_version(1, 3)
		.set_required_features_13(features13)
		.set_required_features_12(features12)
		.set_surface(m_surface)
		.select()
		.value();
    
    vkb::DeviceBuilder vkbBuilder { physicalDevice };

    vkb::Device vkbDevice = vkbBuilder.build().value();

    SAMPLER_COUNT = physicalDevice.properties.limits.maxDescriptorSetSamplers;
    STORAGE_COUNT = physicalDevice.properties.limits.maxDescriptorSetStorageBuffers;
    IMAGE_COUNT = physicalDevice.properties.limits.maxDescriptorSetStorageImages;

    m_physicalDevice = physicalDevice.physical_device;
    m_device = vkbDevice.device;

    m_graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    m_graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    VmaAllocatorCreateInfo allocatorInfo = {
        .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = m_physicalDevice,
        .device = m_device,
        .instance = m_instance
    };
    vmaCreateAllocator(&allocatorInfo, &m_allocator);
}

void Engine::initCommands(){

    VkCommandPoolCreateInfo commandPoolInfo =  {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr, 
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = m_graphicsQueueFamily
    };
	
	for (int i = 0; i < 2; i++) {

		vkCreateCommandPool(m_device, &commandPoolInfo, nullptr, &frameData[i].commandPool);

		VkCommandBufferAllocateInfo cmdAllocInfo = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = frameData[i].commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1
        };

		vkAllocateCommandBuffers(m_device, &cmdAllocInfo, &frameData[i].commandBuffer);
	}
}

void Engine::initSwapchain(){
    windowWidth = 1600;
    windowHeight = 900;
    m_swapchain = std::make_unique<Swapchain>(m_device, m_physicalDevice, m_surface);
    m_swapchain->createSwapchain(windowWidth, windowHeight);
}

void Engine::initDrawResources(){
    //Initialize images
    VkFormat drawImgFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkFormat depthImgFormat = VK_FORMAT_D32_SFLOAT;

    VkImageUsageFlags drawImgUsage = {};
    drawImgUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    drawImgUsage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    drawImgUsage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    VkImageUsageFlags depthImgUsage = {};
    depthImgUsage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VkExtent3D drawImgExtent = {
        .width = windowWidth,
        .height = windowHeight,
        .depth = 1
    };

    drawExtent.width = drawImgExtent.width;
    drawExtent.height = drawImgExtent.height;

    drawImage = std::make_unique<Image>(m_device, m_allocator, 
        drawImgExtent, drawImgFormat, drawImgUsage,
        VK_IMAGE_ASPECT_COLOR_BIT,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
    );

    depthImage = std::make_unique<Image>(m_device, m_allocator, 
        drawImgExtent, depthImgFormat, depthImgUsage,
        VK_IMAGE_ASPECT_DEPTH_BIT,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
    );
}

void Engine::initSynchronization(){
    VkFenceCreateInfo fenceInfo = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT
    };

    VkSemaphoreCreateInfo semInfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = nullptr,
    };
    
	for (int i = 0; i < 2; i++) {
		vkCreateFence(m_device, &fenceInfo, nullptr, &frameData[i].renderFence);

		vkCreateSemaphore(m_device, &semInfo, nullptr, &frameData[i].swapchainSemaphore);
		vkCreateSemaphore(m_device, &semInfo, nullptr, &frameData[i].renderSemaphore);
	}
}

void Engine::initDescriptors(){
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, STORAGE_COUNT},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, SAMPLER_COUNT},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, IMAGE_COUNT},
    };
    
    VkDescriptorPoolCreateInfo descriptorPoolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        .maxSets = 1,
        .poolSizeCount = (uint32_t)poolSizes.size(),
        .pPoolSizes = poolSizes.data()        
    };

    vkCreateDescriptorPool(m_device, &descriptorPoolInfo, nullptr, &m_descriptorPool);

    VkDescriptorSetLayoutBinding storage = {
        .binding = STORAGE_BINDING,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = STORAGE_COUNT,
        .stageFlags = VK_SHADER_STAGE_ALL
    };
    VkDescriptorSetLayoutBinding sampler = {
        .binding = SAMPLER_BINDING,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = SAMPLER_COUNT,
        .stageFlags = VK_SHADER_STAGE_ALL
    };
    VkDescriptorSetLayoutBinding images = {
        .binding = IMAGE_BINDING,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = IMAGE_COUNT,
        .stageFlags = VK_SHADER_STAGE_ALL
    };

    std::vector<VkDescriptorBindingFlags> bindingFlags = {
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT
    };
    VkDescriptorSetLayoutBindingFlagsCreateInfo layoutBindingFlags = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .pNext = nullptr,
        .bindingCount = (uint32_t)bindingFlags.size(),
        .pBindingFlags = bindingFlags.data()
    };

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        storage, sampler, images
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &layoutBindingFlags,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = (uint32_t)bindings.size(),
        .pBindings = bindings.data()
    };

    vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descriptorLayout);

    VkDescriptorSetAllocateInfo setInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = m_descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &m_descriptorLayout
    };

    vkAllocateDescriptorSets(m_device, &setInfo, &m_descriptorSet);
}

void Engine::initPipelines(){
    pb = std::make_unique<PipelineBuilder>();
    initMeshPipeline();    
}

void Engine::initMeshPipeline(){
    VkShaderModule frag;
	if (!loadShader(&frag, "../shaders/simple.frag.spv")) {
		std::cout << "Error when building the triangle fragment shader module" << std::endl;
	}
	else {
		std::cout << "Built the triangle fragment shader module" << std::endl;
	}
    VkShaderModule vert;
	if (!loadShader(&vert, "../shaders/simple.vert.spv")) {
		std::cout << "Error when building the triangle vertex shader module" << std::endl;
	}
	else {
		std::cout << "Built the triangle vertex shader module" << std::endl;
	}

    VkPushConstantRange pc = {
        .stageFlags = VK_SHADER_STAGE_ALL,
        .offset = 0,
        .size = sizeof(PushConstants)
    };

	VkPipelineLayoutCreateInfo layoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &m_descriptorLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pc
    };
	vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &meshPipelineLayout);

    
	pb->clear();
	pb->pipeline_layout = meshPipelineLayout;
	pb->setShaders(vert, frag);
	pb->setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
	pb->setPolygonMode(VK_POLYGON_MODE_FILL);
	pb->setCullingMode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_CLOCKWISE);
	pb->setMultisamplingNone();
	pb->disableBlending();
	pb->enableDepthtest(VK_TRUE, VK_COMPARE_OP_GREATER_OR_EQUAL);
	pb->setColorAttachmentFormat(drawImage->format);
	pb->setDepthAttachmentFormat(depthImage->format);
	meshPipeline = pb->buildPipeline(m_device);

	vkDestroyShaderModule(m_device, vert, nullptr);
	vkDestroyShaderModule(m_device, frag, nullptr);
}
//Utility
bool Engine::loadShader(VkShaderModule* outShader, const char* filePath) {
	std::ifstream file(filePath, std::ios::ate | std::ios::binary);
	if (!file.is_open()) {
		return false;
	}
	size_t fileSize = (size_t)file.tellg();
	std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

	file.seekg(0);
	file.read((char*)buffer.data(), fileSize);
	file.close();

	VkShaderModuleCreateInfo shader = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .codeSize = buffer.size() * sizeof(uint32_t),
        .pCode = buffer.data()
    };

	if(vkCreateShaderModule(m_device, &shader, nullptr, outShader) != VK_SUCCESS){
        return false;
    }
    return true;
}

//Drawing
void Engine::draw(){
    VkResult fenceResult = vkWaitForFences(m_device, 1, &getCurrentFrame().renderFence, VK_TRUE, 1000000000);
    if (fenceResult != VK_SUCCESS) {
        throw std::runtime_error("Fence wait failed!");
    }    
	uint32_t index;
	VkResult acquireResult = vkAcquireNextImageKHR(m_device, m_swapchain->swapchain, 1000000000, getCurrentFrame().swapchainSemaphore, VK_NULL_HANDLE, &index);
	if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        int32_t w = 0;
        int32_t h = 0;
		SDL_GetWindowSizeInPixels(m_pWindow, &w, &h);
        m_swapchain->resizeSwapchain(w, h);
		return;
	}
	else if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acqurie swapchain image!");
	}
    
    vkResetFences(m_device, 1, &getCurrentFrame().renderFence);
    VkCommandBuffer cmd = getCurrentFrame().commandBuffer;
    vkResetCommandBuffer(cmd, 0);
    
    VkCommandBufferBeginInfo commandBufferBeginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr
    };

    vkBeginCommandBuffer(cmd, &commandBufferBeginInfo);

 	drawImage->transitionTo(cmd, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    depthImage->transitionTo(cmd, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    //Draw
    drawMeshes(cmd);

    drawImage->transitionTo(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
	m_swapchain->images.at(index).transitionTo(cmd, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    drawImage->copyTo(cmd, m_swapchain->images.at(index));
    m_swapchain->images.at(index).transitionTo(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

	vkEndCommandBuffer(cmd);

    VkCommandBufferSubmitInfo commandBufferSubmitInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext = nullptr,
        .commandBuffer = cmd,
        .deviceMask = 0
    };
    VkSemaphoreSubmitInfo semaphoreWaitSubmitInfo{ 
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .semaphore = getCurrentFrame().swapchainSemaphore,
        .value = 1,
        .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR,
        .deviceIndex = 0
    };
	VkSemaphoreSubmitInfo semaphoreSignalSubmitInfo{ 
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .semaphore = getCurrentFrame().renderSemaphore,
        .value = 1,
        .stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
        .deviceIndex = 0
    };
    VkSubmitInfo2 queueSubmitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext = nullptr,
        .flags = 0,
        .waitSemaphoreInfoCount = 1,
        .pWaitSemaphoreInfos = &semaphoreWaitSubmitInfo,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &commandBufferSubmitInfo,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &semaphoreSignalSubmitInfo
    };

    VkResult submitRes = vkQueueSubmit2(m_graphicsQueue, 1, &queueSubmitInfo, getCurrentFrame().renderFence);
    if(submitRes != VK_SUCCESS){
        std::cout << "queue submit failed" << std::endl;
    }

    VkPresentInfoKHR presentInfo = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &getCurrentFrame().renderSemaphore,
        .swapchainCount = 1,
        .pSwapchains = &m_swapchain->swapchain,
        .pImageIndices = &index
    };

	VkResult presentRes = vkQueuePresentKHR(m_graphicsQueue, &presentInfo);
	if (presentRes == VK_ERROR_OUT_OF_DATE_KHR || presentRes == VK_SUBOPTIMAL_KHR) {
        //resizeSwwapchain
        std::cout << "needs to resize" << std::endl;
	}
	else if (presentRes != VK_SUCCESS) {
		std::cout << "Failed to present to swapchain" << std::endl;
	};
	frameNumber++;
}

void Engine::drawMeshes(VkCommandBuffer cmd){
    VkRenderingAttachmentInfo color_attachment = {};
    color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color_attachment.pNext = nullptr;
    color_attachment.imageView = drawImage->view;
    color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    
    VkRenderingAttachmentInfo depth_attachment = {};
    depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth_attachment.pNext = nullptr;
    depth_attachment.imageView = depthImage->view;
    depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_attachment.clearValue.depthStencil.depth = 0.0f;
    
    VkRenderingInfo rendering_info = {};
    rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering_info.pNext = nullptr;
    rendering_info.renderArea = VkRect2D{ VkOffset2D { 0, 0 }, drawExtent };
    rendering_info.layerCount = 1;
    rendering_info.colorAttachmentCount = 1;
    rendering_info.pColorAttachments = &color_attachment;
    rendering_info.pDepthAttachment = &depth_attachment;
    rendering_info.pStencilAttachment = nullptr;
    
    vkCmdBeginRendering(cmd, &rendering_info);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);
    
    VkViewport viewport = {};
    viewport.x = 0;
    viewport.y = 0;
    viewport.width = (float)drawExtent.width;
    viewport.height = (float)drawExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    
    VkRect2D scissor = {};
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent.width = drawExtent.width;
    scissor.extent.height = drawExtent.height;
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    
    
    // PushConstants pcs;
    // for (const MeshData& mesh : meshes) {
    //     pcs.vb_addr = mesh.vertex_buffer_address;
    //     pcs.model = mesh.model_mat;
    //     vkCmdPushConstants(cmd, mesh_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants), &pcs);
    //     vkCmdBindIndexBuffer(cmd, mesh.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    //     vkCmdDrawIndexed(cmd, mesh.index_count, 1, 0, 0, 0);
    // }

    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRendering(cmd);
}

void Engine::run(){
    SDL_Event e;
    bool quit = false;
    while(!quit){

        while(SDL_PollEvent(&e) != 0){
            switch (e.type)
            {
            case SDL_EVENT_QUIT:
                quit = true;
                break;
            case SDL_EVENT_WINDOW_MINIMIZED:
                //stopRendering =true
                break;
            case SDL_EVENT_WINDOW_MAXIMIZED:
                //stopRendering = false;
                break;
            case SDL_EVENT_KEY_DOWN:
                std::cout << "Pressed Key: " << SDL_GetKeyName(e.key.key) << std::endl;
            default:
                break;
            }
        }

        //stopRendering
            //continue
        
        //future imgui stuff
        //ImplVulkanNewFrame, ImpleSDL3NewFrame
        
        draw();
    }
}