#include "Engine.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <VkBootstrap.h>

#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
#define STB_IMAGE_IMPLEMENTATION
#include "../thirdparty/stb_image.h"

#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>

#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <variant>
#include <random>

#include "Swapchain.hpp"
#include "Camera.hpp"
#include "Image.hpp"
#include "Buffer.hpp"
#include "PipelineBuilder.hpp"
#include "MeshLoader.hpp"

#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#define PARTICLE_COUNT 1000000
                                                          
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

    meshThread.join();
    
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    vkDestroyPipelineLayout(m_device, meshPipelineLayout, nullptr);
    vkDestroyPipeline(m_device, meshPipelineOpaque, nullptr);
    vkDestroyPipeline(m_device, meshPipelineTransparent, nullptr);
    vkDestroyPipelineLayout(m_device, particleDrawPipelineLayout, nullptr);
    vkDestroyPipelineLayout(m_device, particleComputePipelineLayout, nullptr);
    vkDestroyPipeline(m_device, particleDrawPipeline, nullptr);
    vkDestroyPipeline(m_device, particleComputePipeline, nullptr);
    
    
    vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
    vkDestroyDescriptorPool(m_device, m_ImguiPool, nullptr);
	vkDestroyDescriptorSetLayout(m_device, m_descriptorLayout, nullptr);
    
    for (int i = 0; i < 2; i++) {
        vkDestroyCommandPool(m_device, frameData[i].graphicsCommandPool, nullptr);
        vkDestroyCommandPool(m_device, frameData[i].computeCommandPool, nullptr);

        vkDestroyFence(m_device, frameData[i].renderFence, nullptr);
        vkDestroySemaphore(m_device, frameData[i].renderSemaphore, nullptr);
        vkDestroySemaphore(m_device, frameData[i].swapchainSemaphore, nullptr);
    }
    
    vkDestroyCommandPool(m_device, m_immTransfer.pool, nullptr);
    vkDestroyFence(m_device, m_immTransfer.fence, nullptr);

    drawImage->destroy();
    resolveImage->destroy();
    depthImage->destroy();

    for(MeshAsset& mesh : testMeshes){
        mesh.data.indexBuffer->destroy();
        mesh.data.vertexBuffer->destroy();
    }

    for(auto& [sceneName, scene] : loadedGLTFs){
        for(VkSampler sampler : scene->samplers){
            vkDestroySampler(m_device, sampler, nullptr);
        }
        
        for(auto& [meshName, mesh] : scene->meshes){
            mesh->data.indexBuffer->destroy();
            mesh->data.vertexBuffer->destroy();
        }

        for(auto& [imageName, image] : scene->images){
            image->destroy();
        }
    }

    for(const TextureData& td : textures){
        td.texture->destroy();
    }

    uboBuffer->destroy();
    materialBuffer->destroy();
    lightBuffer->destroy();

    hostPositionBuffer->destroy();
    hostVelocityBuffer->destroy();
    devicePositionBufferA->destroy();
    devicePositionBufferB->destroy();
    deviceVelocityBuffer->destroy();

    checkerboardImage->destroy();

    vkDestroySampler(m_device, defaultLinearSampler, nullptr);
    vkDestroySampler(m_device, defaultNearestSampler, nullptr);

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

    initData();

    initDearImGui();

    initializationTime = SDL_GetTicksNS();
    stats.initTime = (float)initializationTime / 1e6f;
}
//Initialization
void Engine::initSDL3(){
    SDL_InitFlags SdlInitFlags = 0;
    SdlInitFlags |= SDL_INIT_VIDEO;
    
    SDL_WindowFlags SdlWindowFlags = 0;
    SdlWindowFlags |= SDL_WINDOW_VULKAN;
    SdlWindowFlags |= SDL_WINDOW_RESIZABLE;
    
    SDL_Init(SdlInitFlags);
    m_pWindow = SDL_CreateWindow("Sun", 1600, 900, SdlWindowFlags);
    mouseCaptured = false;
    SDL_SetWindowRelativeMouseMode(m_pWindow, mouseCaptured);
    SDL_CaptureMouse(mouseCaptured);
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
    uint32_t sampler = physicalDevice.properties.limits.maxDescriptorSetSamplers;
    uint32_t buffer = physicalDevice.properties.limits.maxDescriptorSetStorageBuffers;
    uint32_t images = physicalDevice.properties.limits.maxDescriptorSetStorageImages;
    if(sampler < SAMPLER_COUNT){
        SAMPLER_COUNT = sampler;
    }
    if(buffer < STORAGE_COUNT){
        STORAGE_COUNT = buffer;
    }
    if(images < IMAGE_COUNT){
        IMAGE_COUNT = images;
    }

    m_physicalDevice = physicalDevice.physical_device;
    m_device = vkbDevice.device;

    m_graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    m_graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();
    m_computeQueue = vkbDevice.get_queue(vkb::QueueType::compute).value();
    m_computeQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::compute).value();
    m_transferQueue = vkbDevice.get_queue(vkb::QueueType::transfer).value();
    m_transferQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::transfer).value();

    VmaAllocatorCreateInfo allocatorInfo = {
        .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = m_physicalDevice,
        .device = m_device,
        .instance = m_instance
    };
    vmaCreateAllocator(&allocatorInfo, &m_allocator);
}

void Engine::initCommands(){

    VkCommandPoolCreateInfo graphicsCommandPoolInfo =  {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr, 
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = m_graphicsQueueFamily
    };

    VkCommandPoolCreateInfo computeCommandPoolInfo =  {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr, 
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = m_computeQueueFamily
    };
	
	for (int i = 0; i < 2; i++) {

		vkCreateCommandPool(m_device, &graphicsCommandPoolInfo, nullptr, &frameData[i].graphicsCommandPool);
		vkCreateCommandPool(m_device, &computeCommandPoolInfo, nullptr, &frameData[i].computeCommandPool);

		VkCommandBufferAllocateInfo cmdAllocInfo = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = frameData[i].graphicsCommandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1
        };

		vkAllocateCommandBuffers(m_device, &cmdAllocInfo, &frameData[i].graphicsCommandBuffer);

        cmdAllocInfo.commandPool = frameData[i].computeCommandPool;
		vkAllocateCommandBuffers(m_device, &cmdAllocInfo, &frameData[i].computeCommandBuffer);
	}

    VkCommandPoolCreateInfo transferCommandPoolInfo =  {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr, 
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = m_transferQueueFamily
    };
    vkCreateCommandPool(m_device, &transferCommandPoolInfo, nullptr, &m_immTransfer.pool);
    VkCommandBufferAllocateInfo cmdAllocInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = m_immTransfer.pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    vkAllocateCommandBuffers(m_device, &cmdAllocInfo, &m_immTransfer.buffer);
    
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
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        VK_SAMPLE_COUNT_4_BIT, 1
    );

    resolveImage = std::make_unique<Image>(m_device, m_allocator, 
        drawImgExtent, drawImgFormat, drawImgUsage,
        VK_IMAGE_ASPECT_COLOR_BIT,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        VK_SAMPLE_COUNT_1_BIT, 1
    );

    depthImage = std::make_unique<Image>(m_device, m_allocator, 
        drawImgExtent, depthImgFormat, depthImgUsage,
        VK_IMAGE_ASPECT_DEPTH_BIT,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        VK_SAMPLE_COUNT_4_BIT, 1
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
		vkCreateFence(m_device, &fenceInfo, nullptr, &frameData[i].computeFence);

		vkCreateSemaphore(m_device, &semInfo, nullptr, &frameData[i].swapchainSemaphore);
		vkCreateSemaphore(m_device, &semInfo, nullptr, &frameData[i].renderSemaphore);
	}

    vkCreateFence(m_device, &fenceInfo, nullptr, &m_immTransfer.fence);

    VkSemaphoreTypeCreateInfo semTypeInfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = 0
    };
    VkSemaphoreCreateInfo timelineInfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &semTypeInfo
    };
    vkCreateSemaphore(m_device, &timelineInfo, nullptr, &particleTLSemaphore);

}

void Engine::initDescriptors(){
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, STORAGE_COUNT},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, SAMPLER_COUNT},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, IMAGE_COUNT},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
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
    VkDescriptorSetLayoutBinding uniforms = {
        .binding = UBO_BINDING,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_ALL
    };

    std::vector<VkDescriptorBindingFlags> bindingFlags = {
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        0
    };
    VkDescriptorSetLayoutBindingFlagsCreateInfo layoutBindingFlags = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .pNext = nullptr,
        .bindingCount = (uint32_t)bindingFlags.size(),
        .pBindingFlags = bindingFlags.data()
    };

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        storage, sampler, images, uniforms
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
    initMeshPipelines();
    initParticlePipelines();
}

void Engine::initMeshPipelines(){
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
	pb->setCullingMode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	pb->setMultisampling(VK_SAMPLE_COUNT_4_BIT);
    // pb->setMultisamplingNone();
	pb->disableBlending();
	pb->enableDepthtest(VK_TRUE, VK_COMPARE_OP_GREATER_OR_EQUAL);
    // pb->disableDepthtest();
	pb->setColorAttachmentFormat(drawImage->format);
	pb->setDepthAttachmentFormat(depthImage->format);
	meshPipelineOpaque = pb->buildPipeline(m_device);

    pb->enableBlending(VK_BLEND_FACTOR_ONE);
    meshPipelineTransparent = pb->buildPipeline(m_device);

	vkDestroyShaderModule(m_device, vert, nullptr);
	vkDestroyShaderModule(m_device, frag, nullptr);
}

void Engine::initParticlePipelines(){
    VkShaderModule frag;
	if (!loadShader(&frag, "../shaders/particle.frag.spv")) {
		std::cout << "Error when building the particle fragment shader module" << std::endl;
	}
	else {
		std::cout << "Built the particle fragment shader module" << std::endl;
	}
    VkShaderModule vert;
	if (!loadShader(&vert, "../shaders/particle.vert.spv")) {
		std::cout << "Error when building the particle vertex shader module" << std::endl;
	}
	else {
		std::cout << "Built the particle vertex shader module" << std::endl;
	}

    VkShaderModule comp;
	if (!loadShader(&comp, "../shaders/particle_lorenz.comp.spv")) {
		std::cout << "Error when building the particle compute shader module" << std::endl;
	}
	else {
		std::cout << "Built the particle compute shader module" << std::endl;
	}

    VkPushConstantRange pcDraw = {
        .stageFlags = VK_SHADER_STAGE_ALL,
        .offset = 0,
        .size = sizeof(ParticlePushConstants)
    };    
    VkPushConstantRange pcComp = {
        .stageFlags = VK_SHADER_STAGE_ALL,
        .offset = 0,
        .size = sizeof(ParticleComputePushConstants)
    };

	VkPipelineLayoutCreateInfo layoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &m_descriptorLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcDraw
    };
	vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &particleDrawPipelineLayout);
    layoutInfo.pPushConstantRanges = &pcComp;
	vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &particleComputePipelineLayout);

    VkPipelineShaderStageCreateInfo particleComputeStageInfo {
        .sType=  VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = comp,
        .pName = "main"
    };
    VkComputePipelineCreateInfo computePipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .stage = particleComputeStageInfo,
        .layout = particleComputePipelineLayout
    };
	vkCreateComputePipelines(m_device, nullptr, 1, &computePipelineCreateInfo, nullptr, &particleComputePipeline);
    
	pb->clear();
	pb->pipeline_layout = particleDrawPipelineLayout;
	pb->setShaders(vert, frag);
	pb->setTopology(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);
	pb->setPolygonMode(VK_POLYGON_MODE_FILL);
	pb->setCullingMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	pb->setMultisampling(VK_SAMPLE_COUNT_4_BIT);
	pb->enableBlending(VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA);
	pb->enableDepthtest(VK_TRUE, VK_COMPARE_OP_GREATER_OR_EQUAL);
	pb->setColorAttachmentFormat(drawImage->format);
	pb->setDepthAttachmentFormat(depthImage->format);
	particleDrawPipeline = pb->buildPipeline(m_device);

	vkDestroyShaderModule(m_device, vert, nullptr);
	vkDestroyShaderModule(m_device, frag, nullptr);
    vkDestroyShaderModule(m_device, comp, nullptr);
}

void Engine::initData(){
    cam = std::make_unique<Camera>((float)drawExtent.width, (float)drawExtent.height);
    
    //Create UBO Buffer
    uboBuffer = std::make_unique<Buffer>(m_device, m_allocator, sizeof(UniformBufferObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
    VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT );
    
    ubo.camPos = cam->getPos();
    ubo.viewproj = glm::mat4(1.0f) * cam->getProjMatrix() * cam->getViewMatrix();
    ubo.ambientColor = glm::vec4(0.2f, 0.2f, 0.2f, 1.0f);
    ubo.sunlightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f); //w is sunlight intensity
    ubo.sunlightDirection = glm::normalize(glm::vec4(1.0f, 1.0f, 1.0f, 0.0f));
    
    memcpy(uboBuffer->allocationInfo.pMappedData, &ubo, sizeof(UniformBufferObject));

    //Material Buffer
    materials.reserve(32);
    VkBufferUsageFlags materialUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    materialBuffer = std::make_unique<Buffer>(m_device, m_allocator, sizeof(MaterialData) * materials.capacity(), 
    materialUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);
    
    lightsPoint.reserve(32);
    VkBufferUsageFlags lightUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    lightBuffer = std::make_unique<Buffer>(m_device, m_allocator, sizeof(Light) * lightsPoint.capacity(), 
    lightUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);

    VkBufferDeviceAddressInfo bdaInfo {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = nullptr,
        .buffer = materialBuffer->buffer
    };
    materialBufferAddress = vkGetBufferDeviceAddress(m_device, &bdaInfo);
    bdaInfo.buffer = lightBuffer->buffer;
    lightBufferAddress = vkGetBufferDeviceAddress(m_device, &bdaInfo);


    //Create default texture
	uint32_t black =    glm::packUnorm4x8(glm::vec4(0, 0, 0, 0));	
	uint32_t magenta =  glm::packUnorm4x8(glm::vec4(1, 0, 1, 1));
	std::array<uint32_t, 16 *16 > pixels;
	for (int x = 0; x < 16; x++) {
        for (int y = 0; y < 16; y++) {
            pixels[y*16 + x] = ((x % 2) ^ (y % 2)) ? magenta : black;
		}
	}
    
	checkerboardImage = createImageFromData(pixels.data(), VkExtent3D{16, 16, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT, false);
    
    //Create texture samplers
    VkSamplerCreateInfo sampler = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST
    };
	vkCreateSampler(m_device, &sampler, nullptr, &defaultNearestSampler);
	sampler.magFilter = VK_FILTER_LINEAR;
	sampler.minFilter = VK_FILTER_LINEAR;
	vkCreateSampler(m_device, &sampler, nullptr, &defaultLinearSampler);


    //Create Particle Buffers
    initialPositions.resize(PARTICLE_COUNT);
    initialVelocities.resize(PARTICLE_COUNT);
    std::minstd_rand rng(std::random_device{}());
    std::uniform_real_distribution<float> unitDist(-1.0f, 1.0f);

    for (size_t i = 0; i < PARTICLE_COUNT; ++i) {
        initialPositions.at(i) = glm::vec4(unitDist(rng), unitDist(rng), unitDist(rng), 0);
        initialVelocities.at(i) = glm::vec4(0.0, 1.0, 1.05, 0);
    }

    hostPositionBuffer = std::make_unique<Buffer>(m_device, m_allocator, PARTICLE_COUNT * sizeof(glm::vec4), 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, 
        VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

    devicePositionBufferA = std::make_unique<Buffer>(m_device, m_allocator, PARTICLE_COUNT * sizeof(glm::vec4), 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);
    
    devicePositionBufferB = std::make_unique<Buffer>(m_device, m_allocator, PARTICLE_COUNT * sizeof(glm::vec4), 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);

    hostVelocityBuffer = std::make_unique<Buffer>(m_device, m_allocator, PARTICLE_COUNT * sizeof(glm::vec4), 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, 
        VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
    
    deviceVelocityBuffer = std::make_unique<Buffer>(m_device, m_allocator, PARTICLE_COUNT * sizeof(glm::vec4), 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);

    VkBufferDeviceAddressInfo positionAddressInfoA = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = nullptr,
        .buffer = devicePositionBufferA->buffer
    };
    VkBufferDeviceAddressInfo positionAddressInfoB = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = nullptr,
        .buffer = devicePositionBufferB->buffer
    };
    VkBufferDeviceAddressInfo velocityAddressInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = nullptr,
        .buffer = deviceVelocityBuffer->buffer
    };
    particlePosBufferAddressA = vkGetBufferDeviceAddress(m_device, &positionAddressInfoA);
    particlePosBufferAddressB = vkGetBufferDeviceAddress(m_device, &positionAddressInfoB);
    particleVelBufferAddress = vkGetBufferDeviceAddress(m_device, &velocityAddressInfo);
    
    if (hostPositionBuffer->allocationInfo.pMappedData == nullptr) {
        throw std::runtime_error("Host position buffer not mapped.");
    }
    memcpy(hostPositionBuffer->allocationInfo.pMappedData, initialPositions.data(), sizeof(glm::vec4) * PARTICLE_COUNT);
    if (hostVelocityBuffer->allocationInfo.pMappedData == nullptr) {
        throw std::runtime_error("Host position buffer not mapped.");
    }
    memcpy(hostVelocityBuffer->allocationInfo.pMappedData, initialVelocities.data(), sizeof(glm::vec4) * PARTICLE_COUNT);

    VkBufferCopy posCopy = {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = PARTICLE_COUNT * sizeof(glm::vec4)
    };
    prepImmediateTransfer();
    vkCmdCopyBuffer(m_immTransfer.buffer, hostPositionBuffer->buffer, devicePositionBufferA->buffer, 1, &posCopy);
    vkCmdCopyBuffer(m_immTransfer.buffer, hostPositionBuffer->buffer, devicePositionBufferB->buffer, 1, &posCopy);
    vkCmdCopyBuffer(m_immTransfer.buffer, hostVelocityBuffer->buffer, deviceVelocityBuffer->buffer, 1, &posCopy);
    submitImmediateTransfer();

    //Update descriptors
    VkDescriptorBufferInfo uboInfo = {
        .buffer = uboBuffer->buffer,
        .offset = 0,
        .range = sizeof(UniformBufferObject)
    };
    VkWriteDescriptorSet uboWrite {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr, 
        .dstSet = m_descriptorSet,
        .dstBinding = UBO_BINDING,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pImageInfo = nullptr,
        .pBufferInfo = &uboInfo,
        .pTexelBufferView = nullptr
    };
    vkUpdateDescriptorSets(m_device, 1, &uboWrite, 0, nullptr);

    meshThread = std::thread(&Engine::meshUploader, this);
    pathQueue.push("..\\..\\resources\\cube.glb");
    pathQueue.push("QUIT");
}

void Engine::initDearImGui(){
    VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE };

    VkDescriptorPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 1000,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize
    };

    vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_ImguiPool);

    ImGui::CreateContext();
    ImGui_ImplSDL3_InitForVulkan(m_pWindow);
    ImGui_ImplVulkan_InitInfo imguiVulkanInfo = {
        .Instance = m_instance,
        .PhysicalDevice = m_physicalDevice,
        .Device = m_device,
        .QueueFamily = m_graphicsQueueFamily,
        .Queue = m_graphicsQueue,
        .DescriptorPool = m_ImguiPool,
        .MinImageCount = 2,
        .ImageCount = 2,
        .MSAASamples = VK_SAMPLE_COUNT_1_BIT,
        .UseDynamicRendering = true
    };
    imguiVulkanInfo.PipelineRenderingCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .pNext = nullptr,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &m_swapchain->format
    };
    
    ImGui::CreateContext();
    ImGui_ImplVulkan_Init(&imguiVulkanInfo);
}

//Utility
bool Engine::loadShader(VkShaderModule* outShader, std::string filePath) {
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

std::shared_ptr<Image> Engine::createImageFromData(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped){
    size_t dataSize = size.depth * size.width * size.height * sizeof(float);
    VkBufferUsageFlags stagingUsage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    Buffer stagingBuffer = Buffer(m_device, m_allocator, dataSize, stagingUsage, 
        VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

	if (stagingBuffer.allocationInfo.pMappedData == nullptr) {
		throw std::runtime_error("Staging buffer not mapped.");
	}
	memcpy(stagingBuffer.allocationInfo.pMappedData, data, dataSize);

    std::unique_ptr<Image> image = std::make_unique<Image>(m_device, m_allocator, size, format, 
        usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 
        VK_IMAGE_ASPECT_COLOR_BIT, 0,
        VK_SAMPLE_COUNT_1_BIT, 1
    );


    prepImmediateTransfer();
    
    image->transitionTo(m_immTransfer.buffer, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, true);

    VkBufferImageCopy copyRegion = {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageExtent = size
    };
    copyRegion.imageSubresource = {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1
    };
    vkCmdCopyBufferToImage(m_immTransfer.buffer, stagingBuffer.buffer, image->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
    
    image->transitionTo(m_immTransfer.buffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, true);

	submitImmediateTransfer();

    stagingBuffer.destroy();
	return image;
}

MeshData Engine::uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices){
    size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
	size_t indexBufferSize = indices.size() * sizeof(uint32_t);

	MeshData mesh = {};
	mesh.indexCount = (uint32_t)indices.size();

    VkBufferUsageFlags vertexUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VkBufferUsageFlags indexUsage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBufferUsageFlags stagingUsage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    mesh.vertexBuffer = std::make_unique<Buffer>(m_device, m_allocator, vertexBufferSize, vertexUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);
    mesh.indexBuffer = std::make_unique<Buffer>(m_device, m_allocator, indexBufferSize, indexUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);
    std::unique_ptr<Buffer> stagingBuffer = std::make_unique<Buffer>(m_device, m_allocator, vertexBufferSize + indexBufferSize, stagingUsage, VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);


	VkBufferDeviceAddressInfo addressInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = nullptr,
        .buffer = mesh.vertexBuffer->buffer
    };
	mesh.vertexBufferAddress = vkGetBufferDeviceAddress(m_device, &addressInfo);
	
	if (stagingBuffer->allocationInfo.pMappedData == nullptr) {
		throw std::runtime_error("Staging buffer not mapped.");
	}

	memcpy(stagingBuffer->allocationInfo.pMappedData, vertices.data(), vertexBufferSize);
	memcpy((char*)stagingBuffer->allocationInfo.pMappedData + vertexBufferSize, indices.data(), indexBufferSize);

	prepImmediateTransfer();

	VkBufferCopy vert_copy = {};
	vert_copy.size = vertexBufferSize;
	vert_copy.srcOffset = 0;
	vert_copy.dstOffset = 0;

	vkCmdCopyBuffer(m_immTransfer.buffer, stagingBuffer->buffer, mesh.vertexBuffer->buffer, 1, &vert_copy);

	VkBufferCopy ind_copy = {};
	ind_copy.size = indexBufferSize;
	ind_copy.srcOffset = vertexBufferSize;
	ind_copy.dstOffset = 0;

	vkCmdCopyBuffer(m_immTransfer.buffer, stagingBuffer->buffer, mesh.indexBuffer->buffer, 1, &ind_copy);

	submitImmediateTransfer();

    stagingBuffer->destroy();
	return mesh;
}

void Engine::updateScene(){
    cam->update();

    ubo.camPos = cam->getPos();
    ubo.viewproj = glm::mat4(1.0f) * cam->getProjMatrix() * cam->getViewMatrix();
    ubo.materialBuffer = materialBufferAddress;
    ubo.lightBuffer = lightBufferAddress;
    ubo.numLights = lightsPoint.size();
    memcpy(uboBuffer->allocationInfo.pMappedData, &ubo, sizeof(UniformBufferObject));
}

void Engine::updateGUI(){
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
        if (ImGui::Begin("Stats")) {
            ImGui::Text("Initialization Time: %.4f ms", stats.initTime);
            ImGui::Text("Frame: %f ms", stats.frameTime);
            ImGui::Text("Draw: %f ms", stats.meshDrawTime);
            ImGui::Text("Update: %f ms", stats.sceneUpdateTime);
            ImGui::Text("Particles: %f ms", stats.particleTime);
            ImGui::Text("Tris: %i", stats.triCount);
            ImGui::Text("Draws: %i", stats.drawCallCount);
            ImGui::Text("Yaw: %f ", glm::degrees(cam->yaw));
            ImGui::Text("Pitch: %f ", glm::degrees(cam->pitch));
            ImGui::SliderFloat("timescale", &timeScale, 0.05f, 2.0f);
		}
		ImGui::End();
        ImGui::Render();
}

void Engine::createRenderablesFromNode(std::shared_ptr<GLTFNode> node){
    if(node->mesh != nullptr){
        for(auto surface : node->mesh->surfaces){
            Renderable r{
                .vertexBufferAddress = node->mesh->data.vertexBufferAddress,
                .indexBuffer = node->mesh->data.indexBuffer->buffer,
                .indexCount = surface.count,
                .firstIndex = surface.startIndex,
                .materialIndex = surface.matIndex,
                .modelMat = node->worldTransform,
                .bounds = surface.bounds
            };

            if(renderableVisible(r, ubo.viewproj)){    
                switch (surface.type)
                {
                    case RenderPass::OPAQUE:
                        opaqueRenderables.push_back(r);
                        break;
                    case RenderPass::TRANSPARENT:
                        transparentRenderables.push_back(r);
                        break;
                    default:
                        break;
                }
            }
        }
    }

    for(auto c : node->children){
        createRenderablesFromNode(c);
    }
}

void Engine::sortTransparentRenderables(){
    transparentRenderablesIndices.reserve(transparentRenderables.size());

    for (uint32_t i = 0; i < transparentRenderables.size(); i++) {
        transparentRenderablesIndices.push_back(i);
    }

    std::sort(transparentRenderablesIndices.begin(), transparentRenderablesIndices.end(), 
    [&](const auto& l, const auto& r) {
        const Renderable& L = transparentRenderables.at(l);
        const Renderable& R = transparentRenderables.at(r);
        
        return (glm::length2(L.bounds.origin - glm::vec3(ubo.camPos))) > (glm::length2(R.bounds.origin - glm::vec3(ubo.camPos)));
    });
}

bool Engine::renderableVisible(const Renderable& r, const glm::mat4& viewProj){
    //TODO replace with separating axis theorem (currently: edge/edge not handled and a problem at certain angles)
    std::array<glm::vec3, 8> corners {
        glm::vec3 {  1,  1,  1 },
        glm::vec3 {  1,  1, -1 },
        glm::vec3 {  1, -1,  1 },
        glm::vec3 {  1, -1, -1 },
        glm::vec3 { -1,  1,  1 },
        glm::vec3 { -1,  1, -1 },
        glm::vec3 { -1, -1,  1 },
        glm::vec3 { -1, -1, -1 },
    };
    glm::mat4 MVP = viewProj * r.modelMat;
    glm::vec3 min = {  1.5,  1.5,  1.5 };
    glm::vec3 max = { -1.5, -1.5, -1.5 };

    for (int c = 0; c < 8; c++) {
        glm::vec4 v = MVP * glm::vec4(r.bounds.origin + (corners[c] * r.bounds.extents), 1.0f);
        v.x = v.x / v.w;
        v.y = v.y / v.w;
        v.z = v.z / v.w;

        min = glm::min(glm::vec3 { v.x, v.y, v.z }, min);
        max = glm::max(glm::vec3 { v.x, v.y, v.z }, max);
    }

    // check the clip space box is within the view
    if (min.z > 1.0f || max.z < 0.0f || min.x > 1.0f || max.x < -1.0f || min.y > 1.0f || max.y < -1.0f) {
        return false;
    } else {
        return true;
    }
}


uint32_t Engine::addTexture(const TextureData& data, std::string name){
    if(texNameToIndex.find(name) != texNameToIndex.end()){
        return texNameToIndex.at(name);
    }
        
    uint32_t index = (uint32_t)textures.size();
    texNameToIndex.insert({name, index});

    VkDescriptorImageInfo samplerInfo = {
        .sampler = data.sampler,
        .imageView = data.texture->view,
        .imageLayout = data.texture->layout,
    };
    VkWriteDescriptorSet samplerWrite {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr, 
        .dstSet = m_descriptorSet,
        .dstBinding = SAMPLER_BINDING,
        .dstArrayElement = index,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &samplerInfo,
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr
    };
    vkUpdateDescriptorSets(m_device, 1, &samplerWrite, 0, nullptr);

    textures.push_back(data);
    return index;
}

uint32_t Engine::addMaterial(const MaterialData& data, std::string name){
    if(matNameToIndex.find(name) != matNameToIndex.end()){
        return matNameToIndex.at(name);
    }
    //Check if this material will cause buffer to resize
    bool willResize = false;
    if(materials.size() == materials.capacity()){
        willResize = true;
    }

    //Point staging copy to where next element will be pushed to
	VkBufferCopy stagingCopy = {
        .srcOffset = materials.size() * sizeof(MaterialData),
        .dstOffset = materials.size() * sizeof(MaterialData),
        .size = sizeof(MaterialData)
    };

    //Push next element into spot it'll be read
    uint32_t index = (uint32_t)materials.size();
    materials.push_back(data);
    matNameToIndex.insert({name, index});
    //If that push is going to cause a resize, resize the destination buffer then copy the whole buffer over 
    if(willResize){
        stagingCopy.srcOffset = 0;
        stagingCopy.dstOffset = 0;
        stagingCopy.size = materials.capacity() * sizeof(MaterialData);


        materialBuffer->destroy();
    
        VkBufferUsageFlags materialUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        
        materialBuffer = std::make_unique<Buffer>(m_device, m_allocator, materials.capacity() * sizeof(MaterialData), 
            materialUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);

        VkBufferDeviceAddressInfo bdaInfo {
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .pNext = nullptr,
            .buffer = materialBuffer->buffer
        };
        materialBufferAddress = vkGetBufferDeviceAddress(m_device, &bdaInfo);
    }    
    
    Buffer stagingBuffer = Buffer(m_device, m_allocator, materials.capacity() * sizeof(MaterialData), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
	if (stagingBuffer.allocationInfo.pMappedData == nullptr) {
		throw std::runtime_error("Material staging buffer not mapped.");
	}

	memcpy(stagingBuffer.allocationInfo.pMappedData, materials.data(), materials.size() * sizeof(MaterialData));
	prepImmediateTransfer();

	vkCmdCopyBuffer(m_immTransfer.buffer, stagingBuffer.buffer, materialBuffer->buffer, 1, &stagingCopy);

    submitImmediateTransfer();

    return index;
}

void Engine::addLights(const std::vector<Light>& lights){
    lightsPoint.reserve(lightsPoint.size() + lights.size());
    for(const Light& p : lights){
        lightsPoint.push_back(p);
    }

    lightBuffer->destroy();

    VkBufferCopy stagingCopy = {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = lightsPoint.size() * sizeof(Light)
    };

    VkBufferUsageFlags lightUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        
    lightBuffer = std::make_unique<Buffer>(m_device, m_allocator, lightsPoint.size() * sizeof(Light), 
        lightUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);

    VkBufferDeviceAddressInfo bdaInfo {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = nullptr,
        .buffer = lightBuffer->buffer
    };
    lightBufferAddress = vkGetBufferDeviceAddress(m_device, &bdaInfo);

    Buffer stagingBuffer = Buffer(m_device, m_allocator, lightsPoint.size() * sizeof(Light), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
	
    if (stagingBuffer.allocationInfo.pMappedData == nullptr) {
		throw std::runtime_error("Material staging buffer not mapped.");
	}

	memcpy(stagingBuffer.allocationInfo.pMappedData, lightsPoint.data(), lightsPoint.size() * sizeof(Light));
	prepImmediateTransfer();
	vkCmdCopyBuffer(m_immTransfer.buffer, stagingBuffer.buffer, lightBuffer->buffer, 1, &stagingCopy);
    submitImmediateTransfer();
    
}

bool Engine::loadGLTF(std::filesystem::path filePath){
    std::cout << "Loading GLTF: " << filePath << std::endl;
    
    std::shared_ptr<GLTFScene> scene = std::make_shared<GLTFScene>();
    std::vector<std::shared_ptr<MeshAsset>> meshes;
    std::vector<std::shared_ptr<GLTFNode>> nodes;
    std::vector<std::shared_ptr<Image>> images;
    std::vector<std::string> imageNames;
    std::vector<std::string> materialNames;
    std::vector<Light> lights;
    std::vector<RenderPass> passTypes;

    fastgltf::Parser parser(fastgltf::Extensions::KHR_lights_punctual);

    auto gltfOptions =  fastgltf::Options::DontRequireValidAssetMember | fastgltf::Options::AllowDouble | fastgltf::Options::LoadExternalBuffers;

    auto data = fastgltf::GltfDataBuffer::FromPath(filePath);
    if (data.error() != fastgltf::Error::None) {
        return false;
    }

    auto asset = parser.loadGltf(data.get(), filePath.parent_path(), gltfOptions);
    if (auto error = asset.error(); error != fastgltf::Error::None) {
        return false;
    }

    VkSamplerCreateInfo samplerCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .minLod = 0.0f,
        .maxLod = VK_LOD_CLAMP_NONE,
    };

    //Load samplers
    for (fastgltf::Sampler& sampler : asset->samplers) {
        fastgltf::Filter magF = sampler.magFilter.value_or(fastgltf::Filter::Linear);
        switch (magF)
        {
        case fastgltf::Filter::Linear:
        case fastgltf::Filter::LinearMipMapLinear:
        case fastgltf::Filter::LinearMipMapNearest:
            samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
            break;
        case fastgltf::Filter::Nearest:
        case fastgltf::Filter::NearestMipMapLinear:
        case fastgltf::Filter::NearestMipMapNearest:
            samplerCreateInfo.magFilter = VK_FILTER_NEAREST;
            break;
        default:
            samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
            break;
        }

        fastgltf::Filter minF = sampler.minFilter.value_or(fastgltf::Filter::Linear);
        switch (minF)
        {
        case fastgltf::Filter::Linear:
        case fastgltf::Filter::LinearMipMapLinear:
        case fastgltf::Filter::LinearMipMapNearest:
            samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
            break;
        case fastgltf::Filter::Nearest:
        case fastgltf::Filter::NearestMipMapLinear:
        case fastgltf::Filter::NearestMipMapNearest:
            samplerCreateInfo.minFilter = VK_FILTER_NEAREST;
            break;
        default:
            samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
            break;
        }
        
        switch (minF)
        {
        case fastgltf::Filter::LinearMipMapLinear:
        case fastgltf::Filter::NearestMipMapLinear:
            samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            break;
        case fastgltf::Filter::LinearMipMapNearest:
        case fastgltf::Filter::NearestMipMapNearest:
            samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            break;
        default:
            samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            break;
        }
    
        VkSampler vksampler;
        vkCreateSampler(m_device, &samplerCreateInfo, nullptr, &vksampler);
        scene->samplers.push_back(vksampler);
    }

    //Load images
    for (fastgltf::Image& image : asset->images){
        std::shared_ptr<Image> newImage = nullptr;        
        int width, height, nrChannels;

        std::visit(fastgltf::visitor{
            [&](fastgltf::sources::URI& URI) {
                assert(URI.fileByteOffset == 0);
                assert(URI.uri.isLocalPath());
                std::string stem = filePath.relative_path().remove_filename().generic_string();
                std::string path(URI.uri.path().begin(), URI.uri.path().end());
                std::string concatted = stem + path;
                unsigned char* data = stbi_load(concatted.c_str(), &width, &height, &nrChannels, 4);
                if(data){
                    VkExtent3D size{
                        .width = (uint32_t)width,
                        .height = (uint32_t)height,
                        .depth = 1
                    };
                    newImage = createImageFromData(data, size, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT, false);
                    stbi_image_free(data);
                }
                else{
                    std::cout << "Failed to load URI" << std::endl;
                }
                imageNames.push_back(path.c_str());
                scene->images.insert({path.c_str(), newImage});
                std::cout << "- Loaded Image: " << path.c_str() << std::endl;

            },
            [&](fastgltf::sources::BufferView& BufferView) {
                auto& bufView = asset->bufferViews[BufferView.bufferViewIndex];
                auto& buffer = asset->buffers[bufView.bufferIndex];
                
                unsigned char* data = nullptr;
                
                if(std::holds_alternative<fastgltf::sources::Vector>(buffer.data))
                {   
                    const auto& vec = std::get<fastgltf::sources::Vector>(buffer.data);
                    const unsigned char* vecData = reinterpret_cast<const unsigned char*>(vec.bytes.data());
                    data = stbi_load_from_memory(vecData + bufView.byteOffset, (int)bufView.byteLength, &width, &height, &nrChannels, 4);

                }else if(std::holds_alternative<fastgltf::sources::Array>(buffer.data)){
                    const auto& vec = std::get<fastgltf::sources::Array>(buffer.data).bytes;
                    const unsigned char* vecData = reinterpret_cast<const unsigned char*>(vec.data());
                    data = stbi_load_from_memory(vecData + bufView.byteOffset, (int)bufView.byteLength, &width, &height, &nrChannels, 4);
                }

                if(data){
                    VkExtent3D size{
                        .width = (uint32_t)width,
                        .height = (uint32_t)height,
                        .depth = 1
                    };
                    newImage = createImageFromData(data, size, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT, false);
                    stbi_image_free(data);
                }else{
                    std::cout << "Failed to load BufferView" << std::endl;
                }
                imageNames.push_back(image.name.c_str());
                scene->images.insert({image.name.c_str(), newImage});
                std::cout << "- Loaded Image: " << image.name.c_str() << std::endl;
            },
            [&](fastgltf::sources::Array& arraySource) {
                const auto& vec = arraySource.bytes;
                const auto& vecData = reinterpret_cast<const unsigned char*>(vec.data());
                unsigned char* data = stbi_load_from_memory(vecData, (int)vec.size(), &width, &height, &nrChannels, 4);
                if (data) {                    
                    VkExtent3D size{
                        .width = (uint32_t)width,
                        .height = (uint32_t)height,
                        .depth = 1
                    };
                    newImage = createImageFromData(data, size, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT, false);
                    stbi_image_free(data);
                }else{
                    std::cout << "Failed to load Array" << std::endl;
                }
                
                imageNames.push_back(image.name.c_str());
                scene->images.insert({image.name.c_str(), newImage});
                std::cout << "- Loaded Image: " << image.name.c_str() << std::endl;
            },
            [&](auto&& other) {
                std::cerr << "Unhandled image source type: " << typeid(other).name() << "\n";
            }
        }, image.data);

        if(newImage == nullptr){
            std::cout << "- Failed to load image: " << image.name.c_str() << ". Defaulting to checkerboard." << std::endl;
            newImage = checkerboardImage;
        }
        images.push_back(newImage);
    }

    //Load materials (addtextures, addmaterial)
    for (fastgltf::Material& material : asset->materials){
        
        std::cout << "- Loading Material: " << material.name.c_str() << std::endl;
        MaterialData newMaterial;
        newMaterial.baseColor.x = material.pbrData.baseColorFactor[0];
        newMaterial.baseColor.y = material.pbrData.baseColorFactor[1];
        newMaterial.baseColor.z = material.pbrData.baseColorFactor[2];
        newMaterial.baseColor.w = material.pbrData.baseColorFactor[3];

        newMaterial.metallicFactor = material.pbrData.metallicFactor;
        newMaterial.roughnessFactor = material.pbrData.roughnessFactor;
        
        newMaterial.baseColorIndex = 0;
        newMaterial.metallicRoughnessIndex = 0;
        newMaterial.normalIndex = 0;
        
        if(material.alphaMode == fastgltf::AlphaMode::Blend){
            passTypes.push_back(RenderPass::TRANSPARENT);
        }else{
            passTypes.push_back(RenderPass::OPAQUE);
        }
        //create textureData 
        if(material.pbrData.baseColorTexture.has_value()){
            size_t assetTexIndex = material.pbrData.baseColorTexture.value().textureIndex;
            TextureData texData;

            size_t imageLocalIndex = asset->textures[assetTexIndex].imageIndex.value();
            texData.texture = images.at(imageLocalIndex);

            if(asset->textures[assetTexIndex].samplerIndex.has_value()){
                size_t samplerLocalIndex = asset->textures[assetTexIndex].samplerIndex.value();
                texData.sampler = scene->samplers.at(samplerLocalIndex);
            }else{
                texData.sampler = defaultLinearSampler;
            }            
            newMaterial.baseColorIndex = addTexture(texData, imageNames.at(imageLocalIndex));
        }        
        if(material.pbrData.metallicRoughnessTexture.has_value()){
            size_t assetTexIndex = material.pbrData.metallicRoughnessTexture.value().textureIndex;
            TextureData texData;

            size_t imageLocalIndex = asset->textures[assetTexIndex].imageIndex.value();
            texData.texture = images.at(imageLocalIndex);

            if(asset->textures[assetTexIndex].samplerIndex.has_value()){
                size_t samplerLocalIndex = asset->textures[assetTexIndex].samplerIndex.value();
                texData.sampler = scene->samplers.at(samplerLocalIndex);
            }else{
                texData.sampler = defaultLinearSampler;
            }            
            newMaterial.metallicRoughnessIndex = addTexture(texData, imageNames.at(imageLocalIndex));
        }
        if(material.normalTexture.has_value()){
            size_t assetTexIndex = material.normalTexture.value().textureIndex;
            TextureData texData;

            size_t imageLocalIndex = asset->textures[assetTexIndex].imageIndex.value();
            texData.texture = images.at(imageLocalIndex);

            if(asset->textures[assetTexIndex].samplerIndex.has_value()){
                size_t samplerLocalIndex = asset->textures[assetTexIndex].samplerIndex.value();
                texData.sampler = scene->samplers.at(samplerLocalIndex);
            }else{
                texData.sampler = defaultLinearSampler;
            }            
            newMaterial.normalIndex = addTexture(texData, imageNames.at(imageLocalIndex));
        }

        materialNames.push_back(material.name.c_str());
        uint32_t matIndex = addMaterial(newMaterial, material.name.c_str());
        scene->materialIndices.insert({material.name.c_str(), matIndex});
    }

    std::vector<uint32_t> indices;
    std::vector<Vertex> vertices;
    for (fastgltf::Mesh& mesh : asset->meshes) {
        std::shared_ptr<MeshAsset> newMesh = std::make_shared<MeshAsset>();
        newMesh->name = mesh.name;
        
        std::cout << "- Loading Mesh: " << mesh.name.c_str() << std::endl;

        scene->meshes.insert({mesh.name.c_str(), newMesh});
        meshes.push_back(newMesh);
        
        indices.clear();
        vertices.clear();
        
        for (fastgltf::Primitive& prim : mesh.primitives) {
            Surface newSurface;

            newSurface.startIndex = (uint32_t)indices.size();
            newSurface.count = (uint32_t)asset->accessors[prim.indicesAccessor.value()].count;
            size_t initialVert = vertices.size();
            fastgltf::Accessor& indexAccessor = asset->accessors[prim.indicesAccessor.value()];
            indices.reserve(indices.size() + indexAccessor.count);
            {
            fastgltf::iterateAccessor<std::uint32_t>(asset.get(), indexAccessor,
                        [&](std::uint32_t idx) {
                            indices.push_back(idx + (uint32_t)initialVert);
                        });
            }

            auto posAttribute = prim.findAttribute("POSITION");
            auto posAccessor = asset->accessors[posAttribute->accessorIndex];
            vertices.resize(vertices.size() + posAccessor.count);
            {
            fastgltf::iterateAccessorWithIndex<glm::vec3>(asset.get(), posAccessor,
                    [&](glm::vec3 v, size_t idx) {
                        Vertex vert;
                        vert.position = v;
                        vert.normal = { 1, 0, 0 };
                        vert.color = glm::vec4 { 1.f };
                        vert.uv_x = 0;
                        vert.uv_y = 0;
                        vertices[initialVert + idx] = vert;
                    });
            }

            glm::vec3 minPos = vertices.at(initialVert).position;
            glm::vec3 maxPos = vertices.at(initialVert).position;
            for (size_t i = initialVert; i < vertices.size(); i++) {
                minPos = glm::min(minPos, vertices.at(i).position);
                maxPos = glm::max(maxPos, vertices.at(i).position);
            }
            newSurface.bounds.origin = (maxPos + minPos) / 2.0f;
            newSurface.bounds.extents = (maxPos - minPos) / 2.0f;

            auto normAttribute = prim.findAttribute("NORMAL");
            if (normAttribute != prim.attributes.end()) {
                auto normAccessor = asset->accessors[normAttribute->accessorIndex];
                fastgltf::iterateAccessorWithIndex<glm::vec3>(asset.get(), normAccessor,
                    [&](glm::vec3 v, size_t idx) {
                        vertices[initialVert + idx].normal = v;
                    });
            }

            // TEXCOORD_0 /uvs
            auto texAttribute = prim.findAttribute("TEXCOORD_0");
            if (texAttribute != prim.attributes.end()) {
                auto texAccessor = asset->accessors[texAttribute->accessorIndex];
                fastgltf::iterateAccessorWithIndex<glm::vec2>(asset.get(), texAccessor,
                    [&](glm::vec2 v, size_t idx) {
                        vertices[initialVert + idx].uv_x = v.x;
                        vertices[initialVert + idx].uv_y = v.y;
                    });
            }

            // COLOR_0
            auto colAttribute = prim.findAttribute("COLOR_0");
            if (colAttribute != prim.attributes.end()) {
                auto colAccessor = asset->accessors[colAttribute->accessorIndex];
                fastgltf::iterateAccessorWithIndex<glm::vec4>(asset.get(), colAccessor,
                    [&](glm::vec4 v, size_t idx) {
                        vertices[initialVert + idx].color = v;
                    });
            }

            //TANGENT
            auto tanAttribute = prim.findAttribute("TANGENT");
            if (tanAttribute != prim.attributes.end()) {
                auto tanAccessor = asset->accessors[tanAttribute->accessorIndex];
                fastgltf::iterateAccessorWithIndex<glm::vec4>(asset.get(), tanAccessor,
                    [&](glm::vec4 v, size_t idx) {
                        vertices[initialVert + idx].tangent = v;
                    });
            }

            if(prim.materialIndex.has_value()){
                newSurface.matIndex = matNameToIndex.at(materialNames.at(prim.materialIndex.value()));
                newSurface.type = passTypes.at(prim.materialIndex.value());
            }else{
                newSurface.matIndex = 0;
                newSurface.type = RenderPass::OPAQUE;
            }

            newMesh->surfaces.push_back(newSurface);
        }
        
        newMesh->data = uploadMesh(indices, vertices);
    }

    for (fastgltf::Light& light : asset->lights){
        Light p{};
        p.lightColor.x = light.color.x();
        p.lightColor.y = light.color.y();
        p.lightColor.z = light.color.z();
        p.intensity = 1.0f;
        if(light.intensity > 0.0f){
            p.intensity = light.intensity;
        }
        
        if(light.range.has_value() && light.range.value() > 0.0f){
            p.range = light.range.value();
        }else{
            p.range = FLT_MAX;
        }

        if(light.type == fastgltf::LightType::Point){
            p.type = 0;
        }else if(light.type == fastgltf::LightType::Directional){
            p.type = 1;
        }else{
            p.type = 2; //Spot light
            p.innerConeAngle = light.innerConeAngle.value_or(0.0f);
            p.outerConeAngle = light.outerConeAngle.value_or(30.0f);
        }


        lights.push_back(p);
        std::cout << "Loading light: " << light.name.c_str() << std::endl;
    }

    //Extract all node transforms & meshes
    for (fastgltf::Node& node: asset->nodes){
        std::shared_ptr<GLTFNode> newNode = std::make_shared<GLTFNode>();
        newNode->lightIndex = -1;

        std::cout << "- Loading Node: " << node.name.c_str() << std::endl;

        if(node.meshIndex.has_value()){
            newNode->mesh = meshes.at(node.meshIndex.value());
        }
        if(node.lightIndex.has_value()){
            newNode->lightIndex = node.lightIndex.value();
        }

        nodes.push_back(newNode);

        fastgltf:: math::fmat4x4 matrix = fastgltf::getTransformMatrix(node);
        memcpy(&newNode->localTransform, matrix.data(), sizeof(matrix));

        scene->nodes.insert({node.name.c_str(), newNode});
    }

    //Create node relationships
    for(uint32_t nodeIndex = 0; nodeIndex < asset->nodes.size(); nodeIndex++){
        fastgltf::Node& node = asset->nodes.at(nodeIndex);
        std::shared_ptr<GLTFNode> gltfNode = nodes.at(nodeIndex);

        for (size_t c : node.children){
            gltfNode->children.push_back(nodes[c]);
            nodes[c]->parent = gltfNode;
        }
    }

    //Refresh world node world transforms
    for (std::shared_ptr<GLTFNode> node : nodes) {
        if (node->parent.lock() == nullptr) {
            scene->topNodes.push_back(node);

            node->refreshTransform(glm::mat4(1.0f));
            
            //update light positions
            if(node->lightIndex < 0){
                continue;
            }
            lights.at(node->lightIndex).lightPos = node->worldTransform[3];
            
            if(lights.at(node->lightIndex).type == 1 || lights.at(node->lightIndex).type == 2){ //direcitonal or spot
                glm::vec4 localForward(0.0f, 0.0f, -1.0f, 0.0f);
                glm::vec4 worldDir = node->worldTransform * localForward;
                lights.at(node->lightIndex).lightDir = glm::normalize(glm::vec3(worldDir));
            }else{
                lights.at(node->lightIndex).lightDir = glm::vec3(0.0f);
            }
        }
    }

    if(lights.size() > 0){
        addLights(lights);
    }

    loadedGLTFs.insert({filePath.filename().generic_string(), scene});
    return true;
}

void Engine::meshUploader(){
    while (true) {
		if (pathQueue.empty()) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			continue;
		}
		
		std::filesystem::path res = pathQueue.front();
		pathQueue.pop();
		if (res.generic_string() == "QUIT") {
			break;
		}

        loadGLTF(res);
	}
    std::cout << "Mesh Uploader exiting..." << std::endl;
}

//Transfer
void Engine::prepImmediateTransfer(){
    vkResetFences(m_device, 1, &m_immTransfer.fence);
	vkResetCommandBuffer(m_immTransfer.buffer, 0);

	VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };

	vkBeginCommandBuffer(m_immTransfer.buffer, &begin);
}

void Engine::submitImmediateTransfer(){

    vkEndCommandBuffer(m_immTransfer.buffer);

	VkCommandBufferSubmitInfo cmdInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext = nullptr,
        .commandBuffer = m_immTransfer.buffer
    };
	VkSubmitInfo2 submit {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext = nullptr,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &cmdInfo,
    };
	vkQueueSubmit2(m_transferQueue, 1, &submit, m_immTransfer.fence);

	vkWaitForFences(m_device, 1, &m_immTransfer.fence, true, 9999999999);
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
    VkCommandBuffer cmd = getCurrentFrame().graphicsCommandBuffer;
    vkResetCommandBuffer(cmd, 0);
    
    VkCommandBufferBeginInfo commandBufferBeginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr
    };

    vkBeginCommandBuffer(cmd, &commandBufferBeginInfo);

 	drawImage->transitionTo(cmd, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    resolveImage->transitionTo(cmd, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    depthImage->transitionTo(cmd, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    stats.drawCallCount = 0;
    drawParticles(cmd);
    //Draw
    // drawMeshes(cmd);

    resolveImage->transitionTo(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
	m_swapchain->images.at(index).transitionTo(cmd, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    resolveImage->copyTo(cmd, m_swapchain->images.at(index));

    m_swapchain->images.at(index).transitionTo(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    drawDearImGui(cmd, m_swapchain->images.at(index).view);
    m_swapchain->images.at(index).transitionTo(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

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
    VkSemaphoreSubmitInfo semaphoreWaitSubmitInfo2{ 
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .semaphore = particleTLSemaphore,
        .value = particleTLValue,
        .stageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
        .deviceIndex = 0
    };
    VkSemaphoreSubmitInfo waits[] = {semaphoreWaitSubmitInfo, semaphoreWaitSubmitInfo2};
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
        .waitSemaphoreInfoCount = 2,
        .pWaitSemaphoreInfos = waits,
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
        int32_t w = 0;
        int32_t h = 0;
		SDL_GetWindowSizeInPixels(m_pWindow, &w, &h);
        m_swapchain->resizeSwapchain(w, h);
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
    color_attachment.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attachment.resolveImageView = resolveImage->view;
    color_attachment.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    
    VkRenderingAttachmentInfo depth_attachment = {};
    depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth_attachment.pNext = nullptr;
    depth_attachment.imageView = depthImage->view;
    depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp = 
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
    

    auto start = std::chrono::system_clock::now();
    vkCmdBeginRendering(cmd, &rendering_info);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipelineOpaque);
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
    
    stats.triCount = 0;
    PushConstants pcs;
    for(const Renderable& r : opaqueRenderables){
        pcs.vertexBuffer = r.vertexBufferAddress;
        pcs.materialIndex = r.materialIndex;
        pcs.modelMatrix = r.modelMat;
        vkCmdPushConstants(cmd, meshPipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(PushConstants), &pcs);
        vkCmdBindIndexBuffer(cmd, r.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, r.indexCount, 1, r.firstIndex, 0, 0);

        stats.drawCallCount++;
        stats.triCount += r.indexCount / 3;
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipelineTransparent);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    for(uint32_t index : transparentRenderablesIndices){
        pcs.vertexBuffer = transparentRenderables.at(index).vertexBufferAddress;
        pcs.materialIndex = transparentRenderables.at(index).materialIndex;
        pcs.modelMatrix = transparentRenderables.at(index).modelMat;
        vkCmdPushConstants(cmd, meshPipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(PushConstants), &pcs);
        vkCmdBindIndexBuffer(cmd, transparentRenderables.at(index).indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, transparentRenderables.at(index).indexCount, 1, transparentRenderables.at(index).firstIndex, 0, 0);

        stats.drawCallCount++;
        stats.triCount += transparentRenderables.at(index).indexCount / 3;
    }

    vkCmdEndRendering(cmd);

    auto end = std::chrono::system_clock::now();    
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if(frameNumber % 144){
        stats.meshDrawTime = elapsed.count() / 1000.0f;
    }
}

void Engine::drawParticles(VkCommandBuffer cmd){
    VkRenderingAttachmentInfo color_attachment = {};
    color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color_attachment.pNext = nullptr;
    color_attachment.imageView = drawImage->view;
    color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attachment.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attachment.resolveImageView = resolveImage->view;
    color_attachment.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    
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
    
    auto start = std::chrono::system_clock::now();    
    vkCmdBeginRendering(cmd, &rendering_info);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, particleDrawPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, particleDrawPipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);
    
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
    

    ParticlePushConstants pcs;
    pcs.renderMatrix = cam->getRenderMatrix();
    pcs.camWorldPos = cam->getPos();
    pcs.velocityBuffer = particleVelBufferAddress;        
    if(frameNumber % 2 == 0){
        pcs.positionBuffer = particlePosBufferAddressB;
    }else{
        pcs.positionBuffer = particlePosBufferAddressA;
    }

    vkCmdPushConstants(cmd, particleDrawPipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(ParticlePushConstants), &pcs);
    vkCmdDraw(cmd, PARTICLE_COUNT, 1, 0, 0);
    stats.drawCallCount++;
    vkCmdEndRendering(cmd);
    
    auto end = std::chrono::system_clock::now();    
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if(frameNumber % 144){
        stats.particleTime = elapsed.count() / 1000.0f;
    }
}

void Engine::updateParticles(){
    VkResult fenceResult = vkWaitForFences(m_device, 1, &getCurrentFrame().computeFence, VK_TRUE, 1000000000);
    if (fenceResult != VK_SUCCESS) {
        throw std::runtime_error("Fence wait failed!");
    }
    
    vkResetFences(m_device, 1, &getCurrentFrame().computeFence);
    VkCommandBuffer cmd = getCurrentFrame().computeCommandBuffer;
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo commandBufferBeginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr
    };

    vkBeginCommandBuffer(cmd, &commandBufferBeginInfo);

        VkPipelineBindPoint compute = VK_PIPELINE_BIND_POINT_COMPUTE;
        uint32_t particleX = (PARTICLE_COUNT + 63) / 64;
        
        ParticleComputePushConstants pcs;
        pcs.deltaTime = deltaTime;
        pcs.timeScale = 0.0005f * timeScale;
        pcs.velocityBuffer = particleVelBufferAddress;
        if(frameNumber % 2 == 0){
            pcs.positionBufferB = particlePosBufferAddressB;
            pcs.positionBufferA = particlePosBufferAddressA;
        }else{
            pcs.positionBufferB = particlePosBufferAddressA;
            pcs.positionBufferA = particlePosBufferAddressB;
        }
        
        vkCmdBindPipeline(cmd, compute, particleComputePipeline);
        vkCmdPushConstants(cmd, particleComputePipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(ParticleComputePushConstants), &pcs);
        vkCmdDispatch(cmd, particleX, 1, 1);

	vkEndCommandBuffer(cmd);
    VkCommandBufferSubmitInfo commandBufferSubmitInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext = nullptr,
        .commandBuffer = cmd,
        .deviceMask = 0
    };
    particleTLValue++;
	VkSemaphoreSubmitInfo semaphoreSignalSubmitInfo{ 
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .semaphore = particleTLSemaphore,
        .value = particleTLValue,
        .stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .deviceIndex = 0
    };
    VkSubmitInfo2 queueSubmitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext = nullptr,
        .flags = 0,
        .waitSemaphoreInfoCount = 0,
        .pWaitSemaphoreInfos = nullptr,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &commandBufferSubmitInfo,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &semaphoreSignalSubmitInfo
    };

    VkResult submitRes = vkQueueSubmit2(m_computeQueue, 1, &queueSubmitInfo, getCurrentFrame().computeFence);
    if(submitRes != VK_SUCCESS){
        std::cout << "compute queue submit failed" << std::endl;
    }        
}

void Engine::drawDearImGui(VkCommandBuffer cmd, VkImageView view){
        VkRenderingAttachmentInfo colorAttachment {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .pNext = nullptr,
        .imageView = view,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    };       
    
    VkRenderingInfo renderingInfo {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .pNext = nullptr,
        .renderArea = VkRect2D { VkOffset2D { 0, 0 }, m_swapchain->extent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachment,
    };

    vkCmdBeginRendering(cmd, &renderingInfo);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    vkCmdEndRendering(cmd);
}

void Engine::run(){
    SDL_Event e;
    bool quit = false;
    while(!quit){
        
        auto start = std::chrono::system_clock::now();

        while(SDL_PollEvent(&e) != 0){

            ImGui_ImplSDL3_ProcessEvent(&e);
            if(mouseCaptured){
                cam->processSDLEvent(e);
            }

            switch (e.type)
            {
            case SDL_EVENT_QUIT:
                quit = true;
                break;
            case SDL_EVENT_WINDOW_MINIMIZED:
                minimized = true;
                break;
            case SDL_EVENT_WINDOW_MAXIMIZED:
                minimized = false;
                break;
            case SDL_EVENT_KEY_DOWN:
                switch (e.key.key)
                {
                case SDLK_ESCAPE:
                    quit = true;
                    break;
                case SDLK_Q:
                    mouseCaptured = !mouseCaptured;
                    SDL_CaptureMouse(mouseCaptured);
                    SDL_SetWindowRelativeMouseMode(m_pWindow, mouseCaptured);
                    break;
                default:
                    break;
                }
            default:
                break;
            }
        }
        
        if(minimized){
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        updateGUI();     

        auto updateStart = std::chrono::system_clock::now();
        opaqueRenderables.clear();
        transparentRenderables.clear();
        transparentRenderablesIndices.clear();
        for(auto& [name, scene] : loadedGLTFs){
            for(auto& node : scene->topNodes){
                createRenderablesFromNode(node);
            }
        }
        sortTransparentRenderables();
        updateScene();
        auto updateEnd = std::chrono::system_clock::now();    
        auto updateElapsed = std::chrono::duration_cast<std::chrono::microseconds>(updateEnd - updateStart);
        if(frameNumber % 144){
            stats.sceneUpdateTime = updateElapsed.count() / 1000.0f;
        }
        
        updateParticles();
        draw();
        
        auto end = std::chrono::system_clock::now();    
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        deltaTime = elapsed.count() / 1000.0f;
        currentTime += deltaTime;
        if(frameNumber% 144){
            stats.frameTime = deltaTime;
        }
    }
}