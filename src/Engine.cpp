#include "Engine.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <VkBootstrap.h>

#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <execution>
#include <stdexcept>
#include <random>

#include "Swapchain.hpp"
#include "Camera.hpp"
#include "Image.hpp"
#include "Buffer.hpp"
#include "PipelineBuilder.hpp"
#include "MeshLoader.hpp"

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
                                                          
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

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    
    vkDestroyPipelineLayout(m_device, meshPipelineLayout, nullptr);
    vkDestroyPipeline(m_device, meshPipeline, nullptr);

    vkDestroyDescriptorPool(m_device, m_ImguiPool, nullptr);
    vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
	vkDestroyDescriptorSetLayout(m_device, m_descriptorLayout, nullptr);

    for (int i = 0; i < 2; i++) {
        vkDestroyCommandPool(m_device, frameData[i].commandPool, nullptr);

        vkDestroyFence(m_device, frameData[i].renderFence, nullptr);
        vkDestroySemaphore(m_device, frameData[i].renderSemaphore, nullptr);
        vkDestroySemaphore(m_device, frameData[i].swapchainSemaphore, nullptr);
    }
    
    vkDestroyCommandPool(m_device, m_immTransfer.pool, nullptr);
    vkDestroyFence(m_device, m_immTransfer.fence, nullptr);

    drawImage->destroy();
    depthImage->destroy();

    for(MeshAsset& mesh : testMeshes){
        mesh.data.indexBuffer->destroy();
        mesh.data.vertexBuffer->destroy();
    }

    hostPositionBuffer->destroy();
    devicePositionBuffer->destroy();
    deviceGridCells->destroy();
    deviceGridCounters->destroy();
    neighborCount->destroy();
    neighborList->destroy();

    vmaDestroyAllocator(m_allocator);

    m_swapchain->destroySwapchain();

	vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
	vkDestroyDevice(m_device, nullptr);
		
	vkb::destroy_debug_utils_messenger(m_instance, m_debugMessenger);
	vkDestroyInstance(m_instance, nullptr);
	SDL_DestroyWindow(m_pWindow);

    cleanedUp = true;
}
//TODO
//Simple materials
//Input manage, cam -> shared_ptr
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
    
    initDearImgui();

    initializationTime = SDL_GetTicksNS();
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

    vkCreateFence(m_device, &fenceInfo, nullptr, &m_immTransfer.fence);
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
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
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
    initComputePipelines();
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
	pb->setCullingMode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	pb->setMultisamplingNone();
	pb->disableBlending();
	pb->enableDepthtest(VK_TRUE, VK_COMPARE_OP_GREATER_OR_EQUAL);
    // pb->disableDepthtest();
	pb->setColorAttachmentFormat(drawImage->format);
	pb->setDepthAttachmentFormat(depthImage->format);
	meshPipeline = pb->buildPipeline(m_device);

	vkDestroyShaderModule(m_device, vert, nullptr);
	vkDestroyShaderModule(m_device, frag, nullptr);
}

void Engine::initComputePipelines(){
    VkPushConstantRange pc = {
        .stageFlags = VK_SHADER_STAGE_ALL,
        .offset = 0,
        .size = sizeof(ComputePushConstants)
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
	vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &computePipelineLayout);

    VkShaderModule resetGrid;
    VkShaderModule resetParticles;
    VkShaderModule predictPosition;
    
    VkShaderModule buildGrid;
    VkShaderModule neighbor;
    
    VkShaderModule lambdas;
    VkShaderModule deltas;
    VkShaderModule collisions;
    
    VkShaderModule vorticity;
    VkShaderModule vorticityGradient;
    VkShaderModule viscosity;
	if (!loadShader(&resetGrid, "../shaders/reset_grid_data.comp.spv")) {
		std::cout << "Error when building the reset_grid_data compute shader module" << std::endl;
	}
	else {
		std::cout << "Built the reset_grid_data shader module" << std::endl;
	}		
    if (!loadShader(&resetParticles, "../shaders/reset_particle_data.comp.spv")) {
		std::cout << "Error when building the reset_particle_data compute shader module" << std::endl;
	}
	else {
		std::cout << "Built the reset_particle_data shader module" << std::endl;
	}	
    if (!loadShader(&predictPosition, "../shaders/predict_position.comp.spv")) {
        std::cout << "Error when building the predict_position compute shader module" << std::endl;
    }
    else {
        std::cout << "Built the predict_position shader module" << std::endl;
    }

    if (!loadShader(&buildGrid, "../shaders/build_grid.comp.spv")) {
		std::cout << "Error when building the build_grid compute shader module" << std::endl;
	}
	else {
        std::cout << "Built the build_grid shader module" << std::endl;
	}	
    if (!loadShader(&neighbor, "../shaders/find_neighbors.comp.spv")) {
        std::cout << "Error when building the find_neighbors compute shader module" << std::endl;
    }
    else {
        std::cout << "Built the find_neighbors shader module" << std::endl;
    }

    if (!loadShader(&lambdas, "../shaders/lambdas.comp.spv")) {
		std::cout << "Error when building the lambdas compute shader module" << std::endl;
	}
	else {
		std::cout << "Built the lambdas shader module" << std::endl;
	}	
    if (!loadShader(&deltas, "../shaders/deltas.comp.spv")) {
		std::cout << "Error when building the deltas compute shader module" << std::endl;
	}
	else {
		std::cout << "Built the deltas shader module" << std::endl;
	}
    if (!loadShader(&collisions, "../shaders/collisions.comp.spv")) {
		std::cout << "Error when building the collisions compute shader module" << std::endl;
	}
	else {
		std::cout << "Built the collisions shader module" << std::endl;
	}
    if (!loadShader(&vorticity, "../shaders/finalize.comp.spv")) {
		std::cout << "Error when building the finalize compute shader module" << std::endl;
	}
	else {
		std::cout << "Built the finalize shader module" << std::endl;
	}
    if (!loadShader(&vorticityGradient, "../shaders/finalize2.comp.spv")) {
		std::cout << "Error when building the finalize2 compute shader module" << std::endl;
	}
	else {
		std::cout << "Built the finalize2 shader module" << std::endl;
	}
    if (!loadShader(&viscosity, "../shaders/finalize3.comp.spv")) {
		std::cout << "Error when building the finalize3 compute shader module" << std::endl;
	}
	else {
		std::cout << "Built the finalize3 shader module" << std::endl;
	}
    VkPipelineCache computeCache;
    VkPipelineCacheCreateInfo cacheCreateInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        .pNext = nullptr,
        .initialDataSize = 0
    };
    vkCreatePipelineCache(m_device, &cacheCreateInfo, nullptr, &computeCache);
    

    VkPipelineShaderStageCreateInfo predictStageInfo{
        .sType=  VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = predictPosition,
        .pName = "main"
    };   
    VkPipelineShaderStageCreateInfo resetGridStageInfo{
        .sType=  VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = resetGrid,
        .pName = "main"
    };   
    VkPipelineShaderStageCreateInfo resetParticleStageInfo{
        .sType=  VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = resetParticles,
        .pName = "main"
    };

    VkPipelineShaderStageCreateInfo buildGridStageInfo{
        .sType=  VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = buildGrid,
        .pName = "main"
    };   
    VkPipelineShaderStageCreateInfo neighborStageInfo{
        .sType=  VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = neighbor,
        .pName = "main"
    };

    VkPipelineShaderStageCreateInfo lambdaStageInfo{
        .sType=  VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = lambdas,
        .pName = "main"
    };

    VkPipelineShaderStageCreateInfo deltaStageInfo{
        .sType=  VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = deltas,
        .pName = "main"
    };
    VkPipelineShaderStageCreateInfo collisionStageInfo{
        .sType=  VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = collisions,
        .pName = "main"
    };        
    
    VkPipelineShaderStageCreateInfo vortStageInfo{
        .sType=  VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = vorticity,
        .pName = "main"
    };    
    VkPipelineShaderStageCreateInfo vortGradStageInfo{
        .sType=  VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = vorticityGradient,
        .pName = "main"
    };    
    VkPipelineShaderStageCreateInfo viscStageInfo{
        .sType=  VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = viscosity,
        .pName = "main"
    };    

	VkComputePipelineCreateInfo computePipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .stage = predictStageInfo,
        .layout = computePipelineLayout
    };
	vkCreateComputePipelines(m_device, computeCache, 1, &computePipelineCreateInfo, nullptr, &computePredict);
    computePipelineCreateInfo.stage = resetGridStageInfo;
	vkCreateComputePipelines(m_device, computeCache, 1, &computePipelineCreateInfo, nullptr, &computeResetGrid);
    computePipelineCreateInfo.stage = resetParticleStageInfo;
	vkCreateComputePipelines(m_device, computeCache, 1, &computePipelineCreateInfo, nullptr, &computeResetParticles);
    computePipelineCreateInfo.stage = buildGridStageInfo;
	vkCreateComputePipelines(m_device, computeCache, 1, &computePipelineCreateInfo, nullptr, &computeGrid);
    computePipelineCreateInfo.stage = deltaStageInfo;
	vkCreateComputePipelines(m_device, computeCache, 1, &computePipelineCreateInfo, nullptr, &computeDeltas);
    computePipelineCreateInfo.stage = neighborStageInfo;
    vkCreateComputePipelines(m_device, computeCache, 1, &computePipelineCreateInfo, nullptr, &computeNeighbors);
    computePipelineCreateInfo.stage = lambdaStageInfo;
    vkCreateComputePipelines(m_device, computeCache, 1, &computePipelineCreateInfo, nullptr, &computeLambdas);
    computePipelineCreateInfo.stage = vortStageInfo;
    vkCreateComputePipelines(m_device, computeCache, 1, &computePipelineCreateInfo, nullptr, &computeVorticity);
    computePipelineCreateInfo.stage = vortGradStageInfo;
    vkCreateComputePipelines(m_device, computeCache, 1, &computePipelineCreateInfo, nullptr, &computeVorticityGradient);
    computePipelineCreateInfo.stage = viscStageInfo;
    vkCreateComputePipelines(m_device, computeCache, 1, &computePipelineCreateInfo, nullptr, &computeViscosity);
    computePipelineCreateInfo.stage = collisionStageInfo;
    vkCreateComputePipelines(m_device, computeCache, 1, &computePipelineCreateInfo, nullptr, &computeCollision);
   
    // vkDestroyShaderModule() all the modules
    vkDestroyShaderModule(m_device, resetGrid, nullptr);
    vkDestroyShaderModule(m_device, resetParticles, nullptr);
    vkDestroyShaderModule(m_device, predictPosition, nullptr);
    vkDestroyShaderModule(m_device, buildGrid, nullptr);
    vkDestroyShaderModule(m_device, neighbor, nullptr);
    vkDestroyShaderModule(m_device, lambdas, nullptr);
    vkDestroyShaderModule(m_device, deltas, nullptr);
    vkDestroyShaderModule(m_device, collisions, nullptr);
    vkDestroyShaderModule(m_device, vorticity, nullptr);
    vkDestroyShaderModule(m_device, vorticityGradient, nullptr);
    vkDestroyShaderModule(m_device, viscosity, nullptr);
}

void Engine::initData(){
    
    hostPositionBuffer = std::make_unique<Buffer>(m_device, m_allocator, instanceCount * sizeof(ParticleData), 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, 
        VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

    devicePositionBuffer = std::make_unique<Buffer>(m_device, m_allocator, instanceCount * sizeof(ParticleData), 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);

    deviceGridCounters= std::make_unique<Buffer>(m_device, m_allocator, tableSize * sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);

    deviceGridCells= std::make_unique<Buffer>(m_device, m_allocator, tableSize * sizeof(uint32_t) * 12,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);

    neighborList = std::make_unique<Buffer>(m_device, m_allocator, instanceCount * sizeof(uint32_t) * 12 * 27,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);

    neighborCount = std::make_unique<Buffer>(m_device, m_allocator, instanceCount * sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);
    
    VkBufferDeviceAddressInfo positionAddressInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = nullptr,
        .buffer = devicePositionBuffer->buffer
    };
    VkBufferDeviceAddressInfo gridCounterAddressInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = nullptr,
        .buffer = deviceGridCounters->buffer
    };    
    VkBufferDeviceAddressInfo gridCellsAddressInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = nullptr,
        .buffer = deviceGridCells->buffer
    };    
    VkBufferDeviceAddressInfo neighborListAddressInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = nullptr,
        .buffer = neighborList->buffer
    };    
    VkBufferDeviceAddressInfo neighborCountAddressInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = nullptr,
        .buffer = neighborCount->buffer
    };
    particleBufferAddress = vkGetBufferDeviceAddress(m_device, &positionAddressInfo);    
    gridCountBufferAddress = vkGetBufferDeviceAddress(m_device, &gridCounterAddressInfo); 
    gridCellsBufferAddress = vkGetBufferDeviceAddress(m_device, &gridCellsAddressInfo); 
    neighborCountBufferAddress = vkGetBufferDeviceAddress(m_device, &neighborCountAddressInfo);
    neighborListBufferAddress = vkGetBufferDeviceAddress(m_device, &neighborListAddressInfo);


    if (hostPositionBuffer->allocationInfo.pMappedData == nullptr) {
        throw std::runtime_error("Host position buffer not mapped.");
    }

    maxBoundingPos = glm::vec3(60.0f, 0.0f, 0.0f);
    minBoundingPos = glm::vec3(-60.0f, -55.0f, -120.0f);

    cellStart.resize(tableSize);
    tempCounts.resize(tableSize);

    particleCurrPosition.reserve(instanceCount);
    particlePrevPosition.reserve(instanceCount);
    particleVelocity.reserve(instanceCount);
    
    particleIndices.resize(instanceCount);
    positionDeltas.resize(instanceCount);
    particleCollisions.resize(instanceCount);
    particleLambdas.resize(instanceCount);
    particleNeighbors.resize(instanceCount);
    particleVorticity.resize(instanceCount);
    particleVorticityGradient.resize(instanceCount);
    particleViscosityDelta.resize(instanceCount);

    gridCounters.resize(tableSize);
    gridCells.resize(tableSize * 12);
    
    glm::vec3 step = (maxBoundingPos - minBoundingPos) / float(sideLength - 1);
    for (uint32_t x = 0; x < sideLength; ++x) {
        for (uint32_t y = 0; y < sideLength; ++y) {
            for (uint32_t z = 0; z < sideLength; ++z) {
                glm::vec3 pos = minBoundingPos + glm::vec3(x, y, z) * step;
                particleCurrPosition.push_back(pos);
                particlePrevPosition.push_back(pos);
                particleVelocity.push_back(glm::vec3(0.0f));
            }
        }
    }

    resetPersistentParticleData();
    
    updatePositionBuffer();

    VkDescriptorBufferInfo bufferInfo = {
        .buffer = devicePositionBuffer->buffer,
        .offset = 0,
        .range = instanceCount * sizeof(ParticleData)
    };

    VkWriteDescriptorSet writeDescriptorSet = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr, 
        .dstSet = m_descriptorSet,
        .dstBinding = STORAGE_BINDING,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pImageInfo = nullptr,
        .pBufferInfo = &bufferInfo,
        .pTexelBufferView = nullptr
    };

    vkUpdateDescriptorSets(m_device, 1, &writeDescriptorSet, 0, nullptr);
    
    MeshLoader::loadGltfMeshes(this, testMeshes, "..\\..\\resources\\sphere.glb");

    
    cam = std::make_unique<Camera>((float)drawExtent.width, (float)drawExtent.height);
}

void Engine::initDearImgui(){
    VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE };

    VkDescriptorPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
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

//PBF
void Engine::updatePositionBuffer(){
    uint32_t OFFSET_PREV_POS      	= 3 * instanceCount * sizeof(float);
    uint32_t OFFSET_VELOCITY      	= 6 * instanceCount * sizeof(float);
    uint32_t OFFSET_DELTA         	= 9 * instanceCount * sizeof(float);
    uint32_t OFFSET_COLLISIONS    	= 12 * instanceCount * sizeof(float);
    uint32_t OFFSET_VORTICITY     	= 15 * instanceCount * sizeof(float);
    uint32_t OFFSET_VORTICITY_GRAD	= 18 * instanceCount * sizeof(float);
    uint32_t OFFSET_VISCOSITY     	= 21 * instanceCount * sizeof(float);
    uint32_t OFFSET_LAMBDAS       	= 24 * instanceCount * sizeof(float);

    memcpy(hostPositionBuffer->allocationInfo.pMappedData,                                  particleCurrPosition.data(),        instanceCount * sizeof(glm::vec3));
    memcpy((char*)hostPositionBuffer->allocationInfo.pMappedData + OFFSET_PREV_POS,         particlePrevPosition.data(),        instanceCount * sizeof(glm::vec3));
    memcpy((char*)hostPositionBuffer->allocationInfo.pMappedData + OFFSET_VELOCITY,         particleVelocity.data(),            instanceCount * sizeof(glm::vec3));
    memcpy((char*)hostPositionBuffer->allocationInfo.pMappedData + OFFSET_DELTA,            positionDeltas.data(),              instanceCount * sizeof(glm::vec3));
    memcpy((char*)hostPositionBuffer->allocationInfo.pMappedData + OFFSET_COLLISIONS,       particleCollisions.data(),          instanceCount * sizeof(glm::vec3));
    memcpy((char*)hostPositionBuffer->allocationInfo.pMappedData + OFFSET_VORTICITY,        particleVorticity.data(),           instanceCount * sizeof(glm::vec3));
    memcpy((char*)hostPositionBuffer->allocationInfo.pMappedData + OFFSET_VORTICITY_GRAD,   particleVorticityGradient.data(),   instanceCount * sizeof(glm::vec3));
    memcpy((char*)hostPositionBuffer->allocationInfo.pMappedData + OFFSET_VISCOSITY,        particleViscosityDelta.data(),      instanceCount * sizeof(glm::vec3));
    memcpy((char*)hostPositionBuffer->allocationInfo.pMappedData + OFFSET_LAMBDAS,          particleLambdas.data(),             instanceCount * sizeof(float));

    prepImmediateTransfer();
    
    VkBufferCopy instanceCopy = {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = instanceCount * sizeof(ParticleData)
    };
    
    vkCmdCopyBuffer(m_immTransfer.buffer, hostPositionBuffer->buffer, devicePositionBuffer->buffer, 1, &instanceCopy);
    
    vkCmdFillBuffer(m_immTransfer.buffer, deviceGridCounters->buffer, 0 , VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(m_immTransfer.buffer, deviceGridCells->buffer, 0 , VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(m_immTransfer.buffer, neighborCount->buffer, 0 , VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(m_immTransfer.buffer, neighborList->buffer, 0 , VK_WHOLE_SIZE, 0);

    submitImmediateTransfer();
}

glm::vec3 Engine::clampDeltaToBounds(uint32_t index){
    glm::vec3 pos = particleCurrPosition.at(index);
    glm::vec3 delta = positionDeltas.at(index);
    glm::vec newPos = pos + delta;
    
    float dampingFactor = 1.0f;
    glm::vec3 collision = glm::vec3(1.0f);
    float radius = 0.1f;

    // X axis
    if (newPos.x - radius < minBoundingPos.x){
        delta.x = minBoundingPos.x + radius - pos.x;
        collision.x *= -dampingFactor;
    } else if (newPos.x + radius > maxBoundingPos.x){
        delta.x = maxBoundingPos.x - radius - pos.x;
        collision.x *= -dampingFactor;
    }

    // Y axis
    if (newPos.y - radius < minBoundingPos.y){
        delta.y = minBoundingPos.y + radius - pos.y;
        collision.y *= -dampingFactor;
    } else if (newPos.y + radius > maxBoundingPos.y){
        delta.y = maxBoundingPos.y - radius - pos.y;
        collision.y *= -dampingFactor;
    }

    // Z axis
    if (newPos.z - radius < minBoundingPos.z){
        delta.z = minBoundingPos.z + radius - pos.z;
        collision.z *= -dampingFactor;
    } else if (newPos.z + radius > maxBoundingPos.z) {
        delta.z = maxBoundingPos.z - radius - pos.z;
        collision.z *= -dampingFactor;
    }  
    positionDeltas.at(index) = delta;
    return collision;
}

float Engine::densityKernel(float r){
    float factor = 315.0f / (64.0f * glm::pi<float>() * (float)std::pow(smoothingRadius, 9.0f));
    return factor * (float)std::pow(((smoothingRadius * smoothingRadius) - (r * r)), 3.0f);
}

float Engine::gradientKernel(float r){
    float factor = 45.0f / (glm::pi<float>() * (float)std::pow(smoothingRadius, 6.0f));
    return factor * (float)std::pow((smoothingRadius - r), 2.0f);
}

void Engine::resetPersistentParticleData(){
    for(uint32_t i = 0; i < instanceCount; i++){
        positionDeltas.at(i) = glm::vec3(0.0f);
        particleCollisions.at(i) = glm::vec3(0.0f);
        particleVorticity.at(i) = glm::vec3(0.0f);
        particleVorticityGradient.at(i) = glm::vec3(0.0f);
        particleViscosityDelta.at(i) = glm::vec3(0.0f);
        particleLambdas.at(i) = 0.0f;
        particleNeighbors.at(i).clear();
    }
}

void Engine::resetParticlePositions(){
    particleCurrPosition.clear();
    particlePrevPosition.clear();
    particleVelocity.clear();
    glm::vec3 step = (maxBoundingPos - minBoundingPos) / float(sideLength - 1);
    for (uint32_t x = 0; x < sideLength; ++x) {
        for (uint32_t y = 0; y < sideLength; ++y) {
            for (uint32_t z = 0; z < sideLength; ++z) {
                glm::vec3 pos = minBoundingPos + glm::vec3(x, y, z) * step;
                particleCurrPosition.push_back(pos);
                particlePrevPosition.push_back(pos);
                particleVelocity.push_back(glm::vec3(0.0f));
            }
        }
    }

    updatePositionBuffer();
}

void Engine::randomizeParticlePositions(){
    std::minstd_rand rng(std::random_device{}());
    std::uniform_real_distribution<float> unitDist(0.0f, 1.0f);
    glm::vec3 size = maxBoundingPos - minBoundingPos;

    for (size_t i = 0; i < instanceCount; ++i) {
        glm::vec3 pos(
            minBoundingPos.x + unitDist(rng) * size.x,
            minBoundingPos.y + unitDist(rng) * size.y,
            minBoundingPos.z + unitDist(rng) * size.z
        );
        particleCurrPosition.at(i) = pos;
        particlePrevPosition.at(i) = pos;
        particleVelocity.at(i) = glm::vec3(0.0f);
    }

    updatePositionBuffer();
}

size_t Engine::hashGridCoord(int x, int y, int z) {
    return (x * PRIME1 ^ y * PRIME2 ^ z * PRIME3) % tableSize;
}

void Engine::buildGrid() {
    std::fill(tempCounts.begin(), tempCounts.end(), 0);
    for (int i = 0; i < (int)particleCurrPosition.size(); i++) {
        auto cell = glm::floor(particleCurrPosition.at(i) / smoothingRadius);
        size_t hash = hashGridCoord((int)cell.x, (int)cell.y, (int)cell.z);
        tempCounts[hash]++;
    }

    cellStart[0] = 0;
    for (int i = 1; i < tableSize; i++){
        cellStart[i] = cellStart[i - 1] + tempCounts[i - 1];
    }

    std::fill(tempCounts.begin(), tempCounts.end(), 0);
    for (int i = 0; i < (int)particleCurrPosition.size(); i++) {
        auto cell = glm::floor(particleCurrPosition.at(i) / smoothingRadius);
        size_t hash = hashGridCoord((int)cell.x, (int)cell.y, (int)cell.z);
        int writeIndex = cellStart[hash] + tempCounts[hash]++;
        particleIndices[writeIndex] = i;
    }
}

void Engine::findNeighbors(int i, std::vector<int>& neighbors) {
    glm::ivec3 base = glm::floor(particleCurrPosition.at(i) / smoothingRadius);
    float r2 = smoothingRadius * smoothingRadius;

    for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {
    for (int dz = -1; dz <= 1; dz++) {
        glm::ivec3 neighborCell = base + glm::ivec3(dx, dy, dz);
        size_t hash = hashGridCoord(neighborCell.x, neighborCell.y, neighborCell.z);

        int start = cellStart[hash];
        int end = (hash + 1 < tableSize) ? cellStart[hash + 1] : (int)particleIndices.size();

        for (int j = start; j < end; ++j) {
            int neighborIdx = particleIndices[j];
            if (neighborIdx == i) continue;
            if (glm::length2(particleCurrPosition.at(neighborIdx) - particleCurrPosition.at(i)) < r2)
                neighbors.push_back(neighborIdx);
        }
    }
    }
    }

}

void Engine::buildCellStart() {
    cellStart[0] = 0;
    for (int i = 1; i < tableSize; ++i) {
        cellStart[i] = cellStart[i - 1] + tempCounts[i - 1];
    }
}

void Engine::calculateLambdas(int i){
    glm::vec3 particle = particleCurrPosition.at(i);
            
            // Compute density
            float density = 0.0f;

            for (int j : particleNeighbors[i]) {
                glm::vec3 neighbor = particleCurrPosition.at(j);

                float r = glm::length(particle - neighbor);
                if (r * r < (smoothingRadius * smoothingRadius)){
                    density += densityKernel(r);
                }
            }
    
            float Ci = (density / restDensity) - 1.0f;
    
            // Compute ∑ |∇C_i|²
            float gradSum = 0.0f;
            glm::vec3 grad_i(0.0f);

            for (int j : particleNeighbors[i]) {
                glm::vec3 neighbor = particleCurrPosition.at(j);

                glm::vec3 grad = (gradientKernel(glm::length(particle - neighbor)) * glm::normalize(particle - neighbor)) / restDensity;
                
                gradSum += glm::dot(grad, grad);
                grad_i += grad;
            }

            grad_i = -grad_i;
            gradSum += glm::dot(grad_i, grad_i); // include self term
    
            particleLambdas.at(i) = -Ci / (gradSum + epsilon);
}

void Engine::calculateDpiCollision(int i, float h){
    glm::vec3 particle = particleCurrPosition.at(i);
    glm::vec3 delta(0.0f);
    for (int j : particleNeighbors[i]) {
        glm::vec3 neighbor = particleCurrPosition.at(j);

        glm::vec3 grad = gradientKernel(glm::length(particle - neighbor)) * glm::normalize(particle - neighbor);
        
        delta += (particleLambdas.at(i) + particleLambdas.at(j)) * grad;
    }
    delta /= restDensity;
    positionDeltas.at(i) = delta;
    particleCollisions.at(i) = clampDeltaToBounds(i);
    //Apply dpi
    particleCurrPosition.at(i) += positionDeltas.at(i);
    //Set velocity, bounce 
    particleVelocity.at(i) = (1/h) * (particleCurrPosition.at(i) - particlePrevPosition.at(i));
    particleVelocity.at(i).x *= particleCollisions.at(i).x;
    particleVelocity.at(i).y *= particleCollisions.at(i).y;
    particleVelocity.at(i).z *= particleCollisions.at(i).z;
}

void Engine::updateParticlePositions(){
    //Dispatch reset persistent data
    resetPersistentParticleData();
    float h = 1.0/60.0;

    //Dispatch predict position
    //Unconstrained Step (Position Prediction)
    for(int i = 0; i < particleCurrPosition.size(); i++){
        particlePrevPosition.at(i) = particleCurrPosition.at(i);
        particleVelocity.at(i) += (h / particleMass) * (particleMass * gravityForce); //- damping * velocity)
        particleCurrPosition.at(i) += (particleVelocity.at(i) * h);
    }
    //Build Grid
    buildGrid();
    //Calculate neighbors
    std::for_each(
        std::execution::par_unseq,
        particleCurrPosition.begin(),
        particleCurrPosition.end(),
        [&](const auto& particle) {
            int i = &particle - &particleCurrPosition[0]; // index
            findNeighbors(i, particleNeighbors.at(i));
        }
    );
    
    //Apply Constraints
    for(uint32_t s = 0; s < solverIterations; s++){
        std::for_each(
            std::execution::par_unseq,
            particleCurrPosition.begin(),
            particleCurrPosition.end(),
            [&](const auto& particle) {
                int i = &particle - &particleCurrPosition[0]; // index
                calculateLambdas(i);
                calculateDpiCollision(i, h);
            }
        );
    }

    //Vorticity
    std::for_each(
        std::execution::par_unseq,
        particleCurrPosition.begin(),
        particleCurrPosition.end(),
        [&](const auto& p) {
            int i = &p - &particleCurrPosition[0]; // index
            glm::vec3 particle = particleCurrPosition.at(i);
            for (int j : particleNeighbors[i]) {
                glm::vec3 neighbor = particleCurrPosition.at(j);

                glm::vec3 vij = particleVelocity.at(i) - particleVelocity.at(j);
                glm::vec3 grad = gradientKernel(glm::length(particle - neighbor)) * glm::normalize(particle - neighbor);
                
                particleVorticity[i] += glm::cross(vij, grad);
            } 
        }
    );

    //Vorticity gradient
    std::for_each(
        std::execution::par_unseq,
        particleCurrPosition.begin(),
        particleCurrPosition.end(),
        [&](const auto& p) {
            int i = &p - &particleCurrPosition[0]; // index
            glm::vec3 particle = particleCurrPosition.at(i);
            for (int j : particleNeighbors[i]) {
                glm::vec3 neighbor = particleCurrPosition.at(j);
                float dist = glm::length(neighbor - particle);
                if(dist > epsilon){
                    float diff = glm::length(particleVorticity.at(j)) - glm::length(particleVorticity.at(i));
                    particleVorticityGradient.at(i) += (diff / dist) * ((neighbor - particle) / (dist + epsilon));
                }
            
            }
            if (glm::length(particleVorticityGradient.at(i)) > epsilon){
                particleVorticityGradient.at(i) = glm::normalize(particleVorticityGradient.at(i));
            }
            else{
                particleVorticityGradient.at(i) = glm::vec3(0.0f);
            }
            
            //Apply vorticity force
            particleVelocity.at(i) += (h * (vorticityEpsilon * glm::cross(particleVorticityGradient.at(i), particleVorticity.at(i))));
        }
    );
    
    //Apply viscosity force
    std::for_each(
        std::execution::par_unseq,
        particleCurrPosition.begin(),
        particleCurrPosition.end(),
        [&](const auto& p) {
            int i = &p - &particleCurrPosition[0]; // index
            glm::vec3 particle = particleCurrPosition.at(i);
            for (int j : particleNeighbors[i]) {
                glm::vec3 neighbor = particleCurrPosition.at(j);            
                glm::vec3 vij = particleVelocity.at(i) - particleVelocity.at(j);
                float scaling = densityKernel(glm::length(particle - neighbor));
                particleViscosityDelta.at(i) += scaling * vij;
            }
            particleVelocity.at(i) += viscosity * particleViscosityDelta.at(i);
            particleCurrPosition.at(i) = particlePrevPosition.at(i) + (h * particleVelocity.at(i));
        }
    );
}

void Engine::gpuUpdateParticlePositions(VkCommandBuffer cmd){

        VkMemoryBarrier2 uberBarrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT
        };

        VkDependencyInfo dependencyInfo = {};
        dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependencyInfo.memoryBarrierCount = 1;
        dependencyInfo.pMemoryBarriers = &uberBarrier;

        vkCmdPipelineBarrier2(cmd , &dependencyInfo);
        ComputePushConstants cpcs;
        cpcs.particleBuffer = particleBufferAddress;
        cpcs.gridCounter = gridCountBufferAddress;
        cpcs.gridCells = gridCellsBufferAddress;
        cpcs.neighborCount = neighborCountBufferAddress;
        cpcs.neighborList = neighborListBufferAddress;
        cpcs.timestep = 1.0f/60.0f;
        cpcs.smoothingRadius = smoothingRadius;
        cpcs.restDensity = restDensity;
        cpcs.particleMass = particleMass;
        cpcs.minBoundingPos = minBoundingPos;
        cpcs.maxBoundingPos = maxBoundingPos;

        VkPipelineBindPoint compute = VK_PIPELINE_BIND_POINT_COMPUTE;
        uint32_t particleX = (instanceCount + 63) / 64;
        uint32_t tableX = (tableSize + 63) / 64;
        
        //Dispatch predict position
        vkCmdBindPipeline(cmd, compute, computePredict);
        vkCmdPushConstants(cmd, computePipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(ComputePushConstants), &cpcs);
        vkCmdDispatch(cmd, particleX, 1, 1);
        //Dispatch reset persistent data
        vkCmdBindPipeline(cmd, compute, computeResetParticles);
        vkCmdPushConstants(cmd, computePipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(ComputePushConstants), &cpcs);
        vkCmdDispatch(cmd, particleX, 1, 1);
        //Dispatch reset grid
        vkCmdBindPipeline(cmd, compute, computeResetGrid);
        vkCmdPushConstants(cmd, computePipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(ComputePushConstants), &cpcs);
        vkCmdDispatch(cmd, tableX, 1, 1);
        //Grid depends on current positions, barrier
        vkCmdPipelineBarrier2(cmd, &dependencyInfo);
        
        //Dispatch grid build
        vkCmdBindPipeline(cmd, compute, computeGrid);
        vkCmdPushConstants(cmd, computePipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(ComputePushConstants), &cpcs);
        vkCmdDispatch(cmd, particleX, 1, 1);
        //Neighbors depend on built grid
        vkCmdPipelineBarrier2(cmd, &dependencyInfo);
        
        //Dispatch create neighbor list
        vkCmdBindPipeline(cmd, compute, computeNeighbors);
        vkCmdPushConstants(cmd, computePipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(ComputePushConstants), &cpcs);
        vkCmdDispatch(cmd, particleX, 1, 1);
        //Solving constraints depends on neighbor list
        vkCmdPipelineBarrier2(cmd, &dependencyInfo);
        

        for(uint32_t s = 0; s < solverIterations; s++){
            //Dispatch solve constraints
            vkCmdBindPipeline(cmd, compute, computeLambdas);
            vkCmdPushConstants(cmd, computePipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(ComputePushConstants), &cpcs);
            vkCmdDispatch(cmd, particleX, 1, 1);
            vkCmdPipelineBarrier2(cmd, &dependencyInfo);

            vkCmdBindPipeline(cmd, compute, computeDeltas);
            vkCmdPushConstants(cmd, computePipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(ComputePushConstants), &cpcs);
            vkCmdDispatch(cmd, particleX, 1, 1);    
            vkCmdPipelineBarrier2(cmd, &dependencyInfo);

            

        }
        vkCmdBindPipeline(cmd, compute, computeCollision);
        vkCmdPushConstants(cmd, computePipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(ComputePushConstants), &cpcs);
        vkCmdDispatch(cmd, particleX, 1, 1);    
        vkCmdPipelineBarrier2(cmd, &dependencyInfo);
        
        // vkCmdBindPipeline(cmd, compute, computeVorticity);
        // vkCmdPushConstants(cmd, computePipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(ComputePushConstants), &cpcs);
        // vkCmdDispatch(cmd, particleX, 1, 1);    
        // vkCmdPipelineBarrier2(cmd, &dependencyInfo);
        // vkCmdBindPipeline(cmd, compute, computeVorticityGradient);
        // vkCmdPushConstants(cmd, computePipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(ComputePushConstants), &cpcs);
        // vkCmdDispatch(cmd, particleX, 1, 1);    
        // vkCmdPipelineBarrier2(cmd, &dependencyInfo);
        // vkCmdBindPipeline(cmd, compute, computeViscosity);
        // vkCmdPushConstants(cmd, computePipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(ComputePushConstants), &cpcs);
        // vkCmdDispatch(cmd, particleX, 1, 1);    
        // vkCmdPipelineBarrier2(cmd, &dependencyInfo);
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

    // if(frameNumber < 100){
        gpuUpdateParticlePositions(cmd);
    // }
    //Draw
    drawMeshes(cmd);

    drawImage->transitionTo(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
	m_swapchain->images.at(index).transitionTo(cmd, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    drawImage->copyTo(cmd, m_swapchain->images.at(index));
    m_swapchain->images.at(index).transitionTo(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    drawDearImGui(cmd, m_swapchain->images.at(index).view);

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
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
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
    
    
    PushConstants pcs;
    // for (const MeshData& mesh : meshes) {
    //     pcs.vb_addr = mesh.vertex_buffer_address;
    //     pcs.model = mesh.model_mat;
    //     vkCmdPushConstants(cmd, mesh_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants), &pcs);
    //     vkCmdBindIndexBuffer(cmd, mesh.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    //     vkCmdDrawIndexed(cmd, mesh.index_count, 1, 0, 0, 0);
    // }

    pcs.worldMatrix = cam->getRenderMatrix();
	pcs.vertexBuffer = testMeshes.at(0).data.vertexBufferAddress;
    pcs.particleBuffer = particleBufferAddress;
    pcs.camWorldPos = cam->getPosition();
    
	vkCmdPushConstants(cmd, meshPipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(PushConstants), &pcs);
	vkCmdBindIndexBuffer(cmd, testMeshes.at(0).data.indexBuffer->buffer, 0, VK_INDEX_TYPE_UINT32);
    
	vkCmdDrawIndexed(cmd, testMeshes.at(0).surfaces.at(0).count, instanceCount, testMeshes.at(0).surfaces.at(0).startIndex, 0, 0);

    vkCmdEndRendering(cmd);
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
        uint64_t currentTime = SDL_GetTicksNS() - initializationTime;
        deltaTime = (float)(currentTime - lastTime)/1e9f;
        while(SDL_PollEvent(&e) != 0){
            ImGui_ImplSDL3_ProcessEvent(&e);
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
            {
                switch(e.key.key){
                    case SDLK_ESCAPE:
                        quit = true;
                        break;
                    case SDLK_Q:
                        mouseCaptured = !mouseCaptured;
                        SDL_SetWindowRelativeMouseMode(m_pWindow, mouseCaptured);
                        break;
                    case SDLK_W:
                        cam->updateVelocity(glm::vec3(0.0f, 0.0f, 1.0f));
                        break;
                    case SDLK_S:
                        cam->updateVelocity(glm::vec3(0.0f, 0.0f, -1.0f));
                        break;
                    case SDLK_A:
                        cam->updateVelocity(glm::vec3(-1.0f, 0.0f, 0.0f));
                        break;
                    case SDLK_D:
                        cam->updateVelocity(glm::vec3(1.0f, 0.0f, 0.0f));
                        break;
                    case SDLK_1:
                    {
                        minBoundingPos.x += 0.5;
                        break;
                    }
                    case SDLK_2:
                    {
                        minBoundingPos.x -= 0.5;
                        break;
                    }
                    case SDLK_3:
                    {
                        maxBoundingPos.x += 0.5;
                        break;
                    }
                    case SDLK_4:
                    {
                        maxBoundingPos.x -= 0.5;
                        break;
                    }               
                    case SDLK_R:
                    {
                        resetParticlePositions();
                        break;
                    }
                    case SDLK_G:
                    {
                        randomizeParticlePositions();
                        break;
                    }

                }
                break;
            }
            case SDL_EVENT_KEY_UP:
            {
                switch(e.key.key){
                    case SDLK_ESCAPE:
                        quit = true;
                        break;
                    case SDLK_W:
                        cam->updateVelocity(glm::vec3(0.0f, 0.0f, -1.0f));
                        break;
                    case SDLK_S:
                        cam->updateVelocity(glm::vec3(0.0f, 0.0f, 1.0f));
                        break;
                    case SDLK_A:
                        cam->updateVelocity(glm::vec3(1.0f, 0.0f, 0.0f));
                        break;
                    case SDLK_D:
                        cam->updateVelocity(glm::vec3(-1.0f, 0.0f, 0.0f));
                        break;
                }
                break;
            }
            case SDL_EVENT_MOUSE_MOTION:
            {
                if(mouseCaptured){
                    cam->updateLook(e.motion.xrel, e.motion.yrel);
                }
                break;
            }
            default:
                break;
            }
        }
        cam->updatePosition(deltaTime);

        //stopRendering
            //continue

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
        if (ImGui::Begin("background")) {		
			ImGui::Text("Parameters:");
			ImGui::Text("gravRotation: %.6f", gravityRotation);
			ImGui::InputFloat("radius", &smoothingRadius, 0.25f, 1.0f, "%.6f");
			ImGui::InputFloat("restDensity", &restDensity, 100.0f, 1000.0f, "%.6f");
			ImGui::InputFloat("epsilon", &epsilon, 0.00001f, 0.001f, "%.6f");
			ImGui::InputFloat("vorticity", &vorticityEpsilon, 0.01f, 0.1f, "%.6f");
			ImGui::InputFloat("viscosity", &viscosity, 0.0001f, 0.01f, "%.6f");
			ImGui::InputFloat3("minPoint", (float*)&minBoundingPos);
			ImGui::InputFloat3("maxPoint", (float*)&maxBoundingPos);
		}
		ImGui::End();

        ImGui::Render();

        // if(frameNumber < 100){
            // updateParticlePositions();
            // updatePositionBuffer();
        // }
        draw();
        if(frameNumber % 144 == 0){
            double frameTime = (SDL_GetTicksNS() - initializationTime - currentTime) / 1e6;
            char title[64];
            sprintf(title, "Fluid Sim - %.4f ms / %.4f fps", frameTime, 1.0/ (frameTime / 1e3));
            SDL_SetWindowTitle(m_pWindow, title);
        }
        lastTime = currentTime; 
    }
}



//TODO
//change it all into a compute shader
