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
    vkDeviceWaitIdle(device);

    meshThread.join();
    
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    vkDestroyPipelineLayout(device, meshPipelineLayout, nullptr);
    vkDestroyPipeline(device, meshPipelineOpaque, nullptr);
    vkDestroyPipeline(device, meshPipelineTransparent, nullptr);
    vkDestroyPipelineLayout(device, particleDrawPipelineLayout, nullptr);
    vkDestroyPipelineLayout(device, particleComputePipelineLayout, nullptr);
    vkDestroyPipeline(device, particleDrawPipeline, nullptr);
    
    
    vkDestroyDescriptorPool(device, descriptorPoolDefault, nullptr);
    vkDestroyDescriptorPool(device, descriptorPoolImgui, nullptr);
	vkDestroyDescriptorSetLayout(device, descriptorLayoutBindless, nullptr);
    
    for (int i = 0; i < 2; i++) {
        vkDestroyCommandPool(device, frameData[i].graphicsCommandPool, nullptr);
        vkDestroyCommandPool(device, frameData[i].computeCommandPool, nullptr);

        vkDestroyFence(device, frameData[i].renderFence, nullptr);
        vkDestroyFence(device, frameData[i].computeFence, nullptr);
        vkDestroySemaphore(device, frameData[i].acquireSemaphore, nullptr);
    }

    
    vkDestroyCommandPool(device, immTransfer.pool, nullptr);
    vkDestroyFence(device, immTransfer.fence, nullptr);

    drawImage->destroy();
    resolveImage->destroy();
    depthImage->destroy();

    for(MeshAsset& mesh : testMeshes){
        mesh.data.indexBuffer->destroy();
        mesh.data.vertexBuffer->destroy();
    }

    for(auto& [sceneName, scene] : loadedGLTFs){
        for(VkSampler sampler : scene->samplers){
            vkDestroySampler(device, sampler, nullptr);
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

    vkDestroyPipelineCache(device, particleComputePipelineCache, nullptr);
    for(auto& [type, pipeline]: particlePipelineMap){
        vkDestroyPipeline(device, pipeline, nullptr);
    }

    for(auto& sema : particleSystemGarbage){
        vkDestroySemaphore(device, sema, nullptr);
    }

    for(auto& ps: particleSystems){
        ps.devicePositionBufferA->destroy();
        ps.devicePositionBufferB->destroy();
        ps.deviceVelocityBuffer->destroy();
        vkDestroySemaphore(device, ps.particleTLSemaphore, nullptr);
    }

    checkerboardImage->destroy();

    vkDestroySampler(device, defaultLinearSampler, nullptr);
    vkDestroySampler(device, defaultNearestSampler, nullptr);

    vmaDestroyAllocator(vmaAllocator);

    for(size_t i = 0; i < swapchain->submitSemaphores.size(); i++){
        vkDestroySemaphore(device, swapchain->submitSemaphores.at(i), nullptr);
    }
    swapchain->destroySwapchain();

	vkDestroySurfaceKHR(instance, surface, nullptr);
	vkDestroyDevice(device, nullptr);
		
	vkb::destroy_debug_utils_messenger(instance, debugMessenger);
	vkDestroyInstance(instance, nullptr);
	SDL_DestroyWindow(pWindow);

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
    pWindow = SDL_CreateWindow("Sun", 1600, 900, SdlWindowFlags);
    mouseCaptured = false;
    SDL_SetWindowRelativeMouseMode(pWindow, mouseCaptured);
    SDL_CaptureMouse(mouseCaptured);
}

void Engine::initVulkan(){
    vkb::InstanceBuilder instanceBuilder;

    vkb::SystemInfo sysInfo = vkb::SystemInfo::get_system_info().value();

    if (sysInfo.validation_layers_available && useValidation) {
        instanceBuilder.request_validation_layers();
    }
    if (sysInfo.debug_utils_available && useDebugMessenger) {
        instanceBuilder.use_default_debug_messenger();
    }
    instanceBuilder.require_api_version(1, 3, 0);

    vkb::Result<vkb::Instance> instanceBuilderRet = instanceBuilder.build();
    if (!instanceBuilderRet) {
        throw std::runtime_error(instanceBuilderRet.error().message() + "\n");
    }
    vkb::Instance vkbInstance = instanceBuilderRet.value();
    instance = vkbInstance.instance;

    if(vkbInstance.debug_messenger && useDebugMessenger){
        debugMessenger = vkbInstance.debug_messenger;
    }       

    
    if(!SDL_Vulkan_CreateSurface(pWindow, instance, nullptr, &surface)){
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
        .timelineSemaphore = VK_TRUE,
        .bufferDeviceAddress = VK_TRUE,
    };
	VkPhysicalDeviceVulkan11Features features11{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .shaderDrawParameters = VK_TRUE
    };
    vkb::PhysicalDeviceSelector vkbSelector{ vkbInstance };
	vkb::PhysicalDevice vkbPhysicalDevice = vkbSelector
		.set_minimum_version(1, 3)
		.set_required_features_13(features13)
		.set_required_features_12(features12)
        .set_required_features_11(features11)
		.set_surface(surface)
		.select()
		.value();
    
    vkb::DeviceBuilder vkbBuilder { vkbPhysicalDevice };

    vkb::Device vkbDevice = vkbBuilder.build().value();
    uint32_t sampler = vkbPhysicalDevice.properties.limits.maxDescriptorSetSamplers;
    uint32_t buffer = vkbPhysicalDevice.properties.limits.maxDescriptorSetStorageBuffers;
    uint32_t images = vkbPhysicalDevice.properties.limits.maxDescriptorSetStorageImages;
    if(sampler < SAMPLER_COUNT){
        SAMPLER_COUNT = sampler;
    }
    if(buffer < STORAGE_COUNT){
        STORAGE_COUNT = buffer;
    }
    if(images < IMAGE_COUNT){
        IMAGE_COUNT = images;
    }

    physicalDevice = vkbPhysicalDevice.physical_device;
    device = vkbDevice.device;

    graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();
    computeQueue = vkbDevice.get_queue(vkb::QueueType::compute).value();
    computeQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::compute).value();
    // m_computeQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    // m_computeQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();
    transferQueue = vkbDevice.get_queue(vkb::QueueType::transfer).value();
    transferQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::transfer).value();
    // m_transferQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    // m_transferQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    VmaAllocatorCreateInfo allocatorInfo = {
        .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = physicalDevice,
        .device = device,
        .instance = instance
    };
    vmaCreateAllocator(&allocatorInfo, &vmaAllocator);
}

void Engine::initCommands(){

    VkCommandPoolCreateInfo graphicsCommandPoolInfo =  {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr, 
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = graphicsQueueFamily
    };

    VkCommandPoolCreateInfo computeCommandPoolInfo =  {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr, 
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = computeQueueFamily
    };
	
	for (int i = 0; i < 2; i++) {

		vkCreateCommandPool(device, &graphicsCommandPoolInfo, nullptr, &frameData[i].graphicsCommandPool);
		vkCreateCommandPool(device, &computeCommandPoolInfo, nullptr, &frameData[i].computeCommandPool);

		VkCommandBufferAllocateInfo cmdAllocInfo = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = frameData[i].graphicsCommandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1
        };

		vkAllocateCommandBuffers(device, &cmdAllocInfo, &frameData[i].graphicsCommandBuffer);

        cmdAllocInfo.commandPool = frameData[i].computeCommandPool;
		vkAllocateCommandBuffers(device, &cmdAllocInfo, &frameData[i].computeCommandBuffer);
	}

    VkCommandPoolCreateInfo transferCommandPoolInfo =  {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr, 
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = transferQueueFamily
    };
    vkCreateCommandPool(device, &transferCommandPoolInfo, nullptr, &immTransfer.pool);
    VkCommandBufferAllocateInfo cmdAllocInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = immTransfer.pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    vkAllocateCommandBuffers(device, &cmdAllocInfo, &immTransfer.buffer);
    
}

void Engine::initSwapchain(){
    windowWidth = 1600;
    windowHeight = 900;
    swapchain = std::make_unique<Swapchain>(device, physicalDevice, surface);
    swapchain->createSwapchain(windowWidth, windowHeight);
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

    drawImage = std::make_unique<Image>(device, vmaAllocator, 
        drawImgExtent, drawImgFormat, drawImgUsage,
        VK_IMAGE_ASPECT_COLOR_BIT,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        VK_SAMPLE_COUNT_4_BIT, 1
    );

    resolveImage = std::make_unique<Image>(device, vmaAllocator, 
        drawImgExtent, drawImgFormat, drawImgUsage,
        VK_IMAGE_ASPECT_COLOR_BIT,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        VK_SAMPLE_COUNT_1_BIT, 1
    );

    depthImage = std::make_unique<Image>(device, vmaAllocator, 
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
		vkCreateFence(device, &fenceInfo, nullptr, &frameData[i].renderFence);
		vkCreateFence(device, &fenceInfo, nullptr, &frameData[i].computeFence);

		vkCreateSemaphore(device, &semInfo, nullptr, &frameData[i].acquireSemaphore);
	}

    vkCreateFence(device, &fenceInfo, nullptr, &immTransfer.fence);
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

    vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPoolDefault);

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

    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorLayoutBindless);

    VkDescriptorSetAllocateInfo setInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = descriptorPoolDefault,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptorLayoutBindless
    };

    vkAllocateDescriptorSets(device, &setInfo, &descriptorSetBindless);
}

void Engine::initPipelines(){
    pb = std::make_unique<PipelineBuilder>();
    slang::createGlobalSession(slangGlobalSession.writeRef());
    slangTargets = {
        {
            .format{SLANG_SPIRV},
            .profile{slangGlobalSession->findProfile("spirv_1_4")}
        }
    };
    slangOptions = {
        {
            slang::CompilerOptionName::EmitSpirvDirectly,
            {slang::CompilerOptionValueKind::Int, 1}
        }
    };
    slangDefaultSessionDesc = {
        .targets{slangTargets.data()},
        .targetCount{SlangInt(slangTargets.size())},
        .defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR,
        .compilerOptionEntries{slangOptions.data()},
        .compilerOptionEntryCount{uint32_t(slangOptions.size())}
    };
    initPipelineLayouts();
    initMeshPipelines();
    initParticlePipelines();

    
    registerDefaultParticleSystems();
    createParticleSystem("explode", 100000, 5.0f, glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(3.0f));
    // createParticleSystem("chua", 100000, 30.0f, glm::vec3(0.0f), glm::vec3(2.0f));
}

void Engine::initPipelineLayouts(){
    /////// MESH
    VkPushConstantRange meshPc = {
        .stageFlags = VK_SHADER_STAGE_ALL,
        .offset = 0,
        .size = sizeof(PushConstants)
    };

	VkPipelineLayoutCreateInfo meshLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptorLayoutBindless,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &meshPc
    };
	vkCreatePipelineLayout(device, &meshLayoutInfo, nullptr, &meshPipelineLayout);

    ////// Particle Draw
    VkPushConstantRange partPc = {
        .stageFlags = VK_SHADER_STAGE_ALL,
        .offset = 0,
        .size = sizeof(ParticlePushConstants)
    };    

	VkPipelineLayoutCreateInfo partLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptorLayoutBindless,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &partPc
    };
	vkCreatePipelineLayout(device, &partLayoutInfo, nullptr, &particleDrawPipelineLayout);


    /////// Compute Cache & Layout    if (particlePipelineMap.empty()){
    VkPushConstantRange compPc = {
        .stageFlags = VK_SHADER_STAGE_ALL,
        .offset = 0,
        .size = sizeof(ParticleComputePushConstants)
    };
    VkPipelineLayoutCreateInfo compLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptorLayoutBindless,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &compPc
    };
    vkCreatePipelineLayout(device, &compLayoutInfo, nullptr, &particleComputePipelineLayout); 

    VkPipelineCacheCreateInfo particleComputeCacheInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        .pNext = nullptr
    };
    vkCreatePipelineCache(device, &particleComputeCacheInfo, nullptr, &particleComputePipelineCache);
}

void Engine::initMeshPipelines(){
    
    VkShaderModule phong;
    if(loadShader(&phong, "../../shaders/rendering/phong.slang")){
        std::cout << "Successfully built phong shader." << std::endl;
    }

	pb->clear();
	pb->pipeline_layout = meshPipelineLayout;
	pb->setShaders(phong, phong);
	pb->setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
	pb->setPolygonMode(VK_POLYGON_MODE_FILL);
	pb->setCullingMode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	pb->setMultisampling(VK_SAMPLE_COUNT_4_BIT);
	pb->disableBlending();
	pb->enableDepthtest(VK_TRUE, VK_COMPARE_OP_GREATER_OR_EQUAL);
	pb->setColorAttachmentFormat(drawImage->format);
	pb->setDepthAttachmentFormat(depthImage->format);
	meshPipelineOpaque = pb->buildPipeline(device);

    pb->enableBlending(VK_BLEND_FACTOR_ONE);
    meshPipelineTransparent = pb->buildPipeline(device);

	vkDestroyShaderModule(device, phong, nullptr);
}

void Engine::initParticlePipelines(){
    VkShaderModule particle;
	if (loadShader(&particle, "../../shaders/rendering/particle.slang")) {
		std::cout << "Successfully built particle shader." << std::endl;
	}

    
	pb->clear();
	pb->pipeline_layout = particleDrawPipelineLayout;
	pb->setShaders(particle, particle);
	pb->setTopology(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);
	pb->setPolygonMode(VK_POLYGON_MODE_FILL);
	pb->setCullingMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	pb->setMultisampling(VK_SAMPLE_COUNT_4_BIT);
	pb->enableBlending(VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA);
	pb->enableDepthtest(VK_TRUE, VK_COMPARE_OP_GREATER_OR_EQUAL);
	pb->setColorAttachmentFormat(drawImage->format);
	pb->setDepthAttachmentFormat(depthImage->format);
	particleDrawPipeline = pb->buildPipeline(device);

	vkDestroyShaderModule(device, particle, nullptr);
}

void Engine::initData(){
    cam = std::make_unique<Camera>((float)drawExtent.width, (float)drawExtent.height);
    
    //Create UBO Buffer
    uboBuffer = std::make_unique<Buffer>(device, vmaAllocator, sizeof(UniformBufferObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
    VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT );
    
    ubo.camPos = cam->getPos();
    ubo.viewproj = glm::mat4(1.0f) * cam->getProjMatrix() * cam->getViewMatrix();
    ubo.ambientColor = glm::vec4(0.05f, 0.05f, 0.05f, 1.0f);
    ubo.sunlightColor = glm::vec4(1.0f, 1.0f, 1.0f, 100.0f); //w is sunlight intensity
    ubo.sunlightDirection = glm::normalize(glm::vec4(1.0f, 1.0f, 1.0f, 0.0f));
    
    memcpy(uboBuffer->allocationInfo.pMappedData, &ubo, sizeof(UniformBufferObject));

    //Material Buffer
    materials.reserve(32);
    VkBufferUsageFlags materialUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    materialBuffer = std::make_unique<Buffer>(device, vmaAllocator, sizeof(MaterialData) * materials.capacity(), 
        materialUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);
    
    lightsPoint.reserve(32);
    VkBufferUsageFlags lightUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    lightBuffer = std::make_unique<Buffer>(device, vmaAllocator, sizeof(Light) * lightsPoint.capacity(), 
        lightUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);

    VkBufferDeviceAddressInfo bdaInfo {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = nullptr,
        .buffer = materialBuffer->buffer
    };
    materialBufferAddress = vkGetBufferDeviceAddress(device, &bdaInfo);
    bdaInfo.buffer = lightBuffer->buffer;
    lightBufferAddress = vkGetBufferDeviceAddress(device, &bdaInfo);


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
	vkCreateSampler(device, &sampler, nullptr, &defaultNearestSampler);
	sampler.magFilter = VK_FILTER_LINEAR;
	sampler.minFilter = VK_FILTER_LINEAR;
	vkCreateSampler(device, &sampler, nullptr, &defaultLinearSampler);

    //Update descriptors
    VkDescriptorBufferInfo uboInfo = {
        .buffer = uboBuffer->buffer,
        .offset = 0,
        .range = sizeof(UniformBufferObject)
    };
    VkWriteDescriptorSet uboWrite {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr, 
        .dstSet = descriptorSetBindless,
        .dstBinding = UBO_BINDING,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pImageInfo = nullptr,
        .pBufferInfo = &uboInfo,
        .pTexelBufferView = nullptr
    };
    vkUpdateDescriptorSets(device, 1, &uboWrite, 0, nullptr);

    meshThread = std::thread(&Engine::meshUploader, this);
    
    meshPathQueue.push("..\\..\\resources\\khrgltf_sponza_lit\\Sponza.gltf");
    // pathQueue.push("..\\..\\resources\\structure.glb");
    meshPathQueue.push("..\\..\\resources\\Duck.glb");
    //TODO: figure out what test scene am able to publish (has ok license)
    //billboard quad next?
    meshPathQueue.push("QUIT");
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

    vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPoolImgui);

    ImGui::CreateContext();
    ImGui_ImplSDL3_InitForVulkan(pWindow);
    ImGui_ImplVulkan_InitInfo imguiVulkanInfo = {
        .Instance = instance,
        .PhysicalDevice = physicalDevice,
        .Device = device,
        .QueueFamily = graphicsQueueFamily,
        .Queue = graphicsQueue,
        .DescriptorPool = descriptorPoolImgui,
        .MinImageCount = 2,
        .ImageCount = 2,
        .MSAASamples = VK_SAMPLE_COUNT_1_BIT,
        .UseDynamicRendering = true
    };
    imguiVulkanInfo.PipelineRenderingCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .pNext = nullptr,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &swapchain->format
    };
    
    ImGui::CreateContext();
    ImGui_ImplVulkan_Init(&imguiVulkanInfo);
}

//Utility
bool Engine::loadShader(VkShaderModule* outShader, std::string filePath) {    
    Slang::ComPtr<slang::ISession> slangSession;
    slangGlobalSession->createSession(slangDefaultSessionDesc, slangSession.writeRef());
    Slang::ComPtr<slang::IModule> slangModule{ slangSession->loadModuleFromSource(filePath.c_str(), filePath.c_str(), nullptr, nullptr) };
	Slang::ComPtr<ISlangBlob> spirv;
	slangModule->getTargetCode(0, spirv.writeRef());

	VkShaderModuleCreateInfo shader = { 
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, 
        .pNext = nullptr,
        .codeSize = spirv->getBufferSize(), 
        .pCode = (uint32_t*)spirv->getBufferPointer() 
    };

	if(vkCreateShaderModule(device, &shader, nullptr, outShader) != VK_SUCCESS){
        std::cout << "Failed to create shader module at " << filePath << std::endl;
        return false;
    }
    return true;
}

void Engine::reloadShaders(){
    vkDeviceWaitIdle(device);
    
    vkDestroyPipeline(device, meshPipelineOpaque, nullptr);
    vkDestroyPipeline(device, meshPipelineTransparent, nullptr);
    
    for (auto& [name, pipeline] : particlePipelineMap) {
        vkDestroyPipeline(device, pipeline, nullptr);

        glm::vec3 defVelocity = particleVelocityMap.at(name);
        registerParticleSystem(name, defVelocity);
    };
    
    initMeshPipelines();
    
    std::cout << "Shaders reloaded!" << std::endl;
}

std::shared_ptr<Image> Engine::createImageFromData(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped){
    size_t dataSize = size.depth * size.width * size.height * sizeof(float);
    VkBufferUsageFlags stagingUsage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    Buffer stagingBuffer = Buffer(device, vmaAllocator, dataSize, stagingUsage, 
        VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

	if (stagingBuffer.allocationInfo.pMappedData == nullptr) {
		throw std::runtime_error("Staging buffer not mapped.");
	}
	memcpy(stagingBuffer.allocationInfo.pMappedData, data, dataSize);

    std::unique_ptr<Image> image = std::make_unique<Image>(device, vmaAllocator, size, format, 
        usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 
        VK_IMAGE_ASPECT_COLOR_BIT, 0,
        VK_SAMPLE_COUNT_1_BIT, 1
    );


    prepImmediateTransfer();
    
    image->transitionTo(immTransfer.buffer, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, true);

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
    vkCmdCopyBufferToImage(immTransfer.buffer, stagingBuffer.buffer, image->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
    
    image->transitionTo(immTransfer.buffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, true);

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

    mesh.vertexBuffer = std::make_unique<Buffer>(device, vmaAllocator, vertexBufferSize, vertexUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);
    mesh.indexBuffer = std::make_unique<Buffer>(device, vmaAllocator, indexBufferSize, indexUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);
    std::unique_ptr<Buffer> stagingBuffer = std::make_unique<Buffer>(device, vmaAllocator, vertexBufferSize + indexBufferSize, stagingUsage, VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);


	VkBufferDeviceAddressInfo addressInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = nullptr,
        .buffer = mesh.vertexBuffer->buffer
    };
	mesh.vertexBufferAddress = vkGetBufferDeviceAddress(device, &addressInfo);
	
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

	vkCmdCopyBuffer(immTransfer.buffer, stagingBuffer->buffer, mesh.vertexBuffer->buffer, 1, &vert_copy);

	VkBufferCopy ind_copy = {};
	ind_copy.size = indexBufferSize;
	ind_copy.srcOffset = vertexBufferSize;
	ind_copy.dstOffset = 0;

	vkCmdCopyBuffer(immTransfer.buffer, stagingBuffer->buffer, mesh.indexBuffer->buffer, 1, &ind_copy);

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
    ubo.numLights = static_cast<uint32_t>(lightsPoint.size());
    memcpy(uboBuffer->allocationInfo.pMappedData, &ubo, sizeof(UniformBufferObject));
}

void Engine::updateGUI(){
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        //Engine Stats
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

            ImGui::Separator();

            ImGui::Text("Active Elements");
            ImGui::Text("Scenes");
            for(auto& [name, scene] : loadedGLTFs){
                ImGui::Checkbox(name.c_str(), &scene->enabled);
            }
            ImGui::Text("Particle Systems");
            for(auto& ps: particleSystems){
                ImGui::Checkbox(ps.type.c_str(), &ps.enabled);
            }
        }
		ImGui::End();

        //Particle Spawner
        if (particlePipelineMap.size() > 0){
            std::vector<std::string_view> particleNames;
            particleNames.reserve(particlePipelineMap.size());
            for(const auto& [name, _]: particlePipelineMap){
                particleNames.push_back(name);
            }
            
            if (ImGui::Begin("Add Particles")) {
                if (ImGui::BeginCombo("Particle Type", particleNames.at(particleCreation.selIdx).data())){
                    for (int i = 0; i < particleNames.size(); i++)
                    {
                        if (ImGui::Selectable(particleNames.at(i).data(), particleCreation.selIdx == i)){
                            particleCreation.selIdx = i;
                        }
                    }
                    ImGui::EndCombo();
                }
                ImGui::InputFloat3("Origin", &particleCreation.origin.x);
                ImGui::InputFloat3("Origin Variance", &particleCreation.oVar.x);
                ImGui::InputFloat3("Velocity Variance", &particleCreation.vVar.x);

                ImGui::InputFloat("Lifetime", &particleCreation.lifeTime, 1.0f, 10.0f);
                ImGui::InputInt("Num Particles", &particleCreation.numParticles, 64, 64 * 64);
                particleCreation.numParticles = std::clamp(particleCreation.numParticles, 0, INT32_MAX); //max actually is uint32 max
                
                if(ImGui::Button("Spawn ParticleSystem")){
                    createParticleSystem(particleNames.at(particleCreation.selIdx).data(), particleCreation.numParticles,
                        particleCreation.lifeTime, particleCreation.origin, particleCreation.oVar, particleCreation.vVar);
                }
            }
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
        .dstSet = descriptorSetBindless,
        .dstBinding = SAMPLER_BINDING,
        .dstArrayElement = index,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &samplerInfo,
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr
    };
    vkUpdateDescriptorSets(device, 1, &samplerWrite, 0, nullptr);

    textures.push_back(data);
    return index;
}

uint32_t Engine::addMaterial(const MaterialData& data, std::string name){
    if(matNameToIndex.find(name) != matNameToIndex.end()){
        return matNameToIndex.at(name);
    }
    //Check if this material will cause buffer to resize
    bool willResize = materials.size() == materials.capacity() ? true : false;

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
        
        materialBuffer = std::make_unique<Buffer>(device, vmaAllocator, materials.capacity() * sizeof(MaterialData), 
            materialUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);

        VkBufferDeviceAddressInfo bdaInfo {
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .pNext = nullptr,
            .buffer = materialBuffer->buffer
        };
        materialBufferAddress = vkGetBufferDeviceAddress(device, &bdaInfo);
    }
    
    Buffer stagingBuffer = Buffer(device, vmaAllocator, materials.capacity() * sizeof(MaterialData), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
	if (stagingBuffer.allocationInfo.pMappedData == nullptr) {
		throw std::runtime_error("Material staging buffer not mapped.");
	}

	memcpy(stagingBuffer.allocationInfo.pMappedData, materials.data(), materials.size() * sizeof(MaterialData));
	prepImmediateTransfer();

	vkCmdCopyBuffer(immTransfer.buffer, stagingBuffer.buffer, materialBuffer->buffer, 1, &stagingCopy);

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
        
    lightBuffer = std::make_unique<Buffer>(device, vmaAllocator, lightsPoint.size() * sizeof(Light), 
        lightUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);

    VkBufferDeviceAddressInfo bdaInfo {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = nullptr,
        .buffer = lightBuffer->buffer
    };
    lightBufferAddress = vkGetBufferDeviceAddress(device, &bdaInfo);

    Buffer stagingBuffer = Buffer(device, vmaAllocator, lightsPoint.size() * sizeof(Light), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
	
    if (stagingBuffer.allocationInfo.pMappedData == nullptr) {
		throw std::runtime_error("Material staging buffer not mapped.");
	}

	memcpy(stagingBuffer.allocationInfo.pMappedData, lightsPoint.data(), lightsPoint.size() * sizeof(Light));
	prepImmediateTransfer();
	vkCmdCopyBuffer(immTransfer.buffer, stagingBuffer.buffer, lightBuffer->buffer, 1, &stagingCopy);
    submitImmediateTransfer();
    
}

bool Engine::loadGLTF(std::filesystem::path filePath){
    std::cout << "Loading GLTF: " << filePath << std::endl;
    
    std::shared_ptr<GLTFScene> scene = std::make_shared<GLTFScene>();
    scene->enabled = true;
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
        vkCreateSampler(device, &samplerCreateInfo, nullptr, &vksampler);
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
            newNode->lightIndex = static_cast<int32_t>(node.lightIndex.value());
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
                lights.at(node->lightIndex).lightDir = glm::normalize(worldDir);
            }else{
                lights.at(node->lightIndex).lightDir = glm::vec4(0.0f);
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
		if (meshPathQueue.empty()) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			continue;
		}
		
		std::filesystem::path res = meshPathQueue.front();
		meshPathQueue.pop();
		if (res.generic_string() == "QUIT") {
			break;
		}

        loadGLTF(res);
	}
    std::cout << "Mesh Uploader exiting..." << std::endl;
}

//Transfer
void Engine::prepImmediateTransfer(){
    vkResetFences(device, 1, &immTransfer.fence);
	vkResetCommandBuffer(immTransfer.buffer, 0);

	VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };

	vkBeginCommandBuffer(immTransfer.buffer, &begin);
}

void Engine::submitImmediateTransfer(){

    vkEndCommandBuffer(immTransfer.buffer);

	VkCommandBufferSubmitInfo cmdInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext = nullptr,
        .commandBuffer = immTransfer.buffer
    };
	VkSubmitInfo2 submit {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext = nullptr,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &cmdInfo,
    };
	vkQueueSubmit2(transferQueue, 1, &submit, immTransfer.fence);

	vkWaitForFences(device, 1, &immTransfer.fence, true, 9999999999);
}

//Provides default particle systems.
void Engine::registerDefaultParticleSystems(){
    registerParticleSystem("lorenz", glm::vec3(0.0, 1.0, 1.05)); //?enum for default position, RANDOM, CENTERED : rn all random in 1x1 unit cube
    registerParticleSystem("chen", glm::vec3(0.1, 0.3, -0.6));
    registerParticleSystem("chua", glm::vec3(1.0, 1.0, 0.0));
    registerParticleSystem("rossler", glm::vec3(0.0, 0.0, 0.0));
    registerParticleSystem("explode");
}

//Register a particle system. Save its default values to created and rendered at any time.
void Engine::registerParticleSystem(std::string name, glm::vec3 defaultVelocity){
    VkShaderModule comp;
    std::string shaderFileName = "../../shaders/particle_compute/particle_" + name + ".slang";
	if (loadShader(&comp, shaderFileName)) {
        std::cout << "Successfully built " << shaderFileName << " shader module." << std::endl;
    }

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
    VkPipeline computePipeline;
	vkCreateComputePipelines(device, particleComputePipelineCache, 1, &computePipelineCreateInfo, nullptr, &computePipeline);
    particlePipelineMap.insert_or_assign(name, computePipeline);
    particleVelocityMap.insert_or_assign(name, defaultVelocity);

    vkDestroyShaderModule(device, comp, nullptr);
}

//Create an active instance of a particle system that has been registered.
void Engine::createParticleSystem(std::string name, uint32_t particleCount, float lifeTime, glm::vec3 originPosition, glm::vec3 originVariance, glm::vec3 velocityVariance){
    ParticleSystem ps = {
        .type = name,
        .particleCount = particleCount,
        .lifeTime = lifeTime * 1000.0f,
        .spawnTime =  static_cast<double>(getTime()) / 1e6,
        //device buffers
        //buffer addressess
        //semaphore
        .particleTLValue = 0,
        .framesAlive = 0,
        .originPos = originPosition,
        .enabled = true
    };

    if (particleVelocityMap.count(name) == 0){
        std::cout << "Failed to read " << name << " particle velocity. Make sure you register it before creating it." << std::endl;
        return;
    }    
    if (particlePipelineMap.count(name) == 0){
        std::cout << "Failed to read " << name << " particle pipeline. Make sure you register it before creating it." << std::endl;
        return;
    }

    glm::vec4 defaultVel = glm::vec4(particleVelocityMap.at(name), 0.0f);

    initialPositions.resize(particleCount);
    initialVelocities.resize(particleCount);
    std::minstd_rand rng(std::random_device{}());
    std::uniform_real_distribution<float> unitDist(-1.0f, 1.0f);

    //better locality?
    //originPos added in shader so particle calculations are done at highest precision
    for (size_t i = 0; i < particleCount; ++i) {
        float mag = std::fabs(unitDist(rng));
        glm::vec3 randVec = glm::normalize(glm::vec3(unitDist(rng), unitDist(rng), unitDist(rng))) * mag;
        initialPositions.at(i) = glm::vec4( randVec.x * 0.5f * originVariance.x, 
                                            randVec.y * 0.5f * originVariance.y,
                                            randVec.z * 0.5f * originVariance.z, 0.0f);
    }

    for (size_t i = 0; i < particleCount; ++i) {
        float mag = std::fabs(unitDist(rng));
        glm::vec3 randVec = glm::normalize(glm::vec3(unitDist(rng), unitDist(rng), unitDist(rng))) * mag;
        
        initialVelocities.at(i) = defaultVel + glm::vec4(   randVec.x * 0.5f * velocityVariance.x, 
                                                            randVec.y * 0.5f * velocityVariance.y,
                                                            randVec.z * 0.5f * velocityVariance.z, 0.0f);
    }
    

    Buffer positionStagingBuffer = Buffer(device, vmaAllocator, particleCount * sizeof(glm::vec4),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, 
        VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

    Buffer velocityStagingBuffer = Buffer(device, vmaAllocator, particleCount * sizeof(glm::vec4),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, 
        VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

    ps.devicePositionBufferA = std::make_unique<Buffer>(device, vmaAllocator, particleCount * sizeof(glm::vec4), 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);
    
    ps.devicePositionBufferB = std::make_unique<Buffer>(device, vmaAllocator, particleCount * sizeof(glm::vec4), 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);
    
    ps.deviceVelocityBuffer = std::make_unique<Buffer>(device, vmaAllocator, particleCount * sizeof(glm::vec4), 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);

    VkBufferDeviceAddressInfo positionAddressInfoA = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = nullptr,
        .buffer = ps.devicePositionBufferA->buffer
    };
    VkBufferDeviceAddressInfo positionAddressInfoB = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = nullptr,
        .buffer = ps.devicePositionBufferB->buffer
    };
    VkBufferDeviceAddressInfo velocityAddressInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = nullptr,
        .buffer = ps.deviceVelocityBuffer->buffer
    };
    ps.positionBufferA = vkGetBufferDeviceAddress(device, &positionAddressInfoA);
    ps.positionBufferB = vkGetBufferDeviceAddress(device, &positionAddressInfoB);
    ps.velocityBuffer  = vkGetBufferDeviceAddress(device, &velocityAddressInfo );
    
    if (positionStagingBuffer.allocationInfo.pMappedData == nullptr) {
        throw std::runtime_error("Host position buffer not mapped.");
    }
    memcpy(positionStagingBuffer.allocationInfo.pMappedData, initialPositions.data(), sizeof(glm::vec4) * particleCount);
    if (velocityStagingBuffer.allocationInfo.pMappedData == nullptr) {
        throw std::runtime_error("Host position buffer not mapped.");
    }
    memcpy(velocityStagingBuffer.allocationInfo.pMappedData, initialVelocities.data(), sizeof(glm::vec4) * particleCount);

    VkBufferCopy posCopy = {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = particleCount * sizeof(glm::vec4)
    };
    prepImmediateTransfer();
    vkCmdCopyBuffer(immTransfer.buffer, positionStagingBuffer.buffer, ps.devicePositionBufferA->buffer, 1, &posCopy);
    vkCmdCopyBuffer(immTransfer.buffer, positionStagingBuffer.buffer, ps.devicePositionBufferB->buffer, 1, &posCopy);
    vkCmdCopyBuffer(immTransfer.buffer, velocityStagingBuffer.buffer, ps.deviceVelocityBuffer->buffer,  1, &posCopy);
    submitImmediateTransfer();

    VkSemaphoreTypeCreateInfo semTypeInfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = 0
    };
    VkSemaphoreCreateInfo timelineInfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &semTypeInfo
    };
    vkCreateSemaphore(device, &timelineInfo, nullptr, &ps.particleTLSemaphore);

    particleSystems.push_back(std::move(ps));
}

//Drawing loop. Meshes and particles.
void Engine::draw(){
    VkResult fenceResult = vkWaitForFences(device, 1, &getCurrentFrame().renderFence, VK_TRUE, 1000000000);
    if (fenceResult != VK_SUCCESS) {
        throw std::runtime_error("Fence wait failed!");
    }    
	uint32_t index;
	VkResult acquireResult = vkAcquireNextImageKHR(device, swapchain->swapchain, 1000000000, getCurrentFrame().acquireSemaphore, VK_NULL_HANDLE, &index);
	if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        int32_t w = 0;
        int32_t h = 0;
		SDL_GetWindowSizeInPixels(pWindow, &w, &h);
        swapchain->resizeSwapchain(w, h);
		return;
	}
	else if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acqurie swapchain image!");
	}
    
    vkResetFences(device, 1, &getCurrentFrame().renderFence);
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
    //Draw
    drawParticles(cmd);
    drawMeshes(cmd);

    resolveImage->transitionTo(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
	swapchain->images.at(index).transitionTo(cmd, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    resolveImage->copyTo(cmd, swapchain->images.at(index));

    swapchain->images.at(index).transitionTo(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    drawDearImGui(cmd, swapchain->images.at(index).view);
    swapchain->images.at(index).transitionTo(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

	vkEndCommandBuffer(cmd);

    VkCommandBufferSubmitInfo commandBufferSubmitInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext = nullptr,
        .commandBuffer = cmd,
        .deviceMask = 0
    };

    std::vector<VkSemaphoreSubmitInfo> semaphoreWaits;
    semaphoreWaits.reserve(particleSystems.size() + 1);
    semaphoreWaits.emplace_back(
        VkSemaphoreSubmitInfo { 
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .semaphore = getCurrentFrame().acquireSemaphore,
            .value = 1,
            .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR,
            .deviceIndex = 0
        }
    );

    for(const auto& ps: particleSystems){
        semaphoreWaits.emplace_back(
            VkSemaphoreSubmitInfo { 
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                .pNext = nullptr,
                .semaphore = ps.particleTLSemaphore,
                .value = ps.particleTLValue,
                .stageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                .deviceIndex = 0
            }
        );
    }

	VkSemaphoreSubmitInfo semaphoreSignalSubmitInfo{ 
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .semaphore = swapchain->submitSemaphores.at(index),
        .value = 1,
        .stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
        .deviceIndex = 0
    };
    VkSubmitInfo2 queueSubmitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext = nullptr,
        .flags = 0,
        .waitSemaphoreInfoCount = static_cast<uint32_t>(semaphoreWaits.size()),
        .pWaitSemaphoreInfos = semaphoreWaits.data(),
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &commandBufferSubmitInfo,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &semaphoreSignalSubmitInfo
    };

    VkResult submitRes = vkQueueSubmit2(graphicsQueue, 1, &queueSubmitInfo, getCurrentFrame().renderFence);
    if(submitRes != VK_SUCCESS){
        std::cout << "queue submit failed" << std::endl;
    }

    VkPresentInfoKHR presentInfo = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &swapchain->submitSemaphores.at(index),
        .swapchainCount = 1,
        .pSwapchains = &swapchain->swapchain,
        .pImageIndices = &index
    };

	VkResult presentRes = vkQueuePresentKHR(graphicsQueue, &presentInfo);
	if (presentRes == VK_ERROR_OUT_OF_DATE_KHR || presentRes == VK_SUBOPTIMAL_KHR) {
        int32_t w = 0;
        int32_t h = 0;
		SDL_GetWindowSizeInPixels(pWindow, &w, &h);
        swapchain->resizeSwapchain(w, h);
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
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    
    VkRenderingAttachmentInfo depth_attachment = {};
    depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth_attachment.pNext = nullptr;
    depth_attachment.imageView = depthImage->view;
    depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
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
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipelineLayout, 0, 1, &descriptorSetBindless, 0, nullptr);
    
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
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipelineLayout, 0, 1, &descriptorSetBindless, 0, nullptr);
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
    
    auto start = std::chrono::system_clock::now();    
    vkCmdBeginRendering(cmd, &rendering_info);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, particleDrawPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, particleDrawPipelineLayout, 0, 1, &descriptorSetBindless, 0, nullptr);
    
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

    for(const auto& ps: particleSystems){
        if(!ps.enabled){
            continue;
        }

        pcs.velocityBuffer = ps.velocityBuffer;        
        if(ps.framesAlive % 2 == 0){
            pcs.positionBuffer = ps.positionBufferB;
        }else{
            pcs.positionBuffer = ps.positionBufferA;
        }
        pcs.originPos = glm::vec4(ps.originPos, 0.0f);
        vkCmdPushConstants(cmd, particleDrawPipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(ParticlePushConstants), &pcs);
        vkCmdDraw(cmd, static_cast<uint32_t>(ps.particleCount), 1, 0, 0);
        stats.drawCallCount++;
    }

    vkCmdEndRendering(cmd);
    
    auto end = std::chrono::system_clock::now();    
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if(frameNumber % 144){
        stats.particleTime = elapsed.count() / 1000.0f;
    }
}

void Engine::updateParticles(){
    VkResult fenceResult = vkWaitForFences(device, 1, &getCurrentFrame().computeFence, VK_TRUE, 1000000000);
    if (fenceResult != VK_SUCCESS) {
        throw std::runtime_error("Fence wait failed!");
    }
    vkResetFences(device, 1, &getCurrentFrame().computeFence);
    VkCommandBuffer cmd = getCurrentFrame().computeCommandBuffer;
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo commandBufferBeginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr
    };

    double currTime = static_cast<double>(getTime()) / 1e6;
    //remove dead particleSystems
    for(auto it = particleSystems.begin(); it != particleSystems.end();){
        if (currTime > it->spawnTime + it->lifeTime){
            particleSystemGarbage.push_back(it->particleTLSemaphore);
            it = particleSystems.erase(it);
        }else{
            it++;
        }
    }

    vkBeginCommandBuffer(cmd, &commandBufferBeginInfo);
        ParticleComputePushConstants pcs;
        pcs.deltaTime = deltaTime;
        pcs.timeScale = 0.0005f * timeScale;
        pcs.time = static_cast<float>(currTime);

        for(const auto& ps : particleSystems){
            pcs.particleCount = static_cast<uint32_t>(ps.particleCount);
            uint32_t particleX = (pcs.particleCount + 63) / 64;
            pcs.velocityBuffer = ps.velocityBuffer;
            if(ps.framesAlive % 2 == 0){
                pcs.positionBufferA = ps.positionBufferA;
                pcs.positionBufferB = ps.positionBufferB;
            }else{
                pcs.positionBufferA = ps.positionBufferB;
                pcs.positionBufferB = ps.positionBufferA;
            }
            
            
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, particlePipelineMap.at(ps.type));
            vkCmdPushConstants(cmd, particleComputePipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(ParticleComputePushConstants), &pcs);
            vkCmdDispatch(cmd, particleX, 1, 1);
        }

	vkEndCommandBuffer(cmd);


    VkCommandBufferSubmitInfo commandBufferSubmitInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext = nullptr,
        .commandBuffer = cmd,
        .deviceMask = 0
    };
    std::vector<VkSemaphoreSubmitInfo> semaphoreSubmits;
    semaphoreSubmits.reserve(particleSystems.size());
    for(auto& ps: particleSystems){
        ps.particleTLValue++;
        ps.framesAlive++;
        semaphoreSubmits.emplace_back(
            VkSemaphoreSubmitInfo {
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                .pNext = nullptr,
                .semaphore = ps.particleTLSemaphore,
                .value = ps.particleTLValue,
                .stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .deviceIndex = 0
            }
        );
    }
    VkMemoryBarrier2 memoryBarrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT
    };
    
    VkDependencyInfo dependencyInfo = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext = nullptr,
        .dependencyFlags = 0,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &memoryBarrier,
        .bufferMemoryBarrierCount = 0,
        .pBufferMemoryBarriers = nullptr,
        .imageMemoryBarrierCount = 0,
        .pImageMemoryBarriers = nullptr
    };
    
    vkCmdPipelineBarrier2(cmd, &dependencyInfo);

    VkSubmitInfo2 queueSubmitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext = nullptr,
        .flags = 0,
        .waitSemaphoreInfoCount = 0,
        .pWaitSemaphoreInfos = nullptr,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &commandBufferSubmitInfo,
        .signalSemaphoreInfoCount = static_cast<uint32_t>(semaphoreSubmits.size()),
        .pSignalSemaphoreInfos = semaphoreSubmits.data()
    };

    VkResult submitRes = vkQueueSubmit2(computeQueue, 1, &queueSubmitInfo, getCurrentFrame().computeFence);
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
        .renderArea = VkRect2D { VkOffset2D { 0, 0 }, swapchain->extent },
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
                    SDL_SetWindowRelativeMouseMode(pWindow, mouseCaptured);
                    break;
                case SDLK_R:
                    reloadShaders();
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
            if(!scene->enabled){
                //skip disabled scenes here instead of in render loop, dont waste time doing frustum culling on them 
                continue;
            }

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