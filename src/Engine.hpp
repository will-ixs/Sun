#ifndef ENGINE_HPP
#define ENGINE_HPP

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <vector>
#include <memory>

struct FrameData{
	VkCommandPool commandPool;
	VkCommandBuffer commandBuffer;

    VkSemaphore swapchainSemaphore;
    VkSemaphore renderSemaphore;
    VkFence renderFence;
};

struct Buffer{
    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo info;  
};

struct PushConstants{
    int modelIndex;
    int textureIndex;
    int materialIndex;
};

class Swapchain;
class Image;
class PipelineBuilder;

class Engine
{
private:
    //Initialization
    void initSDL3();
    void initVulkan();
    void initSwapchain();
    void initDrawResources();
    void initCommands();
    void initSynchronization();
    void initDescriptors();
    void initPipelines();

    //Util
    bool loadShader(VkShaderModule* outShader, const char* filePath);
    
    //Pipelines
    void initMeshPipeline();

    //drawing
    void draw();
    void drawMeshes(VkCommandBuffer cmd);

    //Optons
    bool m_bUseValidation = false;
    bool m_bUseDebugMessenger = false;
    
    struct SDL_Window* m_pWindow {nullptr};
    uint32_t windowWidth;
    uint32_t windowHeight;

    //Vulkan Resources
    VkInstance m_instance;
    VkDebugUtilsMessengerEXT m_debugMessenger;
    VkPhysicalDevice m_physicalDevice;
    VkDevice m_device;
    VkSurfaceKHR m_surface;
    VmaAllocator m_allocator;

    //Swapchain
    std::unique_ptr<Swapchain> m_swapchain;

    //Descriptors
    VkDescriptorPool m_descriptorPool;
    VkDescriptorSetLayout m_descriptorLayout;
    VkDescriptorSet m_descriptorSet;
    
    const uint32_t STORAGE_BINDING = 0;
    const uint32_t SAMPLER_BINDING = 1;
    const uint32_t IMAGE_BINDING = 2;
    uint32_t STORAGE_COUNT = 65536;
    uint32_t SAMPLER_COUNT = 65536;
    uint32_t IMAGE_COUNT = 65536;
    
    //Pipelines
    VkPipelineLayout meshPipelineLayout;
    VkPipeline meshPipeline;
    std::unique_ptr<PipelineBuilder> pb;

    //Queue Info
	VkQueue m_graphicsQueue;
	uint32_t m_graphicsQueueFamily;

    //Draw Resources
    std::unique_ptr<Image> drawImage;
    std::unique_ptr<Image> depthImage;
    VkExtent2D drawExtent; 

    bool cleanedUp;
    
public:
    Engine();
    ~Engine();
    void init();
    void run();
    void cleanup();

    FrameData frameData[2];
	FrameData& getCurrentFrame() { return frameData[frameNumber % 2]; };
    uint64_t frameNumber = 0;
};

#endif