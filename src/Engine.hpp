#ifndef ENGINE_HPP
#define ENGINE_HPP

#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>

#include <vector>
#include <memory>

struct FrameData{
	VkCommandPool commandPool;
	VkCommandBuffer commandBuffer;

    VkSemaphore swapchainSemaphore;
    VkSemaphore renderSemaphore;
    VkFence renderFence;
};

struct Image{
    VkImage image;
    VkImageView view;
    VkExtent3D extent;
    VkFormat format;
    VmaAllocation allocation;
};

struct Buffer{
    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo info;  
};

class Swapchain;

class Engine
{
private:
    //Initialization
    void init();
    void initSDL3();
    void initVulkan();
    void initSwapchain();
    void initCommands();
    void initSynchronization();

    //drawing
    void draw();
    
    void transitionImage(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout);

    //Optons
    bool m_bUseValidation = false;
    bool m_bUseDebugMessenger = false;
    
    struct SDL_Window* m_pWindow {nullptr};

    //Vulkan Resources
    VkInstance m_instance;
    VkDebugUtilsMessengerEXT m_debugMessenger;
    VkPhysicalDevice m_physicalDevice;
    VkDevice m_device;
    VkSurfaceKHR m_surface;
    VmaAllocator m_allocator;

    //Swapchain
    std::unique_ptr<Swapchain> m_swapchain;

    //Draw Resources
    Image drawImage;
    VkExtent2D drawExtent; 

    //Queue Info
	VkQueue m_graphicsQueue;
	uint32_t m_graphicsQueueFamily;
    
public:
    Engine();
    ~Engine();
    void run();

    FrameData frameData[2];
	FrameData& getCurrentFrame() { return frameData[frameNumber % 2]; };
    uint64_t frameNumber = 0;
};

#endif