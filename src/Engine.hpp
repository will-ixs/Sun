#ifndef ENGINE_HPP
#define ENGINE_HPP

#include <vulkan/vulkan.h>
#include <vector>

struct FrameData {
	VkCommandPool commandPool;
	VkCommandBuffer commandBuffer;

    VkSemaphore swapchainSemaphore;
    VkSemaphore renderSemaphore;
    VkFence renderFence;
};

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

    VkSwapchainKHR m_swapchain;
    VkFormat m_swapchainImageFormat;

    std::vector<VkImage> m_swapchainImages;
    std::vector<VkImageView> m_swapchainImageViews;
    VkExtent2D m_swapchainExtent;
    void createSwapchain(uint32_t width, uint32_t height);
    void destroySwapchain();

    FrameData frameData[2];
	FrameData& getCurrentFrame() { return frameData[m_frameNumber % 2]; };
    uint64_t m_frameNumber = 0;

	VkQueue m_graphicsQueue;
	uint32_t m_graphicsQueueFamily;
    
public:
    Engine();
    ~Engine();
    void run();
};

#endif