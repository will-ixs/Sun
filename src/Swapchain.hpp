#ifndef SWAPCHAIN_HPP
#define SWAPCHAIN_HPP

#include <vulkan/vulkan.h>
#include <VkBootstrap.h>
#include <vector>

class Image;

class Swapchain
{
private:
    //Engine's resources
    VkDevice m_device;
    VkPhysicalDevice m_physicalDevice;
    VkSurfaceKHR m_surface;

public:    
    //Swapchain's resources
    VkSwapchainKHR swapchain;
    std::vector<Image> images;
    std::vector<VkSemaphore> presentComplete;
    VkExtent2D extent;
    VkFormat format;
    
    Swapchain() = delete;
    Swapchain(VkDevice device, VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);
    ~Swapchain();
    
    void createSwapchain(uint32_t width, uint32_t height, bool recreate = false);
    void resizeSwapchain(uint32_t width, uint32_t height);
    void destroySwapchain();
    
};

#endif