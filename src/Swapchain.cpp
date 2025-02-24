#include "Swapchain.hpp"


Swapchain::Swapchain(VkDevice device, VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) 
:
m_device(device), m_physicalDevice(physicalDevice), m_surface(surface)
{
}

Swapchain::~Swapchain()
{
}

void Swapchain::createSwapchain(uint32_t width, uint32_t height){
    vkb::SwapchainBuilder swapchainBuilder{ m_physicalDevice, m_device, m_surface};

	imageFormat = VK_FORMAT_B8G8R8A8_UNORM;

	vkb::Swapchain vkbSwapchain = swapchainBuilder
		.set_desired_format(
            VkSurfaceFormatKHR{ 
                .format = imageFormat, 
                .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR 
            })
		.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
		.set_desired_extent(width, height)
		.add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
		.build()
		.value();

	extent = vkbSwapchain.extent;
	images = vkbSwapchain.get_images().value();
	imageViews = vkbSwapchain.get_image_views().value();

	swapchain = vkbSwapchain.swapchain;
}

void Swapchain::resizeSwapchain(uint32_t width, uint32_t height){
    vkDeviceWaitIdle(m_device);

    destroySwapchain();
    
    createSwapchain(width, height);
}

void Swapchain::destroySwapchain(){
	vkDestroySwapchainKHR(m_device, swapchain, nullptr);
	
    for (uint32_t i = 0; i < imageViews.size(); i++) {
		vkDestroyImageView(m_device, imageViews[i], nullptr);
	}
}
