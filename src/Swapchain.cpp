#include "Swapchain.hpp"
#include "Image.hpp"

Swapchain::Swapchain(VkDevice device, VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) 
:
m_device(device), m_physicalDevice(physicalDevice), m_surface(surface)
{
}

Swapchain::~Swapchain()
{
}

void Swapchain::createSwapchain(uint32_t width, uint32_t height, bool recreate){
    vkb::SwapchainBuilder swapchainBuilder{ m_physicalDevice, m_device, m_surface};

	format = VK_FORMAT_B8G8R8A8_UNORM;

	vkb::Swapchain vkbSwapchain = swapchainBuilder
		.set_desired_format(
            VkSurfaceFormatKHR{ 
                .format = format, 
                .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR 
            })
		.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
		.set_desired_extent(width, height)
		.add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
		.build()
		.value();

	extent = vkbSwapchain.extent;
	std::vector<VkImage> swapchainImages = vkbSwapchain.get_images().value();
	std::vector<VkImageView> swapchainImageViews = vkbSwapchain.get_image_views().value();
	swapchain = vkbSwapchain.swapchain;

	VkExtent3D imageExtent = {
		.width = extent.width,
		.height = extent.height,
		.depth = 1
	};
	VkSemaphoreCreateInfo semInfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = nullptr,
    };
	for(size_t i=0; i < swapchainImages.size(); i++){
		images.emplace_back(Image(m_device, swapchainImages.at(i), swapchainImageViews.at(i), imageExtent, format, VK_IMAGE_ASPECT_COLOR_BIT, true));
	}

	if(!recreate){
		presentComplete.resize(swapchainImages.size());
		for(size_t i=0; i < swapchainImages.size(); i++){
			vkCreateSemaphore(m_device, &semInfo, nullptr, &presentComplete.at(i));
		}
	}
}

void Swapchain::resizeSwapchain(uint32_t width, uint32_t height){
    vkDeviceWaitIdle(m_device);

    destroySwapchain();
    
    createSwapchain(width, height, true);
}

void Swapchain::destroySwapchain(){
	vkDestroySwapchainKHR(m_device, swapchain, nullptr);
	
    for(size_t i = 0; i < images.size(); i++){
		vkDestroyImageView(m_device, images.at(i).view, nullptr);
	}

	images.clear();
}
