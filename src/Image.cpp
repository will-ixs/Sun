#include "Image.hpp"

Image::Image(VkDevice device, VmaAllocator allocator, VkExtent3D imgExtent, VkFormat imgFormat, 
    VkImageUsageFlags usage, VkImageAspectFlags aspectFlags, VmaAllocationCreateFlags vmaAllocFlags)
:
m_device(device), m_allocator(allocator), extent(imgExtent), format(imgFormat)
{
    createImage(usage, vmaAllocFlags);
    createImageView(aspectFlags);
}

Image::~Image()
{
    if(!cleanedUp){
        destroy();
    }
}

void Image::createImage(VkImageUsageFlags usage, VmaAllocationCreateFlags vmaAllocFlags){
    VkImageCreateInfo imageInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = extent,
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
    };

    VmaAllocationCreateInfo allocInfo{
        .flags = vmaAllocFlags,
        .usage = VMA_MEMORY_USAGE_AUTO
    };

    if (vmaCreateImage(m_allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image");
    }
}

void Image::createImageView(VkImageAspectFlags aspect){

    VkImageSubresourceRange imageRange = {
        .aspectMask = aspect,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1
    };

    VkImageViewCreateInfo viewInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .subresourceRange = imageRange
    };

    if (vkCreateImageView(m_device, &viewInfo, nullptr, &view) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image view");
    }
}

void Image::destroy(){
    if (view != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device, view, nullptr);
        view = VK_NULL_HANDLE;
    }
    if (image != VK_NULL_HANDLE) {
        vmaDestroyImage(m_allocator, image, allocation);
        image = VK_NULL_HANDLE;
        allocation = VK_NULL_HANDLE;
    }
    cleanedUp = true;
}

void Image::transitionImage(){
//TODO
}

void Image::copyImage(){
//TODO
}