#include "Image.hpp"
#include <stdexcept>

Image::Image(VkDevice device, VmaAllocator allocator, VkExtent3D imageExtent, VkFormat imageFormat, 
    VkImageUsageFlags usage, VkImageAspectFlags imageAspect, VmaAllocationCreateFlags vmaAllocFlags)
:
m_device(device), m_allocator(allocator), extent(imageExtent), format(imageFormat), aspect(imageAspect)
{
    createImage(usage, vmaAllocFlags);
    createImageView();
}


Image::Image(VkDevice device, VkImage image, VkImageView imageView, VkExtent3D imageExtent, VkFormat imageFormat, 
    VkImageAspectFlags imageAspect, bool swapchainImage = false)
:
m_device(device), image(image), view(imageView), extent(imageExtent), format(imageFormat), m_swapchain(swapchainImage), aspect(imageAspect)
{
    if(swapchainImage){
        aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    }
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

void Image::createImageView(){
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
    if(m_swapchain){
        return;
    }

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

void Image::transitionTo(VkCommandBuffer cmd, VkImageLayout oldLayout, VkImageLayout newLayout, bool isTransfer){
    layout = newLayout;
    VkImageSubresourceRange range = {
        .aspectMask = aspect,
        .baseMipLevel = 0,
        .levelCount = VK_REMAINING_MIP_LEVELS,
        .baseArrayLayer = 0,
        .layerCount = VK_REMAINING_ARRAY_LAYERS        
    };
    
    VkImageMemoryBarrier2 imageBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .image = image,
        .subresourceRange = range
    };

    switch (oldLayout) {
        case VK_IMAGE_LAYOUT_UNDEFINED:
            imageBarrier.srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            imageBarrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            imageBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            imageBarrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            imageBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            imageBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            imageBarrier.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            imageBarrier.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            break;
        case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
            imageBarrier.srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            imageBarrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            break;
        default:
            throw std::runtime_error("Unsupported oldLayout transition!");
    }

    if(isTransfer){
        imageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        imageBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    }else{
        switch (newLayout) {
            case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
                imageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                imageBarrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                break;
            case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
                imageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                imageBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
                break;
            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
                imageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                imageBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;
                break;
            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                imageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                imageBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
                break;
            case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
                imageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
                imageBarrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT;
                break;
            case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
                imageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
                imageBarrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                break;
            default:
                throw std::runtime_error("Unsupported newLayout transition!");
        }
    }

    

    VkDependencyInfo dependencyInfo{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext = nullptr,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &imageBarrier
    };
    
    vkCmdPipelineBarrier2(cmd, &dependencyInfo);
}

void Image::copyTo(VkCommandBuffer cmd, Image& dstImage){
    VkOffset3D zeroOffset = {
        0, 0, 0
    };
    VkOffset3D srcOffset = {
        (int32_t)extent.width, (int32_t)extent.height, (int32_t)extent.depth
    };
    VkOffset3D dstOffset = {
        (int32_t)dstImage.extent.width, (int32_t)dstImage.extent.height, (int32_t)dstImage.extent.depth
    };
    VkImageSubresourceLayers layers = {
        .aspectMask = aspect,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1
    };

    VkImageBlit2 blit = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
        .pNext = nullptr,
        .srcSubresource = layers,
        .dstSubresource = layers
    };
    blit.srcOffsets[0] = zeroOffset;
    blit.dstOffsets[0] = zeroOffset;
    blit.srcOffsets[1] = srcOffset;
    blit.dstOffsets[1] = dstOffset;

	VkBlitImageInfo2 blitInfo = {
        .sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
        .pNext = nullptr,
        .srcImage = image,
        .srcImageLayout = layout,
        .dstImage = dstImage.image,
        .dstImageLayout = dstImage.layout,
        .regionCount = 1,
        .pRegions = &blit,
        .filter = VK_FILTER_LINEAR
    };

	vkCmdBlitImage2(cmd, &blitInfo);
}
