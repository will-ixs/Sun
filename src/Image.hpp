#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

class Image
{
private:
    VkDevice m_device;
    VmaAllocator m_allocator;
    bool cleanedUp = false;
public:
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_NONE;
    VkExtent3D extent{};
    VkFormat format{};
    VmaAllocation allocation = VK_NULL_HANDLE;

    Image() = delete;
    Image(VkDevice device, VmaAllocator allocator, VkExtent3D extent, VkFormat format, 
        VkImageUsageFlags usage, VkImageAspectFlags aspectFlags, VmaAllocationCreateFlags vmaAllocFlags);
    ~Image();

    void createImage(VkImageUsageFlags usage, VmaAllocationCreateFlags vmaAllocFlags);
    void createImageView();
    void destroy();

    void transitionTo(VkCommandBuffer cmd, VkImageLayout oldLayout, VkImageLayout newLayout);
    void copyTo(VkCommandBuffer cmd, const Image& dstImage);

    static void transitionVulkanImage(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout);
};
