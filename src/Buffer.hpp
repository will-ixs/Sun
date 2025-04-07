#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <iostream>
#include <stdexcept>

class Buffer
{
private:
    VkDevice m_device;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    bool cleanedUp = false;
public:
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation;
    VmaAllocationInfo allocationInfo;
    size_t size;

    Buffer() = delete;
    Buffer(VkDevice device, VmaAllocator allocator, size_t allocSize, VkBufferUsageFlags bufferUsage,
         VmaMemoryUsage memoryUsage, VmaAllocationCreateFlags createFlags);
    ~Buffer();

    void createBuffer(VkBufferUsageFlags bufferUsage, VmaMemoryUsage memoryUsage, VmaAllocationCreateFlags createFlags);
    void destroy();
    
    void copyTo(VkCommandBuffer cmd, Buffer& other, VkDeviceSize dst, VkDeviceSize src);
};
