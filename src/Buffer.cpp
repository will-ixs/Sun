#include "Buffer.hpp"

Buffer::Buffer(VkDevice device, VmaAllocator allocator, size_t allocSize, VkBufferUsageFlags bufferUsage, VmaMemoryUsage memoryUsage, VmaAllocationCreateFlags createFlags)
:
m_device(device), m_allocator(allocator), size(allocSize)
{
    createBuffer(bufferUsage, memoryUsage, createFlags);
}

Buffer::~Buffer()
{
    if(!cleanedUp){
        destroy();
    }
}

void Buffer::createBuffer(VkBufferUsageFlags bufferUsage, VmaMemoryUsage memoryUsage, VmaAllocationCreateFlags createFlags){
    VkBufferCreateInfo bufferInfo{ 
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .size = size,
        .usage = bufferUsage
    };
    
    VmaAllocationCreateInfo allocationCreateInfo{
        .flags = createFlags, //vma flags- should set required flags and preferred flags 
        .usage = memoryUsage, //almost always AUTO or AUTO_PREFER_DEVICE/HOST
    };

    VkResult res = vmaCreateBuffer(m_allocator, &bufferInfo, &allocationCreateInfo, &buffer, &allocation, &allocationInfo);
    if(res != VK_SUCCESS){
        throw std::runtime_error("Failed to create buffer");
    }
}

void Buffer::destroy(){
    if (buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator, buffer, allocation);
        buffer = VK_NULL_HANDLE;
        allocation = VK_NULL_HANDLE;
    }
    cleanedUp = true;
}

void Buffer::copyTo(VkCommandBuffer cmd, Buffer& other, VkDeviceSize src, VkDeviceSize dst){
    VkBufferCopy copy{
        .srcOffset = src,
        .dstOffset = dst,
        .size = size
    };

	vkCmdCopyBuffer(cmd, buffer, other.buffer, 1, &copy);
}