#ifndef TYPES_H
#define TYPES_H

#include <glm/glm.hpp>
#include <memory>

class Buffer;
class Image;

struct FrameData{
	VkCommandPool commandPool;
	VkCommandBuffer commandBuffer;

    VkSemaphore swapchainSemaphore;
    VkSemaphore renderSemaphore;
    VkFence renderFence;
};

struct ImmediateTransfer{
    VkCommandPool pool;
    VkCommandBuffer buffer;
    VkFence fence;
};

struct PushConstants{
    glm::mat4 worldMatrix;
    VkDeviceAddress vertexBuffer; 
    uint32_t textureIndex;
    uint32_t materialIndex;
};

struct Vertex {
	glm::vec3 position;
	float uv_x;
	glm::vec3 normal;
	float uv_y;
	glm::vec4 color;
};

struct MeshData {
    std::unique_ptr<Buffer> indexBuffer;
    std::unique_ptr<Buffer> vertexBuffer;
    VkDeviceAddress vertexBufferAddress;	
    glm::mat4 model_mat;
	uint32_t indexCount;
};


#endif