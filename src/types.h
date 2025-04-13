#ifndef TYPES_H
#define TYPES_H

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <string>
#include <vector>

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
    uint32_t instanceIndex;
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
    glm::mat4 modelMat;
	uint32_t indexCount;
};

struct Surface {
    uint32_t startIndex;
    uint32_t count;
};

struct MeshAsset {
    std::string name;
    std::vector<Surface> surfaces;
    MeshData data;
};

struct ParticleData {
    glm::vec4 currPosition;
    glm::vec4 prevPosition;
    glm::vec4 velocity;
};

#endif