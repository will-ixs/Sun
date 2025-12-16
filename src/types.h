#ifndef TYPES_H
#define TYPES_H

#include <vulkan/vulkan.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>
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
    VkDeviceAddress particleBuffer;
    glm::vec3 camWorldPos;
};

struct alignas(16) ComputePushConstants{
    VkDeviceAddress particleBuffer;
    VkDeviceAddress gridCounter;
    VkDeviceAddress gridCells;
    VkDeviceAddress neighborCount;
    VkDeviceAddress neighborList;
    float timestep;
    float smoothingRadius;
    float restDensity;
    float particleMass;
    alignas(16) glm::vec3 minBoundingPos;
    alignas(16) glm::vec3 maxBoundingPos;
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
    glm::vec3 currPosition;
    glm::vec3 prevPosition;
    glm::vec3 velocity;    
    glm::vec3 positionDeltas;
    glm::vec3 particleCollisions;
    glm::vec3 particleVorticity;
    glm::vec3 particleVorticityGradient;
    glm::vec3 particleViscosityDelta;
    float particleLambdas;
};

#endif