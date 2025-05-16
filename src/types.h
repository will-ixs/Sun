#ifndef TYPES_H
#define TYPES_H

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
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
    glm::mat4 modelMat;
	uint32_t indexCount;
};

struct TextureData {
    std::shared_ptr<Image> texture;
    VkSampler sampler;
};

struct MaterialData{
    glm::vec4 baseColor;
    float metallicFactor;
    float roughnessFactor;

    uint32_t baseColorIndex;
    uint32_t metallicRoughnessIndex;
    uint32_t normalIndex;
};

struct Surface {
    uint32_t startIndex;
    uint32_t count;
    MaterialData matData;
};

struct MeshAsset {
    std::string name;
    std::vector<Surface> surfaces;
    MeshData data;
};

struct GLTFNode{
    uint32_t parent = -1;
    std::vector<uint32_t> childrenIndices;
    
    glm::mat4 modelMat;

    glm::vec3 position;
    glm::vec3 scale;
    glm::quat rotation;

    std::shared_ptr<MeshAsset> mesh;
};


struct GLTFScene{
    std::vector<GLTFNode> nodes;
    std::vector<uint32_t> rootNodes;
};

enum RenderPass {
    OPAQUE,
    TRANSPARENT,
    RenderPassSize
};

struct Renderable{
    RenderPass type;
    uint32_t nodeIndex;
    uint32_t surfaceIndex;
};

struct UniformBufferObject{
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewproj;
    glm::vec4 ambientColor;
    glm::vec4 sunlightDirection;
    glm::vec4 sunlightColor;

    VkDeviceAddress materialBuffer;
};

#endif