#ifndef TYPES_H
#define TYPES_H

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <unordered_map>
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
    glm::mat4 modelMatrix;
    VkDeviceAddress vertexBuffer;
    uint32_t materialIndex;
};

struct Vertex {
	glm::vec3 position;
	float uv_x;
	glm::vec3 normal;
	float uv_y;
	glm::vec4 color;
	glm::vec4 tangent;
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

struct EngineStats {
    float initTime;
    float frameTime;
    float sceneUpdateTime;
    float meshDrawTime;
    int triCount;
    int drawCallCount;
};

struct alignas(16) MaterialData{
    glm::vec4 baseColor;
    float metallicFactor;
    float roughnessFactor;

    uint32_t baseColorIndex;
    uint32_t metallicRoughnessIndex;
    uint32_t normalIndex;
};

enum RenderPass {
    OPAQUE,
    TRANSPARENT,
    RenderPassSize
};

struct Bounds{
    glm::vec3 origin;
    glm::vec3 extents;
};

struct Surface {
    RenderPass type;
    uint32_t startIndex;
    uint32_t count;
    uint32_t matIndex;
    Bounds bounds;
};

struct MeshAsset {
    std::string name;
    std::vector<Surface> surfaces;
    MeshData data;
};

struct GLTFNode{
    std::weak_ptr<GLTFNode> parent;
    std::vector<std::shared_ptr<GLTFNode> > children;
    
    glm::mat4 localTransform;
    glm::mat4 worldTransform;

    std::shared_ptr<MeshAsset> mesh;
    int32_t lightIndex = -1;

    void refreshTransform(const glm::mat4& parentMatrix)
    {
        worldTransform = parentMatrix * localTransform;
        for (auto c : children) {
            c->refreshTransform(worldTransform);
        }
    }
};

struct GLTFScene{
    std::unordered_map<std::string, std::shared_ptr<MeshAsset>> meshes;
    std::unordered_map<std::string, std::shared_ptr<GLTFNode>> nodes;
    std::unordered_map<std::string, std::shared_ptr<Image>> images;
    std::unordered_map<std::string, uint32_t> materialIndices;

    std::vector<std::shared_ptr<GLTFNode>> topNodes;
    std::vector<VkSampler> samplers;
};

struct Renderable{
    VkDeviceAddress vertexBufferAddress;
    VkBuffer indexBuffer;
    uint32_t indexCount;
    uint32_t firstIndex;
    uint32_t materialIndex;
    glm::mat4 modelMat;
    Bounds bounds;
};

struct UniformBufferObject{
    glm::mat4 viewproj;
    glm::vec4 camPos;
    glm::vec4 ambientColor;
    glm::vec4 sunlightDirection;
    glm::vec4 sunlightColor;

    VkDeviceAddress materialBuffer;
    VkDeviceAddress lightBuffer;
    uint32_t numLights;
};

struct alignas(16) Light{
    alignas(16) uint32_t type;
    alignas(16) glm::vec3 lightPos;   
    alignas(16) glm::vec3 lightDir;
    alignas(16) glm::vec3 lightColor; //VVVVV fastgltf descriptions VVVVVVV
    /** Point and spot lights use candela (lm/sr) while directional use lux (lm/m^2) */
    float intensity;
    /** Range for point and spot lights. If not present, range is infinite. */
    float range;
	/** The inner and outer cone angles only apply to spot lights */
    float innerConeAngle;
    float outerConeAngle;
};

#endif