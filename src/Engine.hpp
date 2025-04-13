#ifndef ENGINE_HPP
#define ENGINE_HPP

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <vector>
#include <array>
#include <span>
#include <memory>
#include <SDL3/SDL_timer.h>
#include "types.h"

class Swapchain;
class Image;
class Buffer;
class PipelineBuilder;
class MeshUploader;
class Camera;

constexpr uint32_t sideLength = 15;
constexpr uint32_t instanceCount = sideLength * sideLength * sideLength;
constexpr float particleMass = 10.0f;
constexpr uint32_t solverIterations = 3;

class Engine
{
private:
    //Initialization
    void initSDL3();
    void initVulkan();
    void initSwapchain();
    void initDrawResources();
    void initCommands();
    void initSynchronization();
    void initDescriptors();
    void initPipelines();
    void initData();

    //Util
    bool loadShader(VkShaderModule* outShader, const char* filePath);
    ImmediateTransfer m_immTransfer;
    void prepImmediateTransfer();
    void submitImmediateTransfer();

    //Pipelines
    void initMeshPipeline();

    //drawing
    void draw();
    void drawMeshes(VkCommandBuffer cmd);

    //Optons
    bool m_bUseValidation = false;
    bool m_bUseDebugMessenger = false;
    
    struct SDL_Window* m_pWindow {nullptr};
    uint32_t windowWidth;
    uint32_t windowHeight;

    //Vulkan Resources
    VkInstance m_instance;
    VkDebugUtilsMessengerEXT m_debugMessenger;
    VkPhysicalDevice m_physicalDevice;
    VkDevice m_device;
    VkSurfaceKHR m_surface;
    VmaAllocator m_allocator;

    //Swapchain
    std::unique_ptr<Swapchain> m_swapchain;

    //Descriptors
    VkDescriptorPool m_descriptorPool;
    VkDescriptorSetLayout m_descriptorLayout;
    VkDescriptorSet m_descriptorSet;
    
    const uint32_t STORAGE_BINDING = 0;
    const uint32_t SAMPLER_BINDING = 1;
    const uint32_t IMAGE_BINDING = 2;
    uint32_t STORAGE_COUNT = 65536;
    uint32_t SAMPLER_COUNT = 65536;
    uint32_t IMAGE_COUNT = 65536;
    
    //Pipelines
    VkPipelineLayout meshPipelineLayout;
    VkPipeline meshPipeline;
    std::unique_ptr<PipelineBuilder> pb;

    //Queue Info
	VkQueue m_graphicsQueue;
	uint32_t m_graphicsQueueFamily;
	VkQueue m_transferQueue;
	uint32_t m_transferQueueFamily;

    //Draw Resources
    std::unique_ptr<Image> drawImage;
    std::unique_ptr<Image> depthImage;
    VkExtent2D drawExtent; 

    std::vector<MeshAsset> testMeshes;

    std::unique_ptr<Camera> cam;

    bool cleanedUp;
    bool mouseCaptured = true;
    uint64_t initializationTime = 0;
    uint64_t lastTime = 0;
    float deltaTime = 0;

    glm::vec3 maxBoundingPos;
    glm::vec3 minBoundingPos;
    std::vector<ParticleData> particleInfo;
    std::vector<glm::vec4> positionDeltas;
    std::vector<glm::vec3> particleCollisions;
    std::vector<float> particleLambdas;
    std::vector<std::vector<int>> particleNeighbors;

    std::unique_ptr<Buffer> hostPositionBuffer;
    std::unique_ptr<Buffer> devicePositionBuffer;
    void updatePositionBuffer();
    void updateParticlePositions();
    glm::vec3 clampDeltaToBounds(uint32_t index);

    glm::vec3 gravityForce = glm::vec3(0.0f, -9.81f, 0.0f);
    float gravityRotation = 0.0f;
    
    float smoothingRadius = 2.25f;
    float restDensity = 1000.0f;
    float epsilon = 1e-3f;
    float densityKernel(float r);
    float gradientKernel(float r);
    void resetPersistentParticleData();
    void resetParticlePositions();
    void randomizeParticlePositions();

public:
    Engine();
    ~Engine();
    void init();
    void run();
    void cleanup();
    
    MeshData uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);
    
    FrameData frameData[2];
    uint64_t getTime() {return SDL_GetTicks() - initializationTime; }
	FrameData& getCurrentFrame() { return frameData[frameNumber % 2]; };
    uint64_t frameNumber = 0;
};

#endif