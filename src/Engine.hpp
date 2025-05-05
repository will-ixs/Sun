#ifndef ENGINE_HPP
#define ENGINE_HPP

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <vector>
#include <array>
#include <span>
#include <memory>
#include <unordered_map>
#include <SDL3/SDL_timer.h>
#include "types.h"

class Swapchain;
class Image;
class Buffer;
class PipelineBuilder;
class MeshUploader;
class Camera;

constexpr uint32_t sideLength = 30;
constexpr uint32_t instanceCount = sideLength * sideLength * sideLength;
constexpr float particleMass = 0.1f;
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
    void initDearImgui();

    //Util
    bool loadShader(VkShaderModule* outShader, const char* filePath);
    ImmediateTransfer m_immTransfer;
    void prepImmediateTransfer();
    void submitImmediateTransfer();

    //Pipelines
    void initMeshPipeline();
    void initComputePipeline();

    //drawing
    void draw();
    void drawMeshes(VkCommandBuffer cmd);
    void drawDearImGui(VkCommandBuffer cmd, VkImageView view);

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
    VkDescriptorPool m_ImguiPool;
    
    const uint32_t STORAGE_BINDING = 0;
    const uint32_t SAMPLER_BINDING = 1;
    const uint32_t IMAGE_BINDING = 2;
    uint32_t STORAGE_COUNT = 65536;
    uint32_t SAMPLER_COUNT = 65536;
    uint32_t IMAGE_COUNT = 65536;
    
    //Pipelines
    VkPipelineLayout meshPipelineLayout;
    VkPipeline meshPipeline;

    VkPipelineLayout computePipelineLayout;
    VkPipeline computePredict;
    VkPipeline computeGrid;
    VkPipeline computeConstraints;
    VkPipeline computeResetGrid;
    VkPipeline computeResetParticles;

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


    //PBF
    const int tableSize = 129631;
    const uint32_t PRIME1 = 19349663;
    const uint32_t PRIME2 = 73856093;
    const uint32_t PRIME3 = 83492791;

    std::vector<std::vector<int>> grid;
    size_t hashGridCoord(int x, int y, int z);
    void buildGrid();
    void findNeighbors(int i, std::vector<int>& neighbors);
    void buildCellStart();
    
    //GPU Data
    VkDeviceAddress particleBufferAddress;
    VkDeviceAddress gridCountBufferAddress;
    VkDeviceAddress gridCellsBufferAddress;
    glm::vec3 maxBoundingPos;
    glm::vec3 minBoundingPos;

    std::vector<uint32_t> gridCounters;     //tableSize - contains number of particles in each cell
    std::vector<uint32_t> gridCells;        //tableSize * cellSize(10 if it gets weirdly compressed) - contains indices of particles in each cell
    // std::vector<uint32_t> particleNeighbors //instanceCount * (cellSize * 27) -  
    std::vector<int> cellStart;        // size = tableSize
    std::vector<int> particleIndices;
    std::vector<int> tempCounts;


    std::vector<glm::vec3> particleCurrPosition;
    std::vector<glm::vec3> particlePrevPosition;
    std::vector<glm::vec3> particleVelocity;
    std::vector<glm::vec3> positionDeltas;
    std::vector<glm::vec3> particleCollisions;
    std::vector<glm::vec3> particleVorticity;
    std::vector<glm::vec3> particleVorticityGradient;
    std::vector<glm::vec3> particleViscosityDelta;
    std::vector<float> particleLambdas;

    std::vector<std::vector<int>> particleNeighbors;

    std::unique_ptr<Buffer> hostPositionBuffer;
    std::unique_ptr<Buffer> devicePositionBuffer;
    std::unique_ptr<Buffer> deviceGridCounters;
    std::unique_ptr<Buffer> deviceGridCells;
    void updatePositionBuffer();
    void calculateLambdas(int i);
    void calculateDpiCollision(int i, float h);
    void updateParticlePositions();
    void gpuUpdateParticlePositions(VkCommandBuffer cmd);
    glm::vec3 clampDeltaToBounds(uint32_t index);

    glm::vec3 gravityForce = glm::vec3(0.0f, -9.81f, 0.0f);
    float gravityRotation = 0.0f;
    
    float smoothingRadius = 2.5f;
    float restDensity = 10000.0f;
    float epsilon = 1e-4f;
    float vorticityEpsilon = 0.25f;
    float viscosity = 0.01f;                                                                         
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