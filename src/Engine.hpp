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

constexpr uint32_t sideLength = 20;
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
    void initDearImgui();

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
    const uint32_t tableSize = 20011;
    const uint32_t PRIME1 = 19349663;
    const uint32_t PRIME2 = 73856093;
    const uint32_t PRIME3 = 83492791;

    std::vector<std::vector<int>> grid;
    
    size_t hashGridCoord(int x, int y, int z) {
        return (x * PRIME1 ^ y * PRIME2 ^ z * PRIME3) % tableSize;
    }
    glm::ivec3 getGridCoord(const glm::vec3& pos) {
        return glm::floor(pos / smoothingRadius);
    }
    void buildGrid() {
        std::fill(tempCounts.begin(), tempCounts.end(), 0);
        for (size_t i = 0; i < particleInfo.size(); i++) {
            auto cell = glm::floor(particleInfo.at(i).currPosition / smoothingRadius);
            size_t hash = hashGridCoord(cell.x, cell.y, cell.z);
            tempCounts[hash]++;
        }

        cellStart[0] = 0;
        for (int i = 1; i < tableSize; ++i){
            cellStart[i] = cellStart[i - 1] + tempCounts[i - 1];
        }

        std::fill(tempCounts.begin(), tempCounts.end(), 0);
        for (size_t i = 0; i < particleInfo.size(); i++) {
            auto cell = glm::floor(particleInfo.at(i).currPosition / smoothingRadius);
            size_t hash = hashGridCoord(cell.x, cell.y, cell.z);
            int writeIndex = cellStart[hash] + tempCounts[hash]++;
            particleIndices[writeIndex] = i;
        }
    }
    void findNeighbors(int i, std::vector<int>& neighbors) {
        glm::ivec3 base = glm::floor(particleInfo.at(i).currPosition / smoothingRadius);
        float r2 = smoothingRadius * smoothingRadius;
    
        for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dz = -1; dz <= 1; ++dz) {
            glm::ivec3 neighborCell = base + glm::ivec3(dx, dy, dz);
            size_t hash = hashGridCoord(neighborCell.x, neighborCell.y, neighborCell.z);
    
            // get start and end indices of this cell's range
            int start = cellStart[hash];
            int end = (hash + 1 < tableSize) ? cellStart[hash + 1] : particleIndices.size();
    
            for (int j = start; j < end; ++j) {
                int neighborIdx = particleIndices[j];
                if (neighborIdx == i) continue;
                if (glm::length2(particleInfo.at(neighborIdx).currPosition - particleInfo.at(i).currPosition) < r2)
                    neighbors.push_back(neighborIdx);
            }
        }
    }

    void countParticles() {
        std::fill(tempCounts.begin(), tempCounts.end(), 0);
    
        for (size_t i = 0; i<particleInfo.size(); i++) {
            glm::ivec3 cell = glm::floor(glm::vec3(particleInfo.at(i).currPosition) / smoothingRadius);
            size_t hash = hashGridCoord(cell.x, cell.y, cell.z);
            tempCounts[hash]++;
        }
    }
    void buildCellStart() {
        cellStart[0] = 0;
        for (int i = 1; i < tableSize; ++i) {
            cellStart[i] = cellStart[i - 1] + tempCounts[i - 1];
        }
    }

    std::vector<int> cellStart;        // size = tableSize
    std::vector<int> particleIndices;  // flat index list
    std::vector<int> tempCounts;   

    glm::vec3 maxBoundingPos;
    glm::vec3 minBoundingPos;
    std::vector<ParticleData> particleInfo;
    // std::vector<glm::vec4> particleCurrPosition;
    // std::vector<glm::vec4> particlePrevPosition;
    // std::vector<glm::vec4> particleVelocity;
    std::vector<glm::vec4> positionDeltas;
    std::vector<glm::vec3> particleCollisions;
    std::vector<glm::vec3> particleVorticity;
    std::vector<glm::vec3> particleVorticityGradient;
    std::vector<glm::vec3> particleViscosityDelta;
    std::vector<float> particleLambdas;

    
    std::vector<std::vector<int>> particleNeighbors;

    std::unique_ptr<Buffer> hostPositionBuffer;
    std::unique_ptr<Buffer> devicePositionBuffer;
    void updatePositionBuffer();
    void updateParticlePositions();
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