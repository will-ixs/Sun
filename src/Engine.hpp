#ifndef ENGINE_HPP
#define ENGINE_HPP

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <vector>
#include <array>
#include <filesystem>
#include <span>
#include <memory>
#include <thread>
#include <queue>
#include <unordered_map>
#include <SDL3/SDL_timer.h>
#include "types.h"

class Swapchain;
class Image;
class Buffer;
class PipelineBuilder;
class Camera;

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
    void initDearImGui();
    
    //Util
    ImmediateTransfer m_immTransfer;
    EngineStats stats = {0};
    void prepImmediateTransfer();
    void submitImmediateTransfer();

    void registerDefaultParticleSystems();
    void registerParticleSystem(std::string type, glm::vec3 defaultVelocity = glm::vec3(0.0f));
    void createParticleSystem(std::string name, uint32_t particleCount, float lifeTime, 
        glm::vec3 originPosition = glm::vec3(0.0f), glm::vec3 originVariance = glm::vec3(0.0f), glm::vec3 velocityVariance = glm::vec3(0.0f));
    void updateScene();
    void updateGUI();
    
    bool loadShader(VkShaderModule* outShader, std::string filePath);
    bool loadGLTF(std::filesystem::path filePath);
    std::shared_ptr<Image> createImageFromData(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped);
    
    void createRenderablesFromNode(std::shared_ptr<GLTFNode> node);
    void sortTransparentRenderables();
    bool renderableVisible(const Renderable& r, const glm::mat4& viewProj);
    //Material / Textures util
    void addLights(const std::vector<Light>& lights);
    uint32_t addMaterial(const MaterialData& data, std::string name);
    uint32_t addTexture(const TextureData& data, std::string name);
    MaterialData getMatFromName(std::string name) { return materials.at(matNameToIndex.find(name)->second); };
    TextureData getTexFromName(std::string name) { return textures.at(texNameToIndex.find(name)->second); };
    
    
    void meshUploader();
    std::queue<std::filesystem::path> pathQueue;
    std::thread meshThread;
    
    //Pipelines
    void initMeshPipelines();
    void initParticlePipelines();
    
    //Drawing
    void draw();
    void drawMeshes(VkCommandBuffer cmd);
    void drawParticles(VkCommandBuffer cmd);
    void drawDearImGui(VkCommandBuffer cmd, VkImageView view);
    
    //Compute
    void updateParticles();

    //Optons
    bool m_bUseValidation = false;
    bool m_bUseDebugMessenger = false;
    
    //Window info
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
    VkDescriptorPool m_ImguiPool;
    VkDescriptorPool m_descriptorPool;
    VkDescriptorSetLayout m_descriptorLayout;
    VkDescriptorSet m_descriptorSet;
    
    //Bindless stuff
    const uint32_t STORAGE_BINDING = 0;
    const uint32_t SAMPLER_BINDING = 1;
    const uint32_t IMAGE_BINDING = 2;
    const uint32_t UBO_BINDING = 3;
    uint32_t STORAGE_COUNT = 65536;
    uint32_t SAMPLER_COUNT = 65536;
    uint32_t IMAGE_COUNT = 65536;
    
    //Pipelines
    VkPipelineLayout particleDrawPipelineLayout;
    VkPipeline particleDrawPipeline;

    VkPipelineLayout particleComputePipelineLayout;
    VkPipelineCache particleComputePipelineCache;
    std::unordered_map<std::string, VkPipeline> particlePipelineMap; //use reflection to map to ENUM instead of string?
    std::unordered_map<std::string, glm::vec3> particleVelocityMap;
    std::list<ParticleSystem> particleSystems;
    std::vector<VkSemaphore> particleSystemGarbage;

    VkPipelineLayout meshPipelineLayout;
    VkPipeline meshPipelineOpaque;
    VkPipeline meshPipelineTransparent;
    std::unique_ptr<PipelineBuilder> pb;

    //Queue Info
	VkQueue m_graphicsQueue;
	uint32_t m_graphicsQueueFamily;
    VkQueue m_computeQueue;
    uint32_t m_computeQueueFamily;
	VkQueue m_transferQueue;
	uint32_t m_transferQueueFamily;

    //Draw Resources
    std::unique_ptr<Image> drawImage;
    std::unique_ptr<Image> resolveImage;
    std::unique_ptr<Image> depthImage;
    VkExtent2D drawExtent; 
    std::vector<Renderable> opaqueRenderables;
    std::vector<Renderable> transparentRenderables;
    std::vector<uint32_t> transparentRenderablesIndices;
    std::vector<MeshAsset> testMeshes;
    std::unique_ptr<Camera> cam;

    //Particle Buffers
    std::unique_ptr<Buffer> hostPositionBuffer;
    std::unique_ptr<Buffer> hostVelocityBuffer;
    std::unique_ptr<Buffer> devicePositionBufferA;
    std::unique_ptr<Buffer> devicePositionBufferB;
    std::unique_ptr<Buffer> deviceVelocityBuffer;
    std::vector<glm::vec4> initialPositions;
    std::vector<glm::vec4> initialVelocities;
    VkDeviceAddress particlePosBufferAddressA;
    VkDeviceAddress particlePosBufferAddressB;
    VkDeviceAddress particleVelBufferAddress;

    //Scenes
    std::unordered_map<std::string, std::shared_ptr<GLTFScene>> loadedGLTFs;

    //UBO Stuff
    UniformBufferObject ubo;
    std::unique_ptr<Buffer> uboBuffer;
    
    //Materials
    std::vector<MaterialData> materials;
    std::unordered_map<std::string, uint32_t> matNameToIndex;
    std::unique_ptr<Buffer> materialBuffer;
    VkDeviceAddress materialBufferAddress;

    //Lights
    std::vector<Light> lightsPoint;
    std::unique_ptr<Buffer> lightBuffer;
    VkDeviceAddress lightBufferAddress;
    
    //Textures
    std::vector<TextureData> textures;
    std::unordered_map<std::string, uint32_t> texNameToIndex;

    std::shared_ptr<Image> whiteImage;
    std::shared_ptr<Image> blackImage;
    std::shared_ptr<Image> grayImage;
    std::shared_ptr<Image> checkerboardImage;
    VkSampler defaultLinearSampler;
    VkSampler defaultNearestSampler;

    VkImage videoTexture;
    VkImageView videoTextureView;
    VmaAllocation videoTextureAlloc;
    VkImageLayout videoTextureLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    
    bool cleanedUp;
    bool mouseCaptured = true;
    bool minimized = false;
    uint64_t initializationTime = 0;
    float deltaTime = 0;
    float currentTime = 0;
    float timeScale = 1.0f;
    struct {
        glm::vec3 origin = glm::vec3(0.0f);
        glm::vec3 oVar = glm::vec3(1.0f);
        glm::vec3 vVar = glm::vec3(1.0f);
        uint32_t selIdx = 0;
        int numParticles = 1024;
        float lifeTime = 5.0f;
    } particleCreation;

public:
    Engine();
    ~Engine();
    void init();
    void run();
    void cleanup();
    
    MeshData uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);
    
    FrameData frameData[2];
    uint64_t getTime() {return SDL_GetTicksNS() - initializationTime; }
	FrameData& getCurrentFrame() { return frameData[frameNumber % 2]; };
    uint64_t frameNumber = 0;
};

#endif