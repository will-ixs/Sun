# Sun - WIP Vulkan Engine


![Sponza Screenshot](https://github.com/will-ixs/Sun/blob/main/screenshots/Screenshot5.png)

---
### Features:

+ Bindless Design 
+ Scene Graph 
+ Point, Spot, Directional Lights
+ GLTF/GLB Loading 
+ PBR 
+ Frustum Culling 
+ Multisampling
+ Asynchronous Mesh Uploading 
+ Transparency & Draw Sorting 
+ Vulkan Swapchain/Images/Buffers Abstraction 
+ Builtin FPS Camera 
+ DearImGui Timing 
+ Position Based Fluids (Standalone described [here](https://will-ixs.github.io/projects/particle-based-fluids/))
---

### Requirements:
Vulkan 1.3 > SDK \
Support of dynamic rendering, descriptor indexing, and buffer device address.


### Planned for Future:
+ Postprocessing
+ IBL
+ Cascaded Shadow Mapping
+ Clustered Forward Rendering

---

Thank you to contributors and owners of the following projects and libraries!
+ [SDL3](https://github.com/libsdl-org/SDL)
+ [GLM](https://github.com/g-truc/glm)
+ [vk-bootstrap](https://github.com/charles-lunarg/vk-bootstrap)
+ [DearImGui](https://github.com/ocornut/imgui)
+ [VulkanMemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
+ [fastgltf](https://github.com/spnda/fastgltf)
+ [Khronos Texture](https://github.com/KhronosGroup/KTX-Software)

as well as Crytek for Sponza in the screenshots. I used the one hosted by Khronos [here](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/Sponza) and added lights into it.