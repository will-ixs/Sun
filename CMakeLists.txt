cmake_minimum_required(VERSION 3.10)
project(Sun LANGUAGES C CXX)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(Vulkan REQUIRED)
# set the output directory for built objects.
# This makes sure that the dynamic library goes into the build directory automatically.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")

# FetchContent to pull dependencies
include(FetchContent)
#SDL3
FetchContent_Declare(
    SDL3
    GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
    GIT_TAG release-3.2.4
)
#GLM
FetchContent_Declare(
	glm
	GIT_REPOSITORY	https://github.com/g-truc/glm.git
	GIT_TAG 	1.0.2
)
#VKB
FetchContent_Declare(
    vk_bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap
    GIT_TAG        v1.4.307
)# ImGui
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.8
)
#VMA - set(VMA_DYNAMIC_VULKAN_FUNCTIONS ON)
FetchContent_Declare(
    VulkanMemoryAllocator
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG v3.0.1
)
#fastgltf
FetchContent_Declare(
    fastgltf
    GIT_REPOSITORY https://github.com/spnda/fastgltf.git
    GIT_TAG v0.8.0
)

FetchContent_MakeAvailable(SDL3 glm vk_bootstrap imgui VulkanMemoryAllocator fastgltf)

set(IMGUI_SOURCES
    ${Vulkan_INCLUDE_DIRS}
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)
add_library(imgui STATIC ${IMGUI_SOURCES})

target_include_directories(imgui PUBLIC
    ${Vulkan_INCLUDE_DIRS}
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
find_library(Slang_LIBRARY NAMES slang 
    HINTS "${Vulkan_INCLUDE_DIRS}/../lib"  # SDK include is usually one level up from lib
    REQUIRED)

add_executable(Sun 
    src/main.cpp
    src/Engine.cpp
    src/Swapchain.cpp
    src/Image.cpp
    src/Camera.cpp
    src/Buffer.cpp
    src/PipelineBuilder.cpp
)

# Include Directories
target_include_directories(Sun PRIVATE 
    ${Vulkan_INCLUDE_DIRS}/slang
    ${SDL3_INCLUDE_DIRS}
    ${glm_SOURCE_DIR}
    ${vkbootstrap_SOURCE_DIR}
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
    ${VulkanMemoryAllocator_SOURCE_DIR}/include
)

# Link Libraries
target_link_libraries(Sun PRIVATE 
    Vulkan::Vulkan
    SDL3::SDL3
    glm::glm
    vk-bootstrap::vk-bootstrap
    fastgltf::fastgltf
    imgui
    ${Slang_LIBRARY}
)

# Enable compiler warnings
if (MSVC)
    target_compile_options(Sun PRIVATE /W4)
else()
    target_compile_options(Sun PRIVATE -Wall -Wextra -Wpedantic)
endif()
message(STATUS "Vulkan include: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Slang library: ${Slang_LIBRARY}")