#include "MeshLoader.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "../thirdparty/stb_image.h"
#include <unordered_map>

#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>

#include "Engine.hpp"
#include "Buffer.hpp"

MeshLoader::~MeshLoader()
{
}


bool MeshLoader::loadGltfMeshes(Engine* engine, std::vector<MeshAsset>& meshesOut, std::filesystem::path filePath){
    fastgltf::Parser parser;
    auto gltfOptions = fastgltf::Options::LoadExternalBuffers; // | fastgltf::Options::LoadExternalImages; 

    auto data = fastgltf::GltfDataBuffer::FromPath(filePath);
    if (data.error() != fastgltf::Error::None) {
        return false;
    }

    auto asset = parser.loadGltf(data.get(), filePath.parent_path(), gltfOptions);
    if (auto error = asset.error(); error != fastgltf::Error::None) {
        return false;
    }

    std::vector<uint32_t> indices;
    std::vector<Vertex> vertices;
    for (fastgltf::Mesh& mesh : asset->meshes) {
        MeshAsset newMesh;
        newMesh.name = mesh.name;

        indices.clear();
        vertices.clear();

        for (fastgltf::Primitive& prim : mesh.primitives) {
            Surface newSurface;
            newSurface.startIndex = (uint32_t)indices.size();
            newSurface.count = (uint32_t)asset->accessors[prim.indicesAccessor.value()].count;

            size_t initialVert = vertices.size();
            fastgltf::Accessor& indexAccessor = asset->accessors[prim.indicesAccessor.value()];
            indices.reserve(indices.size() + indexAccessor.count);
            {
            fastgltf::iterateAccessor<std::uint32_t>(asset.get(), indexAccessor,
                        [&](std::uint32_t idx) {
                            indices.push_back(idx + (uint32_t)initialVert);
                        });
            }

            auto posAttribute = prim.findAttribute("POSITION");
            auto posAccessor = asset->accessors[posAttribute->accessorIndex];
            vertices.resize(vertices.size() + posAccessor.count);
            {
            fastgltf::iterateAccessorWithIndex<glm::vec3>(asset.get(), posAccessor,
                    [&](glm::vec3 v, size_t idx) {
                        Vertex vert;
                        vert.position = v;
                        vert.normal = { 1, 0, 0 };
                        vert.color = glm::vec4 { 1.f };
                        vert.uv_x = 0;
                        vert.uv_y = 0;
                        vertices[initialVert + idx] = vert;
                    });
            }

            auto normAttribute = prim.findAttribute("NORMAL");
            auto normAccessor = asset->accessors[normAttribute->accessorIndex];
            if (normAttribute != prim.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<glm::vec3>(asset.get(), normAccessor,
                    [&](glm::vec3 v, size_t idx) {
                        vertices[initialVert + idx].normal = v;
                    });
            }

            // TEXCOORD_0 /uvs
            auto texAttribute = prim.findAttribute("TEXCOORD_0");
            auto texAccessor = asset->accessors[texAttribute->accessorIndex];
            if (texAttribute != prim.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<glm::vec2>(asset.get(), texAccessor,
                    [&](glm::vec2 v, size_t idx) {
                        vertices[initialVert + idx].uv_x = v.x;
                        vertices[initialVert + idx].uv_y = v.y;
                    });
            }

            // COLOR_0
            auto colAttribute = prim.findAttribute("COLOR_0");
            auto colAccessor = asset->accessors[colAttribute->accessorIndex];
            if (colAttribute != prim.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<glm::vec4>(asset.get(), colAccessor,
                    [&](glm::vec4 v, size_t idx) {
                        vertices[initialVert + idx].color = v;
                    });
            }

            newMesh.surfaces.push_back(newSurface);
        }

        // display the vertex normals
        // for (Vertex& vert : vertices) {
        //     vert.color = glm::vec4(vert.normal, 1.f);
        // }

        newMesh.data = engine->uploadMesh(indices, vertices);

        meshesOut.emplace_back(std::move(newMesh));
    }

    return true;
}