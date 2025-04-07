#ifndef MESHLOADER_H
#define MESHLOADER_H

#include "types.h"
#include <filesystem>

class Engine;
class Buffer; 

class MeshLoader
{
private:

public:
    MeshLoader() = delete;
    ~MeshLoader();

    static bool loadGltfMeshes(Engine* engine, std::vector<MeshAsset>& meshesOut, std::filesystem::path filePath);
};


#endif