#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

struct Material{
	vec4 baseColor;
    float metallicFactor;
    float roughnessFactor;

    uint baseColorIndex;
    uint metallicRoughnessIndex;
    uint normalIndex;
};

layout(buffer_reference, std430) readonly buffer MaterialBuffer{ 
	Material materials[];
};

layout(binding = 1) uniform sampler2D Sampler2D[];
layout(binding = 3) uniform  SceneData{   
	mat4 view;
	mat4 proj;
	mat4 viewproj;
	vec4 ambientColor;
	vec4 sunlightDirection; //w for sun power
	vec4 sunlightColor;
	MaterialBuffer matBuffer;
} sceneData;

layout (location = 0) in vec3 inColor;
layout (location = 1) in vec2 uv;
layout (location = 2) flat in uint matIndex;

layout (location = 0) out vec4 outFragColor;

void main() 
{	
	Material mat = sceneData.matBuffer.materials[matIndex];

	outFragColor = texture(Sampler2D[0], uv);
	// outFragColor = vec4(inColor, 1.0);
}