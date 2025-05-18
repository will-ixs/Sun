#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

struct Vertex {
	vec3 position;
	float uv_x;
	vec3 normal;
	float uv_y;
	vec4 color;
}; 

layout(buffer_reference, std430) readonly buffer VertexBuffer{ 
	Vertex vertices[];
};


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

layout( push_constant ) uniform constants
{	
	mat4 modelMatrix;
	VertexBuffer vertexBuffer;
    uint materialIndex;
} PushConstants;

layout (location = 0) in vec3 inColor;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec3 inNormal;

layout (location = 0) out vec4 outFragColor;

void main() 
{	
	Material mat = sceneData.matBuffer.materials[PushConstants.materialIndex];
	vec3 sunDir = sceneData.sunlightDirection.xyz;
	sunDir.y *= -1.0;
	float lightValue = max(dot(inNormal, sunDir), 0.1);

	vec3 color = inColor * texture(Sampler2D[mat.baseColorIndex], inUV).xyz;
	vec3 ambient = color *  sceneData.ambientColor.xyz;

	outFragColor = vec4(color * lightValue *  sceneData.sunlightColor.w + ambient, 1.0);
}