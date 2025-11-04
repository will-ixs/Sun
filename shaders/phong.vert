#version 450
#extension GL_EXT_buffer_reference : require

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

struct Vertex {
	vec3 position;
	float uv_x;
	vec3 normal;
	float uv_y;
	vec4 color;
	vec4 tangent;
}; 

layout(buffer_reference, std430) readonly buffer VertexBuffer{ 
	Vertex vertices[];
};

layout(binding = 3) uniform  SceneData{   
	mat4 viewproj;
	vec4 camPos;
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

layout (location = 0) out vec3 outColor;
layout (location = 1) out vec2 outUV;
layout (location = 2) out vec3 outNormal;
layout (location = 3) out vec4 outWorldPos;

void main() 
{	
	Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];
	Material mat = sceneData.matBuffer.materials[PushConstants.materialIndex];
	
	vec4 position = vec4(v.position, 1.0f);
	vec3 worldPos = vec3(PushConstants.modelMatrix * position);
	gl_Position = sceneData.viewproj * vec4(worldPos, 1.0f);

	outColor = v.color.xyz * mat.baseColor.xyz;	
	outUV.x = v.uv_x;
	outUV.y = v.uv_y;
	outNormal = (PushConstants.modelMatrix * vec4(v.normal, 0.0f)).xyz;
	outWorldPos = vec4(worldPos, 1.0f);
}