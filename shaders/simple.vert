#version 450
#extension GL_EXT_buffer_reference : require

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

struct Vertex {
	vec3 position;
	float uv_x;
	vec3 col;
	float uv_y;
	vec3 normal;
}; 

layout(buffer_reference, std430) readonly buffer VertexBuffer{ 
	Vertex vertices[];
};

layout(push_constant) uniform constants{
	VertexBuffer vertex_buffer;
} pc;

layout (location = 0) out vec3 worldNorm;

void main() 
{	
	Vertex v = pc.vertex_buffer.vertices[gl_VertexIndex];

	//worldNorm = normalize(vec3(ubo.Q * vec4(v.normal, 1.0f)));
    worldNorm = v.normal;
	gl_Position =  ubo.proj * ubo.view * ubo.model * vec4(v.position, 1.0f);
}
