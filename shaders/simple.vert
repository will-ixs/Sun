#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

struct ParticleData {
	vec4 currPosition;
	vec4 prevPosition;
	vec4 velocity;
}; 

layout(set = 0, binding = 0) buffer StorageBuffer {
    ParticleData data[];
} storageBuffers[];

layout (location = 0) out vec3 outColor;
layout (location = 1) out vec2 outUV;

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

//push constants block
layout( push_constant ) uniform constants
{	
	mat4 render_matrix;
	VertexBuffer vertexBuffer;
	uint instanceBufferIndex;
} PushConstants;

void main() 
{	
	Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];
	vec4 instancePosition = storageBuffers[PushConstants.instanceBufferIndex].data[gl_InstanceIndex].currPosition;

	//output data
	vec3 finalPos = v.position + instancePosition.xyz;
	gl_Position = PushConstants.render_matrix * vec4(finalPos, 1.0);
	
	// float velocity = length(storageBuffers[PushConstants.instanceBufferIndex].data[gl_InstanceIndex].velocity);
	// velocity *= 0.1;
	// outColor = vec3(0.2 + velocity, 0.2 + velocity, 1);
	outColor = v.color.xyz;
	outUV.x = v.uv_x;
	outUV.y = v.uv_y;
}
