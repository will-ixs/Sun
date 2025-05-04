#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

#define NUM_PARTICLES 8000

const uint OFFSET_CURR_POS      	= 0;
const uint OFFSET_PREV_POS      	= 3 * NUM_PARTICLES;
const uint OFFSET_VELOCITY      	= 6 * NUM_PARTICLES;
const uint OFFSET_DELTA         	= 9 * NUM_PARTICLES;
const uint OFFSET_COLLISIONS    	= 12 * NUM_PARTICLES;
const uint OFFSET_VORTICITY     	= 15 * NUM_PARTICLES;
const uint OFFSET_VORTICITY_GRAD	= 18 * NUM_PARTICLES;
const uint OFFSET_VISCOSITY     	= 21 * NUM_PARTICLES;
const uint OFFSET_LAMBDAS       	= 24 * NUM_PARTICLES;

struct Vertex {
	vec3 position;
	float uv_x;
	vec3 normal;
	float uv_y;
	vec4 color;
}; 

layout (location = 0) out vec3 outColor;
layout (location = 1) out vec3 outEye;
layout (location = 2) out vec3 outWorldPos;
layout (location = 3) out vec3 outWorldNorm;

layout(buffer_reference, std430) readonly buffer VertexBuffer{ 
	Vertex vertices[];
};

layout(buffer_reference, std430) buffer ParticleBuffer{ 
	float data[];
};

layout( push_constant ) uniform constants
{	
	mat4 render_matrix;
	VertexBuffer vertexBuffer;
	ParticleBuffer particleBuffer;
	vec3 camWorldPos;
} PushConstants;

vec3 loadVec3(uint baseOffset, uint index){
    return vec3(
        PushConstants.particleBuffer.data[baseOffset + (3 * index) + 0],
        PushConstants.particleBuffer.data[baseOffset + (3 * index) + 1],
        PushConstants.particleBuffer.data[baseOffset + (3 * index) + 2]
    );
}

void main() 
{	
	Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];
	vec3 instancePosition = loadVec3(OFFSET_CURR_POS, gl_InstanceIndex);
	
	vec3 finalPos = v.position + instancePosition.xyz;
	gl_Position = PushConstants.render_matrix * vec4(finalPos, 1.0);
	
	float velocity = length(loadVec3(OFFSET_VELOCITY, gl_InstanceIndex));
	velocity *= 0.1;
	outColor = vec3(0.2 + velocity, 0.2 + velocity, 1);
	
	outEye = PushConstants.camWorldPos;
	outWorldPos = finalPos;
	outWorldNorm = v.normal;
}
