#version 450
layout (binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout (location = 0) in vec3 worldNorm;

layout (location = 0) out vec4 outFragColor;

void main() 
{	
	outFragColor = vec4(worldNorm,1.0f);
}
