#version 450

layout(location = 0) out vec2 texCoord;

layout( push_constant ) uniform constants
{	
	mat4 renderMat;
	vec3 camWorldPos;
	vec3 minBoundingPos;
	vec3 maxBoundingPos;
} PushConstants;

void main() {
    texCoord = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(texCoord * 2.0 - 1.0, 0.0, 1.0);
}