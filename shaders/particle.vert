#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

layout(buffer_reference, std430) buffer PositionBuffer{ 
	vec3 pos[];
};
layout(buffer_reference, std430) buffer VelocityBuffer{ 
	vec3 vel[];
};

layout( push_constant ) uniform constants
{	
	mat4 renderMat;
	PositionBuffer positionBuffer;
    VelocityBuffer velocityBuffer;
	vec4 camWorldPos;
	vec4 originPos;
} pcs;

layout (location = 0) out vec3 outColor;
layout (location = 1) out vec3 outEye;
layout (location = 2) out vec3 outWorldPos;

void main() 
{	
	vec3 pPos = pcs.positionBuffer.pos[gl_VertexIndex];
	vec3 pVel = pcs.velocityBuffer.vel[gl_VertexIndex];
	
    outColor.rgb = abs(normalize(pVel)) + 0.2;

	gl_Position = pcs.renderMat * vec4(pPos + pcs.originPos.xyz, 1.0f);
	gl_PointSize = 1.0f;
    outEye = pcs.camWorldPos.xyz;
	outColor.rgb = vec3(1.0);
    outWorldPos = pPos;
}