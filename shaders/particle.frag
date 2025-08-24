#version 450

layout (location = 0) in vec3 inColor;
layout (location = 1) in vec3 inEye;
layout (location = 2) in vec3 inWorldPos;

layout (location = 0) out vec4 outFragColor;

void main() 
{
	// vec3 ka = vec3(0.1, 0.1, 0.1)
	// vec3 kd = vec3(0.1, 0.1, 0.5);
	// float s = 10.0;

	// vec3 lightWorldPos = vec3(0.0, 10.0, -30.0);
	// vec3 lightColor = vec3(1.0, 1.0, 1.0);

	// vec3 currColor = ka * inColor;
	// vec3 eye = vec3(inEye); 
	// vec3 pos = vec3(inWorldPos);
	// vec3 N = normalize(inWorldNorm);

	// vec3 Li =  normalize(lightWorldPos - pos);
	// vec3 kd_vec = kd * max(0, dot(Li, N));
	// currColor += lightColor * kd_vec;

	outFragColor = vec4(inColor, 1.0f);
}