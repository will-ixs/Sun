#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

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

struct Light{
    uint type;
    vec3 lightPos;   
    vec3 lightDir;
    vec3 lightColor; //VVVVV fastgltf descriptions VVVVVVV
    /** Point and spot lights use candela (lm/sr) while directional use lux (lm/m^2) */
    float intensity;
    /** Range for point and spot lights. If not present, range is infinite. */
    float range;
	/** The inner and outer cone angles only apply to spot lights */
    float innerConeAngle;
    float outerConeAngle;
};

layout(buffer_reference, std430) readonly buffer LightBuffer{ 
	Light lights[];
};

layout(binding = 1) uniform sampler2D Sampler2D[];
layout(binding = 3) uniform  SceneData{   
	mat4 viewproj;
	vec4 camPos;
	vec4 ambientColor;
	vec4 sunlightDirection; 
	vec4 sunlightColor;		//w for sun power
	MaterialBuffer matBuffer;
	LightBuffer lightBuffer;
    uint numLights;
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
layout (location = 3) in vec4 inWorldPos;

layout (location = 0) out vec4 outFragColor;

vec3 calculateBP(const vec3 fragPos, const vec3 viewDir, 
				 const vec3 norm, const vec3 lightDir, 
				 const vec3 diffuseColor, const vec3 specularColor, 
				 const vec3 lightColor, const float dist, const float intensity)
{
	vec3 halfDir = normalize(lightDir + viewDir);
	float diffuse = max(dot(norm, lightDir), 0.0);
	float specular = pow(max(dot(norm, halfDir), 0.0), 4.0);


	//diffuseColor * diffuse * lightColor * powerFactor 
	//specularColor * specular * lightColor *  powerFactor
	float powerFactor = 0.05 * intensity / dist;
	return (diffuse * diffuseColor * lightColor * powerFactor) + (specular * specularColor * lightColor * powerFactor);
}

void main() 
{
	const Material mat = sceneData.matBuffer.materials[PushConstants.materialIndex];
	const vec4 albedo = texture(Sampler2D[mat.baseColorIndex], inUV);
	if (albedo.a < 0.5){
		discard;
	} 
	const vec3 fragPos = inWorldPos.xyz;
	const vec3 viewDir = fragPos - sceneData.camPos.xyz;
	const vec3 norm = normalize(inNormal);

    vec3 color = vec3(sceneData.ambientColor);
	//Sun
	color += calculateBP(	fragPos, viewDir, norm, normalize(sceneData.sunlightDirection.xyz), 
						 	albedo.rgb, inColor, sceneData.sunlightColor.rgb,
							10.0f, sceneData.sunlightColor.w);

	for(uint i = 0; i < sceneData.numLights; i++){
        Light l = sceneData.lightBuffer.lights[i];
        if(l.type == 0){
            color += calculateBP(	fragPos, viewDir, norm, normalize(l.lightPos - fragPos), 
									albedo.rgb, inColor, l.lightColor, 
									distance(l.lightPos, fragPos), l.intensity);

        }else if(l.type == 1){
            color += calculateBP(	fragPos, viewDir, norm, normalize(l.lightDir), 
									albedo.rgb, inColor, l.lightColor, 
									distance(l.lightPos, fragPos), l.intensity);
        }else if(l.type == 2){
			//not implemented rn
        }
    }
	outFragColor = vec4(color, 1.0f);

}