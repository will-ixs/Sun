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
	vec4 sunlightDirection; //w for sun power
	vec4 sunlightColor;
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
layout (location = 4) in mat3 inTBN;

layout (location = 0) out vec4 outFragColor;

const float PI = 3.14159265359;

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float ggx1 = GeometrySchlickGGX(max(dot(N, V), 0.0), roughness);
    float ggx2 = GeometrySchlickGGX(max(dot(N, L), 0.0), roughness);
    return ggx1 * ggx2;
}

vec3 calculatePointLight(Light p, vec3 albedo, vec2 metal_rough, vec3 normal){    
    vec3 L = normalize(p.lightPos - inWorldPos.xyz);
    vec3 V = vec3(normalize(sceneData.camPos - inWorldPos)); 
    vec3 H = normalize(V + L);
    vec3 F0 = mix(vec3(0.04f), albedo, metal_rough.x);
        
    float dist  = length(p.lightPos - inWorldPos.xyz);
    if(p.range > 0.0f && dist > p.range){
        return vec3(0.0f);
    }
        
    float attenuation = 1.0f / (dist * dist);
    attenuation *= clamp(1.0 - ((dist * dist) / (p.range * p.range)), 0.0, 1.0);

    vec3 radiance = p.lightColor * attenuation * p.intensity;

    float NDF = DistributionGGX(normal, H, metal_rough.y);
    float G = GeometrySmith(normal, V, L, metal_rough.y);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0f), F0);

    vec3 kS = F;
    vec3 kD = vec3(1.0f) - kS;
    kD *= 1.0f - metal_rough.x;

    vec3 top = NDF * G * F;
    float bot = 4.0f * max(dot(normal, V), 0.0f) * max(dot(normal, L), 0.0f) + 0.0001f;
    vec3 specular = top / bot;  
            
    float NdotL = max(dot(normal, L), 0.0f);                
    return (kD * albedo / PI + specular) * radiance * NdotL; 
}

vec3 calculateDirectionalLight(Light d, vec3 albedo, vec2 metal_rough, vec3 normal){
    vec3 L = normalize(-d.lightDir);
    vec3 V = vec3(normalize(sceneData.camPos - inWorldPos)); 
    vec3 H = normalize(V + L);
    vec3 F0 = mix(vec3(0.04f), albedo, metal_rough.x);

    vec3 radiance = d.lightColor * d.intensity;

    float NDF = DistributionGGX(normal, H, metal_rough.y);
    float G = GeometrySmith(normal, V, L, metal_rough.y);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0f), F0);

    vec3 kS = F;
    vec3 kD = vec3(1.0f) - kS;
    kD *= 1.0f - metal_rough.x;

    vec3 top = NDF * G * F;
    float bot = 4.0f * max(dot(normal, V), 0.0f) * max(dot(normal, L), 0.0f) + 0.0001f;
    vec3 specular = top / bot;  
            
    float NdotL = max(dot(normal, L), 0.0f);                
    return (kD * albedo / PI + specular) * radiance * NdotL; 
}

vec3 calculateSpotLight(Light s, vec3 albedo, vec2 metal_rough, vec3 normal){
    vec3 L = normalize(s.lightPos - inWorldPos.xyz);
    vec3 V = normalize(sceneData.camPos.xyz - inWorldPos.xyz);
    vec3 H = normalize(V + L);
    vec3 F0 = mix(vec3(0.04), albedo, metal_rough.x);

    float dist = length(s.lightPos - inWorldPos.xyz);
    if (s.range > 0.0 && dist > s.range) {
        return vec3(0.0);
    }

    // Spotlight attenuation
    vec3 lightDir = normalize(-s.lightDir); // Assuming lightDir points *from* the light source
    float spotCos = dot(L, lightDir);

    float innerCos = cos(radians(s.innerConeAngle));
    float outerCos = cos(radians(s.outerConeAngle));

    float spotAtten = clamp((spotCos - outerCos) / (innerCos - outerCos), 0.0f, 1.0f);

    float attenuation = 1.0f / (dist * dist);
    attenuation *= clamp(1.0f - ((dist * dist) / (s.range * s.range)), 0.0f, 1.0f);
    attenuation *= spotAtten;

    vec3 radiance = s.lightColor * s.intensity * attenuation;

    float NDF = DistributionGGX(normal, H, metal_rough.y);
    float G = GeometrySmith(normal, V, L, metal_rough.y);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0f), F0);

    vec3 kS = F;
    vec3 kD = vec3(1.0f) - kS;
    kD *= 1.0f- metal_rough.x;

    vec3 numerator = NDF * G * F;
    float denominator = 4.0f * max(dot(normal, V), 0.0f) * max(dot(normal, L), 0.0f) + 0.0001f;
    vec3 specular = numerator / denominator;

    float NdotL = max(dot(normal, L), 0.0f);
    return (kD * albedo / PI + specular) * radiance * NdotL; 
}


const float alphaCutoffTest = 0.5f;
void main() 
{	
	Material mat = sceneData.matBuffer.materials[PushConstants.materialIndex];

	//pbr
    vec4 albedo = texture(Sampler2D[mat.baseColorIndex], inUV);
    if(albedo.a < alphaCutoffTest){
        discard;
    }
    albedo = pow(albedo, vec4(2.2));
	vec2 metal_rough = texture(Sampler2D[mat.metallicRoughnessIndex], inUV).bg;
    
	vec3 normalMap = texture(Sampler2D[mat.normalIndex], inUV).xyz * 2.0 - 1.0;
    vec3 normal = normalize(inTBN * normalMap);
    
    vec3 color = vec3(0.0f);
    for(uint i = 0; i < sceneData.numLights; i++){
        Light l = sceneData.lightBuffer.lights[i];
        if(l.type == 0){
            color += calculatePointLight(l, albedo.xyz, metal_rough, normal);
        }else if(l.type == 1){
            color += calculateDirectionalLight(l, albedo.xyz, metal_rough, normal);
        }else if(l.type == 2){
            color += calculateSpotLight(l, albedo.xyz, metal_rough, normal);
        }
    }

    outFragColor = vec4(pow(color, vec3(1.0f/2.2f)), 1.0f); // gamma correction
}