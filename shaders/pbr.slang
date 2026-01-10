import "types";

struct PushConstants {
    float4x4 model;
    Vertex* vertexBuffer;
    uint materialIndex;
};

struct UniformData {
    float4x4 viewProj;
    float4 camWorldPos;
    float4 ambientColor;
    float4 sunDir; // w is power
    float4 sunCol;
    Material* materialBuffer;
    Light* lightBuffer;
    uint numLights;
};

[[vk::push_constant]]
PushConstants pcs;

[[vk::binding(1, 0)]]
Sampler2D textures[];

[[vk::binding(3, 0)]]
ConstantBuffer<UniformData> ubo;

struct VSInput {
    uint vertexID : SV_VertexID;
};

struct VSOutput {
    float4 projPos : SV_Position;
    float3 color;
    float2 uv;
    float3 norm;
    float4 worldPos;
    float3x3 outTBN;
};

#define PI 3.14159265359

[shader("vertex")]
VSOutput main(VSInput input) {
    Vertex v = pcs.vertexBuffer[input.vertexID];
    Material m = ubo.materialBuffer[pcs.materialIndex];

    float4 pos = float4(v.position, 1.0);
    float4 worldPos = mul(pcs.model, pos);

    float3 T = normalize(mul(float3x3(pcs.model), v.tangent.xyz));
    float3 N = normalize(mul(float3x3(pcs.model), v.normal));
    float3 B = cross(N, T);

    VSOutput output;
    output.projPos = mul(ubo.viewProj, worldPos);
    output.color = v.color.rgb * m.baseColor.rgb;
    output.uv = float2(v.uv_x, v.uv_y);
    output.norm = mul(pcs.model, float4(v.normal, 1.0)).xyz;
    output.worldPos = worldPos;
    output.outTBN = float3x3(T, B, N);
    return output;
}

[shader("pixel")]
float4 main(VSOutput input) : SV_Target0 {
    Material m = ubo.materialBuffer[pcs.materialIndex];

    float4 albedo = float4(0.0);
    if (m.baseColorIndex != 0xFFFFFFFF) {
        albedo = textures[m.baseColorIndex].Sample(input.uv);
    }
    if (albedo.w < 0.5) {
        discard;
    }

    float2 metalRough = textures[m.metallicRoughnessIndex].Sample(input.uv).bg;
    float3 normalMap = textures[m.normalIndex].Sample(input.uv).xyz * 2.0 - 1.0;
    float3 normal = normalize(mul(input.outTBN, normalMap));

    float3 color = float3(0.0);
    for (uint i = 0; i < ubo.numLights; i++) {
        Light l = ubo.lightBuffer[i];
        if (l.type == 0) {
            color += calcPointLight(l, input.worldPos.xyz, albedo.rgb, metalRough, normal);
        }
            // } else if (l.type == 1) {
        //     color += calcDirectionalLight(l, albedo.rgb, metalRough, normal);
        // } else if (l.type == 2) {
        //     color += calcSpotLight(l, albedo.rgb, metalRough, normal);
        // }
    }

    return float4(pow(color, float3(1.0 / 2.2)), 1.0);


}

float3 fresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float DistributionGGX(float3 N, float3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265359 * denom * denom);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(float3 N, float3 V, float3 L, float roughness) {
    float ggx1 = GeometrySchlickGGX(max(dot(N, V), 0.0), roughness);
    float ggx2 = GeometrySchlickGGX(max(dot(N, L), 0.0), roughness);
    return ggx1 * ggx2;
}

float3 calcPointLight(Light p, float3 fragPos, float3 albedo, float2 metalRough, float3 normal) {
    float3 L = normalize(p.pos.xyz - fragPos);
    float3 V = normalize(ubo.camWorldPos.xyz - fragPos);
    float3 H = normalize(V + L);
    float3 F0 = lerp(float3(0.04f), albedo, metalRough.x);

    float dist = length(p.pos.xyz - fragPos);
    if (p.range > 0.0f && dist > p.range) {
        return float3(0.0f);
    }

    float attenuation = 1.0f / (dist * dist);
    attenuation *= clamp(1.0 - ((dist * dist) / (p.range * p.range)), 0.0, 1.0);

    float3 radiance = p.col.rgb * attenuation * p.intensity;

    float NDF = DistributionGGX(normal, H, metalRough.y);
    float G = GeometrySmith(normal, V, L, metalRough.y);
    float3 F = fresnelSchlick(max(dot(H, V), 0.0f), F0);

    float3 kS = F;
    float3 kD = float3(1.0f) - kS;
    kD *= 1.0f - metalRough.x;

    float3 top = NDF * G * F;
    float bot = 4.0f * max(dot(normal, V), 0.0f) * max(dot(normal, L), 0.0f) + 0.0001f;
    float3 specular = top / bot;

    float NdotL = max(dot(normal, L), 0.0f);
    return (kD * albedo / PI + specular) * radiance * NdotL;
}
