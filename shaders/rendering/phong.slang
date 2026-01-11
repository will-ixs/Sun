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
};

[shader("vertex")]
VSOutput main(VSInput input) {
    
    Vertex v = pcs.vertexBuffer[input.vertexID];
    Material m = ubo.materialBuffer[pcs.materialIndex];

    float4 pos = float4(v.position, 1.0);
    float4 worldPos = mul(pcs.model, pos);

    VSOutput output;
    output.projPos = mul(ubo.viewProj, worldPos);
    output.color = v.color.xyz * m.baseColor.xyz;
    output.uv = float2(v.uv_x, v.uv_y);
    output.norm = mul(pcs.model, float4(v.normal, 0.0)).xyz;
    output.worldPos = worldPos;
    return output;
}

[shader("pixel")]
float4 main(VSOutput input) : SV_Target0 {
    Material m = ubo.materialBuffer[pcs.materialIndex];
    float4 albedo = float4(input.color, 1.0);

    if (m.baseColorIndex != 0xFFFFFFFF) {
        albedo *= textures[m.baseColorIndex].Sample(input.uv);
    }
    if (albedo.w < 0.5) {
        discard;
    }

    float3 normal = normalize(input.norm);
    float3 viewDir = normalize(input.worldPos.xyz - ubo.camWorldPos.xyz);
    float3 fragPos = input.worldPos.xyz;

    float3 color = ubo.ambientColor.rgb * albedo.rgb;
    
    float3 lightDir = normalize(ubo.sunDir.xyz);
    float3 halfDir = normalize(lightDir + viewDir);

    float shininess = 4.0; // Convert roughness to shininess
    float diffuseStrength = max(dot(normal, lightDir), 0.0);
    float specularStrength = pow(max(dot(normal, halfDir), 0.0), shininess);

    float3 diffuse = diffuseStrength * ubo.sunCol.rgb * albedo.rgb;
    float3 specular = specularStrength * ubo.sunCol.rgb * albedo.rgb; // w is sun power
    color += diffuse + specular;

    for (uint i = 0; i < ubo.numLights; i++) {
        Light l = ubo.lightBuffer[i];
        lightDir = normalize(l.pos.xyz - fragPos);
        float dist = distance(l.pos.xyz, fragPos);
        halfDir = normalize(lightDir + viewDir);
        
        
        diffuseStrength = max(dot(normal, lightDir), 0.0);
        specularStrength = pow(max(dot(normal, halfDir), 0.0), shininess);

        float power = 0.035 * l.intensity / dist;
        diffuse = diffuseStrength * l.col.rgb * albedo.rgb * power;
        specular = specularStrength * l.col.rgb * albedo.rgb * power; // w is sun power
        color += (diffuse + specular);
    }

    return float4(color, albedo.a);
}