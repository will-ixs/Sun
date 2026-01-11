struct PushConstants {
    float4x4 renderMat;
    float3* positions;
    float3* velocities;
    float4 camWorldPos;
    float4 originPos;
};

struct VSInput {
    uint vertexID : SV_VertexID;
};

struct VSOutput {
    float4 projPos : SV_Position;
    float3 color;
    float3 eye;
    float3 worldPos;
    float pointSize : SV_PointSize;
};

[[vk::push_constant]]
PushConstants pcs;

[shader("vertex")]
VSOutput main(VSInput input) {
    float3 pPos = pcs.positions[input.vertexID];
    float3 pVel = pcs.velocities[input.vertexID];

    VSOutput output;
    output.projPos = mul(pcs.renderMat, float4(pPos + pcs.originPos.xyz, 1.0));
    output.color = abs(normalize(pVel) + 0.2);
    output.eye = pcs.camWorldPos.xyz;
    output.worldPos = pPos;
    output.pointSize = 1.0;

    return output;
}

[shader("pixel")]
float4 main(VSOutput input) : SV_Target0 {
    return float4(input.color, 1.0);
}