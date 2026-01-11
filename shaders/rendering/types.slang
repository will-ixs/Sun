struct Vertex {
	float3 position;
	float uv_x;
	float3 normal;
	float uv_y;
	float4 color;
	float4 tangent;
};
struct Material {
    float4 baseColor;
    float metallicFactor;
    float roughnessFactor;
    uint baseColorIndex;
    uint metallicRoughnessIndex;
    uint normalIndex;
    uint _pad0, _pad1, _pad2;
};
struct Light {
    uint type;
    uint _pad0, _pad1, _pad2;

    float4 pos;
    float4 dir;
    float4 col;
    
    float intensity;
    float range;
    float innerConeAngle;
    float outerConeAngle;
};
