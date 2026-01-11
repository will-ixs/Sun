struct PushConstants {
    float3 *positionsA;
    float3 *positionsB;
    float3 *velocities;
    float deltaTime;
    float timeScale;
    float time;
    uint particleCount;
};

[[vk::push_constant]]
PushConstants pcs;

#define PI 3.14159265359

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;
    if (index >= pcs.particleCount) {
        return;
    }
    float3 pos = pcs.positionsA[index];

    float a = 36.0f;
    float b = 3.0f;
    float c = 20.0f;
    float u = -15.15f;

    float dx = a * (pos.y - pos.x);
    float dy = pos.x - pos.x * pos.z + c * pos.y + u;
    float dz = pos.x * pos.y - b * pos.z;
    float3 vel = float3(dx, dy, dz);

    pos += vel * pcs.deltaTime * pcs.timeScale;

    pcs.positionsB[index] = pos;
    pcs.velocities[index] = vel;
}