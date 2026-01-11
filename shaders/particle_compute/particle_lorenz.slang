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

    float sigma = 10;
    float rho = 28;
    float beta = 8.0 / 3.0;

    float dx = sigma * (pos.y - pos.x);
    float dy = (rho * pos.x) - pos.y - (pos.x * pos.z);
    float dz = (pos.x * pos.y) - (beta * pos.z);
    float3 vel = float3(dx, dy, dz);

    pos += vel * pcs.deltaTime * pcs.timeScale;

    pcs.positionsB[index] = pos;
    pcs.velocities[index] = vel;
}