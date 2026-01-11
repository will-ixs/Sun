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

    float alpha = 0.2;
    float beta = 0.2;
    float sigma = 5.7;

    float dx = -pos.y - pos.z;
    float dy = pos.x + alpha * pos.y;
    float dz = beta + (pos.z * (pos.x - sigma));
    float3 vel = float3(dx, dy, dz);

    pos += vel * pcs.deltaTime * pcs.timeScale;

    pcs.positionsB[index] = pos;
    pcs.velocities[index] = vel;
}