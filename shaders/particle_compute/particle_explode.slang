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

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;
    if (index >= pcs.particleCount) {
        return;
    }
    float3 pos = pcs.positionsA[index];
    float3 vel = pcs.velocities[index];

    pos += vel * pcs.deltaTime * pcs.timeScale;
    pcs.positionsB[index] = pos;
    pcs.velocities[index] = vel;
}