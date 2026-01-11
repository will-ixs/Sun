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

    float a = 1.3f;
    float b = 0.11f;
    float c = 7.0f;
    float alpha = 10.82f;
    float beta = 14.286f;

    float h = 0;
    if (pos.x >= 2 * a * c) {
        h = ((b * PI) / (2 * a)) * (pos.x - (2 * a * c));
    } else if (pos.x > -2 * a * c) {
        h = -b * sin((PI * pos.x) / (2 * a));
    } else {
        h = ((b * PI) / (2 * a)) * (pos.x + (2 * a * c));
    }

    float dx = alpha * (pos.y - h);
    float dy = pos.x - pos.y + pos.z;
    float dz = -beta * pos.y;
    float3 vel = float3(dx, dy, dz);

    pos += vel * pcs.deltaTime * pcs.timeScale;

    pcs.positionsB[index] = pos;
    pcs.velocities[index] = vel;
}