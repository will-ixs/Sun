#version 450

layout(location = 0) in vec2 texCoord;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform sampler3D samplers[];
layout(set = 0, binding = 2, r16f) uniform image3D storageImages3D[];

layout( push_constant ) uniform constants
{	
	mat4 renderMat;
	vec3 camWorldPos;
	vec3 minBoundingPos;
	vec3 maxBoundingPos;
} PushConstants;


const float EPSILON = 0.001;

vec3 intersect(vec3 boxMin, vec3 boxMax, vec3 rayOrigin, vec3 rayDir) {
    vec3 invDir = 1.0 / rayDir;
    vec3 t0 = (boxMin - rayOrigin) * invDir;
    vec3 t1 = (boxMax - rayOrigin) * invDir;
    
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    
    float tNear = max(max(tmin.x, tmin.y), tmin.z);
    float tFar = min(min(tmax.x, tmax.y), tmax.z);
    
    if (tNear > tFar || tFar < 0.0) {
        discard;
    }
    
    //return closest intersection with box or camera world pos if ray is inside box
    return rayOrigin + rayDir * tNear;
}

bool inBounds(vec3 samplePos){
    if( samplePos.x < -EPSILON || samplePos.x > 1.0 + EPSILON ||
        samplePos.y < -EPSILON || samplePos.y > 1.0 + EPSILON ||
        samplePos.z < -EPSILON || samplePos.z > 1.0 + EPSILON) {
            return false;
    }
    return true;
}

float sampleDensity(vec3 samplePos){
        samplePos.y = 1.0 - samplePos.y;
        return texture(samplers[0], samplePos).r; //r16f
}

void main() {
    vec4 clipSpace = vec4(texCoord * 2.0 - 1.0, 0.0, 1.0);
    vec4 worldPos = PushConstants.renderMat * clipSpace;
    worldPos /= worldPos.w;
    
    const vec3 rayOrigin = PushConstants.camWorldPos;
    const vec3 rayDir = normalize(worldPos.xyz - rayOrigin);
    const vec3 minPos = PushConstants.minBoundingPos;
    const vec3 maxPos = PushConstants.maxBoundingPos;
    vec3 volumeSize = maxPos - minPos;
    float maxDim = max(max(volumeSize.x, volumeSize.y), volumeSize.z);
    const float marchStep = 1.0 / 1.250;
    const float densityScale = 12.0;
    const vec3 scalingFactors = vec3(0.1, 0.2, 0.6);

    // Raymarch
    vec3 rayPos = intersect(minPos, maxPos, rayOrigin, rayDir);
    float accumulated = 0.0;
    vec3 totalLight = vec3(0.0);
    int i = 0;
    for (i = 0; i < 256; i++) {
        vec3 samplePos = (rayPos - minPos) / (maxPos - minPos);
        if(!inBounds(samplePos)) break;
        float sampledDensity = sampleDensity(samplePos);
        accumulated += sampledDensity;

        float densityOccludingFluid = 0.0;
        float sampleStepSize = (samplePos.y + 0.1 )/ 16.0; 
        for(int j = 0; j < 16; j++) {
            vec3 aboveSamplePos = samplePos + (sampleStepSize * j);
            if(!inBounds(aboveSamplePos)) break;
            densityOccludingFluid += sampleDensity(aboveSamplePos);
        }

        vec3 inScatteredLight = exp(-densityOccludingFluid * scalingFactors) * sampledDensity * scalingFactors;
        vec3 transmittance = exp(-accumulated * scalingFactors);
        totalLight += inScatteredLight * transmittance;

        if (accumulated > densityScale) break;
        
        rayPos += rayDir * marchStep;
    }

    outColor = vec4(totalLight, 1.0);
}