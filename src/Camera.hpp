#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "types.h"

class Camera
{
private:
const glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 position;
    glm::vec2 rotations;
    
    glm::vec3 velocity;
    glm::vec3 forward;

    float zNear;
    float zFar;
public:
    float fov;
    float viewportWidth;
    float viewportHeight;

    Camera();
    Camera(float width, float height);
    ~Camera();

    glm::mat4 getRenderMatrix();
    void updateLook(float dx, float dy);
    void updateVelocity(glm::vec3 dp);
    void updatePosition(uint64_t deltaTime);
};

#endif