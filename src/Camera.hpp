#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "types.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_events.h>

class Camera
{
private:
const glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 position;
    glm::vec3 velocity;    

    glm::vec3 forward;
    
    float zNear;
    float zFar;
public:
    float fov;
    float viewportWidth;
    float viewportHeight;
    float pitch { 0.f };
    float yaw { 0.f };
    Camera();
    Camera(float width, float height);
    ~Camera();

    glm::vec4 getPos() {return glm::vec4(position, 1.0f);};
    
    glm::mat4 getViewMatrix();
    glm::mat4 getRotationMatrix();
    glm::mat4 getRenderMatrix();
    glm::mat4 getProjMatrix();
    void update();
    void processSDLEvent(SDL_Event& e);
};

#endif