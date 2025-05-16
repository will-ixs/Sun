#include "Camera.hpp"

#include <iostream>
#include <algorithm>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

Camera::Camera()
:
position(0.0f, 2.0f, 5.0f), yaw(0.0f), pitch(0.0f), forward(0.0f, 0.0f, 1.0f), velocity(0.0f, 0.0f, 0.0f),
zNear(500.0f), zFar(0.01f), fov(glm::radians(70.0f)), viewportWidth(1600.0f), viewportHeight(900.0f)
{
}
Camera::Camera(float width, float height)
:
position(0.0f, 2.0f, 5.0f), yaw(0.0f), pitch(0.0f), forward(0.0f, 0.0f, 1.0f), velocity(0.0f, 0.0f, 0.0f),
zNear(500.0f), zFar(0.01f), fov(glm::radians(70.0f)), viewportWidth(width), viewportHeight(height)
{
}

Camera::~Camera()
{
}

void Camera::update()
{       

    float sinPhi = std::sin(pitch);
    float cosPhi = std::cos(pitch);
    float sinTheta = std::sin(yaw);
    float cosTheta = std::cos(yaw);
    forward = glm::normalize(glm::vec3(cosPhi * cosTheta, sinPhi, cosPhi * sinTheta));

    glm::vec3 right = glm::cross(forward, worldUp);
    glm::vec3 adjustedVelocity = forward * velocity.z + right * velocity.x;
    position += adjustedVelocity * 0.125f;
}

void Camera::processSDLEvent(SDL_Event& e)
{
    if (e.type == SDL_EVENT_KEY_DOWN) {
        if (e.key.key == SDLK_W) { velocity.z = 1; }
        if (e.key.key == SDLK_S) { velocity.z = -1; }
        if (e.key.key == SDLK_A) { velocity.x = -1; }
        if (e.key.key == SDLK_D) { velocity.x = 1; }
    }

    if (e.type == SDL_EVENT_KEY_UP) {
        if (e.key.key == SDLK_W) { velocity.z = 0; }
        if (e.key.key == SDLK_S) { velocity.z = 0; }
        if (e.key.key == SDLK_A) { velocity.x = 0; }
        if (e.key.key == SDLK_D) { velocity.x = 0; }
    }

    if (e.type == SDL_EVENT_MOUSE_MOTION) {
        yaw += (float)e.motion.xrel / 400.f;
        pitch -= (float)e.motion.yrel / 400.f;
    }
}

glm::mat4 Camera::getViewMatrix()
{
    return glm::lookAt(position, position + forward, glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::mat4 Camera::getProjMatrix(){
	glm::mat4 projection = glm::perspectiveFovZO(fov, viewportWidth, viewportHeight, zNear, zFar);	
    projection[1][1] *= -1;
    return projection;
}

glm::mat4 Camera::getRenderMatrix(){
    glm::mat4 view = glm::lookAt(position, position + forward, glm::vec3(0.0f, 1.0f, 0.0f));

	glm::mat4 projection = glm::perspectiveFovZO(fov, viewportWidth, viewportHeight, zNear, zFar);	
    projection[1][1] *= -1;
    
    return glm::mat4(1.0f) * projection * view;
}