#include "Camera.hpp"

#include <iostream>
#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>

Camera::Camera()
:
position(-10.0f, 2.0f, 5.0f), rotations(0.0f, -90.0f), forward(0.0f, 0.0f, 1.0f), velocity(0.0f, 0.0f, 0.0f),
zNear(500.0f), zFar(0.01f), fov(glm::radians(70.0f)), viewportWidth(1600.0f), viewportHeight(900.0f)
{
    updateLook(0.0f, 0.0f);
}
Camera::Camera(float width, float height)
:
position(-10.0f, 2.0f, 5.0f), rotations(0.0f, -90.0f), forward(0.0f, 0.0f, 1.0f), velocity(0.0f, 0.0f, 0.0f),
zNear(500.0f), zFar(0.01f), fov(glm::radians(70.0f)), viewportWidth(width), viewportHeight(height)
{
    updateLook(0.0f, 0.0f);
}

Camera::~Camera()
{
}

glm::mat4 Camera::getRenderMatrix(){

    glm::mat4 view = glm::lookAt(position, position + forward, glm::vec3(0.0f, 1.0f, 0.0f));

	glm::mat4 projection = glm::perspectiveFovZO(fov, viewportWidth, viewportHeight, zNear, zFar);	
    projection[1][1] *= -1;
    
    return glm::mat4(1.0f) * projection * view;

}

glm::mat4 Camera::getOrthoRenderMatrix(){

    glm::mat4 view = glm::lookAt(position, position + forward, glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 projection = glm::orthoZO(-160.0f, 160.0f, -90.0f, 90.0f, zNear, zFar);
    projection[1][1] *= -1;
    
    return glm::mat4(1.0f) * projection * view;

}

void Camera::updateLook(float dx, float dy){
    float sensitivity = 0.4f;
    rotations.y += (dx * sensitivity);
    rotations.x += (dy * sensitivity);

    // float pitch = std::min(glm::radians(89.999f), std::max(glm::radians(-89.999f), glm::radians(rotations.x)));
    float pitch = glm::radians(rotations.x);
    float yaw = glm::radians(rotations.y);
    // float yaw = std::min(0.0f, std::max(glm::radians(359.999f), glm::radians(rotations.y)));
    float sinPhi = std::sin(pitch);
    float cosPhi = std::cos(pitch);
    float sinTheta = std::sin(yaw);
    float cosTheta = std::cos(yaw);
    
    forward = glm::normalize(glm::vec3(cosPhi * cosTheta, sinPhi, cosPhi * sinTheta));
}
void Camera::updateVelocity(glm::vec3 dp){
    velocity += dp;
}
void Camera::updatePosition(float deltaTime){
    velocity *= 0.95;    
    glm::vec3 right = glm::cross(forward, worldUp);

    glm::vec3 deltaPos = glm::vec3(0.0f);
    deltaPos += forward * velocity[2];
    deltaPos += right * velocity[0];

    position += (deltaPos * deltaTime);
}