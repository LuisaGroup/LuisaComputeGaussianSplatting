/**
 * @file camera.cpp
 * @brief The Camera Impl
 * @author sailing-innocent
 * @date 2025-03-29
 */

#include "lcgs/util/camera.h"

namespace lcgs {


luisa::float4x4 local_to_world_transform(Camera& cam) noexcept {
    luisa::float4x4 m{
    luisa::make_float4(cam.right, 0.0f), // camera x 
    luisa::make_float4(cam.up, 0.0f), // camera y 
    luisa::make_float4(cam.front, 0.0f), // camera z 
    luisa::make_float4(cam.position, 1.0f)}; // trans
    return m;
}

luisa::float4x4 world_to_local_transform(Camera& cam) noexcept {
    // Calculate the dot products for the transformed translation
    float tx = -luisa::dot(cam.position, cam.right);
    float ty = -luisa::dot(cam.position, cam.up);
    float tz = -luisa::dot(cam.position, cam.front);
    
    return luisa::float4x4{
        luisa::make_float4(cam.right.x, cam.up.x, cam.front.x, 0.0f),     // First column of transpose
        luisa::make_float4(cam.right.y, cam.up.y, cam.front.y, 0.0f),     // Second column of transpose
        luisa::make_float4(cam.right.z, cam.up.z, cam.front.z, 0.0f),     // Third column of transpose
        luisa::make_float4(tx, ty, tz, 1.0f)                              // Fourth column with translation
    };
}

// fovx/fovy in rad
luisa::float4x4 projection_transform(float fovx, float fovy, float znear, float zfar) noexcept {
    float zsign = 1.0f;
    
    // Calculate scale factors
    float tan_half_fovy = std::tan(fovy * 0.5f);
    float tan_half_fovx = std::tan(fovx * 0.5f);
    
    float fx = 1.0f / tan_half_fovx;
    float fy = 1.0f / tan_half_fovy;
    
    // Compute projection matrix elements
    float z_range = zfar - znear;
    float a = zfar / z_range;
    float b = -zfar * znear / z_range;
    
    // Right-handed projection matrix (looking down negative z-axis)
    return luisa::float4x4{
        luisa::make_float4(fx, 0.0f, 0.0f, 0.0f),                // First column
        luisa::make_float4(0.0f, fy, 0.0f, 0.0f),                // Second column
        luisa::make_float4(0.0f, 0.0f, a * zsign, zsign),        // Third column
        luisa::make_float4(0.0f, 0.0f, b, 0.0f)                  // Fourth column
    };
}

Camera get_lookat_cam(luisa::float3 pos, luisa::float3 target, luisa::float3 world_up) {
    Camera cam;
    cam.position = pos;
    cam.front = luisa::normalize(target - pos);
    cam.right = luisa::normalize(luisa::cross(cam.front, world_up));
    cam.up = luisa::normalize(luisa::cross(cam.right, cam.front));
    return cam;
}

} // namespace lcgs