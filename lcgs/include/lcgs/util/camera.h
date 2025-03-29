#pragma once 
/**
 * @file camera.h
 * @brief The LCGS Camera impl
 * @author sailing-innocent
 * @date 2025-03-29
 */

#include "lcgs/config.h"
#include "luisa/dsl/sugar.h"

namespace lcgs {

struct Camera {
    luisa::float3 position;
    luisa::float3 front;
    luisa::float3 up;
    luisa::float3 right;
    float fov;
};

Camera LCGS_API get_lookat_cam(luisa::float3 pos, luisa::float3 target, luisa::float3 world_up = {0.0f, 0.0f, 1.0f});
luisa::float4x4 LCGS_API local_to_world_transform(Camera& cam) noexcept;
luisa::float4x4 LCGS_API world_to_local_transform(Camera& cam) noexcept;
luisa::float4x4 LCGS_API projection_transform(float fovx, float fovy, float znear = 0.1f, float zfar=100.0f) noexcept;

}// namespace lcgs