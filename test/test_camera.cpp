/**
 * @file test_camera.cpp
 * @brief Camera Test Suite
 * @author sailing-innocent
 * @date 2025-03-29
 */

#include "test_util.h"
#include "lcgs/util/camera.h"
#include <cmath>

namespace lcgs::test {

constexpr float M_PI = 3.14159265359f;
constexpr float M_SQRT2 = 1.41421356f;

// Helper function to check if two float3 values are approximately equal
bool approx_equal(const luisa::float3& a, const luisa::float3& b, float epsilon = 1e-5f) {
    // LUISA_INFO("{} {} {}", a, b, epsilon);
    return std::abs(a.x - b.x) < epsilon && 
           std::abs(a.y - b.y) < epsilon && 
           std::abs(a.z - b.z) < epsilon;
}

// Helper function to check if two float4x4 matrices are approximately equal
bool approx_equal_matrix(const luisa::float4x4& a, const luisa::float4x4& b, float epsilon = 1e-5f) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (std::abs(a[i][j] - b[i][j]) > epsilon) {
                return false;
            }
        }
    }
    return true;
}

bool test_cam() {
    // Test get_lookat_cam
    luisa::float3 position = {-4.0f, -4.0f, 0.0f};
    luisa::float3 target = {0.0f, 0.0f, 0.0f};
    luisa::float3 world_up = {0.0f, 0.0f, 1.0f};
    
    Camera camera = get_lookat_cam(position, target, world_up);
    
    // Verify position
    CHECK(approx_equal(camera.position, position));
    
    // Verify front direction (should point toward target)
    luisa::float3 expected_front = luisa::make_float3(1.0f / M_SQRT2, 1.0f/M_SQRT2, 0.0f);
    CHECK(approx_equal(camera.front, expected_front));
    
    // Verify right and up vectors are orthogonal
    CHECK(std::abs(luisa::dot(camera.up, camera.right)) < 1e-5f);
    CHECK(std::abs(luisa::dot(camera.up, camera.front)) < 1e-5f);
    CHECK(std::abs(luisa::dot(camera.right, camera.front)) < 1e-5f);
    
    // Test local_to_world and world_to_local transforms are inverse of each other
    luisa::float4x4 l2w = local_to_world_transform(camera);
    luisa::float4x4 w2l = world_to_local_transform(camera);
    
    // Create a test point in local space
    luisa::float4 local_point = {2.0f * M_SQRT2, 3.0f, 2.0f * M_SQRT2,  1.0f};
    
    // Transform to world space and back
    luisa::float4 world_point = l2w * local_point;
    luisa::float3 world_point_exp = {0.0f, -4.0f, 3.0f};
    CHECK(approx_equal(world_point.xyz(), world_point_exp));

    luisa::float4 round_trip = w2l * world_point;
    // Check round-trip is approximately the original point
    CHECK(std::abs(round_trip.x - local_point.x) < 1e-5f);
    CHECK(std::abs(round_trip.y - local_point.y) < 1e-5f);
    CHECK(std::abs(round_trip.z - local_point.z) < 1e-5f);
    CHECK(std::abs(round_trip.w - local_point.w) < 1e-5f);
    
    // Test projection matrix
    float fovx = 60.0f * M_PI / 180.0f;  // 60 degrees in radians
    float fovy = 45.0f * M_PI / 180.0f;  // 45 degrees in radians
    float near = 0.1f;
    float far = 100.0f;
    
    luisa::float4x4 proj = projection_transform(fovx, fovy, near, far);
    
    // Check some expected properties of the projection matrix
    // For example, points at z=near should map to NDC z=0
    luisa::float4 near_point = {0.0f, 0.0f, near, 1.0f};
    luisa::float4 near_projected = proj * near_point;
    CHECK(std::abs(near_projected.z / near_projected.w) < 1e-5f);
    
    // Points at z=far should map to NDC z=1
    luisa::float4 far_point = {0.0f, 0.0f, far, 1.0f};
    luisa::float4 far_projected = proj * far_point;
    CHECK(std::abs(far_projected.z / far_projected.w - 1.0f) < 1e-5f);
    
    luisa::float4 test_point = {0.2f, .3f, 2.0f, 1.0f};
    luisa::float4 projected_point = proj * test_point;
    luisa::float2 expected_projected = { 0.2f/ std::tan(fovx/2) / 2.0f, .3f / std::tan(fovy/2) / 2.0f};
    // NDC 
    CHECK(std::abs(projected_point.x / projected_point.w - expected_projected.x) < 1e-5);
    CHECK(std::abs(projected_point.y / projected_point.w - expected_projected.y) < 1e-5);
    
    return true;
}

bool test_special_camera_cases() {
    // Test camera looking along world up axis
    luisa::float3 position = {0.0f, 0.0f, 5.0f};
    luisa::float3 target = {0.0f, 0.0f, 10.0f};  // Looking along +Z
    luisa::float3 world_up = {0.0f, 1.0f, 0.0f};
    
    Camera camera = get_lookat_cam(position, target, world_up);
    
    // Verify the camera's right vector is orthogonal to world_up and front
    CHECK(approx_equal(camera.right, luisa::float3{1.0f, 0.0f, 0.0f}));
    
    // Test transformation of a point
    luisa::float4x4 l2w = local_to_world_transform(camera);
    luisa::float4 local_point = {0.0f, 0.0f, 1.0f, 1.0f};  // 1 unit in front of camera
    luisa::float4 world_point = l2w * local_point;
    
    // Expected position is 1 unit along front vector from camera position
    luisa::float3 expected_position = camera.position + camera.front;
    CHECK(approx_equal(
        luisa::float3{world_point.x, world_point.y, world_point.z},
        expected_position
    ));
    
    return true;
}

}// namespace lcgs::test

TEST_SUITE("basic") {
    TEST_CASE("camera") {
        CHECK(lcgs::test::test_cam());
    }
    
    TEST_CASE("camera-special-cases") {
        CHECK(lcgs::test::test_special_camera_cases());
    }
}