#pragma once
/**
 * @file proxy.h
 * @brief The Gaussian Splatter Proxy
 * @author sailing-innocent
 * @date 2025-03-06
 */

#include <luisa/runtime/buffer.h>

namespace lcgs
{

// TODO: set to a max value
struct GSSplatBufferAllocFunc {
    std::function<char*(size_t N)> geometryBuffer;
    std::function<char*(size_t N)> screenBuffer;
    std::function<char*(size_t N)> tileBuffer;
};

struct GSSplatForwardInputProxy {
    int                               num_gaussians = 0;
    luisa::compute::BufferView<float> xyz;
    luisa::compute::BufferView<float> color;
    luisa::compute::BufferView<float> opacity;
    luisa::compute::BufferView<float> scale;
    luisa::compute::BufferView<float> rotq;
    luisa::float3                     bg_color = { 0.0f, 0.0f, 0.0f };
};

struct GSSplatSHForwardInputProxy {
    int                               num_gaussians  = 0;
    int                               sh_deg         = 3;
    int                               feat_dim       = 3;
    float                             scale_modifier = 1.0f;
    luisa::compute::BufferView<float> xyz;
    luisa::compute::BufferView<float> feature;
    luisa::compute::BufferView<float> opacity;
    luisa::compute::BufferView<float> scale;
    luisa::compute::BufferView<float> rotq;
    luisa::float3                     bg_color = { 0.0f, 0.0f, 0.0f };
};

struct GSTileSplatterInputProxy {
    int           num_gaussians;
    luisa::float3 bg_color;
    // param used to build accel structure

    luisa::compute::BufferView<float> means_2d;       // 2 * P
    luisa::compute::BufferView<float> depth_features; // P
    luisa::compute::BufferView<float> conic;          // 3 * P

    // payload
    luisa::compute::BufferView<float> color_features;   // 3 * P
    luisa::compute::BufferView<float> opacity_features; // P
};

struct GSTileSplatterAccelProxy {
    luisa::compute::BufferView<uint32_t>     temp_storage;
    luisa::compute::BufferView<luisa::uint>  tiles_touched;            // P
    luisa::compute::BufferView<luisa::uint>  point_offsets;            // P
    luisa::compute::BufferView<luisa::ulong> point_list_keys_unsorted; // L
    luisa::compute::BufferView<luisa::uint>  point_list_unsorted;      // L
    luisa::compute::BufferView<luisa::ulong> point_list_keys;          // L
    luisa::compute::BufferView<luisa::uint>  point_list;               // L
    luisa::compute::BufferView<luisa::uint>  ranges;                   // TW x TH x 2
};

struct GSSplatForwardOutputProxy {
    int                               height;
    int                               width;
    luisa::compute::BufferView<float> target_img; // hwc
    luisa::compute::BufferView<int>   radii;      // P
};

} // namespace lcgs