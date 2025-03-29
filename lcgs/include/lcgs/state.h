#pragma once
/**
 * @file state.h
 * @brief The Gaussian State
 * @author sailing-innocent
 * @date 2025-03-06
 */

#include <luisa/dsl/soa.h>
#include <luisa/luisa-compute.h>

namespace lcgs
{

using namespace luisa;
using namespace luisa::compute;
// Screen Space Points List
struct PointState {
    float2 means_2d;        // the 2D mean of the gaussian
    float  depth_feature;   // the depth feature of the gaussian
    float3 color_feature;   // the color feature of the gaussian
    float  opacity_feature; // the opacity feature of the gaussian
    float3 conic;           // the inverse of the covariance matrix of the gaussian
    uint   tiles_touched;   // the number of tiles touched by the gaussian
    uint   point_offsets;   // the offset of the point in the point list
    uint   clamped;         // whether the point color is clamped
};

// Tiled Instance State
struct InstanceState {
    ulong key_unsorted;
    ulong key;
    uint  value_unsorted;
    uint  value;
};

// Tile TW x TH
// attributes of tiles
struct TileState {
    uint ranges; // from which index to which index of points belong to this tile
};

// Pixel W x H
// attributes of pixels
struct ScreenState {
    uint  n_contrib;   // how much gaussians contribute to this pixel
    float accum_alpha; // accumulated alpha on this pixel
};

} // namespace lcgs

LUISA_STRUCT(lcgs::PointState, means_2d, depth_feature, color_feature, opacity_feature, conic, tiles_touched, point_offsets, clamped){};
LUISA_STRUCT(lcgs::InstanceState, key_unsorted, key, value_unsorted, value){};
LUISA_STRUCT(lcgs::TileState, ranges){};
LUISA_STRUCT(lcgs::ScreenState, n_contrib, accum_alpha){};

// SOA<>
// SOAVar<>