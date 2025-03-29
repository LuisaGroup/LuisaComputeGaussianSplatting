#pragma once
/**
 * @file gs_tile_splatter.h
 * @brief The Gaussian Tile Splatter
 * @author sailing-innocent
 * @date 2024-11-27
 */

#include "lcgs/config.h"
#include "lcgs/module.h"
#include "lcgs/proxy.h"
#include "lcgs/util/buffer_filler.h"
#include "lcgs/util/device_parallel.h"
#include "proxy.h"

namespace lcgs
{

class LCGS_API GSTileSplatter : public GSModule
{
public:
    int num_rendered          = 0;
    GSTileSplatter()          = default;
    virtual ~GSTileSplatter() = default;

    virtual void create(Device& device) noexcept;
    virtual int  forward(
         Device&                   device,
         Stream&                   stream,
         GSTileSplatterAccelProxy  accel,
         GSTileSplatterInputProxy  input,
         GSSplatForwardOutputProxy output,
         bool                      use_focal = true
     ) noexcept;

    BufferFiller*   mp_buffer_filler;
    void            set_buffer_filler(BufferFiller* buffer_filler) noexcept { mp_buffer_filler = buffer_filler; }
    DeviceParallel* mp_device_parallel;
    void            set_device_parallel(DeviceParallel* device_parallel) noexcept { mp_device_parallel = device_parallel; }

protected:
    virtual void compile(Device& device) noexcept;
    virtual void compile_forward_shader(Device& device) noexcept;
    virtual void compile_impl_shader(Device& device) noexcept;

    U<Shader<1, int,        // P
             uint2, uint2,  // resolution, grids
             Buffer<float>, // depth_features // P
             Buffer<float>, // means_2d // 2 * P
             Buffer<float>, // covs_2d // 3 * P
             Buffer<uint>,  // tiles_touched // P
             Buffer<int>,   // radii // P
             bool           // use_focal
             >>
        shad_allocate_tiles;

    U<Shader<1, int,        // P
             Buffer<float>, // means_2d
             Buffer<uint>,  // offsets
             Buffer<int>,   // radii
             Buffer<float>, // depth
             Buffer<ulong>, // keys_unsorted
             Buffer<uint>,  // values_unsorted
             uint2, uint2   // blocks & grids
             >>
        shad_copy_with_keys;

    U<Shader<1, int,        // num_rendered
             Buffer<ulong>, // point_list_keys
             Buffer<uint>   // ranges
             >>
        shad_get_ranges;

    U<Shader<2,
             uint2,         // resolution
             int, int,      // P, L // for debug
             Buffer<float>, // target img
             // params
             uint2,  // grids
             float3, // bg_color
             // input buffers
             Buffer<uint>,  // ranges
             Buffer<uint>,  // point_list
             Buffer<float>, // means_2d, P x 2
             Buffer<float>, // conic, P x 3
             Buffer<float>, // opacity_features, P
             Buffer<float>  // color_features, P * 3
             >>
        m_forward_render_shader;
};

} // namespace lcgs
