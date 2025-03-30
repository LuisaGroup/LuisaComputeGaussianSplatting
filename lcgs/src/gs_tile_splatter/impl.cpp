/**
 * @file gs_tile_diff_splatter
 * @brief The Gaussian Splatting Tile Splatter
 * @author sailing-innocent
 * @date 2024-11-27
 */

#include "lcgs/gs_tile_splatter.h"
#include "lcgs/state.h"

namespace lcgs
{

using namespace luisa;
using namespace luisa::compute;

void GSTileSplatter::create(Device& device) noexcept
{
    compile(device);
    LUISA_INFO("Tile Splatter created");
}

int GSTileSplatter::forward(
    Device&                   device,
    Stream&                   stream,
    GSTileSplatterAccelProxy  accel,
    GSTileSplatterInputProxy  input,
    GSSplatForwardOutputProxy output,
    bool                      use_focal
) noexcept
{
    auto width      = output.width;
    auto height     = output.height;
    auto resolution = luisa::make_uint2(width, height);

    auto grids = luisa::make_uint2(
        (unsigned int)((width + m_blocks.x - 1u) / m_blocks.x),
        (unsigned int)((height + m_blocks.y - 1u) / m_blocks.y)
    );
    LUISA_INFO("grids: ({}, {})", grids.x, grids.y);

    int  num_gaussians   = input.num_gaussians;
    auto d_point_offsets = accel.point_offsets.subview(0, num_gaussians);
    auto d_tiles_touched = accel.tiles_touched.subview(0, num_gaussians);

    CommandList cmdlist;

    // screen_state->tile state->image state
    cmdlist
        << (*shad_allocate_tiles)(
               num_gaussians,
               resolution,
               grids,
               input.depth_features,
               input.means_2d,
               input.conic,
               d_tiles_touched,
               output.radii,
               use_focal
           )
               .dispatch(num_gaussians);

    // inclusive scan
    mp_device_parallel->scan_inclusive_sum<uint>(
        cmdlist,
        accel.temp_storage,
        d_tiles_touched,
        d_point_offsets,
        0, num_gaussians
    );

    cmdlist << accel.point_offsets.subview(input.num_gaussians - 1, 1).copy_to(&num_rendered);
    stream << cmdlist.commit() << synchronize();

    if (num_rendered <= 0) { return 0; }

    LUISA_INFO("num_rendered: {}", num_rendered);

    auto d_point_list_unsorted      = accel.point_list_unsorted.subview(0, num_rendered);
    auto d_point_list_keys_unsorted = accel.point_list_keys_unsorted.subview(0, num_rendered);
    auto d_point_list               = accel.point_list.subview(0, num_rendered);
    auto d_point_list_keys          = accel.point_list_keys.subview(0, num_rendered);

    // init point list keys to 0
    cmdlist << mp_buffer_filler->fill(device, d_point_list_unsorted, 0u);
    cmdlist << mp_buffer_filler->fill(device, d_point_list_keys_unsorted, 0ull);

    // duplicate keys
    cmdlist << (*shad_copy_with_keys)(
                   num_gaussians,
                   input.means_2d,
                   d_point_offsets,
                   output.radii,
                   input.depth_features,
                   d_point_list_keys_unsorted,
                   d_point_list_unsorted,
                   m_blocks, grids
    )
                   .dispatch(num_gaussians);

    // sort keys
    mp_device_parallel->radix_sort<ulong, uint>(
        cmdlist,
        accel.point_list_keys_unsorted,
        accel.point_list_unsorted,
        accel.point_list_keys,
        accel.point_list,
        accel.temp_storage,
        num_rendered, 64
    );
    stream << cmdlist.commit();

    auto d_ranges = accel.ranges.subview(0, grids.x * grids.y * 2);

    cmdlist << mp_buffer_filler->fill(device, d_ranges, 0u);

    // get range
    LUISA_INFO("get ranges");
    cmdlist
        << (*shad_get_ranges)(
               num_rendered,
               accel.point_list_keys,
               accel.ranges
           )
               .dispatch(num_rendered);

    LUISA_INFO("forward render");
    cmdlist
        << (*m_forward_render_shader)(
               resolution,
               input.num_gaussians,
               num_rendered,
               output.target_img,
               grids,
               input.bg_color,
               // accel structure
               accel.ranges,
               accel.point_list,
               // payload
               input.means_2d,
               input.conic,
               input.opacity_features,
               input.color_features
           )
               .dispatch(resolution);

    stream << cmdlist.commit() << synchronize();

    return num_rendered;
}

} // namespace lcgs