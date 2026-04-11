/**
 * @file gs_tile_diff_splatter
 * @brief The Gaussian Splatting Tile Splatter
 * @author sailing-innocent
 * @date 2024-11-27
 */

#include "lcgs/gs_tile_splatter.h"
#include "lcgs/util/misc.hpp"
#include <lcpp/common/utils.h>

namespace lcgs
{

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;

// Helper to convert bytes to uint count (round up)
inline size_t bytes_to_uint_count(size_t byte_size)
{
    return (byte_size + sizeof(uint) - 1) / sizeof(uint);
}

void GSTileSplatter::create(Device& device) noexcept
{
    compile(device);
    LUISA_INFO("Tile Splatter created");
}

void GSTileSplatter::ensure_scan_temp_buffer(Device& device, size_t num_items)
{
    using ScannerT = luisa::parallel_primitive::DeviceScan<>;
    size_t temp_bytes = ScannerT::GetTempStorageBytes<uint>(num_items);
    size_t required_uint_count = bytes_to_uint_count(temp_bytes);
    
    if (m_scan_temp_buffer == nullptr || m_scan_temp_buffer_size < required_uint_count)
    {
        // Grow the buffer: use max of required size and double current size to avoid frequent reallocations
        size_t new_size = m_scan_temp_buffer_size == 0 ? required_uint_count : std::max(required_uint_count, m_scan_temp_buffer_size * 2);
        m_scan_temp_buffer = luisa::make_unique<Buffer<uint>>(device.create_buffer<uint>(new_size));
        m_scan_temp_buffer_size = new_size;
        LUISA_INFO("GSTileSplatter: scan temp buffer resized to {} uints ({} bytes)", new_size, new_size * sizeof(uint));
    }
}

void GSTileSplatter::ensure_radix_sort_temp_buffer(Device& device, size_t num_items)
{
    using RadixSorterT = luisa::parallel_primitive::DeviceRadixSort<>;
    size_t temp_bytes = RadixSorterT::GetSortPairsTempStorageBytes<ulong, uint>(static_cast<uint>(num_items));
    size_t required_uint_count = bytes_to_uint_count(temp_bytes);
    
    if (m_radix_sort_temp_buffer == nullptr || m_radix_sort_temp_buffer_size < required_uint_count)
    {
        // Grow the buffer: use max of required size and double current size to avoid frequent reallocations
        size_t new_size = m_radix_sort_temp_buffer_size == 0 ? required_uint_count : std::max(required_uint_count, m_radix_sort_temp_buffer_size * 2);
        m_radix_sort_temp_buffer = luisa::make_unique<Buffer<uint>>(device.create_buffer<uint>(new_size));
        m_radix_sort_temp_buffer_size = new_size;
        LUISA_INFO("GSTileSplatter: radix sort temp buffer resized to {} uints ({} bytes)", new_size, new_size * sizeof(uint));
    }
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
    stream << cmdlist.commit() << synchronize();
    
    // Ensure scan temp buffer is large enough and perform inclusive sum
    ensure_scan_temp_buffer(device, num_gaussians);
    mp_device_scan->InclusiveSum(cmdlist, m_scan_temp_buffer->view(), d_tiles_touched, d_point_offsets, num_gaussians);
    
    cmdlist << accel.point_offsets.subview(input.num_gaussians - 1, 1).copy_to(&num_rendered);
    stream << cmdlist.commit() << synchronize();

    if (num_rendered <= 0) { return 0; }
    LUISA_INFO("num_rendered: {}", num_rendered);

    auto d_point_list_unsorted      = accel.point_list_unsorted.subview(0, num_rendered);
    auto d_point_list_keys_unsorted = accel.point_list_keys_unsorted.subview(0, num_rendered);
    auto d_point_list               = accel.point_list.subview(0, num_rendered);
    auto d_point_list_keys          = accel.point_list_keys.subview(0, num_rendered);

    cmdlist << mp_buffer_filler->fill(device, d_point_list_unsorted, 0u);
    cmdlist << mp_buffer_filler->fill(device, d_point_list_keys_unsorted, 0ull);

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
    stream << cmdlist.commit() << synchronize();
    
    // Ensure radix sort temp buffer is large enough and perform sort
    ensure_radix_sort_temp_buffer(device, num_rendered);
    mp_device_radix_sort->SortPairs<ulong, uint>(
        cmdlist,
        m_radix_sort_temp_buffer->view(),
        d_point_list_keys_unsorted,
        d_point_list_keys,
        d_point_list_unsorted,
        d_point_list,
        num_rendered
    );
    stream << cmdlist.commit() << synchronize();
    auto d_ranges = accel.ranges.subview(0, grids.x * grids.y * 2);
    stream << cmdlist.commit() << synchronize();
    cmdlist << mp_buffer_filler->fill(device, d_ranges, 0u);

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
               accel.ranges,
               accel.point_list,
               input.means_2d,
               input.conic,
               input.opacity_features,
               input.color_features
           )
               .dispatch(resolution);

    // stream << cmdlist.commit() << synchronize();
    stream << cmdlist.commit();

    return num_rendered;
}

} // namespace lcgs