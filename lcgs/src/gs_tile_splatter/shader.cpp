/**
 * @file gs_tile_splatter/shader.cpp
 * @brief The Gaussian Tile Splatter Forward Shader
 * @author sailing-innocent
 * @date 2024-11-25
 */

#include "lcgs/gs_tile_splatter.h"
#include "lcgs/core/sugar.h"

namespace lcgs
{

void GSTileSplatter::compile(Device& device) noexcept
{
    GSModule::compile_callables(device);
    compile_impl_shader(device);
    compile_forward_shader(device);
}

void GSTileSplatter::compile_impl_shader(Device& device) noexcept
{
    using namespace luisa;
    using namespace luisa::compute;

    lazy_compile(
        device, shad_copy_with_keys,
        [&](
            Int              P,
            BufferVar<float> points_xy,       // P x 2
            BufferVar<uint>  offsets,         // P x 1
            BufferVar<int>   radii,           // P x 1
            BufferVar<float> depth_features,  // P x 1
            BufferVar<ulong> keys_unsorted,   // L x 1
            BufferVar<uint>  values_unsorted, // L x 1
            UInt2            blocks,
            UInt2            grids
        ) {
            auto idx = dispatch_id().x;
            $if(idx >= UInt(P)) { $return(); };
            auto radius = radii.read(idx);
            $if(radius <= 0) { $return(); };
            // generate key/value
            UInt off = 0u;
            $if(idx >= 1u)
            {
                off = offsets.read(idx - 1);
            };

            Float2 point_xy = read_float2(points_xy, idx);
            UInt2  rect_min, rect_max;

            (*mp_get_rect)(point_xy, radius, rect_min, rect_max, blocks, grids);

            $for(j, rect_min.y, rect_max.y)
            {
                $for(i, rect_min.x, rect_max.x)
                {
                    ULong key = ULong(i + j * grids.x);
                    key <<= 32ull;
                    auto depth = depth_features.read(idx);
                    key |= ULong(depth.as<UInt>()) & 0x00000000FFFFFFFFull;
                    keys_unsorted.write(off, key);
                    values_unsorted.write(off, idx);
                    off = off + 1u;
                };
            };
        }
    );

    lazy_compile(
        device, shad_get_ranges,
        [&](Int L, BufferVar<ulong> point_list_keys, BufferVar<uint> ranges) {
            set_block_size(256);
            UInt idx = dispatch_id().x;
            $if(idx >= L) { $return(); };
            ULong key       = point_list_keys.read(idx);
            UInt  curr_tile = UInt(key >> 32ull);
            UInt  prev_tile = 0u;

            $if(idx == 0u)
            {
                ranges.write(2 * curr_tile + 0u, 0u);
            }
            $else
            {
                ULong prev_key  = point_list_keys.read(idx - 1);
                UInt  prev_tile = UInt(prev_key >> 32ull);
                $if(curr_tile != prev_tile)
                {
                    ranges.write(2 * prev_tile + 1u, idx);
                    ranges.write(2 * curr_tile + 0u, idx);
                };
            };
            $if(idx == L - 1)
            {
                ranges.write(2 * curr_tile + 1u, UInt(L));
            };
        }
    );

    lazy_compile(
        device, shad_allocate_tiles,
        [&](
            Int              P,
            UInt2            resolution,
            UInt2            grids,
            BufferVar<float> depth_features,
            BufferVar<float> means_2d,
            BufferVar<float> covs_2d,
            BufferVar<uint>  tiles_touched,
            BufferVar<int>   radii
        ) {
            set_block_size(m_blocks.x * m_blocks.y);
            auto idx = dispatch_id().x;
            $if(idx >= UInt(P)) { $return(); };
            // -----------------------------
            radii.write(idx, 0);
            tiles_touched.write(idx, 0u);

            // near culling
            auto depth = depth_features.read(idx);
            $if(depth < 0.2f) { $return(); };

            // -----------------------------
            Float2 point_image_ndc = read_float2(means_2d, idx);
            Float3 cov_2d          = read_float3(covs_2d, idx);
            // ndc -> image space
            // cov_2d = make_float3(5.0f, 0.0f, 5.0f);
            cov_2d.x = cov_2d.x * resolution.x * resolution.x * 0.25f;
            cov_2d.y = cov_2d.y * resolution.x * resolution.y * 0.25f;
            cov_2d.z = cov_2d.z * resolution.y * resolution.x * 0.25f;
            // low-pass filter
            cov_2d.x += 0.3f;
            cov_2d.z += 0.3f;

            Float  det     = cov_2d.x * cov_2d.z - cov_2d.y * cov_2d.y;
            Float  inv_det = 1.0f / (det + 1e-6f);
            Float3 conic   = inv_det * make_float3(cov_2d.z, -cov_2d.y, cov_2d.x);
            // Float3 conic = make_float3(100, 0, 100);
            // inv: [0][0] [0][1] transpose [1][1] inverse

            Float mid       = 0.5f * (cov_2d.x + cov_2d.z);
            Float lambda1   = mid + sqrt(max(0.1f, mid * mid - det));
            Float lambda2   = mid - sqrt(max(0.1f, mid * mid - det));
            Int   my_radius = ceil(3.0f * sqrt(max(lambda1, lambda2)));

            UInt2 rect_min, rect_max;

            auto point_image = make_float2((*mp_ndc2pix)(point_image_ndc.x, resolution.x), (*mp_ndc2pix)(point_image_ndc.y, resolution.y));
            (*mp_get_rect)(point_image, my_radius, rect_min, rect_max, m_blocks, grids);
            auto N_tiles_touched = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);

            // write out
            radii.write(idx, my_radius);
            tiles_touched.write(idx, N_tiles_touched);
            // here we write the inverse of cov2d back to cov2d
            write_float3(covs_2d, idx, conic);
            write_float2(means_2d, idx, point_image);
        }
    );
    compile_forward_shader(device);
}

void GSTileSplatter::compile_forward_shader(Device& device) noexcept
{
    using namespace luisa;
    using namespace luisa::compute;
    lazy_compile(
        device,
        m_forward_render_shader,
        [&](
            UInt2 resolution,
            Int P, Int L, // for check and debug
            // output
            BufferVar<float> target_img,
            // params
            UInt2  grids,
            Float3 bg_color,
            // input buffers
            BufferVar<uint>  ranges,           // W x H x 2
            BufferVar<uint>  point_list,       // L
            BufferVar<float> means_2d,         // 2 * P
            BufferVar<float> conic,            // 3 * P
            BufferVar<float> opacity_features, // P
            BufferVar<float> color_features    // 3 * P
        ) {
            set_block_size(m_blocks);
            auto xy         = dispatch_id().xy();
            auto w          = resolution.x;
            auto h          = resolution.y;
            auto thread_idx = thread_id().x + thread_id().y * block_size().x;
            Bool inside     = Bool(xy.x < resolution.x) & Bool(xy.y < resolution.y);
            Bool done       = !inside;
            auto pix_id     = xy.x + resolution.x * xy.y;
            auto pix_f      = Float2(
                static_cast<Float>(xy.x),
                static_cast<Float>(xy.y)
            );

            auto tile_xy     = block_id();
            UInt tile_id     = tile_xy.x + tile_xy.y * grids.x;
            UInt range_start = ranges.read(2 * tile_id + 0u);
            UInt range_end   = ranges.read(2 * tile_id + 1u);

            const size_t shared_mem_size = m_blocks.x * m_blocks.y;
            const UInt   round_step      = (UInt)shared_mem_size;
            const UInt   rounds          = ((range_end - range_start + round_step - 1u) / round_step);
            UInt         todo            = range_end - range_start;

            Shared<uint>*   collected_ids           = new Shared<uint>(shared_mem_size);
            Shared<float2>* collected_means         = new Shared<float2>(shared_mem_size);
            Shared<float4>* collected_conic_opacity = new Shared<float4>(shared_mem_size);

            Float  T                = 1.0f;
            Float3 C                = make_float3(0.0f, 0.0f, 0.0f);
            UInt   contributor      = 0u;
            UInt   last_contributor = 0u;
            Float3 white            = make_float3(1.0f, 1.0f, 1.0f);
            Float3 black            = make_float3(0.0f, 0.0f, 0.0f);
            Bool   is_black         = (tile_xy.x + tile_xy.y) % 2 == 0;

            $for(i, rounds)
            {
                sync_block();
                // collect num_done

                Int progress = i * round_step + thread_idx;
                $if(progress + range_start < range_end)
                {
                    UInt coll_id = point_list.read(progress + range_start);
                    collected_ids->write(thread_idx, coll_id);
                    Float2 means = make_float2(
                        means_2d.read(2 * coll_id + 0),
                        means_2d.read(2 * coll_id + 1)
                    );
                    collected_means->write(thread_idx, means);
                    Float  opacity       = opacity_features.read(coll_id);
                    Float4 conic_opacity = make_float4(
                        conic.read(3 * coll_id + 0),
                        conic.read(3 * coll_id + 1),
                        conic.read(3 * coll_id + 2),
                        opacity
                    );
                    collected_conic_opacity->write(thread_idx, conic_opacity);
                };
                sync_block();

                $for(j, min(round_step, todo))
                {
                    $if(done) { $break; }; // inside or filled
                    contributor  = contributor + 1u;
                    Float2 mean  = collected_means->read(j);
                    Float2 d     = mean - pix_f;
                    Float4 con_o = collected_conic_opacity->read(j);

                    Float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
                    $if(power > 0.0f) { $continue; };
                    Float alpha = min(0.99f, con_o.w * exp(power));
                    $if(alpha < 1.0f / 255.0f) { $continue; };
                    Float test_T = T * (1.0f - alpha);
                    $if(test_T < 0.0001f)
                    {
                        done = true;
                        $continue;
                    };

                    auto   id   = collected_ids->read(j);
                    Float3 feat = make_float3(
                        color_features->read(3 * id + 0),
                        color_features->read(3 * id + 1),
                        color_features->read(3 * id + 2)
                    );
                    C                = C + T * alpha * feat;
                    T                = test_T;
                    last_contributor = contributor;
                };

                todo = todo - round_step;
            };

            $if(inside)
            {
                auto color = bg_color * T + C;
                $for(i, 0, 3)
                {
                    target_img.write(pix_id + i * h * w, color[i]);
                };
            };
        }
    );
}

} // namespace lcgs