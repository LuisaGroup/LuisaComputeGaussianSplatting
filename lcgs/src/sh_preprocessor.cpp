/**
 * @file sh_processor.cpp
 * @brief The SH Processor Implementation
 * @author sailing-innocent
 * @date 2024-11-24
 */

#include "lcgs/sh_preprocessor.h"
#include "lcgs/util/sh.hpp"
#include "lcgs/core/sugar.h"
#include "lcgs/util/camera.h"

namespace lcgs
{

void SHProcessor::create(Device& device) noexcept
{
    compile(device);
    LUISA_INFO("SH Preprocessor created");
}

void SHProcessor::compile(Device& device) noexcept
{
    using namespace luisa;
    using namespace luisa::compute;

    mp_compute_color_from_sh = luisa::make_unique<Callable<float3(
        int, int, int,
        float3,
        Buffer<float>,
        Buffer<float>
    )>>(
        [&](
            Int idx, Int channel, Int deg,
            Float3           campos,
            BufferVar<float> means,
            BufferVar<float> shs
        ) {
            auto feat_dim     = (deg + 1) * (deg + 1);
            Int  sh_idx_start = idx * feat_dim * 3;
            // (N, feat_dim, 3)
            Float3 sh_00 = make_float3(
                shs.read(sh_idx_start + 0 * 3 + 0),
                shs.read(sh_idx_start + 0 * 3 + 1),
                shs.read(sh_idx_start + 0 * 3 + 2)
            );

            // 1
            Float3 result = sh_00;
            $if(deg > -1)
            {
                result = compute_color_from_sh_level_0(sh_00);
                $if(deg > 0)
                {
                    Float3 pos = make_float3(means.read(idx * 3 + 0), means.read(idx * 3 + 1), means.read(idx * 3 + 2));
                    Float3 dir = luisa::compute::normalize(pos - campos);
                    // 3
                    auto sh_10 = make_float3(
                        shs.read(sh_idx_start + 1 * 3 + 0),
                        shs.read(sh_idx_start + 1 * 3 + 1),
                        shs.read(sh_idx_start + 1 * 3 + 2)
                    );
                    auto sh_11 = make_float3(
                        shs.read(sh_idx_start + 2 * 3 + 0),
                        shs.read(sh_idx_start + 2 * 3 + 1),
                        shs.read(sh_idx_start + 2 * 3 + 2)
                    );
                    auto sh_12 = make_float3(
                        shs.read(sh_idx_start + 3 * 3 + 0),
                        shs.read(sh_idx_start + 3 * 3 + 1),
                        shs.read(sh_idx_start + 3 * 3 + 2)
                    );

                    result = result + compute_color_from_sh_level_1(dir, sh_10, sh_11, sh_12);

                    $if(deg > 1)
                    {
                        // 5
                        auto sh_20 = make_float3(
                            shs.read(sh_idx_start + 4 * 3 + 0),
                            shs.read(sh_idx_start + 4 * 3 + 1),
                            shs.read(sh_idx_start + 4 * 3 + 2)
                        );
                        auto sh_21 = make_float3(
                            shs.read(sh_idx_start + 5 * 3 + 0),
                            shs.read(sh_idx_start + 5 * 3 + 1),
                            shs.read(sh_idx_start + 5 * 3 + 2)
                        );
                        auto sh_22 = make_float3(
                            shs.read(sh_idx_start + 6 * 3 + 0),
                            shs.read(sh_idx_start + 6 * 3 + 1),
                            shs.read(sh_idx_start + 6 * 3 + 2)
                        );
                        auto sh_23 = make_float3(
                            shs.read(sh_idx_start + 7 * 3 + 0),
                            shs.read(sh_idx_start + 7 * 3 + 1),
                            shs.read(sh_idx_start + 7 * 3 + 2)
                        );
                        auto sh_24 = make_float3(
                            shs.read(sh_idx_start + 8 * 3 + 0),
                            shs.read(sh_idx_start + 8 * 3 + 1),
                            shs.read(sh_idx_start + 8 * 3 + 2)
                        );

                        result = result + compute_color_from_sh_level_2(dir, sh_20, sh_21, sh_22, sh_23, sh_24);

                        $if(deg > 2)
                        {
                            // 7
                            auto sh_30 = make_float3(
                                shs.read(sh_idx_start + 9 * 3 + 0),
                                shs.read(sh_idx_start + 9 * 3 + 1),
                                shs.read(sh_idx_start + 9 * 3 + 2)
                            );
                            auto sh_31 = make_float3(
                                shs.read(sh_idx_start + 10 * 3 + 0),
                                shs.read(sh_idx_start + 10 * 3 + 1),
                                shs.read(sh_idx_start + 10 * 3 + 2)
                            );
                            auto sh_32 = make_float3(
                                shs.read(sh_idx_start + 11 * 3 + 0),
                                shs.read(sh_idx_start + 11 * 3 + 1),
                                shs.read(sh_idx_start + 11 * 3 + 2)
                            );
                            auto sh_33 = make_float3(
                                shs.read(sh_idx_start + 12 * 3 + 0),
                                shs.read(sh_idx_start + 12 * 3 + 1),
                                shs.read(sh_idx_start + 12 * 3 + 2)
                            );
                            auto sh_34 = make_float3(
                                shs.read(sh_idx_start + 13 * 3 + 0),
                                shs.read(sh_idx_start + 13 * 3 + 1),
                                shs.read(sh_idx_start + 13 * 3 + 2)
                            );
                            auto sh_35 = make_float3(
                                shs.read(sh_idx_start + 14 * 3 + 0),
                                shs.read(sh_idx_start + 14 * 3 + 1),
                                shs.read(sh_idx_start + 14 * 3 + 2)
                            );
                            auto sh_36 = make_float3(
                                shs.read(sh_idx_start + 15 * 3 + 0),
                                shs.read(sh_idx_start + 15 * 3 + 1),
                                shs.read(sh_idx_start + 15 * 3 + 2)
                            );
                            result = result + compute_color_from_sh_level_3(dir, sh_30, sh_31, sh_32, sh_33, sh_34, sh_35, sh_36);
                        };
                    };
                };

                result = result + 0.5f;
            };
            // clamp
            result = clamp(result, 0.0f, 1.0f);

            return result;
        }
    );

    lazy_compile(device, shad_sh_process, [&](Int P, Int channel, Int deg, Float3 cam_pos, BufferVar<float> xyz, BufferVar<float> sh, BufferVar<float> color) {
        auto idx = dispatch_id().x;
        $if(idx >= P) { $return(); };
        auto pos    = read_float3(xyz, idx);
        auto dir    = luisa::compute::normalize(pos - cam_pos);
        auto result = (*mp_compute_color_from_sh)((Int)idx, channel, deg, cam_pos, xyz, sh);
        write_float3(color, idx, result);
    });
}

void SHProcessor::process(
    CommandList&      cmdlist,
    GPUPointsProxy    proxy,
    lcgs::Camera&     camera,
    BufferView<float> sh, BufferView<float> color,
    int level, int channel
) noexcept
{
    cmdlist
        << (*shad_sh_process)(
               proxy.N,
               channel,
               level,
               make_float3(camera.position),
               proxy.pos,
               sh,
               color
           )
               .dispatch(proxy.N);
}

} // namespace lcgs