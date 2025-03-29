/**
 * @file gs_projecto_forward_shader.cpp
 * @brief The GS Projector Forward Shader
 * @author sailing-innocent
 * @date 2024-11-25
 */

#include "lcgs/gs_projector.h"
#include "lcgs/core/sugar.h"
#include "lcgs/util/gaussian.hpp"

namespace lcgs
{

using namespace luisa;
using namespace luisa::compute;

void GSProjector::compile_gs_project_shader(Device& device) noexcept
{
    lazy_compile(
        device, shad_project_gs,
        [&](
            Int P,
            // input
            BufferVar<float> means_3d,
            BufferVar<float> scale_buffer,
            BufferVar<float> rotq_buffer,
            // params
            Float scale_modifier,
            // output
            BufferVar<float> means_2d,
            BufferVar<float> depth_features,
            BufferVar<float> covs_2d,
            // camera
            Float    tanfovx,
            Float    tanfovy,
            Float4x4 view_matrix,
            Float4x4 proj_matrix
        ) {
            set_block_size(m_blocks.x * m_blocks.y);
            auto idx = dispatch_id().x;
            $if(idx >= UInt(P)) { $return(); };

            // -----------------------------
            // project to screen space
            // -----------------------------
            auto   mean_3d    = read_float3(means_3d, idx);
            Float4 p_hom      = make_float4(mean_3d, 1.0f);
            Float4 p_view_hom = view_matrix * p_hom;
            Float3 p_view     = p_view_hom.xyz();
            Float4 p_proj_hom = proj_matrix * p_view_hom;
            Float  p_w        = 1.0f / (p_proj_hom.w + 1e-6f);
            Float3 p_proj     = p_proj_hom.xyz() * p_w; // p_proj in NDC
            // Float2 xy_ndc = make_float2(p_proj.x / tanfovx, p_proj.y / tanfovy);
            Float2 xy_ndc = p_proj.xy();

            $if(p_view.z < 0.2f) { $return(); };
            depth_features.write(idx, p_view.z);

            write_float2(means_2d, idx, xy_ndc);
            // calculate 3d covariance
            Float3 s     = read_float3(scale_buffer, idx);
            Float4 rotq  = read_float4(rotq_buffer, idx); // r, x, y, z
            auto   scale = scale_modifier * s;
            // auto scale = make_float3(0.001f);
            auto qvec = rotq.yzwx(); // rxyz -> xyzw
            // auto qvec = make_float4(0.0f, 0.0f, 0.0f, 1.0f); // rxyz -> xyzw

            Float3x3 cov_3d = calc_cov<Float3, Float4, Float3x3>(scale, qvec);
            Float3   t      = (*mp_cam_clamp)(p_view_hom.xyz(), tanfovx, tanfovy);
            Float3x3 cov    = ewasplat_cov<Float3x3, Float4x4, Float3>(cov_3d, t, view_matrix);

            Float3 cov_2d = make_float3(cov[0][0], cov[0][1], cov[1][1]);
            cov_2d.x      = cov_2d.x * 1.0f / (tanfovx * tanfovx);
            cov_2d.y      = cov_2d.y * 1.0f / (tanfovx * tanfovy);
            cov_2d.z      = cov_2d.z * 1.0f / (tanfovy * tanfovy);

            write_float3(covs_2d, idx, cov_2d);
        }
    );

    lazy_compile(
        device, shad_project_gs_focal,
        [&](
            Int P,
            // input
            BufferVar<float> means_3d,
            BufferVar<float> scale_buffer,
            BufferVar<float> rotq_buffer,
            // params
            Float scale_modifier,
            // output
            BufferVar<float> means_2d,
            BufferVar<float> depth_features,
            BufferVar<float> covs_2d,
            // camera
            Float    tanfovx,
            Float    tanfovy,
            Float    focalx,
            Float    focaly,
            Float4x4 view_matrix,
            Float4x4 proj_matrix
        ) {
            set_block_size(m_blocks.x * m_blocks.y);
            auto idx = dispatch_id().x;
            $if(idx >= UInt(P)) { $return(); };

            // -----------------------------
            // project to screen space
            // -----------------------------
            auto   mean_3d    = read_float3(means_3d, idx);
            Float4 p_hom      = make_float4(mean_3d, 1.0f);
            Float4 p_view_hom = view_matrix * p_hom;
            Float3 p_view     = p_view_hom.xyz();
            Float4 p_proj_hom = proj_matrix * p_view_hom;
            Float  p_w        = 1.0f / (p_proj_hom.w + 1e-6f);
            Float3 p_proj     = p_proj_hom.xyz() * p_w; // p_proj in NDC
            // Float2 xy_ndc = make_float2(p_proj.x / tanfovx, p_proj.y / tanfovy);
            Float2 xy_ndc = p_proj.xy();

            $if(p_view.z < 0.2f) { $return(); };
            depth_features.write(idx, p_view.z);

            write_float2(means_2d, idx, xy_ndc);
            // calculate 3d covariance
            Float3 s     = read_float3(scale_buffer, idx);
            Float4 rotq  = read_float4(rotq_buffer, idx); // r, x, y, z
            auto   scale = scale_modifier * s;
            // auto scale = make_float3(0.001f);
            auto qvec = rotq.yzwx(); // rxyz -> xyzw
            // auto qvec = make_float4(0.0f, 0.0f, 0.0f, 1.0f); // rxyz -> xyzw

            Float3x3 cov_3d = calc_cov<Float3, Float4, Float3x3>(scale, qvec);
            Float3   t      = (*mp_cam_clamp)(p_view_hom.xyz(), tanfovx, tanfovy);
            Float3x3 cov    = ewasplat_cov_focal<Float3x3, Float4x4, Float3, Float>(cov_3d, t, view_matrix, focalx, focaly);
            Float3   cov_2d = make_float3(cov[0][0], cov[0][1], cov[1][1]);
            write_float3(covs_2d, idx, cov_2d);
        }
    );
}

void GSProjector::compile_callables(Device& device) noexcept
{
    GSModule::compile_callables(device);

    mp_cam_clamp = luisa::make_unique<Callable<float3(float3, float, float)>>(
        [](Float3 p, Float tanfovx, Float tanfovy) {
            auto t    = p;
            auto limx = 1.3f * tanfovx;
            auto limy = 1.3f * tanfovy;
            auto txtz = t.x / t.z;
            auto tytz = t.y / t.z;
            // use luisa::compute::clamp for CallOp::CLAMP
            t.x = luisa::compute::clamp(txtz, -limx, limx) * t.z;
            t.y = luisa::compute::clamp(tytz, -limy, limy) * t.z;
            return t;
        }
    );
}

} // namespace lcgs