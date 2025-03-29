#pragma once
/**
 * @file gs_projector.h
 * @brief Gaussian 3D -> Gaussian 2D Projector
 * @author sailing-innocent
 * @date 2024-11-24
 */

#include "lcgs/config.h"
#include "lcgs/module.h"
#include "lcgs/util/camera.h"

namespace lcgs
{

struct GSProjectorInputProxy {
    int                               num_gaussians;
    luisa::compute::BufferView<float> pos;
    luisa::compute::BufferView<float> scale;
    luisa::compute::BufferView<float> rotq;
    float                             scale_modifier;
};

struct GSProjectorOutputProxy {
    luisa::compute::BufferView<float> means_2d;
    luisa::compute::BufferView<float> covs_2d;
    luisa::compute::BufferView<float> depth;
};

class LCGS_API GSProjector : public GSModule
{
public:
    GSProjector()  = default;
    ~GSProjector() = default;
    void create(Device& device) noexcept;

    void forward(
        CommandList&           cmdlist,
        GSProjectorInputProxy  input,
        GSProjectorOutputProxy output,
        lcgs::Camera&          cam,
        bool                   use_focal = true
    ) noexcept;

protected:
    uint2 m_blocks = { 16u, 16u };
    void  compile(Device& device) noexcept;
    void  compile_callables(Device& device) noexcept override;
    void  compile_gs_project_shader(Device& device) noexcept;

    // callables
    UCallable<float3(float3, float, float)> mp_cam_clamp;
    // shaers
    U<Shader<1, int,        // P
             Buffer<float>, // means_3d
             Buffer<float>, // scale_buffer
             Buffer<float>, // rotq_buffer
             // params
             float, // scale_modifier
             // output
             Buffer<float>, // means_2d // 2 * P
             Buffer<float>, // depth_features // P
             Buffer<float>, // conic // 3 * P
             // PARAMS
             float, float, // tanfov x, tanfov y
             float4x4,     // view_matrix
             float4x4      // proj_matrix
             >>
        shad_project_gs;

    U<Shader<1, int,        // P
             Buffer<float>, // means_3d
             Buffer<float>, // scale_buffer
             Buffer<float>, // rotq_buffer
             // params
             float, // scale_modifier
             // output
             Buffer<float>, // means_2d // 2 * P
             Buffer<float>, // depth_features // P
             Buffer<float>, // conic // 3 * P
             // PARAMS
             float, float, // tanfov x, tanfov y
             float, float, // focalx, focaly
             float4x4,     // view_matrix
             float4x4      // proj_matrix
             >>
        shad_project_gs_focal;
};

} // namespace lcgs