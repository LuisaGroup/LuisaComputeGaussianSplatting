/**
 * @file gs_projector.cpp
 * @brief The Gaussian Projector
 * @author sailing-innocent
 * @date 2024-11-24
 */

#include "lcgs/gs_projector.h"
#include "luisa/dsl/builtin.h"

namespace lcgs
{

void GSProjector::create(Device& device) noexcept
{
    compile(device);
    LUISA_INFO("GS Projector created");
}

void GSProjector::compile(Device& device) noexcept
{
    compile_callables(device);
    compile_gs_project_shader(device);
}

void GSProjector::forward(
    CommandList&           cmdlist,
    GSProjectorInputProxy  input,
    GSProjectorOutputProxy output,
    lcgs::Camera&          cam,
    bool                   use_focal
) noexcept
{
    auto fovy     = cam.fov / 180.0f * 3.1415926536f;
    auto tanfovy  = tan(fovy * 0.5f);
    auto tanfovx  = tanfovy * cam.aspect_ratio;
    auto view_mat = world_to_local_matrix(cam);
    auto proj_mat = projection_matrix(tanfovx, tanfovy);
    LUISA_INFO("view mat {}", view_mat);
    LUISA_INFO("proj mat {}", proj_mat);
    auto focalx = cam.width / (2.0f * tanfovx);
    auto focaly = cam.height / (2.0f * tanfovy);

    // world -> screen -> ndc
    if (use_focal)
    {
        cmdlist
            << (*shad_project_gs_focal)(
                   input.num_gaussians,
                   // input
                   input.pos,
                   input.scale,
                   input.rotq,
                   // params
                   input.scale_modifier,
                   // output
                   output.means_2d,
                   output.depth,
                   output.covs_2d,
                   // camera
                   tanfovx,
                   tanfovy,
                   focalx,
                   focaly,
                   view_mat,
                   proj_mat
               )
                   .dispatch(input.num_gaussians);
    }
    else
    {
        cmdlist
            << (*shad_project_gs)(
                   input.num_gaussians,
                   // input
                   input.pos,
                   input.scale,
                   input.rotq,
                   // params
                   input.scale_modifier,
                   // output
                   output.means_2d,
                   output.depth,
                   output.covs_2d,
                   // camera
                   tanfovx,
                   tanfovy,
                   view_mat,
                   proj_mat
               )
                   .dispatch(input.num_gaussians);
    }
}

} // namespace lcgs