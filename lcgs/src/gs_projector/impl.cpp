/**
 * @file gs_projector.cpp
 * @brief The Gaussian Projector
 * @author sailing-innocent
 * @date 2024-11-24
 */

#include "lcgs/gs_projector.h"

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
    lcgs::Camera&            cam
) noexcept
{
    // auto tanfovy = tan(cam.fov * 0.5f);
    // auto tanfovx = tanfovy * cam.aspect_ratio;

    // auto view_mat = cam.world_to_local_matrix();
    // auto proj_mat = cam.projection_matrix();
    // world -> screen -> ndc
    // cmdlist << (*shad_project_gs)(
    //                input.num_gaussians,
    //                // input
    //                input.pos,
    //                input.scale,
    //                input.rotq,
    //                // params
    //                input.scale_modifier,
    //                // output
    //                output.means_2d,
    //                output.depth,
    //                output.covs_2d,
    //                // camera
    //                tanfovx,
    //                tanfovy,
    //                view_mat,
    //                proj_mat
    // )
    //                .dispatch(input.num_gaussians);
}

} // namespace lcgs