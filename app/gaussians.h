#pragma once
/**
 * @file gaussians.h
 * @brief The Gaussians IO
 * @author sailing-innocent
 * @date 2025-03-29
 */

#include <luisa/luisa-compute.h>

namespace lcgs
{

// Host Gaussians Data
struct GaussiansData {
    int                  num_gaussians = 0;
    int                  sh_deg        = 3;
    luisa::vector<float> pos;
    luisa::vector<float> feature;
    luisa::vector<float> opacity;
    luisa::vector<float> scale;
    luisa::vector<float> rotq;

    static float scaling_activation(float x);
    static void  rotation_activation(float& r, float& x, float& y, float& z);
    static float opacity_activation(float x);

    void resize(int N);

    static GaussiansData create_cube(
        float origin_x = 0.0f, float origin_y = 0.0f, float origin_z = 0.0f,
        float side_x = 1.0f, float side_y = 1.0f, float side_z = 1.0f,
        int Nx = 50
    );
};

bool read_gs_ply(GaussiansData& gs, std::filesystem::path fpath);

} // namespace lcgs