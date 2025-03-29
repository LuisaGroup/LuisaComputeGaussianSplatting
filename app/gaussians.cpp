/**
 * @file gaussians.cpp
 * @brief The Implementation of gaussians IO
 * @author sailing-innocent
 * @date 2025-03-29
 */

#include "gaussians.h"
#include <iostream>
#include "happly.h"

namespace lcgs
{

float GaussiansData::opacity_activation(float x)
{
    // sigmoid
    return 1.0f / (1.0f + exp(-x));
}

float GaussiansData::scaling_activation(float x)
{
    // exp
    return exp(x);
}

void GaussiansData::rotation_activation(float& r, float& x, float& y, float& z)
{
    // normalize
    float norm = sqrt(x * x + y * y + z * z + r * r);
    r /= norm;
    x /= norm;
    y /= norm;
    z /= norm;
}

void GaussiansData::resize(int N)
{
    num_gaussians = N;
    pos.resize(N * 3);
    feature.resize(N * (sh_deg + 1) * (sh_deg + 1) * 3);
    opacity.resize(N);
    scale.resize(N * 3);
    rotq.resize(N * 4);
}

GaussiansData GaussiansData::create_cube(
    float origin_x, float origin_y, float origin_z,
    float side_x, float side_y, float side_z,
    int Nx
)
{
    GaussiansData data;
    data.resize(Nx * Nx * Nx);

    for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            for (int k = 0; k < Nx; k++)
            {
                int   idx             = i * Nx * Nx + j * Nx + k;
                float u               = float(i) / Nx;
                float v               = float(j) / Nx;
                float w               = float(k) / Nx;
                data.pos[idx * 3 + 0] = origin_x + side_x * u;
                data.pos[idx * 3 + 1] = origin_y + side_y * v;
                data.pos[idx * 3 + 2] = origin_z + side_z * w;
            }
        }
    }
    return data;
}

bool read_gs_ply(GaussiansData& gs, std::filesystem::path fpath)
{
    int             sh_deg = gs.sh_deg;
    happly::PLYData plyIn(fpath.string());
    if (!plyIn.hasElement("vertex"))
    {
        std::cerr << "No vertex element in the ply file" << std::endl;
        return false;
    }
    // x,y,z
    // f_dc_0,1,2
    // f_rest 0,1,2... 45
    // opacity
    // scale_0,1,2
    // rotation_0,1,2,3
    int N = plyIn.getElement("vertex").count;
    gs.resize(N);
    std::cout << "N: " << N << std::endl;
    std::vector<float> x = plyIn.getElement("vertex").getProperty<float>("x");
    std::vector<float> y = plyIn.getElement("vertex").getProperty<float>("y");
    std::vector<float> z = plyIn.getElement("vertex").getProperty<float>("z");

    std::cout << gs.pos.size() << std::endl;

    for (auto i = 0; i < N; i++)
    {
        gs.pos[3 * i + 0] = x[i];
        gs.pos[3 * i + 1] = y[i];
        gs.pos[3 * i + 2] = z[i];
    }

    int stride = (sh_deg + 1) * (sh_deg + 1);
    int channel;
    int offset;

    for (auto i = 0; i < 3; i++)
    {
        std::string dc_feat_name = "f_dc_" + std::to_string(i);
        std::cout << "dc_feat_name: " << dc_feat_name << std::endl;
        channel                    = i;
        offset                     = 0;
        std::vector<float> dc_feat = plyIn.getElement("vertex").getProperty<float>(dc_feat_name);

        for (auto j = 0; j < N; j++)
        {
            gs.feature[offset * 3 + channel + j * stride * 3] = dc_feat[j];
        }
    }

    for (auto i = 0; i < (stride - 1) * 3; i++)
    {
        std::string rest_feat_name = "f_rest_" + std::to_string(i);
        channel                    = i / (stride - 1);
        offset                     = i % (stride - 1) + 1;
        std::cout << "rest_feat_name: " << rest_feat_name << std::endl;
        std::vector<float> rest_feat = plyIn.getElement("vertex").getProperty<float>(rest_feat_name);
        for (auto j = 0; j < N; j++)
        {
            gs.feature[offset * 3 + channel + j * stride * 3] = rest_feat[j];
        }
    }

    std::vector<float> _opacity = plyIn.getElement("vertex").getProperty<float>("opacity");
    for (auto i = 0; i < N; i++)
    {
        gs.opacity[i] = gs.opacity_activation(_opacity[i]);
    }

    for (auto i = 0; i < 3; i++)
    {
        std::string scale_name = "scale_" + std::to_string(i);
        std::cout << "scale_name: " << scale_name << std::endl;
        std::vector<float> scale_feat = plyIn.getElement("vertex").getProperty<float>(scale_name);
        for (auto j = 0; j < N; j++)
        {
            gs.scale[3 * j + i] = gs.scaling_activation(scale_feat[j]);
        }
    }

    for (auto i = 0; i < 4; i++)
    {
        std::string rotation_name = "rot_" + std::to_string(i);
        std::cout << "rotation_name: " << rotation_name << std::endl;
        std::vector<float> rotation_feat = plyIn.getElement("vertex").getProperty<float>(rotation_name);
        for (auto j = 0; j < N; j++)
        {
            gs.rotq[4 * j + i] = rotation_feat[j];
        }
    }
    // rotation activation
    for (auto i = 0; i < N; i++)
    {
        gs.rotation_activation(gs.rotq[4 * i], gs.rotq[4 * i + 1], gs.rotq[4 * i + 2], gs.rotq[4 * i + 3]);
    }

    return true;
}

} // namespace lcgs