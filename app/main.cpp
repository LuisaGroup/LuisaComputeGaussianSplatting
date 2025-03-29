/**
 * @file main.cpp
 * @brief The LuisaCompute Gaussian Splatting App
 * @author sailing-innocent
 * @date 2025-03-29
 */

#include <luisa/luisa-compute.h>
#include <luisa/gui/window.h>
#include <luisa/dsl/sugar.h>

#include "command_parser.hpp"
#include "gaussians.h"
#include "lcgs/gs_projector.h"
#include "lcgs/gs_tile_splatter.h"
#include "lcgs/sh_preprocessor.h"
#include "lcgs/util/buffer_filler.h"
#include "lcgs/util/camera.h"
#include "lcgs/util/device_parallel.h"
#include "luisa/runtime/rhi/stream_tag.h"
#include <stb/stb_image_write.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char** argv)
{
    luisa::log_level_info();
    luisa::uint2 resolution = luisa::make_uint2(1600, 1063);

    constexpr auto default_ply_path = "D:/ws/data/assets/samples/gsplat.ply";
    auto           ply_path         = std::filesystem::path{ default_ply_path };
    std::string    backend          = "dx";
    Context        context{ argv[0] };
    // validate command line args
    {
        vstd::HashMap<vstd::string, vstd::function<void(vstd::string_view)>> cmds;
        cmds.emplace("ply", [&](vstd::string_view str) {
            luisa::filesystem::path path{ str };
            if (path.is_relative())
            {
                ply_path = luisa::filesystem::path{ argv[0] }.parent_path() / path;
            }
            else
            {
                ply_path = std::move(path);
            }
        });
        cmds.emplace("backend", [&](vstd::string_view str) {
            backend = str;
        });
        // parse command
        parse_command(cmds, argc, argv, {});
    }
    LUISA_INFO("Rendering {} with backend {}", ply_path.string(), backend);
    Device  device   = context.create_device(backend.c_str());
    Device* p_device = &device;

    lcgs::GaussiansData data;
    lcgs::read_gs_ply(data, ply_path);

    int P = data.num_gaussians;
    LUISA_INFO("num_gaussians: {}", P);

    lcgs::GSProjector projector;
    projector.create(device);
    lcgs::BufferFiller   bf;
    lcgs::DeviceParallel dp;
    dp.create(device);

    auto d_pos   = p_device->create_buffer<float>(P * 3);
    auto d_scale = p_device->create_buffer<float>(P * 3);
    auto d_rotq  = p_device->create_buffer<float>(P * 4);
    // payload
    auto d_sh      = p_device->create_buffer<float>(P * 16 * 3);
    auto d_color   = p_device->create_buffer<float>(P * 3);
    auto d_opacity = p_device->create_buffer<float>(P);

    // luisa::float3 pos = { 0.0f, -3.0f, 3.0f };
    luisa::float3 pos      = { 3.0f, -3.0f, 3.0f };
    luisa::float3 target   = { 0.0f, 0.0f, 0.0f };
    luisa::float3 world_up = { 0.0f, 0.0f, 1.0f };

    auto cam         = lcgs::get_lookat_cam(pos, target, world_up);
    cam.aspect_ratio = (float)resolution.x / (float)resolution.y;

    luisa::compute::CommandList cmd_list;
    lcgs::SHProcessor           sh_processor;
    sh_processor.create(*p_device);

    // upload host gaussian data onto device
    cmd_list << d_pos.view(0, P * 3).copy_from(data.pos.data())
             << d_scale.view(0, P * 3).copy_from(data.scale.data())
             << d_rotq.view(0, P * 4).copy_from(data.rotq.data())
             //  << d_color.copy_from(data.color.data())
             //  << d_color.copy_from(h_color.data())
             << d_sh.view(0, P * 3 * 16).copy_from(data.feature.data())
             << d_opacity.view(0, P * 1).copy_from(data.opacity.data());

    auto  stream   = device.create_stream(StreamTag::GRAPHICS);
    auto* p_stream = &stream;
    stream << cmd_list.commit() << synchronize();

    luisa::Clock clk;
    clk.tic();

    sh_processor.process(cmd_list, { P, 3, d_pos }, cam, d_sh, d_color, 3, 3);

    luisa::vector<float> h_means_2d(P * 2);
    luisa::vector<float> h_depth_features(P);
    luisa::vector<float> h_covs_2d(P * 3);
    auto                 d_means_2d       = p_device->create_buffer<float>(P * 2);
    auto                 d_depth_features = p_device->create_buffer<float>(P);
    auto                 d_covs_2d        = p_device->create_buffer<float>(P * 3);

    projector.forward(cmd_list, { P, d_pos, d_scale, d_rotq, 1.0f }, { d_means_2d, d_covs_2d, d_depth_features }, cam);
    (*p_stream) << cmd_list.commit();

    // debug BEGIN
    // stream << d_means_2d.copy_to(h_means_2d.data())
    //        << d_depth_features.copy_to(h_depth_features.data())
    //        << d_covs_2d.copy_to(h_covs_2d.data())
    //        << synchronize();
    // // sample first 30 elements
    // for (int i = 0; i < 30; i++)
    // {
    //     LUISA_INFO("means_2d: {0} {1}", h_means_2d[i * 2], h_means_2d[i * 2 + 1]);
    // }
    // for (int i = 100; i < 130; i++)
    // {
    //     LUISA_INFO("depth_features: {0}", h_depth_features[i]);
    // }
    // for (int i = 0; i < 30; i++)
    // {
    //     LUISA_INFO("covs_2d: {0} {1} {2}", h_covs_2d[i * 3], h_covs_2d[i * 3 + 1], h_covs_2d[i * 3 + 2]);
    // }
    // debug END

    lcgs::GSTileSplatter tile_splatter;
    tile_splatter.create(*p_device);
    tile_splatter.set_buffer_filler(&bf);
    tile_splatter.set_device_parallel(&dp);

    // acceleration structure
    auto d_tiles_touched = p_device->create_buffer<uint>(P);
    auto d_points_offset = p_device->create_buffer<uint>(P);
    // for temp storage
    size_t temp_space_size;
    dp.scan_inclusive_sum<uint>(
        temp_space_size,
        d_tiles_touched,
        d_points_offset, 0, P
    );
    LUISA_INFO("temp_space_size: {}", temp_space_size);

    int  w   = resolution.x;
    int  h   = resolution.y;
    auto bx  = tile_splatter.m_blocks.x;
    auto by  = tile_splatter.m_blocks.y;
    auto TWH = luisa::make_uint2(
        (unsigned int)((w + bx - 1u) / bx),
        (unsigned int)((h + by - 1u) / by)
    );

    auto L = 20000000; // max num rendered

    auto d_point_list_keys_unsorted = p_device->create_buffer<luisa::ulong>(L);
    auto d_point_list_unsorted      = p_device->create_buffer<uint>(L);
    auto d_point_list_keys          = p_device->create_buffer<luisa::ulong>(L);
    auto d_point_list               = p_device->create_buffer<uint>(L);
    auto d_ranges                   = p_device->create_buffer<uint>(TWH.x * TWH.y * 2);

    size_t sort_temp_size;
    dp.enable_radix_sort<luisa::ulong, luisa::uint>(*p_device);
    dp.radix_sort<luisa::ulong, luisa::uint>(
        sort_temp_size,
        d_point_list_keys_unsorted,
        d_point_list_unsorted,
        d_point_list_keys,
        d_point_list,
        L, 64
    );

    LUISA_INFO("sort_temp_size: {0}", sort_temp_size);

    if (sort_temp_size > temp_space_size)
    {
        temp_space_size = sort_temp_size;
    }

    // maximum temp space size
    auto temp_storage = p_device->create_buffer<uint>(temp_space_size);

    auto d_img   = p_device->create_buffer<float>(w * h * 3);
    auto d_radii = p_device->create_buffer<int>(P);

    lcgs::GSSplatForwardOutputProxy output{
        .height     = h,
        .width      = w,
        .target_img = d_img,
        .radii      = d_radii
    };

    lcgs::GSTileSplatterAccelProxy accel{
        .temp_storage             = temp_storage,
        .tiles_touched            = d_tiles_touched,
        .point_offsets            = d_points_offset,
        .point_list_keys_unsorted = d_point_list_keys_unsorted,
        .point_list_unsorted      = d_point_list_unsorted,
        .point_list_keys          = d_point_list_keys,
        .point_list               = d_point_list,
        .ranges                   = d_ranges
    };

    lcgs::GSTileSplatterInputProxy input{
        .num_gaussians    = P,
        .bg_color         = luisa::make_float3(0.0f, 0.0f, 0.0f),
        .means_2d         = d_means_2d,
        .depth_features   = d_depth_features,
        .conic            = d_covs_2d,
        .color_features   = d_color,
        .opacity_features = d_opacity,
    };

    int num_rendered = tile_splatter.forward(*p_device, *p_stream, accel, input, output);
    LUISA_INFO("num_rendered: {}", num_rendered);

    luisa::vector<float> h_img(w * h * 3);
    luisa::vector<int>   h_radii(P);

    (*p_stream) << d_img.copy_to(h_img.data())
                << d_radii.copy_to(h_radii.data())
                << luisa::compute::synchronize();

    // 3 x H x W -> W x H x 3
    luisa::vector<uint8_t> h_img_rgb(w * h * 3); // Change to uint8_t for proper image format
    // Fill with red (255,0,0)
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            int pixel_idx = (i * w + j) * 3;
            // int idx = i * w + j;
            int idx                  = (h - i - 1) * w + j;
            h_img_rgb[pixel_idx + 0] = h_img[0 * h * w + idx] * 255; // R
            h_img_rgb[pixel_idx + 1] = h_img[1 * h * w + idx] * 255; // G
            h_img_rgb[pixel_idx + 2] = h_img[2 * h * w + idx] * 255; // B
        }
    }
    auto img_name = "gs_splat.png";
    stbi_write_png(img_name, w, h, 3, h_img_rgb.data(), 0);
    LUISA_INFO("result saved in {}", img_name);

    return 0;
}