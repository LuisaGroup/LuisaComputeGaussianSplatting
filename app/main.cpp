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

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char** argv)
{
    luisa::log_level_info();
    luisa::uint2 resolution = luisa::make_uint2(1600, 1063);

    constexpr auto default_ply_path = "D:/ws/data/assets/samples/gsplat.ply";
    auto           ply_path         = std::filesystem::path{ default_ply_path };
    std::string    backend          = "dx";
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

    lcgs::GaussiansData data;
    lcgs::read_gs_ply(data, ply_path);

    int P = data.num_gaussians;
    LUISA_INFO("num_gaussians: {}", P);

    return 0;
}