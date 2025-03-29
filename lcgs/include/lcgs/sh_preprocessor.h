#pragma once
/**
 * @file sh_processor
 * @brief The SH Processor (SH + camera -> Color)
 * @author sailing-innocent
 * @date 2024-11-24
 */

#include "lcgs/config.h"
#include "lcgs/core/runtime.h"
#include "lcgs/util/camera.h"

namespace lcgs
{

struct GPUPointsProxy {
    int                               N      = 0;
    int                               stride = 3;
    luisa::compute::BufferView<float> pos;
};

class LCGS_API SHProcessor : public LuisaModule
{
    bool _enabled = false;

public:
    SHProcessor()  = default;
    ~SHProcessor() = default;
    void create(Device& device) noexcept;
    void process(
        CommandList&      cmdlist,
        GPUPointsProxy    proxy,
        lcgs::Camera&     camera,
        BufferView<float> sh,
        BufferView<float> color,
        int channel = 3, int level = 3
    ) noexcept;

private:
    void compile(Device& device) noexcept;

    UCallable<float3(  // output: color
        int, int, int, // P, channel, deg
        float3,        // cam_pos
        Buffer<float>, // xyz
        Buffer<float>  // sh
    )>
        mp_compute_color_from_sh;

    U<Shader<1, int, int, int, // P, channel, deg
             float3,           // cam_pos
             Buffer<float>,    // xyz
             Buffer<float>,    // sh
             // ouitput
             Buffer<float> // color
             >>
        shad_sh_process;
};

} // namespace lcgs