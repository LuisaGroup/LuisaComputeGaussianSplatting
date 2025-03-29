/**
 * @file gs_module.cpp
 * @brief The Gaussian Module
 * @author sailing-innocent
 * @date 2024-11-25
 */

#include "lcgs/module.h"

namespace lcgs
{

void GSModule::compile_callables(Device& device) noexcept
{
    using namespace luisa;
    using namespace luisa::compute;

    mp_ndc2pix = luisa::make_unique<Callable<float(float, uint)>>([](Float v, UInt S) {
        return ((v + 1.0f) * S - 1.0f) * 0.5f;
    });

    mp_get_rect = luisa::make_unique<Callable<void(float2, int, uint2&, uint2&, uint2, uint2)>>(
        [](
            Float2 p,
            Int max_radius,
            UInt2& rect_min,
            UInt2& rect_max,
            UInt2 blocks, UInt2 grids) {
            // clamp
            rect_min = make_uint2(
                clamp(UInt((p.x - max_radius) / blocks.x), Var(0u), grids.x - 1),
                clamp(UInt((p.y - max_radius) / blocks.y), Var(0u), grids.y - 1));
            rect_max = make_uint2(
                clamp(UInt(p.x + max_radius + blocks.x - 1) / blocks.x, Var(0u), grids.x - 1),
                clamp(UInt(p.y + max_radius + blocks.y - 1) / blocks.y, Var(0u), grids.y - 1));
        });
}

} // namespace lcgs