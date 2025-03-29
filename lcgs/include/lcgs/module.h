#pragma once
/**
 * @file gs_module.h
 * @brief The Gaussian Module Base for common callables
 * @author sailing-innocent
 * @date 2024-11-25
 */

#include "lcgs/core/runtime.h"

namespace lcgs
{

class GSModule : public LuisaModule
{
public:
    uint2 m_blocks = { 16u, 16u };

protected:
    UCallable<float(float, uint)> mp_ndc2pix;
    UCallable<void(float2, int, uint2&, uint2&, uint2, uint2)> mp_get_rect;
    virtual void compile_callables(Device& device) noexcept;
};

} // namespace lcgs