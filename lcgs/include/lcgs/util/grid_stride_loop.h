#pragma once
/**
 * @file grid_stride_loop.h
 * @author sailing-innocent
 * @brief The Grid Stride Loop Semantic
 * @date 2023-08-18
 */

#include "luisa/dsl/sugar.h"

namespace lcgs
{

template <typename F>
concept GridStrideLoopF = requires(F) {
    std::is_invocable_v<F, int>;
};

struct DefaultOutRangeF {
    void operator()(luisa::compute::Int) {};
};

struct DefaultAlwaysF {
    void operator()(luisa::compute::Int) {};
};

template <
    GridStrideLoopF InRangeF,
    GridStrideLoopF AlwaysF   = DefaultAlwaysF,
    GridStrideLoopF OutRangeF = DefaultOutRangeF>
inline void grid_stride_loop(
    const ::luisa::compute::Int& begin,
    const ::luisa::compute::Int& end,
    const ::luisa::compute::Int& step,
    InRangeF&&                   f,
    AlwaysF&&                    always_f    = DefaultAlwaysF{},
    OutRangeF&&                  out_range_f = DefaultOutRangeF{}
) noexcept
{
    using namespace ::luisa::compute;

    Int  k           = dispatch_x();
    Int  stride      = dispatch_size_x();
    auto iceil       = [](auto num, auto denom) { return (num + denom - 1) / denom; };
    Int  total       = iceil(abs(end - begin), abs(step));
    Int  upper_bound = iceil(total, stride);
    Int  cur         = 0;
    $while(cur < upper_bound)
    {
        Int i = begin + step * k;
        k += stride;
        cur += 1;
        // if i is in range
        auto builder = $if(i < end) { f(i); };
        // when OutRangeF == DefaultOutRangeF, remove the code.
        if constexpr (!std::is_same_v<OutRangeF, DefaultOutRangeF>)
        {
            std::move(builder).else_([&] {
                out_range_f(i);
            });
        }
        // when AlwaysF == DefaultAlwaysF, remove the code.
        if constexpr (!std::is_same_v<AlwaysF, DefaultAlwaysF>)
        {
            always_f(i);
        }
    };
}

template <GridStrideLoopF InRangeF, GridStrideLoopF AlwaysF = DefaultAlwaysF, GridStrideLoopF OutRangeF = DefaultOutRangeF>
inline void grid_stride_loop(
    const ::luisa::compute::Int& begin, const ::luisa::compute::Int& count,
    InRangeF&& f, AlwaysF&& always_f = DefaultAlwaysF{}, OutRangeF&& out_range_f = DefaultOutRangeF{}
) noexcept
{
    grid_stride_loop(begin, begin + count, 1, std::forward<InRangeF>(f), std::forward<AlwaysF>(always_f), std::forward<OutRangeF>(out_range_f));
}

template <GridStrideLoopF InRangeF, GridStrideLoopF AlwaysF = DefaultAlwaysF, GridStrideLoopF OutRangeF = DefaultOutRangeF>
inline void grid_stride_loop(const ::luisa::compute::Int& count, InRangeF&& f, AlwaysF&& always_f = DefaultAlwaysF{}, OutRangeF&& out_range_f = DefaultOutRangeF{}) noexcept
{
    grid_stride_loop(0, count, std::forward<InRangeF>(f), std::forward<AlwaysF>(always_f), std::forward<OutRangeF>(out_range_f));
}

} // namespace lcgs