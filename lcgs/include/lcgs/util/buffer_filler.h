#pragma once
/**
 * @file helper/buffer_filler
 * @author sailing-innocent
 * @date 2023-12-27
 * @brief The Buffer Filler Reimplementation
 */

#include "lcgs/config.h"
#include "lcgs/core/runtime.h"
#include "grid_stride_loop.h"

namespace lcgs
{

class LCGS_API BufferFiller
{
    template <typename T>
    using Buffer = luisa::compute::Buffer<T>;
    template <typename T>
    using BufferView = luisa::compute::BufferView<T>;
    template <size_t I, typename... Ts>
    using Shader      = luisa::compute::Shader<I, Ts...>;
    using Device      = luisa::compute::Device;
    using CommandList = luisa::compute::CommandList;
    using uint        = luisa::uint;
    using ulong       = luisa::ulong;
    using float2      = luisa::float2;
    using float3      = luisa::float3;
    using float4      = luisa::float4;
    using uint2       = luisa::uint2;
    using uint3       = luisa::uint3;
    using uint4       = luisa::uint4;

public:
    uint block_size = 256u;

    // clang-format off
// filler_shader macro start
#define filler_shader(basic_type) \
public: \
	auto fill(Device& device, BufferView<basic_type> buffer_view, const basic_type& v) const noexcept {\
		using namespace luisa;\
		using namespace luisa::compute;\
		lazy_compile(device, m_fill_##basic_type, [&](Int count, BufferVar<basic_type> buffer, Var<basic_type> value) {\
			set_block_size(this->block_size);\
			grid_stride_loop(\
				count,\
				[&](Int i) noexcept {\
				buffer.write(i, value);\
			});\
		});\
        return (*m_fill_##basic_type)(buffer_view.size(), buffer_view, v).dispatch(buffer_view.size());\
	}\
protected:\
	mutable U<Shader<1, int, Buffer<basic_type>, basic_type>> m_fill_##basic_type = nullptr;

// filler_shader macro end
    // clang-format on

    filler_shader(int);
    filler_shader(float);
    filler_shader(float2);
    filler_shader(float3);
    filler_shader(float4);
    filler_shader(uint);
    filler_shader(ulong);
    filler_shader(uint2);
    filler_shader(uint3);
    filler_shader(uint4);
}; // namespace inno

} // namespace lcgs